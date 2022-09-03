import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np 
import matplotlib.pyplot as plt


def transform_get_labels(data, norm='coord', ncategs=10, precalc_stats=None):
    
    ### co-ordinate-wise normalization
    if(precalc_stats != None):
        max_day = precalc_stats[0]
        min_day = precalc_stats[1]
        norm_den = precalc_stats[2]
        rainfall_normalized = (data-min_day)/norm_den
        slack = precalc_stats[3]

        rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)+slack
        ### log_transform
        rainfall_stats = np.log(rainfall_stats)
        bins = precalc_stats[4]
        categs = np.digitize(rainfall_stats, bins, right=False)

        return rainfall_stats, precalc_stats, categs

    elif(norm == 'coord' and precalc_stats == None):
        max_day = np.max(data, axis=0)
        min_day = np.min(data, axis=0)
        norm_den = max_day-min_day
        eps = 1e-4
        norm_den = np.where(norm_den == 0, eps, norm_den)
        norm_stats = [max_day, min_day, norm_den]

    elif(norm == 'all' and precalc_stats == None):
    ### normalization over all pixels
        max_day = np.max(data)
        min_day = np.min(data)
        norm_den = max_day-min_day
        norm_stats = [max_day, min_day, norm_den]
        
    rainfall_normalized = (data-min_day)/norm_den
    slack = np.min(np.where(rainfall_normalized>0, rainfall_normalized, 1))*1e-1

    norm_stats.append(slack)

    rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)+slack
    ### log_transform
    rainfall_stats = np.log(rainfall_stats)
    nbins = ncategs
    _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
    ### include max value in bin
    bins[-1]+=1
    bins[0]-= 1
    norm_stats.append(bins)
    categs = np.digitize(rainfall_stats, bins, right=False)
    return rainfall_normalized, norm_stats, categs

class ChirpsDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.labels = torch.FloatTensor(labels)
        self.imgs = torch.FloatTensor(imgs[:, np.newaxis, :, :])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #image = read_image(img_path)
        image = self.imgs[idx]
        label = self.labels[idx]
        #no_pred_mask = torch.ones_like(label)
        if self.transform:
            image = self.transform(image)
        return image, label



def get_sampler(keys, sampler_type="uniform", k_val=1e-2, num_quantiles=10):
    
    if(sampler_type == 'uniform'):
        weights = np.ones(len(keys))
    
    elif(sampler_type == 'categorical'):
        if np.isinf(k_val):
            weights = np.ones(len(keys))
        else:
            categs, counts = np.unique(keys, return_counts=True)
            weights = np.zeros_like(keys)
            for i in range(len(categs)):
                weights+=np.where(keys==categs[i], 1.0/(counts[i]+ np.sum(counts)*k_val), 0)    
            ### add normalization
            weights = weights/np.sum(weights)
            print("Min/Max Sampler Weights:", np.max(weights), np.min(weights))

    elif(sampler_type == 'ranked'):
        if np.isinf(k_val):
            weights = np.ones(len(keys))
        else:
            ranks = np.argsort(np.argsort(-1 * keys))
            weights = 1.0 / (k_val * len(keys) + ranks)
            ### add normalization
            weights = weights/np.sum(weights)
            print("Min/Max Sampler Weights:", np.max(weights), np.min(weights))

    elif(sampler_type == 'ranked'):
        if np.isinf(k_val):
            weights = np.ones(len(keys))
        else:
            ranks = np.argsort(np.argsort(-1 * keys))
            weights = 1.0 / (k_val * len(keys) + ranks)
            ### add normalization
            weights = weights/np.sum(weights)
            print("Min/Max Sampler Weights:", np.max(weights), np.min(weights))
    
    elif(sampler_type =='quantiles'):
        list_quantiles = []
        q_ranks = np.ones(len(keys))*11
        
        for i in range(num_quantiles):
            q = np.quantile(keys, (i*1.0)/num_quantiles)
            q_ranks = np.where(q_ranks >= q, q_ranks-1)
        
        print("Min/Max Sampler Ranks:", np.max(q_ranks), np.min(q_ranks))
        weights = 1.0 / (k_val * len(keys) + q_ranks)
        ### add normalization
        weights = weights/np.sum(weights)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(keys), replacement=True)
    return sampler


def prep_dataloaders(data_path, train_val_split = 0.9, seed=129, ncategs=10, norm_type='coord', bs = 16, workers=2, train_sampler=True):
    
    data = np.load(data_path)
    train_len = len(data)*train_val_split
    val_len = len(data) - len(data)*train_val_split
    train, val = random_split(data, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    train_data, train_stats, train_categs = transform_get_labels(train, norm_type, ncategs)
    val_data, _, val_categs = transform_get_labels(val, norm_type, ncategs, train_stats)

    chirps_train_dataset = ChirpsDataset(train_data, train_categs)
    chirps_val_dataset = ChirpsDataset(val_data, val_categs)

    if train_sampler==True:
        weighted_sampler = get_sampler(train_categs, sampler_type='categorical', k_val=1e-1)
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, shuffle=True, sampler=weighted_sampler)
    else:
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, shuffle=True)

    val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, shuffle=False)

    return train_dataloader, val_dataloader, train_stats



    

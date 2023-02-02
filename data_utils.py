import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np 
import matplotlib.pyplot as plt
import random
import torchvision.utils as vutils



def transform_get_labels(data, norm='log_transform', ncategs=10, precalc_stats=None):
    
    ### co-ordinate-wise normalization
    slack = -1
    if(norm == 'coord' and precalc_stats == None):
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
    
    elif(norm == 'log_transform'):
    ### log + co-ordinate-wise normalization
        #slack = np.min(np.where(data>0, data, 1))*1e-1
        #print(slack)
        slack=1
        data = np.log(data+slack)
        #min_day = np.min(data, axis=0)
        #max_day = np.max(data, axis=0)
        min_day = np.min(data)
        max_day = np.max(data)
        #norm_den = np.std(data, axis=0)
        norm_den = max_day-min_day
        #norm_den = np.where(norm_den == 0, 1, norm_den)
        norm_stats = [max_day, min_day, norm_den]
        
    rainfall_normalized = (data-min_day)/norm_den
    rainfall_stats = rainfall_normalized.mean(axis=2).mean(axis=1)
    # #print(len(indices_max), len(indices_mean))


    if(norm != 'log_transform'):
        slack = np.min(np.where(rainfall_normalized>0, rainfall_normalized, 1))*1e-1
        rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)+slack
        rainfall_stats = np.log(rainfall_stats)

    

    #print(slack)
    norm_stats.append(slack)

    ### log_transform
    nbins = ncategs
    _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
    ### include max value in bin
    bins[-1]+=1
    bins[0]-= 1
    norm_stats.append(bins)
    categs = np.digitize(rainfall_stats, bins, right=False)-1
    return rainfall_normalized, norm_stats, categs

class ChirpsDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.labels = torch.LongTensor(labels)
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



def get_sampler(keys, sampler_type="uniform", k_val=1e-6, num_quantiles=10):
    
    if(sampler_type == 'uniform'):
        weights = np.ones(len(keys))
    
    elif(sampler_type == 'categorical'):
        if np.isinf(k_val):
            weights = np.ones(len(keys))
        else:
            categs, counts = np.unique(keys, return_counts=True)
            weights = np.zeros_like(keys)*1.0
            for i in range(len(categs)):
                my_array = np.where(keys==categs[i], 1.0/((counts[i])+ np.sum(counts)*k_val), 0)
                #print(categs[i], counts[i], 1.0/((counts[i])+ np.sum(counts)*k_val))
                weights += my_array    
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


def prep_dataloaders(data_path, ncategs=10, norm_type='coord', bs = 32, workers=2, sampler=True):
    
    data = np.load(data_path)
    train_len = 30*data.shape[1]
    test_len = 10*data.shape[1]
    val_len = data.shape[1]
    
    ### reshape
    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
    data, data_stats, data_categs = transform_get_labels(data, norm_type, ncategs)
    train_data, train_categs = data[0:train_len], data_categs[0:train_len]

    val_data, val_categs = data[train_len: train_len+val_len], data_categs[train_len: train_len+val_len]
    test_data, test_categs = data[train_len+val_len: train_len+val_len+test_len], data_categs[train_len+val_len: train_len+val_len+test_len]

    #remove zeros from training
    train_rainfall_mean = train_data.mean(axis=2).mean(axis=1)
    non_zero = [i for i in range(len(train_rainfall_mean)) if train_rainfall_mean[i] > 0]
    zero = [i for i in range(len(train_rainfall_mean)) if train_rainfall_mean[i] == 0]
    test_rainfall_mean = test_data.mean(axis=2).mean(axis=1)
    print(np.sum(np.where(test_rainfall_mean < 0, 1, 0)), "yo")

    combined_indices = (non_zero + zero[:5])
    combined_indices.sort()
    train_data, train_categs = train_data[combined_indices, :, :], train_categs[combined_indices]

    ### get train samples
    categ0 = [i for i in range(len(train_categs)) if train_categs[i]==0]
    categ1 = [i for i in range(len(train_categs)) if train_categs[i]==1]
    categ2 = [i for i in range(len(train_categs)) if train_categs[i]==2]
    categ3 = [i for i in range(len(train_categs)) if train_categs[i]==3]
    categ4 = [i for i in range(len(train_categs)) if train_categs[i]==4]

    random.shuffle(categ0)
    train_data0 = train_data[categ0[:8]]

    random.shuffle(categ1)
    train_data1 = train_data[categ1[:8]]

    random.shuffle(categ2)
    train_data2 = train_data[categ2[:8]]

    random.shuffle(categ3)
    train_data3 = train_data[categ3[:8]]

    random.shuffle(categ4)
    train_data4 = train_data[categ4[:8]]
    grid = np.concatenate([train_data0, train_data1, train_data2, train_data3, train_data4])
    grid_torch = torch.Tensor(grid).unsqueeze(1)

    img = vutils.make_grid(grid_torch, padding=2, normalize=True)[:1, :, :].squeeze(0)

    plt.figure(figsize=(8,8))
    plt.imshow(img)#, cmap='gray')
    plt.savefig('/home/moulikc2/expose/Climate Generation/training_samples.png')
    plt.close()



    chirps_train_dataset = ChirpsDataset(train_data, train_categs)
    chirps_val_dataset = ChirpsDataset(val_data, val_categs)
    chirps_test_dataset = ChirpsDataset(test_data, test_categs)
    

    if sampler==True:
        weighted_sampler = get_sampler(train_categs, sampler_type='categorical', k_val=1e-5)
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)
        print("Training Length", len(chirps_train_dataset))

        weighted_sampler = get_sampler(val_categs, sampler_type='categorical', k_val=1e-4)
        val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)

        weighted_sampler = get_sampler(test_categs, sampler_type='categorical', k_val=1e-5)
        test_dataloader = DataLoader(chirps_test_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)
        print("Test Length", len(chirps_test_dataset))

    else:
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, shuffle=True)
        val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, shuffle=False)
        test_dataloader = DataLoader(chirps_test_dataset, batch_size=bs, num_workers=workers, shuffle=False)



    return train_dataloader, val_dataloader, test_dataloader, data_stats



    

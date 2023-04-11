import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np 
import matplotlib.pyplot as plt
import random
import torchvision.utils as vutils
from sklearn.utils import shuffle 



def transform_get_labels_wrf(data, norm_type='global', norm='min_max', ncategs=5, log=True):
    
    ### co-ordinate-wise normalization
    data = data[:, 19:83, 19:83]
    slack = -1
    if norm == 'min_max' and log:
        log_sum = True 
        log_slack=1
    else:
        log_sum = False

    print(norm_type, norm)
 

    if norm_type == 'coord':
    ### normalization over time, spatial co-ordinate wise
        if norm == 'min_max':
            max_day = np.max(data, axis=0)
            min_day = np.min(data, axis=0)
            norm_den = max_day-min_day
            eps = 1e-4
            norm_den = np.where(norm_den == 0, eps, norm_den)
            norm_stats = [max_day, min_day, norm_den]
        
        elif(norm == 'normalize'):
            min_day = np.mean(data, axis=0)
            #min_day = np.min(data)
            norm_den = np.std(data, axis=0)
            eps = 1e-4
            norm_den = np.where(norm_den == 0, eps, norm_den)
            norm_stats = [min_day, norm_den]

        elif(norm == 'log_transform_min_max'):
            slack=1
            data = np.log(data+slack)
            min_day = np.min(data, axis=0)
            max_day = np.max(data, axis=0)
            norm_den = max_day-min_day
            norm_stats = [max_day, min_day, norm_den]

    elif(norm_type == 'global'):
    ### normalization over all pixels
        if norm == 'min_max':
            max_day = np.max(data)
            min_day = np.min(data)
            norm_den = max_day-min_day
            norm_stats = [max_day, min_day, norm_den]

        elif(norm == 'normalize'):
        ### normalization over all pixels
            min_day = np.mean(data)
            norm_den = np.std(data)
            eps = 1e-4
            norm_den = np.where(norm_den == 0, eps, norm_den)
            norm_stats = [min_day, norm_den]

        elif(norm == 'log_transform_min_max'):
        ### normalization over all pixels
            slack=1
            data = np.log(data+slack)
            min_day = np.min(data)
            max_day = np.max(data)
            norm_den = max_day-min_day
            norm_stats = [max_day, min_day, norm_den]
        
    rainfall_normalized = (data-min_day)/norm_den
    rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)
    rainfall_normalized = (rainfall_normalized-0.5)*2

    if log_sum==True:
        print("Min/Max Total Rainfall:", np.min(rainfall_stats), np.max(rainfall_stats))
        if np.min(rainfall_stats) < 1:
            rainfall_stats+=log_slack
            rainfall_stats = np.log(rainfall_stats)

    #print(slack)
    norm_stats.append(slack)

    ### log_transform
    nbins = ncategs
    _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
    plt.xlabel('Amount of Rainfall')
    plt.ylabel('Counts')
    plt.savefig('hist_monsoon_wrf.png')
    ### include max value in bin
    bins[-1]+=1
    bins[0]-= 1
    norm_stats.append(bins)
    categs = np.digitize(rainfall_stats, bins, right=False)-1
    print(np.unique(categs), len(categs))
    result = [np.count_nonzero(rainfall_stats[categs == j]) for j in range(nbins)]
    print(result)
    X, y = shuffle(rainfall_normalized, categs, random_state=0)
    print(X.shape, y.shape)
    return X, norm_stats, y

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
    

class pix2pixDataset(Dataset):
    def __init__(self, imgs, maps, transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.labels = torch.FloatTensor(maps[:, np.newaxis, :, :])
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
    

class pix2pixDataset_stoch(Dataset):
    def __init__(self, imgs, maps, noise, transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.labels = torch.FloatTensor(maps[:, np.newaxis, :, :])
        self.imgs = torch.FloatTensor(imgs[:, np.newaxis, :, :])
        self.noise = torch.FloatTensor(noise[:, np.newaxis, :, :])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #image = read_image(img_path)
        image = self.imgs[idx]
        label = self.labels[idx]
        noise = self.noise[idx]
        #no_pred_mask = torch.ones_like(label)
        if self.transform:
            image = self.transform(image)
        return image, label, noise



def get_sampler(keys, sampler_type="uniform", k_val=1e-6, num_quantiles=10, label_noise=False, noise_ratio=1e-1):
    
    if(sampler_type == 'uniform'):
        weights = np.ones(len(keys))
    
    elif(sampler_type == 'categorical'):
        if np.isinf(k_val):
            weights = np.ones(len(keys))
        else:
            categs, counts = np.unique(keys, return_counts=True)
            weights = np.zeros_like(keys)*1.0
            categs.sort()
            for i in categs:
                my_array = np.where(keys==i, 1.0/((counts[i])+ np.sum(counts)*k_val), 0)
                #print(categs[i], counts[i], 1.0/((counts[i])+ np.sum(counts)*k_val))
                weights += my_array
                if label_noise==True and i>0:
                    noise = np.where(keys==i-1, (1.0*noise_ratio)/((counts[i-1])+ np.sum(counts)*k_val), 0)
                    weights += noise        
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


def prep_dataloaders(data_path, ncategs=5, norm_type='global', bs = 64, workers=2, sampler=True, norm="log_transform_min_max"):
    
    data = np.load(data_path)
    total_samples = data.shape[0]
    train_len = int(0.8*total_samples)
    val_len = int(0.05*total_samples)

    test_len = total_samples-train_len-val_len
    
    ### reshape
    if ("WRF" not in data_path):
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
    #data, data_stats, data_categs = transform_get_labels(data, norm_type, norm, ncategs)
    data, data_stats, data_categs = transform_get_labels_wrf(data, norm_type, norm, ncategs, True)
    train_data, train_categs = data[0:train_len], data_categs[0:train_len]

    val_data, val_categs = data[train_len: train_len+val_len], data_categs[train_len: train_len+val_len]
    test_data, test_categs = data[train_len+val_len: train_len+val_len+test_len], data_categs[train_len+val_len: train_len+val_len+test_len]

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

    print(train_data.shape)

    img = vutils.make_grid(grid_torch, padding=2, normalize=True)[:1, :, :].squeeze(0)

    plt.figure(figsize=(8,8))
    plt.imshow(img)#, cmap='gray')
    plt.savefig('/home/moulikc2/expose/Climate Generation/training_samples_wrf.png')
    plt.close()



    chirps_train_dataset = ChirpsDataset(train_data, train_categs)
    chirps_val_dataset = ChirpsDataset(val_data, val_categs)
    chirps_test_dataset = ChirpsDataset(test_data, test_categs)
    

    if sampler==True:
        weighted_sampler = get_sampler(train_categs, sampler_type='categorical', k_val=1e-4)
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)
        print("Training Length", len(chirps_train_dataset))

        weighted_sampler = get_sampler(val_categs, sampler_type='categorical', k_val=1e-4)
        val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)

        weighted_sampler = get_sampler(test_categs, sampler_type='categorical', k_val=1e-4)
        test_dataloader = DataLoader(chirps_test_dataset, batch_size=bs, num_workers=workers, sampler=weighted_sampler)
        print("Test Length", len(chirps_test_dataset))

    else:
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, shuffle=True)
        val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, shuffle=False)
        test_dataloader = DataLoader(chirps_test_dataset, batch_size=bs, num_workers=workers, shuffle=False)



    return train_dataloader, val_dataloader, test_dataloader, data_stats



def get_pix2pix_maps_wrf(data, norm='min_max', patch_dim=8, ncategs=5, map = 'avg'):

    data = data[:, 19:83, 19:83]
    
    if norm == 'min_max':
        max_day = np.max(data)
        min_day = np.min(data)
        norm_den = max_day-min_day
        norm_stats = [max_day, min_day, norm_den]

    elif(norm == 'normalize'):
    ### normalization over all pixels
        min_day = np.mean(data)
        norm_den = np.std(data)
        eps = 1e-4
        norm_den = np.where(norm_den == 0, eps, norm_den)
        norm_stats = [min_day, norm_den]

    elif(norm == 'log_transform_min_max'):
    ### normalization over all pixels
        slack=1
        data = np.log(data+slack)
        min_day = np.min(data)
        max_day = np.max(data)
        norm_den = max_day-min_day
        norm_stats = [max_day, min_day, norm_den]
        
    rainfall_normalized = (data-min_day)/norm_den
    loop_x = rainfall_normalized.shape[1]//patch_dim
    loop_y = rainfall_normalized.shape[1]//patch_dim
    patch_sums = []
    avg_rainfall_map = np.zeros_like(rainfall_normalized)
    conditional_rainfall_map = np.zeros_like(rainfall_normalized)
    for i in range(rainfall_normalized.shape[0]):
        for j in range(loop_x):
            for k in range(loop_y):
                x = j*patch_dim
                y = k*patch_dim
                avg_pool = np.mean(rainfall_normalized[i, x:x+patch_dim, y:y+patch_dim])
                patch_sums.append(avg_pool)
                avg_rainfall_map[i, x:x+patch_dim, y:y+patch_dim] = avg_pool

    ### normalize avg_rainfall_map
    avg_rainfall_map = (avg_rainfall_map - np.min(avg_rainfall_map))/(np.max(avg_rainfall_map) - np.min(avg_rainfall_map))
    nbins = ncategs
    _, bins, _ = plt.hist(patch_sums, bins=nbins)
    #plt.xlabel('Amount of Rainfall')
    #plt.ylabel('Counts')
    #plt.savefig('hist_pix2pix_patches_wrf.png')
    ### include max value in bin
    bins[-1]+=1
    bins[0]-= 1
    norm_stats.append(bins)
    categs = np.digitize(patch_sums, bins, right=False)-1

    count=0
    for i in range(rainfall_normalized.shape[0]):
        for j in range(loop_x):
            for k in range(loop_y):
                conditional_rainfall_map[i, x:x+patch_dim, y:y+patch_dim] = categs[count]
                count+=1

    #X, y = shuffle(rainfall_normalized, conditional_rainfall_map, random_state=0)
    if map == 'avg':
        X1, y1 = shuffle(rainfall_normalized, avg_rainfall_map, random_state=0)
    else:
        X1, y1 = shuffle(rainfall_normalized, conditional_rainfall_map, random_state=0)


    rainfall_stats = X1.sum(axis=2).sum(axis=1)

    ### log_transform
    nbins = ncategs
    _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
    ### include max value in bin
    bins[-1]+=1
    bins[0]-= 1
    norm_stats.append(bins)
    categs_overall = np.digitize(rainfall_stats, bins, right=False)-1
    val_indices = []
    train_indices = set([i for i in range(len(categs_overall))])
    for i in range(nbins):
        
        indices = list(np.where(categs_overall == i)[0])
        random.shuffle(indices)
        val_indices= val_indices + indices[:5]
    
    train_indices= train_indices - set(val_indices)

    #return X, norm_stats, y
    ### normalize between -1 and 1
    X1 = (X1-0.5)*2
    
    ### normalize y_maps
    y1 = (y1 - np.min(y1))/(np.max(y1) - np.min(y1))
    y1 = (y1-0.5)*2

    return X1, norm_stats, y1, list(train_indices), val_indices, categs_overall



def prep_dataloaders_pix2pix(data_path, bs = 64, workers=2, ncategs=5, sampler=False, norm="log_transform_min_max", patch_dim=8, map='avg', add_noise=True):
    
    data = np.load(data_path)

    
    ### reshape
    if ("WRF" not in data_path):
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3])
    #data, data_stats, data_categs = transform_get_labels(data, norm_type, norm, ncategs)
    data, data_stats, maps, rest_indices, val_indices, categs_overall = get_pix2pix_maps_wrf(data, norm, patch_dim, ncategs, map)

    rest_indices = [int(i) for i in rest_indices]
    val_indices = [int(i) for i in val_indices]

    train_len = int(len(rest_indices)*0.9)
    train_data, train_maps, train_categs = data[rest_indices][0:train_len], maps[rest_indices][0:train_len], categs_overall[rest_indices][0:train_len]

    val_data, val_maps = data[val_indices], maps[val_indices]
    test_data, test_maps = data[rest_indices][train_len:], maps[rest_indices][train_len:]

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

    print(train_data.shape)

    img = vutils.make_grid(grid_torch, padding=2, normalize=True)[:1, :, :].squeeze(0)

    plt.figure(figsize=(8,8))
    plt.imshow(img)#, cmap='gray')
    plt.savefig('/home/moulikc2/expose/Climate Generation/training_samples_pix2pix_wrf.png')
    plt.close()

    chirps_train_dataset = pix2pixDataset(train_data, train_maps)
    if add_noise:
        fixed_noise = np.random.normal(size=val_maps.shape)
        chirps_val_dataset = pix2pixDataset_stoch(val_data, val_maps, fixed_noise)
    else:
        chirps_val_dataset = pix2pixDataset_stoch(val_data, val_maps)
    chirps_test_dataset = pix2pixDataset(test_data, test_maps)
    

    if sampler==True:
        raise NotImplementedError

    else:
        train_dataloader = DataLoader(chirps_train_dataset, batch_size=bs, num_workers=workers, shuffle=True)
        val_dataloader = DataLoader(chirps_val_dataset, batch_size=bs, num_workers=workers, shuffle=False)
        test_dataloader = DataLoader(chirps_test_dataset, batch_size=bs, num_workers=workers, shuffle=False)



    return train_dataloader, val_dataloader, test_dataloader, data_stats

    

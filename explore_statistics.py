import os
from statistics import mean
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

sv_path = "/home/moulikc2/expose/Climate Generation/data_chirps/chirps_filtered.npy"
data = np.load(sv_path)
print(data.shape)
norm = 'log_transform_normalize'
ncategs = 10
#normalize_array = (final_array-mean_day)/std_day
if(norm == 'coord'):
    max_day = np.max(data, axis=0)
    min_day = np.min(data, axis=0)
    norm_den = max_day-min_day
    eps = 1e-4
    norm_den = np.where(norm_den == 0, eps, norm_den)
    norm_stats = [max_day, min_day, norm_den]

elif(norm == 'all'):
### normalization over all pixels
    max_day = np.max(data)
    min_day = np.min(data)
    norm_den = max_day-min_day
    norm_stats = [max_day, min_day, norm_den]

elif(norm == 'normalize'):
### normalization over all pixels
    min_day = np.mean(data, axis=0)
    #min_day = np.min(data)
    norm_den = np.std(data, axis=0)
    eps = 1e-4
    norm_den = np.where(norm_den == 0, eps, norm_den)
    #norm_stats = [max_day, min_day, norm_den]

elif(norm == 'log_transform_normalize'):
### normalization over all pixels
    slack = np.min(np.where(data>0, data, 1))*1e-1
    print(slack, np.min(data))
    data = np.log(data+slack)
    min_day = np.min(data, axis=0)
    max_day = np.max(data, axis=0)
    #norm_den = np.std(data, axis=0)
    norm_den = max_day-min_day
    print(np.count_nonzero(norm_den))
    eps = 1e-4
    norm_den = np.where(norm_den == 0, eps, norm_den)
    #norm_stats = [max_day, min_day, norm_den]
    
rainfall_normalized = (data-min_day)/norm_den
print(np.min(data), np.max(data))
print(np.min(rainfall_normalized), np.max(rainfall_normalized))
#slack = np.min(np.where(rainfall_normalized>0, rainfall_normalized, 1))*1e-1
#print(slack)

#norm_stats.append(slack)

rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)#+slack
### log_transform
#rainfall_stats = np.log(rainfall_stats)
nbins = ncategs
_, bins, _ = plt.hist(rainfall_stats, bins=nbins)
print(bins)
### include max value in bin
bins[-1]+=1
bins[0]-= 1

#rainfall_stats = np.log(rainfall_stats)


#print(np.min([rainfall_stats[i] for i in range(len(rainfall_stats))if rainfall_stats[i]>0]))
# nbins = 5
# _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
# ### include max value in bin
# bins[-1]+=1
# bins[0]-= 1e-1
# #print(bins)
plt.savefig('hist_test.png')

#vals = np.random.random(1e8)

#bins = np.linspace(0, 1, nbins+1)
ind = np.digitize(rainfall_stats, bins, right=False)
#print(ind, np.max(ind))
#ind = np.where(ind < 4, 4, ind)
#print(np.unique(ind))

result = [np.count_nonzero(rainfall_stats[ind == j]) for j in range(1, nbins+1)]
#result = [vals[ind == j] for j in range(1, nbins)]
print(result)
    
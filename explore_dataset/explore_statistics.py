import os
from statistics import mean
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sv_path = "/home/moulikc2/expose/Climate Generation/data_chirps/chirps_filtered_monsoon.npy"
data = np.load(sv_path)
s_tuple = data.shape
data = data.reshape(s_tuple[0]*s_tuple[1], s_tuple[2], s_tuple[3])

print(data.shape)
norm = 'log_transform_normalize'
ncategs = 5
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
    #slack = np.min(np.where(data>0, data, 1))*1e-1
    slack=1
    #print(slack, np.min(data))
    old_data = data
    data = np.log(data+slack)
    #min_day = np.min(data, axis=0)
    #max_day = np.max(data, axis=0)
    min_day = np.min(data)
    max_day = np.max(data)
    #print(np.min(max_day), np.max(max_day))
    #norm_den = np.std(data, axis=0)
    norm_den = max_day-min_day
    #print(np.count_nonzero(norm_den))
    eps = 1
    #norm_den = np.where(norm_den == 0, eps, norm_den)
    #norm_stats = [max_day, min_day, norm_den]
    
rainfall_normalized = (data-min_day)/norm_den
#print(np.min(data), np.max(data))
#print(np.min(rainfall_normalized), np.max(rainfall_normalized))
#rainfall_normalized = rainfall_normalized.reshape(rainfall_normalized.shape[0]*rainfall_normalized.shape[1]*rainfall_normalized.shape[2])
#print(rainfall_normalized.shape)
#slack = np.min(np.where(rainfall_normalized>0, rainfall_normalized, 1))*1e-1
#print(slack)
rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)#+slack
non_zero = [i for i in range(len(rainfall_stats)) if rainfall_stats[i] > 0]
rainfall_stats = rainfall_stats[non_zero]
#norm_stats.append(slack)
#rainfall_stats_max = rainfall_normalized.max(axis=2).max(axis=1)#+slack
#rainfall_stats_mean = rainfall_normalized.mean(axis=2).mean(axis=1)#+slack

#indices_max = [ind for ind in range(len(rainfall_stats_max)) if rainfall_stats_max[ind]>0.01]
#indices_mean = [ind for ind in range(len(rainfall_stats_mean)) if rainfall_stats_mean[ind]>0.001]
# #print(len(indices_max), len(indices_mean))

#final_indices = list(set(indices_max).intersection(set(indices_mean)))
# #final_indices = indices_max
#rainfall_stats = rainfall_stats_mean[final_indices]
#rainfall_stats = rainfall_stats_max[indices_max]

print(rainfall_stats.shape)

#print(np.min(rainfall_stats)/(128*128))
#print(rainfall_stats.shape)
#rainfall_stats =  rainfall_normalized.reshape(rainfall_normalized.shape[0]* rainfall_normalized.shape[1]*rainfall_normalized.shape[2])
# print(np.min(rainfall_stats), np.max(rainfall_stats))
# check_val_0 = np.sum(np.where(rainfall_stats>0.1,1.0,0.0))/len(rainfall_stats)
# check_val_1 = np.sum(np.where(rainfall_stats<0.1,1.0,0.0))/len(rainfall_stats)
# check_val_2 = np.sum(np.where(rainfall_stats<0.01,1.0,0.0))/len(rainfall_stats)
# check_val_3 = np.sum(np.where(rainfall_stats<0.001,1.0,0.0))/len(rainfall_stats)
# check_val_4 = np.sum(np.where(rainfall_stats<0.0001,1.0,0.0))/len(rainfall_stats)
# check_val_5 = np.sum(np.where(rainfall_stats<0.00001,1.0,0.0))/len(rainfall_stats)
# check_val_6 = np.sum(np.where(rainfall_stats<0.000001,1.0,0.0))/len(rainfall_stats)
# check_val_7 = np.sum(np.where(rainfall_stats<0.0000001,1.0,0.0))/len(rainfall_stats)
# check_val_8 = np.sum(np.where(rainfall_stats<0.00000001,1.0,0.0))/len(rainfall_stats)
# check_val_9 = np.sum(np.where(rainfall_stats<0.000000001,1.0,0.0))/len(rainfall_stats)
#print("check_vals", check_val_0, check_val_1, check_val_2, check_val_3, check_val_4, check_val_5, check_val_6, check_val_7, check_val_8, check_val_9)
### log_transform
#rainfall_stats = np.log(rainfall_stats)
#nbins = ncategs
nbins = 5
sns.set()
#bins=np.logspace(start=-4, stop=0, num=nbins)
_, bins, _ = plt.hist(rainfall_stats, bins=nbins)
#plt.xscale('log')
#plt.yscale('log')
print(bins)
### include max value in bin
bins[-1]+=1
#bins[0]-= 1

#rainfall_stats = np.log(rainfall_stats)


#print(np.min([rainfall_stats[i] for i in range(len(rainfall_stats))if rainfall_stats[i]>0]))
# nbins = 5
# _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
# ### include max value in bin
# bins[-1]+=1
# bins[0]-= 1e-1
# #print(bins)
plt.xlabel('Amount of Rainfall')
plt.ylabel('Counts')
plt.savefig('hist_monsoon.png')

#vals = np.random.random(1e8)

#bins = np.linspace(0, 1, nbins+1)
ind = np.digitize(rainfall_stats, bins, right=False)
# #print(ind, np.max(ind))
# #ind = np.where(ind < 4, 4, ind)
# #print(np.unique(ind))

result = [np.count_nonzero(rainfall_stats[ind == j]) for j in range(1, nbins+1)]
print(np.sum(result))
# #result = [vals[ind == j] for j in range(1, nbins)]
print(result)

plt.close()

#sns.set_style('whitegrid')
swarm_plot = sns.kdeplot(rainfall_stats)
plt.xlabel('Amount of Rainfall')
plt.ylabel('Density')
plt.tight_layout()

#fig = swarm_plot.get_figure()
plt.savefig("kde.png") 
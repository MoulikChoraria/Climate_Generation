import os
from statistics import mean
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sv_path = "/home/moulikc2/expose/Climate Generation/data_WRF/data_WRF_9km/APCP.npy"
data = np.load(sv_path)
#count=0

# for i in range(data.shape[0]):
#     if np.sum(data[i, :, :])==0:
#         count+=1

# print(count)

# s_tuple = data.shape
# data = data.reshape(s_tuple[0]*s_tuple[1], s_tuple[2], s_tuple[3])

print(data.shape)
#norm_type = 'coord'
norm_type= 'coord'
norm = 'min_max' ### min_max, normalize, log_min_max 

log_sum = True
log_slack = 1

print(norm_type, norm)
ncategs = 5
#normalize_array = (final_array-mean_day)/std_day
### reshape data
data = data[:, 19:83, 19:83]
print(data.shape)

if norm_type == 'coord':
### normalization over time, spatial co-ordinate wise

    if norm == 'min_max':
        max_day = np.max(data, axis=0)
        min_day = np.min(data, axis=0)
        norm_den = max_day-min_day
        eps = 1e-4
        norm_den = np.where(norm_den == 0, eps, norm_den)
        #norm_stats = [max_day, min_day, norm_den]
    
    elif(norm == 'normalize'):
        min_day = np.mean(data, axis=0)
        #min_day = np.min(data)
        norm_den = np.std(data, axis=0)
        eps = 1e-4
        norm_den = np.where(norm_den == 0, eps, norm_den)
        #norm_stats = [min_day, norm_den]

    elif(norm == 'log_transform_min_max'):
        slack=1
        old_data = data
        data = np.log(data+slack)
        min_day = np.min(data, axis=0)
        max_day = np.max(data, axis=0)
        norm_den = max_day-min_day
        #print(np.count_nonzero(norm_den))
        eps = 1

elif(norm_type == 'global'):
### normalization over all pixels
    if norm == 'min_max':
        max_day = np.max(data)
        min_day = np.min(data)
        norm_den = max_day-min_day
        #norm_stats = [max_day, min_day, norm_den]

    elif(norm == 'normalize'):
    ### normalization over all pixels
        min_day = np.mean(data)
        norm_den = np.std(data)
        eps = 1e-4
        norm_den = np.where(norm_den == 0, eps, norm_den)
        #norm_stats = [max_day, min_day, norm_den]

    elif(norm == 'log_transform_min_max'):
    ### normalization over all pixels
        #slack = np.min(np.where(data>0, data, 1))*1e-1
        slack=1
        old_data = data
        data = np.log(data+slack)
        min_day = np.min(data)
        max_day = np.max(data)
        norm_den = max_day-min_day
        eps = 1
        #norm_stats = [max_day, min_day, norm_den]
    
rainfall_normalized = (data-min_day)/norm_den


rainfall_stats = rainfall_normalized.sum(axis=2).sum(axis=1)#+slack
if (log_sum == True):
    rainfall_stats+=log_slack
    rainfall_stats = np.log(rainfall_stats)
#non_zero = [i for i in range(len(rainfall_stats)) if rainfall_stats[i] > 0]
#rainfall_stats = rainfall_stats[non_zero]
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

#print(rainfall_stats.shape)

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

plt.xlabel('Amount of Rainfall')
plt.ylabel('Counts')
fig_name = 'hist_monsoon_wrf_' + norm_type + '_' + norm + '_' + str(log_sum)
plt.savefig(fig_name+ '.png')

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
extremes = rainfall_normalized[ind == nbins]
### post-proc
extremes = (extremes - np.min(extremes))/(np.max(extremes) - np.min(extremes))
#sns.set_style('whitegrid')
## print extreme values

f, axarr = plt.subplots(3,3, figsize=(12,12))
axarr[0,0].imshow(extremes[0])
axarr[0,1].imshow(extremes[1])
axarr[0,2].imshow(extremes[2])
axarr[1,0].imshow(extremes[3])
axarr[1,1].imshow(extremes[4])
axarr[1,2].imshow(extremes[5])
axarr[2,0].imshow(extremes[6])
axarr[2,1].imshow(extremes[7])
axarr[2,2].imshow(extremes[8])

fig_name = 'extremes_wrf_' + norm_type + '_' + norm + '_' + str(log_sum) 
plt.savefig(fig_name+'.png') 
# swarm_plot = sns.kdeplot(rainfall_stats)
# plt.xlabel('Amount of Rainfall')
# plt.ylabel('Density')
# plt.tight_layout()

# #fig = swarm_plot.get_figure()
# plt.savefig("kde_wrf_without_log.png") 
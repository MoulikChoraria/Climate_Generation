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
rainfall = np.load(sv_path)

norm_type= 'global'
norm = 'min_max' ### min_max, normalize, log_min_max 

rainfall = rainfall[:, 19:83, 19:83]

sv_path = "/home/moulikc2/expose/Climate Generation/data_WRF/data_WRF_9km/T2.npy"
temp = np.load(sv_path)

temp = temp[:, 19:83, 19:83]

sv_path = "/home/moulikc2/expose/Climate Generation/data_WRF/data_WRF_9km/U10.npy"
u1 = np.load(sv_path)

u1 = u1[:, 19:83, 19:83]

sv_path = "/home/moulikc2/expose/Climate Generation/data_WRF/data_WRF_9km/V10.npy"
v1 = np.load(sv_path)

v1 = v1[:, 19:83, 19:83]


variables = [rainfall, temp, u1, v1]
variables_normalized = []

for data in variables:
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
        
    data_normalized = (data-min_day)/norm_den
    variables_normalized.append(data_normalized)

sns.set()
#bins=np.logspace(start=-4, stop=0, num=nbins)


f, axarr = plt.subplots(3,6, figsize=(18, 9))
random_time = 3179

for i in range(len(variables_normalized)-1):
    if i==2:

        ux = variables_normalized[i][random_time: random_time+6]
        vx = variables_normalized[i+1][random_time: random_time+6]
        z = np.zeros_like(vx)

        print(ux.shape, vx.shape, z.shape)

        extremes = np.stack([z, ux, vx], axis=3)

        axarr[i,0].imshow(extremes[0])
        axarr[i,1].imshow(extremes[1])
        axarr[i,2].imshow(extremes[2])
        axarr[i,3].imshow(extremes[3])
        axarr[i,4].imshow(extremes[4])
        axarr[i,5].imshow(extremes[5])
    
    else:
        extremes = variables_normalized[i][random_time: random_time+6]

        axarr[i,0].imshow(extremes[0])
        axarr[i,1].imshow(extremes[1])
        axarr[i,2].imshow(extremes[2])
        axarr[i,3].imshow(extremes[3])
        axarr[i,4].imshow(extremes[4])
        axarr[i,5].imshow(extremes[5])

fig_name = 'temporal_continuity' + norm_type + '_' + norm
plt.savefig(fig_name+'.png')
plt.tight_layout()
plt.close()
# swarm_plot = sns.kdeplot(rainfall_stats)
# plt.xlabel('Amount of Rainfall')
# plt.ylabel('Density')

# #fig = swarm_plot.get_figure()
# plt.savefig("kde_wrf_without_log.png") 
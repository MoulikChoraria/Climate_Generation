import os
from statistics import mean
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path, yr):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds, yr)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    yr = 1981
    datasets=[]
    for p in paths:
        #print(p, yr)
        if(yr==2000):
            datasets.append(process_one_path(p, yr).to_array().to_numpy())
            yr_stats = process_one_path(p, yr).to_array().to_numpy().squeeze(0)
            rainfall_stats = yr_stats.sum(axis=2).sum(axis=1)
            sns.set()
            plt.plot([(i+1) for i in range(len(rainfall_stats))], rainfall_stats)
            plt.xlabel('Days')
            plt.ylabel('Total Rainfall')
            plt.tight_layout()

            #fig = swarm_plot.get_figure()
            plt.savefig("year_stats_2000.png") 
            break
        yr+=1


    #datasets = [process_one_path(p) for p in paths]
    #print((datasets[-1].to_array().to_numpy()).shape)
    #combined = xr.concat(datasets, dim)
    #print((combined.to_array().to_numpy().squeeze(0)).shape)
    #combined = np.concatenate(datasets, axis=0)
    #return combined
    #print(paths)


def transform_df(df, yr):
    init = '{}-01-01'.format(yr)
    final = '{}-12-31'.format(yr)
    init_date = datetime.strptime(init, '%Y-%m-%d')
    final_date = datetime.strptime(final, '%Y-%m-%d')
    #print(init_date, final_date)

    bbox = [18.0, 74.0, 24.4, 80.4]
    df_sel = df.sel(time=slice(init_date, final_date),
            latitude=slice(bbox[0], bbox[2]),
            longitude=slice(bbox[1], bbox[3]))
    #print(len(df_sel))
    return df_sel


dir_path = "/home/moulikc2/expose/Climate Generation/data_chirps"
# final_array = read_netcdfs(dir_path+'/*.nc', dim='time', transform_func=transform_df)
read_netcdfs(dir_path+'/*.nc', dim='time', transform_func=transform_df)
# # final_array = combined.to_array().to_numpy().squeeze(0)
# print(final_array.shape)

# ###normalize after removing 0 samples
# rainfall_stats = final_array.sum(axis=2).sum(axis=1)#+slack
# non_zero = [i for i in range(len(rainfall_stats)) if rainfall_stats[i] > 0]
# final_array = np.take(final_array, non_zero, axis=0)
# print("Non-zero data size:", final_array.shape)
# sv_path = "/home/moulikc2/expose/Climate Generation/data_chirps/chirps_filtered_monsoon.npy"

# #np.save(sv_path, final_array)

# #sv_path = "/home/moulikc2/expose/Climate Generation/data_chirps/chirps_filtered_monsoon.npy"
# final_array = np.load(sv_path)
# s_tuple = final_array.shape


# reshape_array = final_array.reshape(s_tuple[0]*s_tuple[1], s_tuple[2], s_tuple[3])

# rainfall_stats = reshape_array.sum(axis=2).sum(axis=1)#+slack
# non_zero = [i for i in range(len(rainfall_stats)) if rainfall_stats[i] > 0]
# print(len(non_zero))


# mean_day = np.mean(final_array, axis=0)
# std_day = np.std(final_array, axis=0)
# std_day = np.where(std_day == 0, std_day+1, std_day)

# #normalize_array = (final_array-mean_day)/std_day


# #mean_array = [mean_day for _ in range(final_array.shape[0])]
# #std_array = [std_day for _ in range(final_array.shape[0])]
# #np.stack(arrays, axis=0).shape

# #normalize_array = (final_array-mean_array)/std_array
# slack = 1e-1
# rainfall_stats = final_array.sum(axis=2).sum(axis=1)+slack
# #rainfall_stats = normalize_array.sum(axis=2).sum(axis=1)#+slack
# #sample_non_zero = [rainfall_stats[i] for i in range(len(rainfall_stats)) if rainfall_stats[i] > 0]
# #rainfall_stats = np.log(rainfall_stats)
# #rainfall_stats = np.array(sample_non_zero)
# rainfall_stats = np.log(rainfall_stats)


# #print(np.min([rainfall_stats[i] for i in range(len(rainfall_stats))if rainfall_stats[i]>0]))
# nbins = 5
# _, bins, _ = plt.hist(rainfall_stats, bins=nbins)
# ### include max value in bin
# bins[-1]+=1
# bins[0]-= 1e-1
# #print(bins)
# plt.savefig('hist_log_transform_rem0_5bins.png')

# #vals = np.random.random(1e8)

# #bins = np.linspace(0, 1, nbins+1)
# ind = np.digitize(rainfall_stats, bins, right=False)
# #print(ind, np.max(ind))
# #print(np.unique(ind))

# result = [np.count_nonzero(rainfall_stats[ind == j]) for j in range(1, nbins+1)]
# #result = [vals[ind == j] for j in range(1, nbins)]
# print(result)
    
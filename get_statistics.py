import os
from statistics import mean
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


def transform_df(df):
    init_date = datetime.strptime('1980-01-01', '%Y-%m-%d')
    final_date = datetime.strptime('2021-12-31', '%Y-%m-%d')
    bbox = [18.0, 74.0, 25.0, 81.0]
    df_sel = df.sel(time=slice(init_date, final_date),
            latitude=slice(bbox[0], bbox[2]),
            longitude=slice(bbox[1], bbox[3]))
    return df_sel


dir_path = "dataset"
combined = read_netcdfs(dir_path+'/*.nc', dim='time', transform_func=transform_df)
final_array = combined.to_array().to_numpy().squeeze(0)
print(final_array.shape)

mean_day = np.mean(final_array, axis=0)
std_day = np.std(final_array, axis=0)
std_day = np.where(std_day == 0, std_day+1, std_day)

#mean_array = [mean_day for _ in range(final_array.shape[0])]
#std_array = [std_day for _ in range(final_array.shape[0])]
#np.stack(arrays, axis=0).shape

#normalize_array = (final_array-mean_array)/std_array

rainfall_stats = final_array.sum(axis=2).sum(axis=1)
_, bins, _ = plt.hist(rainfall_stats, bins=10)
### include max value in bin
bins[-1]+=1
print(bins)
plt.savefig('hist_1980.png')

#vals = np.random.random(1e8)
nbins = 10
#bins = np.linspace(0, 1, nbins+1)
ind = np.digitize(rainfall_stats, bins, right=False)
#print(ind-1)

result = [np.count_nonzero(rainfall_stats[ind == j]) for j in range(1, nbins+1)]
#result = [vals[ind == j] for j in range(1, nbins)]
print(result)
    
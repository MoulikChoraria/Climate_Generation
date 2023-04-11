import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils as vutils
from sklearn.utils import shuffle 

sv_path = "/home/moulikc2/expose/Climate Generation/data_WRF/data_WRF_9km/APCP.npy"
data = np.load(sv_path)


print(data.shape)
norm = 'min_max' ### min_max, normalize, log_min_max 
ncategs = 5
patch_dim = 8

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


nbins = ncategs
_, bins, _ = plt.hist(patch_sums, bins=nbins)
plt.xlabel('Amount of Rainfall')
plt.ylabel('Counts')
plt.savefig('hist_pix2pix_patches_wrf.png')
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
X1, y1 = shuffle(rainfall_normalized, avg_rainfall_map, random_state=0)

print(X1.shape, y1.shape)

#return X, norm_stats, y

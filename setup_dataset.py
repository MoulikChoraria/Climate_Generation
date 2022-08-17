import os
import urllib.request


url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/chirps-v2.0.1981.days_p05.nc"
sv_path = "dataset/chirps-v2.0.1981.days_p05.nc"
urllib.request.urlretrieve(url, sv_path)
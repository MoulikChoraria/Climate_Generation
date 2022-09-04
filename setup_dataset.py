import os
import urllib.request


start_year = 1981
base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
sv_path_base = "/home/moulikc2/expose/Climate Generation/data_chirps/"

for i in range(41):
    print(start_year)
    file_name = "chirps-v2.0."+str(start_year)+".days_p05.nc"
    urllib.request.urlretrieve(base_url+file_name, sv_path_base+file_name)
    start_year+=1
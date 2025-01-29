# %%
import os
import json
import h5py
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import imageio
import tqdm
from robomimic.utils.file_utils import create_hdf5_filter_key
import shutil


dataset_path = "/home/ns1254/data_franka/drawer/mixed_o40z5tal3l5taz5.hdf5"




f = h5py.File(dataset_path, "r+")
demos = list(f["data"].keys())

lengths=[]
for demo_name in demos:
    demo=f['data'][demo_name]
    num_samples=demo.attrs['num_samples']
    lengths.append(num_samples)

lengths=np.array(lengths)

print('Number of demos: ', len(demos))
print('Max length: ', np.max(lengths))
print('Min length: ', np.min(lengths))
print('Mean length: ', np.mean(lengths))
  
demo_name = demos[0]
demo=f['data'][demo_name]
num_samples=demo.attrs['num_samples']

demo_name, num_samples,  demo['obs'].keys(), demo.keys()

for demo_name in demos: 
    num_samples=f['data'][demo_name].attrs['num_samples']
    break 


demo_name="demo_1"


demo_no=np.ones(num_samples)*int(demo_name.split("_")[1])
demo_indices = np.arange(num_samples)
demo_indices


for demo_name in demos: 
    num_samples=f['data'][demo_name].attrs['num_samples']
    # demo_info = [f"{demo_name}_{i}" for i in range(num_samples)]
    # f["data"][demo_name].create_dataset(f'obs/uid', data=demo_info)
    demo_nos=np.ones(num_samples)*int(demo_name.split("_")[1])
    demo_indices = np.arange(num_samples)

    f["data"][demo_name].create_dataset(f'obs/demo_no', data=demo_nos)
    f["data"][demo_name].create_dataset(f'obs/index_in_demo', data=demo_indices)


demo_name = demos[0]
demo=f['data'][demo_name]
num_samples=demo.attrs['num_samples']

demo_name, num_samples,  demo['obs'].keys(), demo.keys()

demo_name = demos[10]
print(demo_name)
demo=f['data'][demo_name]
demo['obs/demo_no'][:]
demo['obs/index_in_demo'][:]

f.close()
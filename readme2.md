### coffee task


```
python train.py --config-name=train_diffusion_unet \
     task_name=coffee_d2 \
     dataset_path="/home/ubuntu/dataset_mimicgen/coffee_gfs_109_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=coffee_d2 \
     dataset_path="/home/ubuntu/dataset_mimicgen/coffee_gfs_109_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ubuntu/equidiff/segs/coffee_g40b30/segs_coffee_md40_g40b30_0ind.txt" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=coffee_d2 \
     dataset_path="/home/ubuntu/dataset_mimicgen/coffee_gfs_109_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ubuntu/equidiff/segs/coffee_g40b30/segs_coffee_lof_g40b30_0ind.txt" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=coffee_d2 \
     dataset_path="/home/ubuntu/dataset_mimicgen/coffee_gfs_109_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ubuntu/equidiff/segs/coffee_g40b30/segs_coffee_bed_g40b30_0ind.txt" 
```





### square task

```
python train.py --config-name=train_diffusion_unet \
     task_name=square_d2 \
     dataset_path="/home/ns1254/dataset_mimicgen/gib/square134_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=square_d2 \
     dataset_path="/home/ns1254/dataset_mimicgen/gib/square134_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ns1254/equidiff/segs/square_g40b30/segs_square_md40_g40b30_0ind.txt" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=square_d2 \
     dataset_path="/home/ns1254/dataset_mimicgen/gib/square134_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ns1254/gib/segs/square_g40b30/segs_square_lof40_g40b30_0ind.txt" 
```

```
python train.py --config-name=train_diffusion_unet \
     task_name=square_d2 \
     dataset_path="/home/ns1254/dataset_mimicgen/gib/square134_2_0ind_abs.hdf5" \
     dataset_filter_key="g40b30" \
     segments_toremove_file="/home/ns1254/gib/segs/square_g40b30/segs_square_bed_g40b30_0ind.txt" 
```



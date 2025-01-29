"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from equi_diffpo.workspace.base_workspace import BaseWorkspace
import json

max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 1200,
    'threading_d2': 400,
    'coffee_d2': 1200,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 1200,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
    "real": 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    
     

   
    # segs_file ='/home/ns1254/gib/segs/segs_square_g40f10s10.txt'

    segs_file = "/home/ns1254/gib/segs/segs_square_md40_g40b30_0ind.txt"

    # segs_file = None
    
    segs_toremove = {}
    if segs_file !=None:
        print('using segs file: ', segs_file)
        with open(segs_file, 'r') as f:
            data = json.load(f)

        segs_todo= data['segs_todo']
        dataset_path = data['dataset_path']
        dataset_filter_key = data['dataset_filter_key']
        fn_seg = data['fn_seg']
        data = data['data']
        segs_toremove=json.loads(data)

        print(f"segs_todo: {segs_todo}")
        print(f"dataset_path: {dataset_path}")
        print(f"dataset_filter_key: {dataset_filter_key}")
        print(f"fn_seg: {fn_seg}") 
        print(f"data: {data}")


    if len(segs_toremove)>1:
        print('fresh loading without cache...')
        cfg.task.dataset.use_cache = False

    workspace: BaseWorkspace = cls(cfg, segs_toremove=segs_toremove)
    workspace.run()




if __name__ == "__main__":
    # import sys
    # print("Arguments:", sys.argv)

    main()


# python train.py --config-name=train_equi_diffusion_unet_abs task_name=square_d2 n_demo=100
# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100
# python train.py --config-name=train_diffusion_unet task_name=mug_cleanup_d1 n_demo=100

# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/square134_2_0ind_abs.hdf5
# /home/ns1254/equidiff/data/outputs/2024.12.27/08.14.15_diff_c_square_d2/checkpoints/epoch=0340-test_mean_score=0.220.ckpt
# filter: good
# /home/ns1254/equidiff/data/outputs/2024.12.28/06.20.02_diff_c_square_d2/checkpoints/epoch=0060-test_mean_score=0.140.ckpt
# /home/ns1254/equidiff/data/outputs/2024.12.28/06.20.02_diff_c_square_d2/checkpoints/epoch=0170-test_mean_score=0.160.ckpt


#after max_step =1200
# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/square134_2_0ind_abs.hdf5 dataset_filter_key="good"
# /home/ns1254/equidiff/data/outputs/2025.01.01/23.14.22_diff_c_square_d2/checkpoints/epoch=0320-test_mean_score=0.460.ckpt

# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/square134_2_0ind_abs.hdf5 dataset_filter_key="g40b30"
# /home/ns1254/equidiff/data/outputs/2025.01.03/02.15.19_diff_c_square_d2/checkpoints/epoch=0370-test_mean_score=0.460.ckpt

# with filter.
# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/gib/square134_2_0ind_abs.hdf5 dataset_filter_key="g40b30"


# with filter
# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/square134_2_0ind_abs.hdf5 dataset_filter_key="g40f10s10" 
# /home/ns1254/equidiff/data/outputs/2025.01.07/02.14.32_diff_c_square_d2/checkpoints/epoch=0290-test_mean_score=0.420.ckpt
# without filter
# python train.py --config-name=train_diffusion_unet task_name=square_d2 n_demo=100 dataset_path=/home/ns1254/dataset_mimicgen/square134_2_0ind_abs.hdf5 dataset_filter_key="g40f10s10" 
# /home/ns1254/equidiff/data/outputs/2025.01.08/10.12.07_diff_c_square_d2/checkpoints/epoch=0200-test_mean_score=0.400.ckpt


# real robot training.
# python train.py --config-name=train_diffusion_unet_real task_name=real n_demo=58 dataset_path=/home/ns1254/data_franka/drawer/mixed_o40z5tal3l5taz5.hdf5
# python train.py --config-name=train_diffusion_unet_real task_name=real n_demo=58 dataset_path=/home/ns1254/data_franka/drawer/mixed_o40z5tal3l5taz5.hdf5 dataset_filter_key="g40" 



 
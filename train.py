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
import os 

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
    'can': 800,
    'lift': 400,
    'square': 800,
    "real": 800,
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
    
    segments_toremove_file = cfg.segments_toremove_file

    
    if os.path.exists(segments_toremove_file):
        print('using segs file: ', segments_toremove_file)
        with open(segments_toremove_file, 'r') as f:
            data = json.load(f) 
        data = data['data']
        segs_toremove=json.loads(data) 
    else:
        segs_toremove = {}  
    
    print(f"segs_toremove: {segs_toremove}") 

    if len(segs_toremove)>1:
        print('Fresh loading without cache...')
        cfg.task.dataset.use_cache = False

    workspace: BaseWorkspace = cls(cfg, segs_toremove=segs_toremove)
    workspace.run() 


if __name__ == "__main__": 
    main()

# see readme2.md

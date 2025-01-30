"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

# copied from diff_policy by ns
# python eval.py --checkpoint data/outputs/2025.01.10/01.15.16_diff_c_mug_cleanup_d1/checkpoints/epoch=0150-test_mean_score=0.740.ckpt -o data/mug_eval_output
# /home/ubuntu/equidiff/data/mug_eval_output/eval_log.json
# python eval.py --checkpoint data/outputs/2025.01.09/06.38.02_diff_c_mug_cleanup_d1/checkpoints/epoch=0060-test_mean_score=0.820.ckpt -o data/mug_eval_output2
# /home/ubuntu/equidiff/data/mug_eval_output2/eval_log.json

# 
# python eval.py \
#     --checkpoint /home/ubuntu/equidiff/data/outputs/2025.01.29/11.30.08_diff_c_mug_cleanup_d1/checkpoints/epoch=0105-test_mean_score=0.820.ckpt \
#     -o data/mug_eval_output \
#     --seed 40



import sys
import torch 
import numpy as np 
import random 

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from equi_diffpo.workspace.base_workspace import BaseWorkspace


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed (for all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables benchmark mode for reproducibility



@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-s', '--seed', default=42)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, seed, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f'-------------using seed {seed} -----------')
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
     
    cfg.task.env_runner.test_start_seed = seed 
    set_seed(cfg.task.env_runner.test_start_seed)
    
    # test_seed = cfg.task.env_runner.test_start_seed
    # print('test_seed: ', test_seed)
    
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()

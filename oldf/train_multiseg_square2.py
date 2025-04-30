# %%
import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import mimicgen_environments
import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

import h5py
 
# import mimicgen.utils.file_utils as MG_FileUtils
# import mimicgen.utils.robomimic_utils as RobomimicUtils
 

from dataset_mod_seg import MultiSegmentSequenceDataset

# %%


# %%
config_file = "/home/ubuntu/bc_trans124.json"

# %%


# %%
ext_cfg = json.load(open(config_file, 'r'))
config = config_factory(ext_cfg["algo_name"])
# update config with external json - this will throw errors if
# the external config has keys not present in the base algo config
with config.values_unlocked():
    config.update(ext_cfg)
config.lock()

device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

print('config_file: ', config_file)
print('data file: ', config.train.data) 
print('device: ', device)

# %%

 

# %%
# first set seeds
np.random.seed(config.train.seed)
torch.manual_seed(config.train.seed)

# torch.set_num_threads(2)

# read config to set up metadata for observation modalities (e.g. detecting rgb observations)
ObsUtils.initialize_obs_utils_with_config(config)

# make sure the dataset exists
dataset_path = os.path.expanduser(config.train.data)
if not os.path.exists(dataset_path):
    raise Exception("Dataset at provided path {} not found!".format(dataset_path))
 
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
shape_meta = FileUtils.get_shape_metadata_from_dataset(
    dataset_path=config.train.data,
    all_obs_keys=config.all_obs_keys,
    verbose=True
)

# %%


# %%


# %%
data_path=config.train.data
file=h5py.File(data_path, 'r')
demos=file['data'].keys()
# demos=[b.decode('utf-8') for b in file['mask'][config.train.hdf5_filter_key]]

segs_orginal={}
for demo in demos: 
    segs_orginal[demo]=[ [0, file['data'][demo].attrs['num_samples']-1]  ]

file.close()
len(segs_orginal)

# %%
segs_orginal

# %% [markdown]
# ### subtasks good

# %%
dataset_path = config.train.data
filter_key = config.train.hdf5_filter_key

# %%


# %%
config.train.hdf5_filter_key


# %%
segments_toremove_file = "/home/ubuntu/dataset_mimicgen/segs/segs_index_s2i_square134_2_0ind_abs_s2i_g40b30_s2i.json"

# %%
import json 
print('using segs file: ', segments_toremove_file)
with open(segments_toremove_file, 'r') as f:
    data = json.load(f) 
data = data['data']
segs_toremove=data

# %%
demo_name="demo_0"

segs_toremove[demo_name]

# %%
segs_orginal[demo_name]
 

# %%
def subtract_segments(base, removals):
    start, end = base
    points = [(start, 'start')] + [(r[0], 'remove_start') for r in removals] + \
             [(r[1] + 1, 'remove_end') for r in removals] + [(end, 'end')]
    points.sort()

    active = 0
    result = []
    curr_start = None

    for p, typ in points:
        if typ == 'start' or typ == 'remove_end':
            if active == 0:
                curr_start = p
            active += 1
        elif typ == 'remove_start':
            active -= 1
            if active == 0 and curr_start is not None and curr_start < p:
                result.append([curr_start, p - 1])
        elif typ == 'end':
            if active == 1 and curr_start is not None and curr_start < p:
                result.append([curr_start, p])

    return result 

# %%
# original_seg = [0, 332]
# to_remove = [[10, 20], [30, 100]]


# result = subtract_segments(original_seg, to_remove)
# print(result)


# %%
for demo_name in segs_toremove.keys():
    original_seg = segs_orginal[demo_name]
    to_remove = segs_toremove[demo_name]
    if len(to_remove) > 0:
        result = subtract_segments(original_seg[0], to_remove)
        segs_orginal[demo_name] = result

# %%
segs_toremove['demo_14']

# %%


# %%
segments_map=segs_orginal

# %%


# %%
data_path=config.train.data
obs_keys=shape_meta["all_obs_keys"]

# %%
ds_kwargs = dict(
    hdf5_path=data_path,
    obs_keys=obs_keys,
    dataset_keys=['actions'], 
    frame_stack=10,   
    seq_length=1,
    segments_map=segments_map,     # new parameter
    pad_frame_stack=True,
    pad_seq_length=True,
    get_pad_mask=False, 
    hdf5_cache_mode= None, #'low_dim', #'all',
    hdf5_use_swmr=True 
)

trainset=MultiSegmentSequenceDataset(**ds_kwargs)
len(trainset)

# %%


# %%
config.experiment.env

# %%


# %%
log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

if config.experiment.logging.terminal_output_to_txt:
    # log stdout and stderr to a text file
    logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger

 
envs = OrderedDict()
if config.experiment.rollout.enabled:
    # create environments for validation runs
    env_names = [env_meta["env_name"]]

    if config.experiment.additional_envs is not None:
        for name in config.experiment.additional_envs:
            env_names.append(name)

    for env_name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name, 
            render=False, 
            render_offscreen=config.experiment.render_video,
            use_image_obs=shape_meta["use_images"], 
        )
        env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
        envs[env.name] = env
        print(envs[env.name])

print("")

# setup for a new training run
data_logger = DataLogger(
    log_dir,
    config,
    log_tb=config.experiment.logging.log_tb,
    log_wandb=config.experiment.logging.log_wandb,
)
model = algo_factory(
    algo_name=config.algo_name,
    config=config,
    obs_key_shapes=shape_meta["all_shapes"],
    ac_dim=shape_meta["ac_dim"],
    device=device,
)

# save the config as a json file
with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
    json.dump(config, outfile, indent=4)

 
# load training data
# trainset, validset = TrainUtils.load_data_for_training(
#     config, obs_keys=shape_meta["all_obs_keys"])

validset=None


train_sampler = trainset.get_dataset_sampler() 
print("\n============= Training Dataset =============")
print(trainset)
print("") 

# maybe retreve statistics for normalizing observations
obs_normalization_stats = None
if config.train.hdf5_normalize_obs:
    obs_normalization_stats = trainset.get_obs_normalization_stats()

# initialize data loaders
train_loader = DataLoader(
    dataset=trainset,
    sampler=train_sampler,
    batch_size=config.train.batch_size,
    shuffle=(train_sampler is None),
    num_workers=config.train.num_data_workers,
    drop_last=True
)

if config.experiment.validate:
    # cap num workers for validation dataset at 1
    num_workers = min(config.train.num_data_workers, 1)
    valid_sampler = validset.get_dataset_sampler()
    valid_loader = DataLoader(
        dataset=validset,
        sampler=valid_sampler,
        batch_size=config.train.batch_size,
        shuffle=(valid_sampler is None),
        num_workers=num_workers,
        drop_last=True
    )
else:
    valid_loader = None

# print all warnings before training begins
print("*" * 50)
print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
flush_warnings()
print("*" * 50)
print("")

# main training loop
best_valid_loss = None
best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
last_ckpt_time = time.time()

# number of learning steps per epoch (defaults to a full dataset pass)
train_num_steps = config.experiment.epoch_every_n_steps
valid_num_steps = config.experiment.validation_epoch_every_n_steps

for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
    step_log = TrainUtils.run_epoch(
        model=model,
        data_loader=train_loader,
        epoch=epoch,
        num_steps=train_num_steps,
        obs_normalization_stats=obs_normalization_stats,
    )
    model.on_epoch_end(epoch)

    # setup checkpoint path
    epoch_ckpt_name = "model_epoch_{}".format(epoch)

    # check for recurring checkpoint saving conditions
    should_save_ckpt = False
    if config.experiment.save.enabled:
        time_check = (config.experiment.save.every_n_seconds is not None) and \
            (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
        epoch_check = (config.experiment.save.every_n_epochs is not None) and \
            (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
        epoch_list_check = (epoch in config.experiment.save.epochs)
        should_save_ckpt = (time_check or epoch_check or epoch_list_check)
    ckpt_reason = None
    if should_save_ckpt:
        last_ckpt_time = time.time()
        ckpt_reason = "time"

    print("Train Epoch {}".format(epoch))
    print(json.dumps(step_log, sort_keys=True, indent=4))
    for k, v in step_log.items():
        if k.startswith("Time_"):
            data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
        else:
            data_logger.record("Train/{}".format(k), v, epoch)

    # Evaluate the model on validation set
    if config.experiment.validate:
        with torch.no_grad():
            step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Valid/{}".format(k), v, epoch)

        print("Validation Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))

        # save checkpoint if achieve new best validation loss
        valid_check = "Loss" in step_log
        if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
            best_valid_loss = step_log["Loss"]
            if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                should_save_ckpt = True
                ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

    # Evaluate the model by by running rollouts

    # do rollouts at fixed rate or if it's time to save a new ckpt
    video_paths = None
    rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
    if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

        # wrap model as a RolloutPolicy to prepare for rollouts
        rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

        num_episodes = config.experiment.rollout.n
        all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
            policy=rollout_model,
            envs=envs,
            horizon=config.experiment.rollout.horizon,
            use_goals=config.use_goals,
            num_episodes=num_episodes,
            render=False,
            video_dir=video_dir if config.experiment.render_video else None,
            epoch=epoch,
            video_skip=config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
        )

        # summarize results from rollouts to tensorboard and terminal
        for env_name in all_rollout_logs:
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

        # checkpoint and video saving logic
        updated_stats = TrainUtils.should_save_from_rollout_logs(
            all_rollout_logs=all_rollout_logs,
            best_return=best_return,
            best_success_rate=best_success_rate,
            epoch_ckpt_name=epoch_ckpt_name,
            save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
            save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
        )
        best_return = updated_stats["best_return"]
        best_success_rate = updated_stats["best_success_rate"]
        epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
        should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
        if updated_stats["ckpt_reason"] is not None:
            ckpt_reason = updated_stats["ckpt_reason"]

    # Only keep saved videos if the ckpt should be saved (but not because of validation score)
    should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
    if video_paths is not None and not should_save_video:
        for env_name in video_paths:
            os.remove(video_paths[env_name])

    # Save model checkpoints based on conditions (success rate, validation loss, etc)
    if should_save_ckpt:
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
            obs_normalization_stats=obs_normalization_stats,
        )

    # Finally, log memory usage in MB
    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000)
    data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
    print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

# terminate logging
data_logger.close()

# %%


# %%


# %%


# %%


# %%


# %%


# %%




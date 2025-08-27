#!/bin/bash

THIS_COMPUTER_ID="vast_126"  #TODO: A name and last part of the ip address.
CONFIG_NAME="train_diffusion_unet"
SEEDS=(42 43 44)


#------------------Edit below------------------
TASK_NAME="square_d2" #TODO: change to your task name
DATASET_PATH="/root/square_s4_abs.hdf5" #TODO: change to your dataset path
TOREMOVE_FILE="/root/segs_index_rm40_gib_gib_square_abs_s4_70_param_h200.json" #TODO: change to your segments to remove file path
N_DEMO=70 #TODO: change to the number of demos in the dataset. (square,coffee has 70, mug has 60, kitchen has 50)
SUFFIX="bed_rm30" #TODO: change to a suffix that describes your experiment
#------------------Edit above------------------


for SEED in "${SEEDS[@]}"; do
    RUN_DIR="data/outputs/${TASK_NAME}_${SUFFIX}_${THIS_COMPUTER_ID}/SEED_${SEED}_$(date +%Y.%m.%d)_$(date +%H.%M.%S)"
    python train.py --config-name=$CONFIG_NAME \
        task_name=$TASK_NAME \
        dataset_path=$DATASET_PATH \
        training.seed=$SEED \
        training.num_epochs=401 \
        hydra.run.dir="$RUN_DIR" \
        segments_toremove_file=$TOREMOVE_FILE \
        n_demo=$N_DEMO
done


# run from equidiff directory.
# activate equidiff conda environment first.
# data saved to: equidiff/data/outputs/<here>


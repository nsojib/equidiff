#!/bin/bash

# Define the base output directory and checkpoint path
BASE_OUTPUT_DIR="data/mug_eval_output"
CHECKPOINT_PATH="/home/ubuntu/equidiff/data/outputs/2025.01.29/11.30.08_diff_c_mug_cleanup_d1/checkpoints/epoch=0105-test_mean_score=0.820.ckpt"


SEEDS=(42 100 1000)

# Loop through each seed and run the command
for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_seed${SEED}"  # Append seed to output dir
    echo "Running evaluation with seed ${SEED}, output will be saved to ${OUTPUT_DIR}"
    
    python eval.py \
        --checkpoint "$CHECKPOINT_PATH" \
        -o "$OUTPUT_DIR" \
        --seed "$SEED"
done

echo "All evaluations completed."


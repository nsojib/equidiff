#!/bin/bash


TASK="coffee"
CHECKPOINT_PATH="/home/ubuntu/equidiff/data/outputs/2025.01.29/09.58.46_diff_c_coffee_d2/checkpoints/epoch=0180-test_mean_score=0.680.ckpt"


BASE_OUTPUT_DIR="data/{$TASK}_eval_output"
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


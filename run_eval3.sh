#!/bin/bash

TASK="pusht_dppd"
CHECKPOINT_PATH="/root/diffusion_policy/data/outputs/2025.05.10/21.26.43_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0350-test_mean_score=0.969.ckpt"

BASE_OUTPUT_DIR="data/${TASK}_eval_output"
SEEDS=(42 100 1000)

SCORES_FILE="${BASE_OUTPUT_DIR}/eval_scores.txt"
mkdir -p "$BASE_OUTPUT_DIR"  # Ensure directory exists
echo "checkpoint: ${CHECKPOINT_PATH}" > "$SCORES_FILE"

# Array to store scores
SCORE_LIST=()

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed${SEED}"
    echo "Running evaluation with seed ${SEED}, output will be saved to ${OUTPUT_DIR}"

    python eval.py \
        --checkpoint "$CHECKPOINT_PATH" \
        -o "$OUTPUT_DIR" \
        --seed "$SEED"

    SCORE=$(jq -r '."test/mean_score"' "${OUTPUT_DIR}/eval_log.json")
    
    echo "seed ${SEED}, mean_score ${SCORE}" >> "$SCORES_FILE"
    SCORE_LIST+=("$SCORE")
done

# Compute average and std
stats=$(printf "%s\n" "${SCORE_LIST[@]}" | awk '
    {
        x[NR]=$1; sum+=$1
    }
    END {
        mean = sum/NR
        for (i=1; i<=NR; i++) {
            sq_diff += (x[i] - mean)^2
        }
        std = sqrt(sq_diff/NR)
        printf "average, %.15f\nstd, %.15f\n", mean, std
    }
')

echo "$stats" >> "$SCORES_FILE"

echo "All evaluations completed. Summary saved to ${SCORES_FILE}"

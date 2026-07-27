#!/bin/bash
# AGZO: facebook/opt-2.7b on SST-2 (GPU)
# Shared settings match scripts/ENGINE/MeZO_opt27b_mnli.sh.


# Change to repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Hugging Face Mirror (for faster downloads in China)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export WANDB_MODE=online
export WANDB_PROJECT=ENGINE
MODEL="facebook/opt-2.7b"
TASK="sst2"
ALGORITHM="AGZO"

# Shared SST-2 hyperparameters (matched to MeZO)
LR=5e-6
EPOCHS=1000
# Three two-sided directions; AGZO adds one activation-collection forward.
NUM_QUERIES=6
BATCH_SIZE=64
BATCH_SIZE_LIMIT=64
SAMPLE_SIZE=512
SAMPLE_SIZE_VAL=256
MAX_SEQ_LENGTH=128

# AGZO-specific settings from the paper
RANK=1
POWER_ITER_STEPS=3

# Other settings
DEVICE=0
TRIAL=0
SEED=42

python main.py \
    --epochs $EPOCHS \
    --model_name $MODEL \
    --task $TASK \
    --algorithm $ALGORITHM \
    --lr $LR \
    --weight_decay 0.0 \
    --num_queries $NUM_QUERIES \
    --perturbation_scale 1e-3 \
    --rank $RANK \
    --agzo_power_iter_steps $POWER_ITER_STEPS \
    --samplesize $SAMPLE_SIZE \
    --samplesize_validation $SAMPLE_SIZE_VAL \
    --batchsize $BATCH_SIZE \
    --batchsize_limit $BATCH_SIZE_LIMIT \
    --full_parameter \
    --max_seq_length $MAX_SEQ_LENGTH \
    --device $DEVICE \
    --save_every 10000000 \
    --logging "wandb+csv" \
    --init_seed $SEED \
    --trial $TRIAL \
    --early_stopping --patience 1000

#!/bin/bash
# ZO-Muon: facebook/opt-2.7b on MNLI (GPU)
# Data, batch, query budget, precision, and evaluation match the ENGINE baselines.


# Change to repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Hugging Face Mirror (for faster downloads in China)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export WANDB_MODE=online
MODEL="facebook/opt-2.7b"
TASK="mnli"
ALGORITHM="ZO-Muon"

# Shared MNLI setup
LR=1e-2
EPOCHS=1000
NUM_QUERIES=6
BATCH_SIZE=64
BATCH_SIZE_LIMIT=64
SAMPLE_SIZE=512
SAMPLE_SIZE_VAL=256
MAX_SEQ_LENGTH=128

# ZO-Muon settings
NONMATRIX_LR=1e-7
RANK=64
STEP_INTERVAL=100

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
    --zomuon_nonmatrix_lr $NONMATRIX_LR \
    --weight_decay 0.0 \
    --num_queries $NUM_QUERIES \
    --perturbation_scale 1e-3 \
    --rank $RANK \
    --step_interval $STEP_INTERVAL \
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

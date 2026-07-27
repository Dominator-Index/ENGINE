#!/bin/bash
# ZO-GV: facebook/opt-2.7b on SST-2, using the ENGINE MeZO comparison setup.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export WANDB_MODE=online
export WANDB_PROJECT=ENGINE

MODEL="facebook/opt-2.7b"
TASK="sst2"
ALGORITHM="ZO-GV"

LR=1e-6
EPOCHS=1000
NUM_QUERIES=6
GV_SPLIT_RATIO=0.5
BATCH_SIZE=64
BATCH_SIZE_LIMIT=64
SAMPLE_SIZE=512
SAMPLE_SIZE_VAL=256
MAX_SEQ_LENGTH=128

DEVICE=0
TRIAL=0
SEED=42

python main.py \
    --epochs "$EPOCHS" \
    --model_name "$MODEL" \
    --task "$TASK" \
    --algorithm "$ALGORITHM" \
    --lr "$LR" \
    --weight_decay 0.0 \
    --num_queries "$NUM_QUERIES" \
    --gv_split_ratio "$GV_SPLIT_RATIO" \
    --perturbation_scale 1e-3 \
    --samplesize "$SAMPLE_SIZE" \
    --samplesize_validation "$SAMPLE_SIZE_VAL" \
    --batchsize "$BATCH_SIZE" \
    --batchsize_limit "$BATCH_SIZE_LIMIT" \
    --full_parameter \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --device "$DEVICE" \
    --save_every 10000000 \
    --logging "wandb+csv" \
    --init_seed "$SEED" \
    --trial "$TRIAL" \
    --early_stopping --patience 1000

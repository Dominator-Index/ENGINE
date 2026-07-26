#!/bin/bash
# SubZero: gpt2-xl on MNLI
# Model: gpt2-xl
# Task: mnli (GLUE)
# Algorithm: SubZero
# Matched to MeZO/LOZO MNLI settings, including LR


# Change to repository root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


# GPU Configuration
export CUDA_VISIBLE_DEVICES=1

# Hugging Face Mirror (for faster downloads in China)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
MODEL="gpt2-xl"
TASK="mnli"
ALGORITHM="SubZero"

# Shared MNLI hyperparameters (LR fixed to MeZO: 5e-6)
LR=5e-6
EPOCHS=1000
NUM_QUERIES=6
BATCH_SIZE=64
BATCH_SIZE_LIMIT=64
SAMPLE_SIZE=512
SAMPLE_SIZE_VAL=256
MAX_SEQ_LENGTH=128

# SubZero-specific parameters
RANK=8
STEP_INTERVAL=1000

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

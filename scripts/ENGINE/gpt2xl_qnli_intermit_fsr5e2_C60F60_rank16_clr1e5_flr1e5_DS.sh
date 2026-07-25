#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline

export HF_HOME="${HF_HOME:-/wutailin/.cache/huggingface}"   # OPT-2.7b lives in this hub cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.." || exit 1


LOG_FILE="/tmp/gpt2xl_qnli_intermit_fsr5e2_C60F60_rank16_DS_clr1e5_flr1e5_$(date +%Y%m%d_%H%M%S).log"


python main.py \
    --algorithm ENGINE \
    --model_name gpt2-xl \
    --task qnli \
    --lr 1e-5 \
    --weight_decay 0.0 \
    --num_queries 6 \
    --perturbation_scale 1e-3 \
    --damping 0.1 \
    --lr_cov 1e-3 \
    --samplesize 512 \
    --samplesize_validation 256 \
    --batchsize 64 \
    --batchsize_limit 64 \
    --full_parameter \
    --max_seq_length 128 \
    --device 0 \
    --logging "wandb+csv" \
    --alignment_diag \
    --alignment_diag_interval 100 \
    --alignment_fd_radius 1e-3 \
    --alignment_kappa_delta 1e-12 \
    --init_seed 42 \
    --epochs 1000 \
    --eval_every 1 \
    --lora_rank 16 \
    --ema_gamma 0.9 \
    --inject_method intermit \
    --fine_sample_ratio 0.05 \
    --tau_scheme adaptive \
    --tau_min 0.3 \
    --tau_max 0.8 \
    --tau_warmup_epochs 30 \
    --vcycle_coarse_epochs 60 \
    --vcycle_fine_epochs 60 \
    --phase3_lr 1e-5 \
    --checkpoint_interval 1000 \
    --early_stopping --patience 1000 \
    --curvature_multigrid --curvature_mg_beta 0.3 \
    --basis_method reuse \
    --no_coarse_isotropic \
    --checkpoint_dir results_demo/gpt2xl/qnli/ckpt_intermit_fsr5e2_C60F60_rank16_clr1e5_flr1e5_DS \
    --no_checkpoint \
    --deep_search \
    2>&1 | tee "$LOG_FILE"

echo "Done. Log: $LOG_FILE"

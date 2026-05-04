## ENGINE: Endogenous Variational MultiScale Optimization for Zeroth-Order LLM Fine-Tuning

This repository provides the official implementation of **ENGINE**, a zeroth-order LLM fine-tuning method that brings the **Variational MultiScale (VMS)** view from computational mechanics into the LoRA–LOREN setting. ENGINE achieves **higher accuracy** and **faster convergence** than existing ZO methods.



## Overview

ENGINE treats the LoRA subspace as the resolved coarse scale and the full parameter space as the unresolved fine scale: the **VMS-EO correction** reinjects the fine-grid curvature that a LoRA-only restriction would discard, and a V-cycle schedule with bidirectional state transfer keeps the endogenous Hessian fresh between coarse phases. 




## Models

| Family | Model                                   |
| :----- | :-------------------------------------- |
| GPT    | `gpt2-xl`                               |
| OPT    | `facebook/opt-2.7b`, `facebook/opt-13b` |
| LLaMA  | `meta-llama/Meta-Llama-3-8B`            |

## Tasks

**GLUE:** `mnli`, `qnli`, `sst2`, `cola`  
**SuperGLUE:** `rte`, `boolq`, `wic`, `cb`



## Algorithms
| Algorithm | Description                                         | Reference                                                    |
| :-------- | :-------------------------------------------------- | :----------------------------------------------------------- |
| `ZO`      | Baseline memory-efficient ZO                        | [Malladi et al., 2023, *Fine-Tuning Language Models with Just Forward Passes*](https://proceedings.neurips.cc/paper_files/paper/2023/file/a627810151be4d13f907ac898ff7e948-Paper-Conference.pdf) |
| `ZOAdam`  | MeZO with Adam's adaptive update                    |                                                              |
| `ZOSVRG`  | MeZO with stochastic variance-reduced gradients     | [Gautam et al., 2024, *Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models*](https://openreview.net/pdf?id=yc758jO4i5) |
| `LOZO`    | Low-rank ZO gradient                                | [Chen et al., 2025, *Enhancing Zeroth-Order Fine-Tuning for Language Models with Low-Rank Structures*](https://openreview.net/pdf?id=9BiVepgmWW) |
| `HiZOO`   | Hessian informed ZO                                 | [Zhao et al., 2025, *Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer*](https://openreview.net/forum?id=bEqI61iBue) |
| `LOREN`   | Low-rank curvature-aware ZO with variance reduction | Seung et al., 2026, *Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning* |
| `ENGINE`(ours)   | ZO with VMS-EO correction | Zhuoli Ouyang et al., 2026, *ENGINE: Endogenous Variational MultiScale Optimization for Zeroth-Order LLM Fine-Tuning* |



## How to Run

Fine-tune a model with ENGINE:

``` shell
python main.py \
    --epochs $epochs \            #epochs
    --algorithm ENGINE \
    --model_name $model \         # model
    --task $task \                # task
    --lr $fine_lr \               # lr of Fine phase
    --weight_decay 0.0 \
    --num_queries $num_queries \     # Default: 6
    --perturbation_scale 1e-3 \
    --damping 0.1 \
    --lr_cov 1e-3 \
    --samplesize 512 \
    --samplesize_validation 256 \
    --batchsize $bsz \                    # Default: 64
    --batchsize_limit $bsz_lim \          # Default: 64
    --full_parameter \
    --max_seq_length 128 \
    --device 0 \
    --logging "wandb+csv" \
    --init_seed 42 \
     --eval_every 1 \
    --lora_rank 16 \
    --ema_gamma 0.9 \
    --inject_method intermit \
    --fine_sample_ratio $fsr \          # Default: 0.05
    --tau_scheme adaptive \
    --tau_min 0.3 \
    --tau_max 0.8 \
    --tau_warmup_epochs 30 \
    --vcycle_coarse_epochs $c_ep \      # Default: 60
    --vcycle_fine_epochs $f_ep \        # Default: 60
    --phase3_lr $coarse_lr \            # lr of Coarse phase
    --checkpoint_interval 1000 \
    --early_stopping --patience 1000 \
    --curvature_multigrid --curvature_mg_beta 0.3 \
    --basis_method reuse \
    --no_coarse_isotropic \
    --checkpoint_dir results_demo \
    --no_checkpoint \
    --deep_search 

```



Fine-tune a model with other algorithm (https://github.com/hseung88/loren):

``` shell
python main.py\
    --epochs $epochs \
    --model_name $model \
    --task $task \
    --algorithm $algorithm \
    --lr $lr \
    --weight_decay 0.0 \
    --num_queries $num_queries \                    # Default: 2 (MeZO, LOZO, MeZO-Adam), 3 (MeZO-SVRG, HiZOO)
    --perturbation_scale 1e-3  \
    --lr_mezosvrg_mb 1e-6 --q 2 --anneal 5 \        # MeZO-SVRG
    --eps 1e-8 \                                    # MeZO-Adam
    --rank 2 --step_interval 50 \                   # LOZO
    --hessian_smooth_type constant1e-8 \            # HiZOO
    --lr_cov 1e-3 --damping 0.1 \                   # LOREN
    --samplesize 512 --samplesize_validation 256 \
    --batchsize $bsz \
    --batchsize_limit $bsz_lim \
    --full_parameter \
    --max_seq_length 128 \
    --device 0 --save_every 10000000 --logging csv \
    --init_seed 42 --trial $trial \
    --early_stopping --patience 1000
```



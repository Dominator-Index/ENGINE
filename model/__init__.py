from .glue_transformer import GLUETransformer
from .glue_transformer_fo import GLUETransformerFO
from .glue_transformer_mezo import GLUETransformerMeZo
from .glue_transformer_svrg import GLUETransformerMeZoSVRG
from .glue_transformer_adam import GLUETransformerMeZoAdam
from .glue_transformer_lozo import GLUETransformerLoZo
from .glue_transformer_hizoo import GLUETransformerHiZOO
from .glue_transformer_loren import GLUETransformerLOREN
from .glue_transformer_adasmooth import GLUETransformerAdaSmooth
from .glue_transformer_evvms import GLUETransformerEVVMS
from .opt import OPTDecoder, OPTForCausalLM, OPTModel
from .roberta import RobertaConfig, RobertaLMHead, RobertaModel, RobertaPreTrainedModel
from .prompt_finetune import (RobertaModelForPromptFinetuning, OPTModelForPromptFinetuning,
                              LlamaModelForPromptFinetuning)


__all__ = ['GLUETransformer', 'GLUETransformerMeZo',
           'GLUETransformerMeZoSVRG', 'GLUETransformerMeZoAdam',
           'GLUETransformerHiZOO', 'GLUETransformerLOREN', 'GLUETransformerAdaSmooth',
           'GLUETransformerEVVMS', 
           'RobertaPreTrainedModel', 'RobertaModel', 'RobertaLMHead',
           'RobertaModelForPromptFinetuning', 'RobertaConfig', 'OPTModelForPromptFinetuning',
           'OPTModel', 'OPTForCausalLM', 'OPTDecoder',  'LlamaModelForPromptFinetuning']


model_class = {
    'FOSGD': GLUETransformerFO,
    'FOAdam': GLUETransformerFO,
    'MeZO': GLUETransformerMeZo,
    'ZOSVRG': GLUETransformerMeZoSVRG,
    'ZOAdam': GLUETransformerMeZoAdam,
    'LOZO': GLUETransformerLoZo,
    'HiZOO': GLUETransformerHiZOO,
    'LOREN': GLUETransformerLOREN,
    'AdaSmooth': GLUETransformerAdaSmooth,
    'ENGINE': GLUETransformerEVVMS,
}


def get_model(args, dm):
    glue_transformer = model_class[args.algorithm]
    transformer = glue_transformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.lr_mezosvrg_mb if 'SVRG' in args.algorithm else args.lr,
        learning_rate_aux=args.lr if 'SVRG' in args.algorithm else None,
        lr_anneal=args.anneal,
        full_parameter=args.full_parameter,
        batchsize_limit=args.batchsize_limit,
        zero_order_eps=args.perturbation_scale,
        soft_prompt=args.soft_prompt,
        use_SGD=True if 'SGD' in args.algorithm else False,
        logger_type=args.logging,
        hf_token=args.hf_token,
        model_init_seed=args.init_seed,
        q=args.q,
        step_interval=args.step_interval,
        rank=args.rank,
        weight_decay=args.weight_decay,
        eps=args.eps,
        damping=args.damping,
        lr_cov=args.lr_cov,
        num_queries=args.num_queries,
        clipping=args.clipping,
        hessian_smooth_type=args.hessian_smooth_type,
        num_epochs=args.epochs,
        beta_init=args.beta_init,
        beta_decay=args.beta_decay,
        # EVVMS specific parameters
        lora_rank=getattr(args, 'lora_rank', 8),
        ema_gamma=getattr(args, 'ema_gamma', 0.9),
        fine_sample_ratio=getattr(args, 'fine_sample_ratio', 0.2),
        use_fine_residual=not getattr(args, 'no_fine_residual', False),
        # v1.13 fixes
        tau_min=getattr(args, 'tau_min', 0.5),
        tau_max=getattr(args, 'tau_max', 1.0),
        tau_scheme=getattr(args, 'tau_scheme', 'adaptive'),
        tau_warmup_epochs=getattr(args, 'tau_warmup_epochs', 0),
        lora_fp32=not getattr(args, 'no_lora_fp32', False),
        tau_lr_scale=getattr(args, 'tau_lr_scale', False),
        vms_lr_ref=getattr(args, 'vms_lr_ref', 1e-5),
        inject_method=getattr(args, 'inject_method', 'continuous'),
        # CT-LOREN parameters
        curvature_transfer=getattr(args, 'curvature_transfer', False),
        ct_interval=getattr(args, 'ct_interval', 5),
        ct_beta=getattr(args, 'ct_beta', 0.2),
        # Per-row preconditioner
        perrow_precond=getattr(args, 'perrow_precond', False),
        # Adam for curvature ODE
        adam_cov=getattr(args, 'adam_cov', False),
        adam_cov_beta1=getattr(args, 'adam_cov_beta1', 0.9),
        adam_cov_beta2=getattr(args, 'adam_cov_beta2', 0.999),
        # lr_cov schedule
        lr_cov_schedule=getattr(args, 'lr_cov_schedule', 'constant'),
        lr_cov_warmup_epochs=getattr(args, 'lr_cov_warmup_epochs', 20),
        lr_cov_min=getattr(args, 'lr_cov_min', None),
        # LoRA basis selection + B-only coarse grid
        basis_method=getattr(args, 'basis_method', 'random'),
        svd_momentum_ema=getattr(args, 'svd_momentum_ema', 0.0),
        coarse_b_only=getattr(args, 'coarse_b_only', False),
        oja_lr=getattr(args, 'oja_lr', 0.1),
        # TA-MIGSS parameters
        nq_lr_coupling=getattr(args, 'nq_lr_coupling', False),
        nq_ref=getattr(args, 'nq_ref', 4),
        sigma_adaptive_lr=getattr(args, 'sigma_adaptive_lr', False),
        coarse_isotropic=getattr(args, 'coarse_isotropic', True),
        evec_a_diagnostic=getattr(args, 'evec_a_diagnostic', False),
        evec_a_diag_interval=getattr(args, 'evec_a_diag_interval', 1),
        alignment_diag=getattr(args, 'alignment_diag', False),
        alignment_diag_interval=getattr(args, 'alignment_diag_interval', 100),
        alignment_fd_radius=getattr(args, 'alignment_fd_radius', 1e-3),
        alignment_kappa_delta=getattr(args, 'alignment_kappa_delta', 1e-12),
    )
    return transformer

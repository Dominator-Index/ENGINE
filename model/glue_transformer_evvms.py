"""
EV-VMS (MultiGrid Dynamics LOREN) Implementation

Based on cmp.md: Single-loop LoRA optimization with VMS feedback.

Core idea:
- Use LoRA-style parameterization: W = W_0 + B @ A
- Train only A and B (low-rank parameters)
- Periodically compute fine grid residual and inject via VMS feedback
- Each matrix (A, B) has its own curvature vector for LOREN preconditioning
"""
import math
import os
import pickle
import time
from typing import Dict, Optional

import numpy as np
import torch

from .glue_transformer_loren import GLUETransformerLOREN
from common.alg_utils import sgd_with_wdecay


def _force_symlink(target, link_path):
    """Create or replace a symlink using relative path for portability."""
    import tempfile
    rel_target = os.path.relpath(target, os.path.dirname(link_path))
    # Atomic symlink replacement: create tmp symlink then rename
    tmp_link = link_path + f".tmp.{os.getpid()}"
    try:
        os.symlink(rel_target, tmp_link)
        os.replace(tmp_link, link_path)
    except OSError:
        # Fallback: non-atomic
        try:
            os.unlink(tmp_link)
        except FileNotFoundError:
            pass
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.unlink(link_path)
        os.symlink(rel_target, link_path)


class GLUETransformerEVVMS(GLUETransformerLOREN):
    """
    EV-VMS: LoRA-based zeroth-order optimization with VMS feedback.

    Key differences from LOREN:
    - Only trains LoRA parameters (A, B) instead of full model
    - Uses fine grid residual feedback for VMS correction
    - Separate curvature vectors for A and B matrices
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        zero_order_eps: float = 1e-3,
        learning_rate_aux: float = 1e-3,
        minibatch: int = 64,
        max_norm: float = 18000.0,
        z_std: float = 1.0,
        lr_anneal: float = 1.0,
        full_parameter: bool = True,
        batchsize_limit: int = 16,
        eval_splits: Optional[list] = None,
        soft_prompt: bool = False,
        use_SGD: bool = True,
        logger_type: str = 'wandb',
        hf_token: Optional[str] = None,
        lora: bool = False,
        model_init_seed: Optional[int] = None,
        weight_decay: float = 0.01,
        damping: float = 0.1,
        lr_cov: float = 2e-3,
        beta1: float = 0.9,
        num_queries: int = 6,  # Default K=6 for EV-VMS (avg 6.4 with fine residual)
        # EV-VMS specific parameters
        lora_rank: int = 8,
        ema_gamma: float = 0.9,  # EMA smoothing coefficient
        fine_sample_ratio: float = 0.2,  # Ratio of steps with fine residual
        use_fine_residual: bool = True,
        # v1.13 fixes based on expert analysis
        tau_min: float = 0.5,  # τ 下界，防止 VMS 过早消失
        tau_max: float = 1.0,  # τ 上界（方案2使用）
        tau_scheme: str = 'adaptive',  # τ 计算方案: 'fixed', 'adaptive', 'ema'
        tau_warmup_epochs: int = 0,  # τ warmup epochs (0=禁用)
        lora_fp32: bool = True,  # LoRA 矩阵使用 FP32，避免 BF16 灾难性抵消
        tau_lr_scale: bool = False,  # lr 感知 τ 缩放开关（默认关闭）
        vms_lr_ref: float = 1e-5,  # lr 感知 τ 缩放的参考学习率
        inject_method: str = 'continuous',  # VMS 注入方式: 'intermit'(仅fine_residual步) / 'continuous'(每步)
        # CT-LOREN (Curvature Transfer)
        curvature_transfer: bool = False,
        ct_interval: int = 5,
        ct_beta: float = 0.2,
        # Per-row preconditioner
        perrow_precond: bool = False,
        # Adam for curvature ODE
        adam_cov: bool = False,
        adam_cov_beta1: float = 0.9,
        adam_cov_beta2: float = 0.999,
        # lr_cov schedule
        lr_cov_schedule: str = 'constant',
        lr_cov_warmup_epochs: int = 20,
        lr_cov_min: float = None,
        num_epochs: int = 200,
        # LoRA basis selection + B-only coarse grid
        basis_method: str = 'random',
        svd_momentum_ema: float = 0.0,
        coarse_b_only: bool = False,
        oja_lr: float = 0.1,
        # FAS residual correction at restriction
        fas_correction: bool = False,
        fas_alpha: float = 0.5,
        fas_w0_alpha: float = 0.0,
        # Curvature multigrid cycle (evec_a ↔ evec_a_B/B bidirectional transfer)
        curvature_multigrid: bool = False,
        curvature_mg_beta: float = 0.3,  # EMA blend coefficient for prolongation (evec_a_B/B → evec_a)
        **kwargs,
    ):
        super().__init__(
            model_name_or_path,
            num_labels,
            zero_order_eps,
            learning_rate_aux,
            minibatch,
            max_norm,
            z_std,
            lr_anneal,
            full_parameter,
            batchsize_limit,
            eval_splits,
            soft_prompt,
            use_SGD,
            logger_type,
            hf_token,
            lora,
            model_init_seed,
            weight_decay,
            damping,
            lr_cov,
            beta1,
            num_queries,
            **kwargs
        )

        # EV-VMS specific parameters
        self.lora_rank = lora_rank
        self.ema_gamma = ema_gamma
        self.fine_sample_ratio = fine_sample_ratio
        self.use_fine_residual = use_fine_residual and fine_sample_ratio > 0
        self.fine_residual_interval = max(1, int(1.0 / fine_sample_ratio)) if fine_sample_ratio > 0 else 1

        # v1.13 fixes based on expert analysis
        self.tau_min = tau_min  # τ 下界，防止 VMS 过早消失
        self.tau_max = tau_max  # τ 上界（方案2使用）
        self.tau_scheme = tau_scheme  # τ 计算方案: 'fixed', 'adaptive', 'ema'
        self.tau_warmup_epochs = tau_warmup_epochs  # τ warmup epochs (0=禁用)
        self._tau_current_epoch = 0  # 当前 epoch，用于 warmup 计算
        self._cycle_start_epoch = 0  # 当前 cycle 开始时的 epoch，用于 per-cycle warmup
        self.lora_fp32 = lora_fp32  # LoRA 矩阵使用 FP32，避免 BF16 灾难性抵消
        self.tau_lr_scale = tau_lr_scale  # lr 感知 τ 缩放开关
        self.vms_lr_ref = vms_lr_ref  # lr 感知 τ 缩放的参考学习率
        self.inject_method = inject_method  # VMS 注入方式: 'intermit' / 'continuous'

        # CT-LOREN (Curvature Transfer)
        self.curvature_transfer = curvature_transfer
        self.ct_interval = ct_interval
        self.ct_beta = ct_beta
        self.ct_warmup_steps = 20  # Don't run CT until LoRA is non-trivial
        self.perrow_precond = perrow_precond  # Per-row independent preconditioner

        # Adam for curvature ODE
        self.adam_cov = adam_cov
        self.adam_cov_beta1 = adam_cov_beta1
        self.adam_cov_beta2 = adam_cov_beta2
        self.adam_cov_step = 0  # Global step counter for bias correction

        # lr_cov schedule
        self.lr_cov_schedule = lr_cov_schedule
        self.lr_cov_warmup_epochs = lr_cov_warmup_epochs
        self.lr_cov_base = lr_cov  # Store base lr_cov
        self.lr_cov_min = lr_cov_min if lr_cov_min is not None else 0.1 * lr_cov
        self.num_epochs = num_epochs

        # τ EMA state (for scheme 3)
        self.tau_ema: Dict[str, float] = {}

        # LoRA basis selection + B-only coarse grid
        self.basis_method = basis_method
        self.oja_basis = (basis_method == 'oja')  # Backward compat
        self.svd_momentum_ema = svd_momentum_ema
        self.coarse_b_only = coarse_b_only
        self.oja_lr = oja_lr
        self._lora_basis_history: Dict[str, torch.Tensor] = {}
        self._oja_cycle_idx = 0
        self._momentum_ema_buffer: Dict[str, torch.Tensor] = {}
        self._svd_spectrum_log: Dict[str, dict] = {}

        # FAS residual correction at restriction
        self.fas_correction = fas_correction
        self.fas_alpha = fas_alpha
        self.fas_w0_alpha = fas_w0_alpha

        # Curvature multigrid cycle
        self.curvature_multigrid = curvature_multigrid
        self.curvature_mg_beta = curvature_mg_beta

        # Gradient SVD diagnostic (disabled: diagnostics complete)
        self._grad_svd_diagnostic = False
        self._grad_svd_interval = 4  # Every N fine-grid steps
        self._grad_svd_history: list = []

        # LoRA structures
        self.lora_B: Dict[str, torch.Tensor] = {}  # A matrices
        self.lora_A: Dict[str, torch.Tensor] = {}  # B matrices
        self.lora_applied: bool = False  # Track if LoRA delta is in model params

        # Separate curvature vectors for A and B
        self.evec_a_B: Dict[str, torch.Tensor] = {}
        self.evec_a_A: Dict[str, torch.Tensor] = {}

        # Momentum buffers
        self.momentum_B: Dict[str, torch.Tensor] = {}
        self.momentum_A: Dict[str, torch.Tensor] = {}

        # Adam buffers for curvature ODE
        self.m_evec_B: Dict[str, torch.Tensor] = {}  # first moment
        self.v_evec_B: Dict[str, torch.Tensor] = {}  # second moment
        self.m_evec_A: Dict[str, torch.Tensor] = {}
        self.v_evec_A: Dict[str, torch.Tensor] = {}

        # Parameter info
        self.lora_param_info: Dict[str, dict] = {}

    def configure_lora_params(self):
        """
        Initialize LoRA parameterization for all 2D parameters.
        W = W_0 + B @ A where B is (m, r) left factor, A is (r, n) right factor

        Memory optimization: W_0 is NOT stored. We use incremental updates:
        - Model params start as W_0 (pretrained weights)
        - Apply LoRA: param += B @ A
        - Update LoRA: param += (new_delta - old_delta)
        """
        for name, param in self.params:
            if param.ndim >= 2:
                m = param.shape[0]
                n = param.view(m, -1).shape[1]
                r = self.lora_rank
                device = param.device
                # v1.13: Use FP32 for LoRA matrices to avoid BF16 catastrophic cancellation
                # in incremental updates: diff = new_B@new_A - old_B@old_A
                dtype = torch.float32 if self.lora_fp32 else param.dtype

                # Initialize B (left factor, m×r): use stored basis if available, else random
                if self.basis_method != 'random' and name in self._lora_basis_history:
                    A_hist = self._lora_basis_history[name]
                    if A_hist.device != device:
                        A_hist = A_hist.to(device)
                    A_hist = A_hist.to(dtype).clone()
                    r_eff = A_hist.shape[1]  # May be < r for small-dim params
                    # Normalize columns to initial scale: reuse preserves DIRECTIONS
                    # (subspace), not magnitudes. Trained A columns grow during SGD,
                    # making B@A too large and destroying model accuracy at F2C.
                    std_A = 1.0 / math.sqrt(r_eff)
                    col_norms = A_hist.norm(dim=0, keepdim=True)  # (1, r_eff)
                    target_norm = std_A * math.sqrt(m)  # Expected col norm for N(0, std_A²)
                    A_hist = A_hist * (target_norm / (col_norms + 1e-8))
                    self.lora_B[name] = A_hist
                else:
                    std_A = 1.0 / math.sqrt(r)
                    self.lora_B[name] = torch.randn(m, r, device=device, dtype=dtype) * std_A
                    r_eff = r

                # A (right factor, r×n) initialized to zero (ensures B@A = 0 at init,
                # so model starts exactly at W_0). Any non-zero A init creates
                # ||B@A|| ≈ 0.1√m per layer, destroying model accuracy at F2C.
                # The optimization direction is bootstrapped via momentum injection
                # (momentum_A = γ·B.t()@M) in restriction_to_coarse() Step 3.
                self.lora_A[name] = torch.zeros(r_eff, n, device=device, dtype=dtype)

                # Curvature vectors for LOREN preconditioning
                if self.perrow_precond:
                    # Per-row: each row has independent curvature direction
                    # evec_a_B: (m, r_eff) - per-row curvature for B (left factor)
                    # evec_a_A: (r_eff, n) - per-row curvature for A (right factor)
                    self.evec_a_B[name] = torch.randn(m, r_eff, device=device, dtype=dtype) * 0.01
                    self.evec_a_A[name] = torch.randn(r_eff, n, device=device, dtype=dtype) * 0.01
                else:
                    # Shared: all rows share one curvature direction (original LOREN)
                    # evec_a_B: (r_eff,) - for B's column space (left factor)
                    # evec_a_A: (n,) - for A's row space (right factor)
                    self.evec_a_B[name] = torch.randn(r_eff, device=device, dtype=dtype) * 0.01
                    self.evec_a_A[name] = torch.randn(n, device=device, dtype=dtype) * 0.01

                # Momentum buffers
                self.momentum_B[name] = torch.zeros_like(self.lora_B[name])
                self.momentum_A[name] = torch.zeros_like(self.lora_A[name])

                # Adam buffers for curvature ODE (same shape as evec_a_B/B)
                if self.adam_cov:
                    self.m_evec_B[name] = torch.zeros_like(self.evec_a_B[name])
                    self.v_evec_B[name] = torch.zeros_like(self.evec_a_B[name])
                    self.m_evec_A[name] = torch.zeros_like(self.evec_a_A[name])
                    self.v_evec_A[name] = torch.zeros_like(self.evec_a_A[name])

                self.lora_param_info[name] = {
                    'm': m, 'n': n, 'r': r_eff,
                    'shape': param.shape,
                }
            # 1D params: no LoRA, keep as is (no need to store)

        self.lora_applied = False

    def _apply_lora_to_params(self):
        """
        Apply LoRA: W = W_0 + B @ A (incremental update).

        If LoRA is not yet applied, adds delta to params.
        If already applied, this is a no-op (use _update_lora_in_params for changes).
        If LoRA structures have been released, this is a no-op.
        """
        if self.lora_applied:
            return  # Already applied

        # Safety check: LoRA structures might have been released in Phase 2
        if not hasattr(self, 'lora_B') or not self.lora_B:
            return

        for name, param in self.params:
            if name in self.lora_B:
                delta = self.lora_B[name] @ self.lora_A[name]
                if param.ndim > 2:
                    delta = delta.view(param.shape)
                param.data.add_(delta)

        self.lora_applied = True

    def _remove_lora_from_params(self):
        """
        Remove LoRA: W = W_0 (subtract current delta).

        Restores params to W_0 by subtracting the current B @ A.
        If LoRA structures have been released, this is a no-op.
        """
        if not self.lora_applied:
            return  # Not applied

        # Safety check: LoRA structures might have been released in Phase 2
        if not hasattr(self, 'lora_B') or not self.lora_B:
            return

        for name, param in self.params:
            if name in self.lora_B:
                delta = self.lora_B[name] @ self.lora_A[name]
                if param.ndim > 2:
                    delta = delta.view(param.shape)
                param.data.sub_(delta)

        self.lora_applied = False

    def _update_lora_in_params(self, old_B: Dict, old_A: Dict):
        """
        Update LoRA delta in params incrementally.

        Computes: param += (new_B @ new_A) - (old_B @ old_A)
        This avoids storing W_0 by using the difference of deltas.

        Args:
            old_B: Previous A matrices (before perturbation)
            old_A: Previous B matrices (before perturbation)
        """
        for name, param in self.params:
            if name in self.lora_B:
                old_delta = old_B[name] @ old_A[name]
                new_delta = self.lora_B[name] @ self.lora_A[name]
                diff = new_delta - old_delta
                if param.ndim > 2:
                    diff = diff.view(param.shape)
                param.data.add_(diff)

    @torch.no_grad()
    def evolve_lora_basis_oja(self, name: str, M: torch.Tensor, cycle_idx: int):
        """
        Oja's rule: evolve LoRA basis A to track top-r subspace of E[MM^T].

        Uses ||M||^2 normalization and eta decay for numerical stability.
        Converges to top-r eigenvectors of accumulated momentum covariance (Oja 1982).
        """
        M_fp32 = M.to(torch.float32)
        if M_fp32.ndim > 2:
            M_fp32 = M_fp32.view(M_fp32.shape[0], -1)
        m = M_fp32.shape[0]
        r = min(self.lora_rank, m)  # Cap rank for small-dim parameters

        if name in self._lora_basis_history:
            A_old = self._lora_basis_history[name].to(M_fp32.device)
        else:
            A_old = torch.randn(m, r, device=M_fp32.device, dtype=torch.float32)
            A_old, _ = torch.linalg.qr(A_old)

        # eta decay: eta_t = eta_0 / (1 + 0.1 * cycle_idx)
        eta_t = self.oja_lr / (1.0 + 0.1 * cycle_idx)
        # ||M||^2 normalization to prevent numerical explosion on large layers
        M_norm_sq = M_fp32.norm() ** 2 + 1e-8

        proj = M_fp32.T @ A_old       # (n, r)
        update = M_fp32 @ proj         # (m, r)
        A_new = A_old + eta_t * update / M_norm_sq

        # Re-orthogonalize
        A_new, _ = torch.linalg.qr(A_new)

        self._lora_basis_history[name] = A_new.clone()
        return A_new

    @torch.no_grad()
    def select_basis_svd(self, name, M):
        """
        SVD of fine-grid momentum M → top-r left singular vectors as LoRA A basis.

        Optimal rank-r approximation (Eckart-Young), one-shot (no iterative convergence).
        Optionally accumulates M across cycles via EMA for more stable spectrum.
        """
        M_fp32 = M.to(torch.float32)
        if M_fp32.ndim > 2:
            M_fp32 = M_fp32.view(M_fp32.shape[0], -1)
        m, n = M_fp32.shape
        r = min(self.lora_rank, m, n)

        # Optional EMA accumulation across V-cycles
        if self.svd_momentum_ema > 0:
            gamma = self.svd_momentum_ema
            if name in self._momentum_ema_buffer:
                M_fp32 = gamma * self._momentum_ema_buffer[name].to(M_fp32.device) + (1 - gamma) * M_fp32
            self._momentum_ema_buffer[name] = M_fp32.clone().cpu()  # CPU to save GPU mem

        U, S, Vh = torch.linalg.svd(M_fp32, full_matrices=False)
        A_new = U[:, :r]

        # Diagnostic: singular value spectrum
        total_energy = (S ** 2).sum().item()
        captured = (S[:r] ** 2).sum().item()
        self._svd_spectrum_log[name] = {
            'top_r_sv': S[:r].cpu().tolist(),
            'all_sv_top20': S[:20].cpu().tolist() if len(S) >= 20 else S.cpu().tolist(),
            'kappa': (captured / (total_energy + 1e-10)) ** 0.5,
            'energy_ratio': captured / (total_energy + 1e-10),
            'sv_ratio_1_r': (S[0] / (S[min(r, len(S)) - 1] + 1e-10)).item(),
            'effective_rank': (S.sum() ** 2 / ((S ** 2).sum() + 1e-10)).item(),
        }

        self._lora_basis_history[name] = A_new.clone()
        return A_new

    @torch.no_grad()
    def select_basis_svd_hybrid(self, name, M):
        """
        Hybrid basis: column 1 from evec_a (curvature direction), columns 2-r from SVD(M).

        Combines curvature information with momentum structure.
        """
        M_fp32 = M.to(torch.float32)
        if M_fp32.ndim > 2:
            M_fp32 = M_fp32.view(M_fp32.shape[0], -1)
        m, n = M_fp32.shape
        r = min(self.lora_rank, m, n)

        # Column 1: evec_a direction (high-curvature direction)
        if name in self.evec_a:
            a_f = self.evec_a[name].to(torch.float32)
            # evec_a is (n,) or (n,1), project M onto it to get row-space direction
            if a_f.ndim == 1:
                v1 = M_fp32 @ a_f  # (m,)
            else:
                v1 = (M_fp32 @ a_f).squeeze()
            v1_norm = v1.norm()
            if v1_norm > 1e-8:
                v1 = v1 / v1_norm
            else:
                v1 = torch.randn(m, device=M_fp32.device, dtype=torch.float32)
                v1 = v1 / v1.norm()
        else:
            v1 = torch.randn(m, device=M_fp32.device, dtype=torch.float32)
            v1 = v1 / v1.norm()

        if r == 1:
            A_new = v1.unsqueeze(1)
        else:
            # Remove v1 component, then SVD for remaining r-1 directions
            M_proj = M_fp32 - v1.unsqueeze(1) @ (v1.unsqueeze(0) @ M_fp32)
            U, S, _ = torch.linalg.svd(M_proj, full_matrices=False)
            A_new = torch.cat([v1.unsqueeze(1), U[:, :r - 1]], dim=1)
            A_new, _ = torch.linalg.qr(A_new)  # Re-orthogonalize

        # Diagnostic
        proj_norm = (A_new.T @ M_fp32).norm().item()
        M_norm = M_fp32.norm().item()
        self._svd_spectrum_log[name] = {
            'kappa': proj_norm / (M_norm + 1e-10),
        }

        self._lora_basis_history[name] = A_new.clone()
        return A_new

    @torch.no_grad()
    def _update_online_oja_basis(self):
        """
        Online Oja: update basis every fine-grid training step using current momentum.

        Unlike batch Oja (1 sample per V-cycle), this sees every momentum update,
        enabling convergence to the true top-r subspace.
        """
        eta = self.oja_lr / (1.0 + 0.01 * self.state.global_training_steps)
        for name, param in self.params:
            if param.ndim >= 2 and name in self.momentum:
                M = self.momentum[name].to(torch.float32)
                if M.ndim > 2:
                    M = M.view(M.shape[0], -1)
                m = M.shape[0]
                r = min(self.lora_rank, m)

                if name in self._lora_basis_history:
                    A = self._lora_basis_history[name].to(M.device)
                else:
                    A = torch.randn(m, r, device=M.device, dtype=torch.float32)
                    A, _ = torch.linalg.qr(A)

                M_norm_sq = M.norm() ** 2 + 1e-8
                A_new = A + eta * (M @ (M.T @ A)) / M_norm_sq
                A_new, _ = torch.linalg.qr(A_new)
                self._lora_basis_history[name] = A_new

    def restriction_operator(self, name: str, Res_f: torch.Tensor):
        """
        Restriction operator R: fine grid -> coarse grid (LoRA space)
        R_A(Res_f) = Res_f @ B^T
        R_B(Res_f) = A^T @ Res_f
        """
        if name not in self.lora_B:
            return None, None

        B = self.lora_B[name]
        A = self.lora_A[name]

        g = Res_f
        if g.ndim > 2:
            g = g.view(g.shape[0], -1)

        R_A = g @ A.t()   # (m, n) @ (n, r) -> (m, r)
        R_B = B.t() @ g   # (r, m) @ (m, n) -> (r, n)

        return R_A, R_B

    def set_epoch(self, epoch: int):
        """Set current epoch for τ warmup and lr_cov schedule."""
        self._tau_current_epoch = epoch

    def _get_lr_cov(self) -> float:
        """Get current lr_cov based on schedule."""
        if self.lr_cov_schedule == 'constant':
            return self.lr_cov
        elif self.lr_cov_schedule == 'warmup_cosine':
            epoch = self._tau_current_epoch  # 1-indexed
            warmup = self.lr_cov_warmup_epochs
            total = self.num_epochs
            lr_max = self.lr_cov_base
            lr_min = self.lr_cov_min
            if epoch <= warmup:
                # Linear warmup: lr_min → lr_max
                return lr_min + (lr_max - lr_min) * epoch / max(1, warmup)
            else:
                # Cosine decay: lr_max → lr_min
                progress = (epoch - warmup) / max(1, total - warmup)
                return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        return self.lr_cov

    def _compute_tau(self, name: str, c_coarse: float = None, a_coarse_expanded=None,
                     c_f_smooth: float = None, R_af=None, branch: str = 'A'):
        """
        Compute per-branch stabilization parameter τ.

        Schemes (controlled by self.tau_scheme):
        - 'fixed': τ = tau_min (constant)
        - 'simple': τ = 1/(α + ||a||²), clamped to [tau_min, tau_max] (bdb2e54 legacy)
        - 'adaptive': τ = clip(δ, tau_min, tau_max) where
            δ = ||c_coarse·ā_coarse - c_f·R(ā_f)|| / (||c_coarse·ā_coarse|| + ε)
            (Theory §5.4: larger curvature discrepancy → larger τ)
        - 'ema': Same as adaptive but with EMA smoothing

        Warmup: τ_effective = τ × min(1, epoch / warmup_epochs)
        Per-branch EMA state keyed by (name, branch).
        """
        if self.tau_scheme == 'fixed':
            tau = self.tau_min

        elif self.tau_scheme == 'adaptive' or self.tau_scheme == 'ema':
            if c_coarse is not None and a_coarse_expanded is not None and c_f_smooth is not None and R_af is not None:
                # δ = ||c_coarse·ā - c_f·R(ā_f)|| / (||c_coarse·ā|| + ε)
                coarse_term = c_coarse * a_coarse_expanded
                fine_term = c_f_smooth * R_af
                diff = coarse_term - fine_term
                diff_norm = diff.norm().item()
                coarse_norm = coarse_term.norm().item()
                delta = diff_norm / (coarse_norm + 1e-8)

                tau_raw = max(self.tau_min, min(delta, self.tau_max))

                if self.tau_scheme == 'ema':
                    ema_key = f"{name}_{branch}"
                    if ema_key not in self.tau_ema:
                        self.tau_ema[ema_key] = tau_raw
                    else:
                        self.tau_ema[ema_key] = 0.9 * self.tau_ema[ema_key] + 0.1 * tau_raw
                    tau = self.tau_ema[ema_key]
                else:
                    tau = tau_raw
            else:
                # Fallback if curvature info not available yet
                tau = self.tau_min

        else:
            # 'simple' scheme (bdb2e54 legacy): τ = 1/(α + ||a||²)
            current_phase = self.get_phase()
            if current_phase == 'coarse':
                sq_norm_a = 0.0
                if name in self.evec_a_B:
                    sq_norm_a += (self.evec_a_B[name] * self.evec_a_B[name]).sum().item()
                if name in self.evec_a_A:
                    sq_norm_a += (self.evec_a_A[name] * self.evec_a_A[name]).sum().item()
            elif name in self.evec_a:
                a_f = self.evec_a[name]
                sq_norm_a = torch.dot(a_f, a_f).item()
            else:
                sq_norm_a = 0.0

            tau = 1.0 / (self.damping + sq_norm_a + 1e-8)
            tau = max(self.tau_min, min(tau, self.tau_max))

        # Warmup: τ_effective = τ × min(1, cycle_local_epoch / warmup_epochs)
        # Uses per-cycle epoch so each new coarse/fine phase warms up from scratch
        if self.tau_warmup_epochs > 0:
            cycle_local_epoch = self._tau_current_epoch - self._cycle_start_epoch
            warmup_factor = min(1.0, cycle_local_epoch / self.tau_warmup_epochs)
            tau = tau * warmup_factor

        return tau

    @torch.no_grad()
    def perturb_lora_parameters(self, random_seed: int, scaling_factor: float = 1.0):
        """
        Perturb LoRA parameters (A, B) with LOREN-style preconditioning.

        Two modes:
        - perrow_precond=False (default): shared curvature, flat vector approach
        - perrow_precond=True: per-row independent curvature, vectorized

        Memory optimization: Uses incremental update to avoid storing W_0.
        Computes: param += (new_B @ new_A) - (old_B @ old_A)
        """
        torch.manual_seed(random_seed)
        params_dict = dict(self.params)

        for name in self.lora_B:
            B = self.lora_B[name]
            A = self.lora_A[name]
            a_B = self.evec_a_B[name]
            a_A = self.evec_a_A[name]
            param = params_dict[name]

            m, r = B.shape
            _, n = A.shape

            # Compute old delta BEFORE perturbation
            old_delta = B @ A

            if self.perrow_precond:
                # === Per-row preconditioning (vectorized) ===
                # a_B: (m, r), a_A: (r, n) — each row independent
                u_B = torch.randn(m, r, device=B.device, dtype=B.dtype)
                sq_norm_B = (a_B * a_B).sum(dim=1, keepdim=True)       # (m, 1)
                dot_au_B = (a_B * u_B).sum(dim=1, keepdim=True)        # (m, 1)
                damp_sq_B = torch.sqrt(self.damping + sq_norm_B)       # (m, 1)
                alpha_B = (self.d_sqrt + damp_sq_B) * dot_au_B / (sq_norm_B * damp_sq_B + 1e-8)
                perturb_A = u_B - alpha_B * a_B

                u_A = torch.randn(r, n, device=A.device, dtype=A.dtype)
                sq_norm_A = (a_A * a_A).sum(dim=1, keepdim=True)       # (r, 1)
                dot_au_A = (a_A * u_A).sum(dim=1, keepdim=True)        # (r, 1)
                damp_sq_A = torch.sqrt(self.damping + sq_norm_A)       # (r, 1)
                alpha_A = (self.d_sqrt + damp_sq_A) * dot_au_A / (sq_norm_A * damp_sq_A + 1e-8)
                perturb_B = u_A - alpha_A * a_A
            else:
                # === Shared preconditioning (original flat vector approach) ===
                # a_B: (r,), a_A: (n,) — tile across rows
                a_B_flat = a_B.view(1, -1).expand(m, -1).reshape(-1)
                u_B_flat = torch.randn(m * r, device=B.device, dtype=B.dtype)
                sq_norm_a_B = torch.dot(a_B_flat, a_B_flat)
                dot_au_B = torch.dot(a_B_flat, u_B_flat)
                damping_sq_B = math.sqrt(self.damping + sq_norm_a_B.item())
                alpha_B = (self.d_sqrt + damping_sq_B) * dot_au_B / (sq_norm_a_B * damping_sq_B + 1e-8)
                perturb_A = (u_B_flat - alpha_B * a_B_flat).view(m, r)

                a_A_flat = a_A.view(1, -1).expand(r, -1).reshape(-1)
                u_A_flat = torch.randn(r * n, device=A.device, dtype=A.dtype)
                sq_norm_a_A = torch.dot(a_A_flat, a_A_flat)
                dot_au_A = torch.dot(a_A_flat, u_A_flat)
                damping_sq_A = math.sqrt(self.damping + sq_norm_a_A.item())
                alpha_A = (self.d_sqrt + damping_sq_A) * dot_au_A / (sq_norm_a_A * damping_sq_A + 1e-8)
                perturb_B = (u_A_flat - alpha_A * a_A_flat).view(r, n)

            # Apply perturbations to B and A
            B.add_(perturb_A, alpha=scaling_factor * self.zero_order_eps)
            A.add_(perturb_B, alpha=scaling_factor * self.zero_order_eps)

            # Compute new delta AFTER perturbation
            new_delta = B @ A

            # Incremental update: param += (new_delta - old_delta)
            diff = new_delta - old_delta
            if param.ndim > 2:
                diff = diff.view(param.shape)
            param.data.add_(diff)

        # Mark that LoRA is applied (or remains applied)
        self.lora_applied = True

    @torch.no_grad()
    def compute_fine_residual(self, model, batch, random_seeds: list):
        """
        Compute fine grid residual using RLOO with num_queries forward passes.

        Uses the same RLOO variance reduction as the coarse-grid LOREN update,
        but on the full parameter space to extract fine-grid curvature info c_f.

        Curvature-VMS Theory:
        - c_f = <ā_f, ĝ_f> / (α + ||ā_f||²)
        - ĝ_f estimated via RLOO: ĝ_f = Σ_k (f_k - f̄) * z_k / (ε * (K-1))

        Returns:
            c_f_dict: Dict[str, float] - curvature projection coefficients
            fmean: float - mean loss across queries
        """
        self._apply_lora_to_params()

        # Evaluate at K perturbed points (one-sided perturbation, RLOO)
        fvals = []
        with torch.inference_mode():
            for k in range(self.num_queries):
                torch.manual_seed(random_seeds[k])
                for name, param in self.params:
                    if param.ndim >= 2:
                        z = torch.randn_like(param)
                        param.data.add_(z, alpha=self.zero_order_eps)

                loss = self.zo_forward_memory_eff(model, batch)
                fvals.append(loss.item())

                # Restore
                torch.manual_seed(random_seeds[k])
                for name, param in self.params:
                    if param.ndim >= 2:
                        z = torch.randn_like(param)
                        param.data.add_(z, alpha=-self.zero_order_eps)

        fmean = sum(fvals) / len(fvals)
        num_queries_inv = 1.0 / (self.num_queries - 1)
        eps_inv = 1.0 / (self.zero_order_eps * (self.num_queries - 1))

        # RLOO: compute c_f, c_f' and update evec_a using all queries
        c_f_dict = {}
        c_f_prime_dict = {}  # c_f' for A-branch (right factor, r×n)
        for k in range(self.num_queries):
            fweight = fvals[k] - fmean

            torch.manual_seed(random_seeds[k])
            for name, param in self.params:
                if param.ndim >= 2 and name in self.evec_a:
                    z = torch.randn_like(param)
                    info = self.param_info[name]
                    evec_size = info['evec_size']
                    mfactor = info['mfactor']
                    a_f = self.evec_a[name]
                    sq_norm_a = torch.dot(a_f, a_f)
                    damping_sq = math.sqrt(self.damping + sq_norm_a.item())

                    # Project z to evec_a dimension (LOREN block structure)
                    z_proj = z.view(mfactor, evec_size).mean(dim=0).to(a_f.dtype)
                    dot_za = torch.dot(z_proj, a_f)

                    # c_f contribution from query k:
                    # c_f = Σ_k fweight_k * <ā_f, z_k> / (ε*(K-1)*(α + m||a_f||²))
                    dot_af_z = mfactor * dot_za
                    c_f_k = fweight * dot_af_z.item() * eps_inv / (self.damping + mfactor * sq_norm_a.item() + 1e-8)
                    if name not in c_f_dict:
                        c_f_dict[name] = 0.0
                    c_f_dict[name] += c_f_k

                    # c_f' for A-branch: c_f' = 1^T(B^T G a_f) / (ρ + r||a_f||²)
                    # Efficient: 1^T(B^T z a_f) = B.sum(1)^T (z @ a_f), avoids (r,m)@(m,n) matmul
                    if name in self.lora_B:
                        B_mat = self.lora_B[name]  # (m, r) = left factor B
                        r_eff = B_mat.shape[1]
                        z_af = z.view(mfactor, evec_size).mv(a_f.to(z.dtype))  # (m,)
                        B_rowsum = B_mat.sum(dim=1).to(z.dtype)  # (m,)
                        dot_af_z_B = torch.dot(B_rowsum, z_af).item()
                        c_f_prime_k = fweight * dot_af_z_B * eps_inv / (self.damping + r_eff * sq_norm_a.item() + 1e-8)
                        if name not in c_f_prime_dict:
                            c_f_prime_dict[name] = 0.0
                        c_f_prime_dict[name] += c_f_prime_k

                    # Update fine grid curvature evec_a (RLOO version)
                    c1 = dot_za * z_proj - a_f
                    c2 = (self.d_sqrt + damping_sq) * (dot_za ** 2 - sq_norm_a) * a_f / (sq_norm_a * damping_sq + 1e-8)
                    ak_grad = fweight * (self.zero_order_eps ** 2) * (c1 - c2) / damping_sq
                    self.evec_a[name].add_(ak_grad, alpha=-self.lr_cov * num_queries_inv)

        return c_f_dict, c_f_prime_dict, fmean

    @torch.no_grad()
    def ct_probe_and_transfer(self, model, batch):
        """
        CT-LOREN: Probe full-space curvature (SPSA, two-sided) and transfer to
        LoRA curvature via J_P(A,B)*.

        Only updates evec_a (full-space curvature direction) and blends the
        projected result into evec_a_B, evec_a_A.  Does NOT update model params.

        Returns dict of diagnostics: rho (projection fidelity), norms.
        """
        import logging as _logging
        self._apply_lora_to_params()

        ct_queries = 2  # SPSA: 2 forward passes per query
        ct_diag = {'rho': [], 'norm_ct_A': [], 'norm_ct_B': []}

        # --- SPSA curvature probe (update evec_a only) ---
        for q in range(ct_queries):
            seed = np.random.randint(1000000000)

            # f+ (positive perturbation)
            torch.manual_seed(seed)
            for name, param in self.params:
                if param.ndim >= 2:
                    z = torch.randn_like(param)
                    param.data.add_(z, alpha=self.zero_order_eps)
            with torch.inference_mode():
                loss_plus = self.zo_forward_memory_eff(model, batch).item()
            torch.manual_seed(seed)
            for name, param in self.params:
                if param.ndim >= 2:
                    z = torch.randn_like(param)
                    param.data.add_(z, alpha=-self.zero_order_eps)

            # f- (negative perturbation)
            torch.manual_seed(seed)
            for name, param in self.params:
                if param.ndim >= 2:
                    z = torch.randn_like(param)
                    param.data.add_(z, alpha=-self.zero_order_eps)
            with torch.inference_mode():
                loss_minus = self.zo_forward_memory_eff(model, batch).item()
            torch.manual_seed(seed)
            for name, param in self.params:
                if param.ndim >= 2:
                    z = torch.randn_like(param)
                    param.data.add_(z, alpha=self.zero_order_eps)

            f_w = (loss_plus - loss_minus) / (2.0 * self.zero_order_eps)

            # Update evec_a via power iteration ODE
            torch.manual_seed(seed)
            for name, param in self.params:
                if param.ndim >= 2 and name in self.evec_a:
                    z = torch.randn_like(param)
                    info = self.param_info[name]
                    z_proj = z.view(info['mfactor'], info['evec_size']).mean(dim=0).to(torch.float32)
                    a_f = self.evec_a[name]
                    sq_norm_a = torch.dot(a_f, a_f)
                    damping_sq = math.sqrt(self.damping + sq_norm_a.item())
                    dot_za = torch.dot(z_proj, a_f)

                    c1 = dot_za * z_proj - a_f
                    c2 = (self.d_sqrt + damping_sq) * (dot_za ** 2 - sq_norm_a) * a_f / (
                        sq_norm_a * damping_sq + 1e-8)
                    ak_grad = f_w * (self.zero_order_eps ** 2) * (c1 - c2) / damping_sq
                    self.evec_a[name].add_(ak_grad, alpha=-self.lr_cov)

        # --- J_P(A,B)* projection: full-space evec_a → LoRA evec_a_B, evec_a_A ---
        for name in self.lora_B:
            if name not in self.evec_a:
                continue
            B = self.lora_B[name]  # (m, r)
            A = self.lora_A[name]  # (r, n)
            m, r = B.shape
            _, n = A.shape
            a_f = self.evec_a[name]  # (n,)

            # Expand a_f to (m, n) — same block structure as LOREN
            a_f_mat = a_f.view(1, -1).expand(m, -1)  # (m, n)

            # J_P* projection
            a_B_CT_mat = a_f_mat @ A.t()  # (m, r)
            a_A_CT_mat = B.t() @ a_f_mat  # (r, n)

            # Aggregate to match evec_a shapes: (r,) and (n,)
            a_B_CT = a_B_CT_mat.sum(0)  # (r,)
            a_A_CT = a_A_CT_mat.sum(0)  # (n,)

            norm_A_CT = a_B_CT.norm().item()
            norm_B_CT = a_A_CT.norm().item()

            ct_diag['norm_ct_A'].append(norm_A_CT)
            ct_diag['norm_ct_B'].append(norm_B_CT)

            # Projection fidelity ρ = ||J_P*(a)||² / ||a_expanded||²
            a_f_norm_sq = a_f.dot(a_f).item() * m
            rho = (norm_A_CT ** 2 + norm_B_CT ** 2) / (a_f_norm_sq + 1e-12)
            ct_diag['rho'].append(rho)

            # Guard: skip blend if projection is too small
            if norm_A_CT < 1e-8 or norm_B_CT < 1e-8:
                continue

            # Normalize projections
            a_B_CT_normed = a_B_CT / norm_A_CT
            a_A_CT_normed = a_A_CT / norm_B_CT

            # EMA blend into LoRA curvature
            beta = self.ct_beta
            blended_A = (1.0 - beta) * self.evec_a_B[name] + beta * a_B_CT_normed
            blended_B = (1.0 - beta) * self.evec_a_A[name] + beta * a_A_CT_normed

            norm_bA = blended_A.norm()
            norm_bB = blended_B.norm()
            if norm_bA > 1e-10:
                self.evec_a_B[name] = blended_A / norm_bA
            if norm_bB > 1e-10:
                self.evec_a_A[name] = blended_B / norm_bB

        self._last_ct_diag = ct_diag
        return ct_diag

    @torch.no_grad()
    def evvms_zo_update(self, random_seeds, fvals, fmean, c_f_dict=None, c_f_prime_dict=None):
        """
        EV-VMS RLOO update on LoRA parameters with per-branch Curvature-VMS feedback.

        Curvature-VMS Theory (Theorem 5.1):
        - R(∇f) = ∇_{A,B}f: gradient-level VMS provides no new info
        - Per-branch VMS mixes curvature corrections independently:
          B-branch: (1-τ_B)·c_B·ā_B + τ_B·c_f·R_B(ā_f)
          A-branch: (1-τ_A)·c_A·ā_A + τ_A·c_f'·R_A(ā_f)

        where:
        - c_B = <ā_A, grad_B_accum> / (α + m·‖a_B‖²): coarse-grid A projection
        - c_A = <ā_B, grad_A_accum> / (α + r·‖a_A‖²): coarse-grid B projection
        - c_f = <ā_f, ĝ_f> / (α + m·||ā_f||²): fine-grid projection for A-branch
        - c_f' = <B^T ā_f, ĝ_f> / (α + r·||ā_f||²): fine-grid projection for B-branch
        - τ_A, τ_B computed independently via per-branch curvature discrepancy
        """
        device = next(iter(self.lora_B.values())).device
        fweight = torch.tensor(fvals, device=device) - fmean
        num_queries_inv = 1.0 / (self.num_queries - 1)
        eps_inv = 1.0 / (self.zero_order_eps * (self.num_queries - 1))
        lr_cov_now = self._get_lr_cov()

        # Accumulators for gradient
        grad_B_accum = {name: torch.zeros_like(A) for name, A in self.lora_B.items()}
        grad_A_accum = {name: torch.zeros_like(B) for name, B in self.lora_A.items()}

        # Accumulators for curvature ODE gradient (for Adam: accumulate first, update once)
        if self.adam_cov:
            cov_grad_B_accum = {name: torch.zeros_like(self.evec_a_B[name]) for name in self.lora_B}
            cov_grad_A_accum = {name: torch.zeros_like(self.evec_a_A[name]) for name in self.lora_A}

        # RLOO gradient estimation with preconditioning
        for k in range(self.num_queries):
            torch.manual_seed(random_seeds[k])

            for name in self.lora_B:
                B = self.lora_B[name]
                A = self.lora_A[name]
                a_B = self.evec_a_B[name]
                a_A = self.evec_a_A[name]

                m, r = B.shape
                _, n = A.shape

                if self.perrow_precond:
                    # === Per-row preconditioning (vectorized) ===
                    # a_B: (m, r), a_A: (r, n)
                    u_B = torch.randn(m, r, device=B.device, dtype=B.dtype)
                    sq_norm_B = (a_B * a_B).sum(dim=1, keepdim=True)       # (m, 1)
                    dot_au_B = (a_B * u_B).sum(dim=1, keepdim=True)        # (m, 1)
                    damp_sq_B = torch.sqrt(self.damping + sq_norm_B)       # (m, 1)
                    alpha_B = (self.d_sqrt + damp_sq_B) * dot_au_B / (sq_norm_B * damp_sq_B + 1e-8)
                    sqrt_cov_u_B = u_B - alpha_B * a_B                    # (m, r)

                    u_A = torch.randn(r, n, device=A.device, dtype=A.dtype)
                    sq_norm_A = (a_A * a_A).sum(dim=1, keepdim=True)       # (r, 1)
                    dot_au_A = (a_A * u_A).sum(dim=1, keepdim=True)        # (r, 1)
                    damp_sq_A = torch.sqrt(self.damping + sq_norm_A)       # (r, 1)
                    alpha_A = (self.d_sqrt + damp_sq_A) * dot_au_A / (sq_norm_A * damp_sq_A + 1e-8)
                    sqrt_cov_u_A = u_A - alpha_A * a_A                    # (r, n)

                    # Gradient estimates
                    grad_B_accum[name].add_(sqrt_cov_u_B, alpha=fweight[k] * eps_inv)
                    grad_A_accum[name].add_(sqrt_cov_u_A, alpha=fweight[k] * eps_inv)

                    # Curvature ODE gradient (per-row, no .sum(0))
                    c1_B = dot_au_B * u_B - a_B                           # (m, r)
                    c2_B = (self.d_sqrt + damp_sq_B) * (dot_au_B ** 2 - sq_norm_B) * a_B / (sq_norm_B * damp_sq_B + 1e-8)
                    ak_grad_B = fweight[k] * (self.zero_order_eps ** 2) * (c1_B - c2_B) / damp_sq_B * num_queries_inv

                    c1_A = dot_au_A * u_A - a_A                           # (r, n)
                    c2_A = (self.d_sqrt + damp_sq_A) * (dot_au_A ** 2 - sq_norm_A) * a_A / (sq_norm_A * damp_sq_A + 1e-8)
                    ak_grad_A = fweight[k] * (self.zero_order_eps ** 2) * (c1_A - c2_A) / damp_sq_A * num_queries_inv

                    if self.adam_cov:
                        cov_grad_B_accum[name].add_(ak_grad_B)
                        cov_grad_A_accum[name].add_(ak_grad_A)
                    else:
                        self.evec_a_B[name].add_(ak_grad_B, alpha=-lr_cov_now)
                        self.evec_a_A[name].add_(ak_grad_A, alpha=-lr_cov_now)
                else:
                    # === Shared preconditioning (original flat vector approach) ===
                    a_B_flat = a_B.view(1, -1).expand(m, -1).reshape(-1)
                    u_B_flat = torch.randn(m * r, device=B.device, dtype=B.dtype)
                    sq_norm_a_B = torch.dot(a_B_flat, a_B_flat)
                    dot_au_B = torch.dot(a_B_flat, u_B_flat)
                    damping_sq_B = math.sqrt(self.damping + sq_norm_a_B.item())
                    alpha_B = (self.d_sqrt + damping_sq_B) * dot_au_B / (sq_norm_a_B * damping_sq_B + 1e-8)
                    sqrt_cov_u_B = (u_B_flat - alpha_B * a_B_flat).view(m, r)

                    a_A_flat = a_A.view(1, -1).expand(r, -1).reshape(-1)
                    u_A_flat = torch.randn(r * n, device=A.device, dtype=A.dtype)
                    sq_norm_a_A = torch.dot(a_A_flat, a_A_flat)
                    dot_au_A = torch.dot(a_A_flat, u_A_flat)
                    damping_sq_A = math.sqrt(self.damping + sq_norm_a_A.item())
                    alpha_A = (self.d_sqrt + damping_sq_A) * dot_au_A / (sq_norm_a_A * damping_sq_A + 1e-8)
                    sqrt_cov_u_A = (u_A_flat - alpha_A * a_A_flat).view(r, n)

                    # Gradient estimates
                    grad_B_accum[name].add_(sqrt_cov_u_B, alpha=fweight[k] * eps_inv)
                    grad_A_accum[name].add_(sqrt_cov_u_A, alpha=fweight[k] * eps_inv)

                    # Curvature ODE gradient (shared, with sum aggregation)
                    c1_B_flat = dot_au_B * u_B_flat - a_B_flat
                    c2_B_flat = (self.d_sqrt + damping_sq_B) * (dot_au_B ** 2 - sq_norm_a_B) * a_B_flat / (sq_norm_a_B * damping_sq_B + 1e-8)
                    ak_grad_B = (fweight[k] * (self.zero_order_eps ** 2) * (c1_B_flat - c2_B_flat) / damping_sq_B).view(m, r).sum(0) * num_queries_inv

                    c1_A_flat = dot_au_A * u_A_flat - a_A_flat
                    c2_A_flat = (self.d_sqrt + damping_sq_A) * (dot_au_A ** 2 - sq_norm_a_A) * a_A_flat / (sq_norm_a_A * damping_sq_A + 1e-8)
                    ak_grad_A = (fweight[k] * (self.zero_order_eps ** 2) * (c1_A_flat - c2_A_flat) / damping_sq_A).view(r, n).sum(0) * num_queries_inv

                    if self.adam_cov:
                        cov_grad_B_accum[name].add_(ak_grad_B)
                        cov_grad_A_accum[name].add_(ak_grad_A)
                    else:
                        self.evec_a_B[name].add_(ak_grad_B, alpha=-lr_cov_now)
                        self.evec_a_A[name].add_(ak_grad_A, alpha=-lr_cov_now)

        # Adam update for curvature ODE: one step using averaged gradient
        if self.adam_cov:
            self.adam_cov_step += 1
            beta1 = self.adam_cov_beta1
            beta2 = self.adam_cov_beta2
            bias_corr1 = 1.0 - beta1 ** self.adam_cov_step
            bias_corr2 = 1.0 - beta2 ** self.adam_cov_step
            for name in self.lora_B:
                # A curvature
                g_B = cov_grad_B_accum[name]
                self.m_evec_B[name].mul_(beta1).add_(g_B, alpha=1.0 - beta1)
                self.v_evec_B[name].mul_(beta2).addcmul_(g_B, g_B, value=1.0 - beta2)
                m_hat_B = self.m_evec_B[name] / bias_corr1
                v_hat_B = self.v_evec_B[name] / bias_corr2
                self.evec_a_B[name].addcdiv_(m_hat_B, v_hat_B.sqrt().add_(1e-8), value=-lr_cov_now)

                # B curvature
                g_A = cov_grad_A_accum[name]
                self.m_evec_A[name].mul_(beta1).add_(g_A, alpha=1.0 - beta1)
                self.v_evec_A[name].mul_(beta2).addcmul_(g_A, g_A, value=1.0 - beta2)
                m_hat_A = self.m_evec_A[name] / bias_corr1
                v_hat_A = self.v_evec_A[name] / bias_corr2
                self.evec_a_A[name].addcdiv_(m_hat_A, v_hat_A.sqrt().add_(1e-8), value=-lr_cov_now)

        # Per-branch Curvature-VMS (Theorem 5.1):
        # B-branch (left, m×r): Δ_B += τ_B·(c_B·ā_B - c_f·R_B(ā_f))
        # A-branch (right, r×n): Δ_A += τ_A·(c_A·ā_A - c_f'·ā_f)
        # c_f and c_f' are independent scalars with different denominators (m vs r).
        #
        # VMS injection runs EVERY coarse step:
        # - When compute_fine_residual ran this step: update EMA with fresh c_f, then inject
        # - Otherwise: inject using cached c_f_smooth from EMA (no EMA update)

        # VMS diagnostics accumulator
        self._last_vms_diag = {'c_B': [], 'c_A': [], 'c_f_raw': [], 'c_f_prime_raw': [],
                               'adj_ratio_B': [], 'adj_ratio_A': [], 'clipped_count': 0}

        # Determine if we have cached EMA state for VMS injection
        has_ema = (hasattr(self, 'c_f_ema') and self.c_f_ema and
                   hasattr(self, 'c_f_prime_ema') and self.c_f_prime_ema)
        has_fresh_cf = c_f_dict is not None and len(c_f_dict) > 0

        # inject_method: 'continuous' = every step (fresh or cached EMA)
        #                'intermit'   = only when compute_fine_residual ran this step
        if self.inject_method == 'intermit':
            should_inject = has_fresh_cf
        else:  # continuous (default)
            should_inject = has_fresh_cf or has_ema

        if should_inject:
            for name in self.lora_B:
                if name not in self.evec_a:
                    continue

                B = self.lora_B[name]  # left factor (m, r)
                A = self.lora_A[name]  # right factor (r, n)
                m, r = B.shape
                _, n = A.shape

                # Coarse-grid curvature vectors
                a_B = self.evec_a_B[name]  # (m,r) if perrow, (r,) if shared
                a_A = self.evec_a_A[name]  # (r,n) if perrow, (n,) if shared

                # Fine-grid curvature vector
                a_f = self.evec_a[name]  # (n,) - same dimension as a_A's last dim

                # === B-branch (left factor, m×r): uses c_f ===
                a_B_expanded = a_B if self.perrow_precond else a_B.view(1, -1).expand(m, -1)
                sq_norm_a_B = (a_B * a_B).sum().item()
                c_B = (grad_B_accum[name] * a_B_expanded).sum().item() / (self.damping + m * sq_norm_a_B + 1e-8)
                # R_B(ā_f) = ā_f·A^T. When A=0 (fresh coarse phase), fall back to
                # A_old from the previous cycle so B-branch fine injection is non-zero
                # from step 1. Consistent with §6.3 restriction which also uses A_old.
                A_for_R_B = A
                if A.abs().max().item() < 1e-10 and hasattr(self, '_lora_A_history') and name in self._lora_A_history:
                    A_for_R_B = self._lora_A_history[name].to(A.device)
                R_B_af = a_f.view(1, -1).expand(m, -1) @ A_for_R_B.t()  # (m, r)

                # EMA: update with fresh c_f if available, otherwise use cached
                if not hasattr(self, 'c_f_ema'):
                    self.c_f_ema = {}
                if not hasattr(self, 'c_f_ema_count'):
                    self.c_f_ema_count = {}
                if has_fresh_cf and name in c_f_dict:
                    c_f = c_f_dict[name]
                    if name not in self.c_f_ema:
                        self.c_f_ema[name] = 0.0
                        self.c_f_ema_count[name] = 0
                    self.c_f_ema[name] = self.ema_gamma * self.c_f_ema[name] + (1 - self.ema_gamma) * c_f
                    self.c_f_ema_count[name] += 1
                if name in self.c_f_ema and self.c_f_ema_count.get(name, 0) > 0:
                    bias_correction = 1.0 - self.ema_gamma ** self.c_f_ema_count[name]
                    c_f_smooth = self.c_f_ema[name] / bias_correction
                else:
                    continue  # no EMA data yet for this layer, skip VMS

                # Per-branch τ_B
                tau_B = self._compute_tau(name, c_coarse=c_B, a_coarse_expanded=a_B_expanded,
                                         c_f_smooth=c_f_smooth, R_af=R_B_af, branch='B')
                if not hasattr(self, '_last_tau_values'):
                    self._last_tau_values = {}
                self._last_tau_values[name] = tau_B

                curvature_adjustment_B = tau_B * (c_B * a_B_expanded - c_f_smooth * R_B_af)

                # VMS adjustment norm constraint
                adj_norm_B = torch.norm(curvature_adjustment_B).item()
                grad_norm_B = torch.norm(grad_B_accum[name]).item() + 1e-8
                max_adj_ratio = 0.5  # VMS adjustment <= 50% of gradient
                adj_ratio_B = adj_norm_B / grad_norm_B
                if adj_ratio_B > max_adj_ratio:
                    scale_B = max_adj_ratio / adj_ratio_B
                    curvature_adjustment_B.mul_(scale_B)
                    self._last_vms_diag['clipped_count'] += 1

                grad_B_accum[name].add_(curvature_adjustment_B)

                # === A-branch (right factor, r×n): uses c_f' ===
                # a_A and a_f both in ℝⁿ: direct broadcast, no restriction
                # (B^T is already absorbed into c_f', not applied to ā_f)
                R_A_af = a_f.view(1, -1).expand(r, -1)  # (r, n)
                a_A_expanded = a_A if self.perrow_precond else a_A.view(1, -1).expand(r, -1)
                sq_norm_a_A = (a_A * a_A).sum().item()
                c_A = (grad_A_accum[name] * a_A_expanded).sum().item() / (self.damping + r * sq_norm_a_A + 1e-8)

                # EMA: update with fresh c_f' if available, otherwise use cached
                if not hasattr(self, 'c_f_prime_ema'):
                    self.c_f_prime_ema = {}
                if not hasattr(self, 'c_f_prime_ema_count'):
                    self.c_f_prime_ema_count = {}
                if has_fresh_cf and c_f_prime_dict and name in c_f_prime_dict:
                    c_f_prime = c_f_prime_dict[name]
                    if name not in self.c_f_prime_ema:
                        self.c_f_prime_ema[name] = 0.0
                        self.c_f_prime_ema_count[name] = 0
                    self.c_f_prime_ema[name] = self.ema_gamma * self.c_f_prime_ema[name] + (1 - self.ema_gamma) * c_f_prime
                    self.c_f_prime_ema_count[name] += 1
                if name in self.c_f_prime_ema and self.c_f_prime_ema_count.get(name, 0) > 0:
                    bias_correction_A = 1.0 - self.ema_gamma ** self.c_f_prime_ema_count[name]
                    c_f_prime_smooth = self.c_f_prime_ema[name] / bias_correction_A
                else:
                    continue  # no EMA data yet for this layer, skip A-branch VMS

                # Per-branch τ_A
                tau_A = self._compute_tau(name, c_coarse=c_A, a_coarse_expanded=a_A_expanded,
                                         c_f_smooth=c_f_prime_smooth, R_af=R_A_af, branch='A')

                curvature_adjustment_A = tau_A * (c_A * a_A_expanded - c_f_prime_smooth * R_A_af)

                adj_norm_A = torch.norm(curvature_adjustment_A).item()
                grad_norm_A = torch.norm(grad_A_accum[name]).item() + 1e-8
                adj_ratio_A = adj_norm_A / grad_norm_A
                if adj_ratio_A > max_adj_ratio:
                    scale_A = max_adj_ratio / adj_ratio_A
                    curvature_adjustment_A.mul_(scale_A)
                    self._last_vms_diag['clipped_count'] += 1

                grad_A_accum[name].add_(curvature_adjustment_A)

                # Collect diagnostics
                c_f_diag = c_f_dict[name] if (has_fresh_cf and name in c_f_dict) else self.c_f_ema.get(name, 0.0)
                c_f_prime_diag = (c_f_prime_dict[name] if (has_fresh_cf and c_f_prime_dict and name in c_f_prime_dict)
                                  else self.c_f_prime_ema.get(name, 0.0))
                self._last_vms_diag['c_B'].append(c_B)
                self._last_vms_diag['c_A'].append(c_A)
                self._last_vms_diag['c_f_raw'].append(c_f_diag)
                self._last_vms_diag['c_f_prime_raw'].append(c_f_prime_diag)
                self._last_vms_diag['adj_ratio_B'].append(adj_ratio_B)
                self._last_vms_diag['adj_ratio_A'].append(adj_ratio_A)

        # Save old LoRA values before SGD update (for incremental update)
        old_B = {name: B.clone() for name, B in self.lora_B.items()}
        old_A = {name: A.clone() for name, A in self.lora_A.items()}

        # Apply momentum and SGD update
        for name in self.lora_B:
            self.momentum_B[name].mul_(self.beta1).add_(grad_B_accum[name])
            self.momentum_A[name].mul_(self.beta1).add_(grad_A_accum[name])

            sgd_with_wdecay(self.lora_B[name], self.momentum_B[name],
                           self.state.learning_rate, self.weight_decay)
            sgd_with_wdecay(self.lora_A[name], self.momentum_A[name],
                           self.state.learning_rate, self.weight_decay)

        # Apply incremental LoRA update to model params
        # param += (new_B @ new_A) - (old_B @ old_A)
        self._update_lora_in_params(old_B, old_A)

    @torch.no_grad()
    def _log_gradient_svd(self, m_before):
        """
        Gradient SVD diagnostic: compute g_eff = m_after - β^K * m_before
        and run SVD to check if individual gradients are low-rank.

        Called from lnes_zo_update overrides in EVVMS
        """
        from common.logging import logger
        beta_K = self.beta1 ** self.num_queries
        r = self.lora_rank
        energy_ratios = []
        eff_ranks = []
        sv_ratios = []
        sample_info = None

        for name in m_before:
            g_eff = self.momentum[name] - beta_K * m_before[name]
            g_mat = g_eff.to(torch.float32)
            if g_mat.ndim > 2:
                g_mat = g_mat.view(g_mat.shape[0], -1)
            if g_mat.ndim < 2:
                continue

            m_dim, n_dim = g_mat.shape
            r_eff = min(r, m_dim, n_dim)
            try:
                U, S, Vh = torch.linalg.svd(g_mat, full_matrices=False)
            except RuntimeError:
                # CUSOLVER can fail on large matrices when GPU memory is tight
                logger.info(f"    [Grad SVD] skipped: cusolver OOM on ({m_dim},{n_dim}) matrix")
                continue
            total_e = (S ** 2).sum().item()
            top_r_e = (S[:r_eff] ** 2).sum().item()
            er = top_r_e / (total_e + 1e-10)
            efr = (S.sum() ** 2 / ((S ** 2).sum() + 1e-10)).item()
            svr = (S[0] / (S[min(r_eff, len(S)) - 1] + 1e-10)).item()

            energy_ratios.append(er)
            eff_ranks.append(efr)
            sv_ratios.append(svr)

            if sample_info is None:
                sample_info = {
                    'name': name,
                    'top_sv': S[:min(8, len(S))].cpu().tolist(),
                    'er': er,
                    'efr': efr,
                }

        if energy_ratios:
            avg_er = sum(energy_ratios) / len(energy_ratios)
            avg_efr = sum(eff_ranks) / len(eff_ranks)
            avg_svr = sum(sv_ratios) / len(sv_ratios)

            logger.info(f"    [Grad SVD] step={self.state.global_training_steps}: "
                       f"avg energy_ratio(r={r})={avg_er:.4f}, "
                       f"avg eff_rank={avg_efr:.1f}, avg σ₁/σᵣ={avg_svr:.1f}")
            if sample_info:
                short_name = '.'.join(sample_info['name'].split('.')[-2:])
                logger.info(f"      Sample ({short_name}): "
                           f"SVs=[{', '.join(f'{s:.4f}' for s in sample_info['top_sv'])}]")

            self._grad_svd_history.append({
                'step': self.state.global_training_steps,
                'avg_energy_ratio': avg_er,
                'avg_effective_rank': avg_efr,
                'avg_sv_ratio': avg_svr,
            })

    @torch.no_grad()
    def lnes_zo_update(self, random_seeds, fvals, fmean):
        """
        Override parent LOREN's update to add gradient SVD diagnostic.

        Captures per-step gradient g = m_after - β^K * m_before and runs SVD
        to check if individual gradients are low-rank (explaining why LoRA works).
        """
        capture = (
            self._grad_svd_diagnostic and
            self.get_phase() == 'fine' and
            self.state.global_training_steps % self._grad_svd_interval == 0
        )

        if capture:
            # Only sample a few layers to avoid cloning all momentum (~model_size).
            # Pick first/mid/last 2D params for representative SVD diagnostics.
            m_before = {}
            candidates = [(n, p) for n, p in self.params
                          if p.ndim >= 2 and n in self.momentum]
            if candidates:
                indices = {0, len(candidates) // 2, len(candidates) - 1}
                for idx in indices:
                    name, _ = candidates[idx]
                    m_before[name] = self.momentum[name].clone()

        # Call parent LOREN update
        super().lnes_zo_update(random_seeds, fvals, fmean)

        if capture and m_before:
            self._log_gradient_svd(m_before)

    def training_step(self, model, batch, tb_writer=None, **kwargs):
        """
        Single EV-VMS training step.
        """
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seeds = [np.random.randint(1000000000) for _ in range(self.num_queries)]
        log_dict = kwargs.get('log', {})

        # Clear stale tau values from previous step (only log fresh values from this step's SPSA)
        if hasattr(self, '_last_tau_values'):
            self._last_tau_values = {}

        # Ensure LoRA is applied
        self._apply_lora_to_params()

        # Evaluate at K perturbed points (RLOO)
        fvals = []
        with torch.inference_mode():
            for k in range(self.num_queries):
                self.perturb_lora_parameters(random_seeds[k], scaling_factor=1)
                loss = self.zo_forward_memory_eff(model, batch)
                fvals.append(loss.item())
                self.perturb_lora_parameters(random_seeds[k], scaling_factor=-1)

        fmean = np.mean(fvals)

        # Curvature-VMS: Extract c_f (curvature projection coefficients) periodically
        # Uses RLOO with num_queries forward passes for variance reduction
        c_f_dict = None
        c_f_prime_dict = None
        fine_grid_queries = 0
        if self.use_fine_residual:
            cycle_local_epoch = self._tau_current_epoch - self._cycle_start_epoch
            warmup_factor = min(1.0, cycle_local_epoch / max(1, self.tau_warmup_epochs))
            # During warmup first half (warmup_factor < 0.5), run RLOO every 5 steps
            # After warmup first half, run RLOO at normal interval
            spsa_interval = self.fine_residual_interval
            if warmup_factor < 0.5:
                spsa_interval = max(self.fine_residual_interval, 5)

            if self.state.global_training_steps % spsa_interval == 0:
                fine_seeds = [np.random.randint(1000000000) for _ in range(self.num_queries)]
                c_f_dict, c_f_prime_dict, fine_loss = self.compute_fine_residual(model, batch, fine_seeds)
                fine_grid_queries = self.num_queries  # RLOO uses num_queries forward passes

        # CT-LOREN: Curvature Transfer from full-space to LoRA
        ct_queries = 0
        if self.curvature_transfer and self.state.global_training_steps >= self.ct_warmup_steps:
            if self.state.global_training_steps % self.ct_interval == 0:
                self.ct_probe_and_transfer(model, batch)
                ct_queries = 4  # 2 SPSA queries × 2 forward passes (two-sided)

        # EV-VMS update with Curvature-VMS
        self.evvms_zo_update(random_seeds, fvals, fmean, c_f_dict, c_f_prime_dict)
        self.state.global_training_steps += 1

        # Store fvals for subclass terrain sensing
        self._last_fvals = fvals
        self._last_fmean = fmean

        total_time = time.perf_counter() - start_time
        self.state.time.append(total_time)
        # Query count = number of forward passes (NOT multiplied by batch size)
        self.state.query.append(self.num_queries + fine_grid_queries + ct_queries)

        log_dict['train_loss'] = fmean
        log_dict['time'] = total_time

        # VMS diagnostic logging
        if hasattr(self, 'c_f_ema') and self.c_f_ema:
            mean_c_f = sum(self.c_f_ema.values()) / len(self.c_f_ema)
            log_dict['vms_c_f_mean'] = mean_c_f
        if hasattr(self, 'c_f_prime_ema') and self.c_f_prime_ema:
            mean_c_f_prime = sum(self.c_f_prime_ema.values()) / len(self.c_f_prime_ema)
            log_dict['vms_c_f_prime_mean'] = mean_c_f_prime
        if hasattr(self, '_last_tau_values') and self._last_tau_values:
            mean_tau = sum(self._last_tau_values.values()) / len(self._last_tau_values)
            log_dict['vms_tau_mean'] = mean_tau
        if hasattr(self, '_last_vms_diag') and self._last_vms_diag:
            diag = self._last_vms_diag
            n = len(diag['c_B'])
            if n > 0:
                log_dict['vms_c_B_mean'] = sum(diag['c_B']) / n
                log_dict['vms_c_A_mean'] = sum(diag['c_A']) / n if diag['c_A'] else 0.0
                log_dict['vms_c_f_raw_mean'] = sum(diag['c_f_raw']) / n
                log_dict['vms_c_f_prime_raw_mean'] = sum(diag['c_f_prime_raw']) / n if diag['c_f_prime_raw'] else 0.0
                log_dict['vms_adj_ratio_B_mean'] = sum(diag['adj_ratio_B']) / n
                log_dict['vms_adj_ratio_A_mean'] = sum(diag['adj_ratio_A']) / n
                log_dict['vms_clipped_count'] = diag['clipped_count']

        # lr_cov schedule logging
        if self.lr_cov_schedule != 'constant':
            log_dict['lr_cov_now'] = self._get_lr_cov()

        # CT-LOREN diagnostic logging
        if self.curvature_transfer and hasattr(self, '_last_ct_diag') and self._last_ct_diag:
            ct_diag = self._last_ct_diag
            n = len(ct_diag['rho'])
            if n > 0:
                log_dict['ct_rho_mean'] = sum(ct_diag['rho']) / n
                log_dict['ct_norm_A_mean'] = sum(ct_diag['norm_ct_A']) / n
                log_dict['ct_norm_B_mean'] = sum(ct_diag['norm_ct_B']) / n

    # =====================================================
    # Override configure_params to also configure LoRA
    # =====================================================
    def configure_params(self):
        """Configure both LOREN params and LoRA params."""
        super().configure_params()
        # Upcast evec_a to fp32 to match LoRA matrices (avoids bf16/fp32 dtype mismatch
        # in VMS correction, CT probe, prolongation, and restriction operations)
        if self.lora_fp32:
            self.evec_a = {k: v.to(torch.float32) for k, v in self.evec_a.items()}
        # Release full-size momentum created by LOREN base class.
        # Coarse phase uses rank-sized momentum_B/B; fine phase recreates
        # self.momentum via prolongation_to_fine(). Saves ~model_size GPU memory
        # (e.g. ~26 GB for OPT-13B bf16).
        self.momentum = {}
        self.configure_lora_params()

    # =====================================================
    # Checkpoint methods
    # =====================================================
    def save_checkpoint(self, checkpoint_path: str, save_full_model: bool = False,
                        shared_dir: str = None, epoch_tag: int = None):
        """
        Save EV-VMS state (lightweight by default).

        Args:
            checkpoint_path: Directory to save checkpoint
            save_full_model: If True, also save full model weights (for Phase 3 resume)
            shared_dir: If provided, save model to shared/model.pt (single copy,
                        overwritten on each save) and create symlinks from checkpoint_path
            epoch_tag: Epoch number for logging (required with shared_dir)

        Checkpoint layout (with shared_dir):
          shared/model.pt        — latest fine model (shared for F2C loading)
          best_coarse/model.pt   — direct save (W + B@A merged, specific to this best)
          best_coarse/lora_state.pt, params_1d.pt, train_state.pkl
          best_fine/model.pt     → ../shared/model.pt (symlink)
          best_fine/fine_state.pt (evec_a + momentum only, no model embedded)
          best_fine/train_state.pkl
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        use_shared = shared_dir is not None and epoch_tag is not None

        # Save training state (small, always direct)
        pickle.dump(self.state, open(os.path.join(checkpoint_path, 'train_state.pkl'), 'wb'))

        # Check if we're in Phase 3 (fine grid mode) where LoRA structures are released
        if self.get_phase() == 'fine':
            # Phase 3: Save model weights and fine-grid LOREN state SEPARATELY.
            # Model goes to shared/model.pt (overwritten, single copy shared by coarse & fine).
            # Fine state (evec_a + momentum) saved directly in checkpoint dir (small, ~12 MB).
            fine_state = {
                'evec_a': {k: v.cpu() for k, v in self.evec_a.items()},
                'momentum': {k: v.cpu() for k, v in self.momentum.items()},
                'phase': 'fine',
                '_lora_A_history': {k: v.cpu() for k, v in self._lora_A_history.items()} if hasattr(self, '_lora_A_history') and self._lora_A_history else {},
            }

            if use_shared:
                os.makedirs(shared_dir, exist_ok=True)
                # Model: overwrite shared/model.pt (single copy, always latest)
                model_shared = os.path.join(shared_dir, 'model.pt')
                torch.save(self.model.state_dict(), model_shared)
                _force_symlink(model_shared, os.path.join(checkpoint_path, 'model.pt'))
                # Clean up legacy epoch-tagged files from old checkpoint format
                import glob as _glob
                for old in (_glob.glob(os.path.join(shared_dir, 'fine_state_ep*.pt')) +
                            _glob.glob(os.path.join(shared_dir, 'model_ep*.pt'))):
                    try:
                        os.remove(old)
                    except (FileNotFoundError, OSError):
                        pass
            else:
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, 'model.pt'))
            # Fine state: saved directly (no model embedded)
            torch.save(fine_state, os.path.join(checkpoint_path, 'fine_state.pt'))
        else:
            # Phase 1/2: Save LoRA state + 1D parameters (lightweight, always direct)
            # Curvature-VMS fix: also save evec_a and c_f_ema (claude-review-2026-02-25-4.md §4.1)
            lora_state = {
                'lora_B': {k: v.cpu() for k, v in self.lora_B.items()},
                'lora_A': {k: v.cpu() for k, v in self.lora_A.items()},
                'evec_a_B': {k: v.cpu() for k, v in self.evec_a_B.items()},
                'evec_a_A': {k: v.cpu() for k, v in self.evec_a_A.items()},
                'momentum_B': {k: v.cpu() for k, v in self.momentum_B.items()},
                'momentum_A': {k: v.cpu() for k, v in self.momentum_A.items()},
                # Curvature-VMS: save fine-grid curvature (used in Phase 1)
                'evec_a': {k: v.cpu() for k, v in self.evec_a.items()},
                'c_f_ema': getattr(self, 'c_f_ema', {}),
                'c_f_ema_count': getattr(self, 'c_f_ema_count', {}),
                'c_f_prime_ema': getattr(self, 'c_f_prime_ema', {}),
                'c_f_prime_ema_count': getattr(self, 'c_f_prime_ema_count', {}),
                'tau_ema': getattr(self, 'tau_ema', {}),
                'curvature_transfer': self.curvature_transfer,
                # Adam curvature ODE state (only when adam_cov=True)
                'adam_cov': self.adam_cov,
                'adam_cov_step': getattr(self, 'adam_cov_step', 0),
                'm_evec_B': {k: v.cpu() for k, v in self.m_evec_B.items()} if self.adam_cov and hasattr(self, 'm_evec_B') else {},
                'v_evec_B': {k: v.cpu() for k, v in self.v_evec_B.items()} if self.adam_cov and hasattr(self, 'v_evec_B') else {},
                'm_evec_A': {k: v.cpu() for k, v in self.m_evec_A.items()} if self.adam_cov and hasattr(self, 'm_evec_A') else {},
                'v_evec_A': {k: v.cpu() for k, v in self.v_evec_A.items()} if self.adam_cov and hasattr(self, 'v_evec_A') else {},
                # Basis selection state (oja/svd/svd_hybrid/online_oja)
                'basis_method': self.basis_method,
                '_lora_basis_history': {k: v.cpu() for k, v in self._lora_basis_history.items()} if self.basis_method != 'random' else {},
                '_lora_A_history': {k: v.cpu() for k, v in self._lora_A_history.items()} if hasattr(self, '_lora_A_history') and self._lora_A_history else {},
                '_oja_cycle_idx': self._oja_cycle_idx,
                '_momentum_ema_buffer': {k: v.cpu() for k, v in self._momentum_ema_buffer.items()} if self._momentum_ema_buffer else {},
            }
            torch.save(lora_state, os.path.join(checkpoint_path, 'lora_state.pt'))

            # Save 1D parameters (bias, LayerNorm, etc.) - these are trained directly
            params_1d = {}
            for name, param in self.params:
                if param.ndim == 1:
                    params_1d[name] = param.data.cpu()
            torch.save(params_1d, os.path.join(checkpoint_path, 'params_1d.pt'))

            # Save full model weights (coarse model = W + B@A, specific to this checkpoint)
            # Coarse model is always saved directly (not shared) because each coarse best
            # has a different LoRA delta merged into the model.
            if save_full_model:
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, 'model.pt'))

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load EV-VMS state from checkpoint.

        Supports three checkpoint types:
        1. Phase 3 checkpoint: model.pt + fine_state.pt (full model + fine grid state)
        2. Phase 1 full checkpoint: model.pt + lora_state.pt (full model + LoRA state)
        3. Phase 1 lightweight checkpoint: lora_state.pt only (requires first checkpoint)
        """
        from common.logging import logger as logging
        device = next(self.model.parameters()).device

        # Load training state
        self.state = pickle.load(open(os.path.join(checkpoint_path, 'train_state.pkl'), 'rb'))

        # Check for Phase 3 checkpoint (fine_state.pt)
        fine_state_path = os.path.join(checkpoint_path, 'fine_state.pt')
        model_path = os.path.join(checkpoint_path, 'model.pt')

        if os.path.exists(fine_state_path):
            # Phase 3 checkpoint: load full model + fine grid LOREN state
            logging.info(f"  Loading Phase 3 checkpoint (fine grid state)")
            fine_state = torch.load(fine_state_path, map_location='cpu')
            if os.path.exists(model_path):
                # Current format: model stored separately (model.pt or symlink → shared/model.pt)
                model_data = torch.load(model_path, map_location='cpu')
                if isinstance(model_data, dict) and 'model' in model_data and 'phase' in model_data:
                    model_data = model_data['model']  # unwrap legacy fine_state format
                self.model.load_state_dict(model_data)
            elif 'model' in fine_state:
                # Legacy embedded format: model weights inside fine_state
                self.model.load_state_dict(fine_state['model'])
            else:
                raise FileNotFoundError(
                    f"fine_state checkpoint has no embedded model and model.pt not found at {model_path}")
            self.evec_a = {k: v.to(device=device, dtype=torch.float32) for k, v in fine_state['evec_a'].items()}
            self.momentum = {k: v.to(device) for k, v in fine_state['momentum'].items()}
            if '_lora_A_history' in fine_state and fine_state['_lora_A_history']:
                if not hasattr(self, '_lora_A_history'):
                    self._lora_A_history = {}
                if not self._lora_A_history:
                    self._lora_A_history = {k: v.to(device) for k, v in fine_state['_lora_A_history'].items()}

            # Set phase to fine and mark LoRA as not applicable
            self._current_phase = 'fine'
            self.lora_applied = True  # Model weights are already in fine grid form
            logging.info(f"  Phase 3 state restored: {len(self.evec_a)} parameter groups")
            return

        # Check if full model weights are available (most reliable)
        if os.path.exists(model_path):
            # Load full model weights — model.pt is a plain state_dict (or symlink → shared/model.pt).
            # Legacy backward compat: may be a fine_state dict with embedded model.
            model_data = torch.load(model_path, map_location='cpu')
            if isinstance(model_data, dict) and 'model' in model_data and 'phase' in model_data:
                model_data = model_data['model']  # unwrap legacy fine_state format
            self.model.load_state_dict(model_data)
            self.lora_applied = True  # Model already has LoRA delta applied

            # Load LoRA state for continued training
            lora_path = os.path.join(checkpoint_path, 'lora_state.pt')
            if os.path.exists(lora_path):
                lora_state = torch.load(lora_path, map_location='cpu')
                self.lora_B = {k: v.to(device) for k, v in lora_state['lora_B'].items()}
                self.lora_A = {k: v.to(device) for k, v in lora_state['lora_A'].items()}
                self.evec_a_B = {k: v.to(device) for k, v in lora_state['evec_a_B'].items()}
                self.evec_a_A = {k: v.to(device) for k, v in lora_state['evec_a_A'].items()}
                self.momentum_B = {k: v.to(device) for k, v in lora_state['momentum_B'].items()}
                self.momentum_A = {k: v.to(device) for k, v in lora_state['momentum_A'].items()}
                # Curvature-VMS: load fine-grid curvature (claude-review-2026-02-25-4.md §4.1)
                if 'evec_a' in lora_state:
                    self.evec_a = {k: v.to(device=device, dtype=torch.float32) for k, v in lora_state['evec_a'].items()}
                if 'c_f_ema' in lora_state:
                    self.c_f_ema = lora_state['c_f_ema']
                if 'c_f_ema_count' in lora_state:
                    self.c_f_ema_count = lora_state['c_f_ema_count']
                elif 'c_f_ema' in lora_state:
                    # Backward compat: old checkpoint without count, set large value so bias_correction ≈ 1
                    self.c_f_ema_count = {k: 100 for k in self.c_f_ema}
                if 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema = lora_state['c_f_prime_ema']
                if 'c_f_prime_ema_count' in lora_state:
                    self.c_f_prime_ema_count = lora_state['c_f_prime_ema_count']
                elif 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema_count = {k: 100 for k in self.c_f_prime_ema}
                if 'tau_ema' in lora_state:
                    self.tau_ema = lora_state['tau_ema']
                # Adam curvature ODE state
                if lora_state.get('adam_cov') and self.adam_cov:
                    if 'm_evec_B' in lora_state and lora_state['m_evec_B']:
                        self.m_evec_B = {k: v.to(device) for k, v in lora_state['m_evec_B'].items()}
                    if 'v_evec_B' in lora_state and lora_state['v_evec_B']:
                        self.v_evec_B = {k: v.to(device) for k, v in lora_state['v_evec_B'].items()}
                    if 'm_evec_A' in lora_state and lora_state['m_evec_A']:
                        self.m_evec_A = {k: v.to(device) for k, v in lora_state['m_evec_A'].items()}
                    if 'v_evec_A' in lora_state and lora_state['v_evec_A']:
                        self.v_evec_A = {k: v.to(device) for k, v in lora_state['v_evec_A'].items()}
                    if 'adam_cov_step' in lora_state:
                        self.adam_cov_step = lora_state['adam_cov_step']
                # Basis selection state: use max/non-empty guard to avoid
                # V-cycle checkpoint loads overwriting more recent in-memory state
                if '_lora_basis_history' in lora_state and self.basis_method != 'random' and not self._lora_basis_history:
                    self._lora_basis_history = {k: v.to(device) for k, v in lora_state['_lora_basis_history'].items()}
                if '_lora_A_history' in lora_state and lora_state['_lora_A_history']:
                    if not hasattr(self, '_lora_A_history'):
                        self._lora_A_history = {}
                    if not self._lora_A_history:
                        self._lora_A_history = {k: v.to(device) for k, v in lora_state['_lora_A_history'].items()}
                if '_oja_cycle_idx' in lora_state:
                    self._oja_cycle_idx = max(self._oja_cycle_idx, lora_state['_oja_cycle_idx'])
                if '_momentum_ema_buffer' in lora_state and not self._momentum_ema_buffer:
                    self._momentum_ema_buffer = lora_state['_momentum_ema_buffer']  # stays on CPU
        else:
            # Lightweight restoration: need base model from first checkpoint
            from common.logging import logger as logging
            import glob
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Dynamically find the first periodic checkpoint with model.pt
            pattern = os.path.join(checkpoint_dir, 'periodic_epoch*')
            periodic_checkpoints = sorted(glob.glob(pattern),
                                          key=lambda x: int(x.split('epoch')[-1]))
            first_checkpoint = None
            for ckpt in periodic_checkpoints:
                if os.path.exists(os.path.join(ckpt, 'model.pt')):
                    first_checkpoint = ckpt
                    break

            if first_checkpoint is None:
                # Try to find initial_model.pt (saved at Phase 1 start)
                initial_model_path = os.path.join(checkpoint_dir, 'initial_model.pt')
                if os.path.exists(initial_model_path):
                    logging.info(f"  Using initial model from Phase 1 start: {initial_model_path}")
                    self.model.load_state_dict(torch.load(initial_model_path, map_location='cpu'))

                    # Load current checkpoint's LoRA state and apply delta
                    lora_path = os.path.join(checkpoint_path, 'lora_state.pt')
                    lora_state = torch.load(lora_path, map_location='cpu')
                    self.lora_B = {k: v.to(device) for k, v in lora_state['lora_B'].items()}
                    self.lora_A = {k: v.to(device) for k, v in lora_state['lora_A'].items()}
                    self.evec_a_B = {k: v.to(device) for k, v in lora_state['evec_a_B'].items()}
                    self.evec_a_A = {k: v.to(device) for k, v in lora_state['evec_a_A'].items()}
                    self.momentum_B = {k: v.to(device) for k, v in lora_state['momentum_B'].items()}
                    self.momentum_A = {k: v.to(device) for k, v in lora_state['momentum_A'].items()}
                    if 'evec_a' in lora_state:
                        self.evec_a = {k: v.to(device=device, dtype=torch.float32) for k, v in lora_state['evec_a'].items()}
                    if 'c_f_ema' in lora_state:
                        self.c_f_ema = lora_state['c_f_ema']
                    if 'c_f_ema_count' in lora_state:
                        self.c_f_ema_count = lora_state['c_f_ema_count']
                    elif 'c_f_ema' in lora_state:
                        self.c_f_ema_count = {k: 100 for k in self.c_f_ema}
                    if 'c_f_prime_ema' in lora_state:
                        self.c_f_prime_ema = lora_state['c_f_prime_ema']
                    if 'c_f_prime_ema_count' in lora_state:
                        self.c_f_prime_ema_count = lora_state['c_f_prime_ema_count']
                    elif 'c_f_prime_ema' in lora_state:
                        self.c_f_prime_ema_count = {k: 100 for k in self.c_f_prime_ema}
                    if 'tau_ema' in lora_state:
                        self.tau_ema = lora_state['tau_ema']
                    if '_lora_A_history' in lora_state and lora_state['_lora_A_history']:
                        if not hasattr(self, '_lora_A_history'):
                            self._lora_A_history = {}
                        if not self._lora_A_history:
                            self._lora_A_history = {k: v.to(device) for k, v in lora_state['_lora_A_history'].items()}
                    # Adam curvature ODE state
                    if lora_state.get('adam_cov') and self.adam_cov:
                        if 'm_evec_B' in lora_state and lora_state['m_evec_B']:
                            self.m_evec_B = {k: v.to(device) for k, v in lora_state['m_evec_B'].items()}
                        if 'v_evec_B' in lora_state and lora_state['v_evec_B']:
                            self.v_evec_B = {k: v.to(device) for k, v in lora_state['v_evec_B'].items()}
                        if 'm_evec_A' in lora_state and lora_state['m_evec_A']:
                            self.m_evec_A = {k: v.to(device) for k, v in lora_state['m_evec_A'].items()}
                        if 'v_evec_A' in lora_state and lora_state['v_evec_A']:
                            self.v_evec_A = {k: v.to(device) for k, v in lora_state['v_evec_A'].items()}
                        if 'adam_cov_step' in lora_state:
                            self.adam_cov_step = lora_state['adam_cov_step']

                    # Load 1D params
                    params_1d_path = os.path.join(checkpoint_path, 'params_1d.pt')
                    if os.path.exists(params_1d_path):
                        params_1d = torch.load(params_1d_path, map_location='cpu')
                        for name, param in self.params:
                            if name in params_1d:
                                param.data.copy_(params_1d[name].to(device))

                    self.lora_applied = False  # Need to apply LoRA delta
                    logging.info(f"  Checkpoint restored using initial_model.pt + LoRA state")
                    return
                else:
                    logging.warning(f"  No periodic checkpoint or initial_model.pt found in {checkpoint_dir}")
                    logging.warning(f"  Falling back to pretrained weights - may cause inconsistency!")
                first_model_path = None
                first_lora_path = None
            else:
                first_model_path = os.path.join(first_checkpoint, 'model.pt')
                first_lora_path = os.path.join(first_checkpoint, 'lora_state.pt')

            if first_model_path and os.path.exists(first_model_path) and os.path.exists(first_lora_path):
                # Step 1: Load base model from first checkpoint (W_0 + delta_10)
                logging.info(f"  Loading base model from first checkpoint: {first_checkpoint}")
                self.model.load_state_dict(torch.load(first_model_path, map_location='cpu'))

                # Step 2: Load first checkpoint's LoRA to remove delta_10
                first_lora = torch.load(first_lora_path, map_location='cpu')
                for name, param in self.params:
                    if name in first_lora['lora_B']:
                        delta_10 = first_lora['lora_B'][name].to(device) @ first_lora['lora_A'][name].to(device)
                        if param.ndim > 2:
                            delta_10 = delta_10.view(param.shape)
                        param.data.sub_(delta_10)  # W_0 + delta_10 - delta_10 = W_0

                # Step 3: Load current checkpoint's LoRA state
                lora_path = os.path.join(checkpoint_path, 'lora_state.pt')
                lora_state = torch.load(lora_path, map_location='cpu')
                self.lora_B = {k: v.to(device) for k, v in lora_state['lora_B'].items()}
                self.lora_A = {k: v.to(device) for k, v in lora_state['lora_A'].items()}
                self.evec_a_B = {k: v.to(device) for k, v in lora_state['evec_a_B'].items()}
                self.evec_a_A = {k: v.to(device) for k, v in lora_state['evec_a_A'].items()}
                self.momentum_B = {k: v.to(device) for k, v in lora_state['momentum_B'].items()}
                self.momentum_A = {k: v.to(device) for k, v in lora_state['momentum_A'].items()}
                # Load fine-grid curvature and c_f EMA for Curvature-VMS
                if 'evec_a' in lora_state:
                    self.evec_a = {k: v.to(device=device, dtype=torch.float32) for k, v in lora_state['evec_a'].items()}
                if 'c_f_ema' in lora_state:
                    self.c_f_ema = lora_state['c_f_ema']
                if 'c_f_ema_count' in lora_state:
                    self.c_f_ema_count = lora_state['c_f_ema_count']
                elif 'c_f_ema' in lora_state:
                    self.c_f_ema_count = {k: 100 for k in self.c_f_ema}
                if 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema = lora_state['c_f_prime_ema']
                if 'c_f_prime_ema_count' in lora_state:
                    self.c_f_prime_ema_count = lora_state['c_f_prime_ema_count']
                elif 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema_count = {k: 100 for k in self.c_f_prime_ema}
                if 'tau_ema' in lora_state:
                    self.tau_ema = lora_state['tau_ema']
                # Adam curvature ODE state
                if lora_state.get('adam_cov') and self.adam_cov:
                    if 'm_evec_B' in lora_state and lora_state['m_evec_B']:
                        self.m_evec_B = {k: v.to(device) for k, v in lora_state['m_evec_B'].items()}
                    if 'v_evec_B' in lora_state and lora_state['v_evec_B']:
                        self.v_evec_B = {k: v.to(device) for k, v in lora_state['v_evec_B'].items()}
                    if 'm_evec_A' in lora_state and lora_state['m_evec_A']:
                        self.m_evec_A = {k: v.to(device) for k, v in lora_state['m_evec_A'].items()}
                    if 'v_evec_A' in lora_state and lora_state['v_evec_A']:
                        self.v_evec_A = {k: v.to(device) for k, v in lora_state['v_evec_A'].items()}
                    if 'adam_cov_step' in lora_state:
                        self.adam_cov_step = lora_state['adam_cov_step']
                if '_lora_A_history' in lora_state and lora_state['_lora_A_history']:
                    if not hasattr(self, '_lora_A_history'):
                        self._lora_A_history = {}
                    if not self._lora_A_history:
                        self._lora_A_history = {k: v.to(device) for k, v in lora_state['_lora_A_history'].items()}

                # Step 4: Apply current checkpoint's delta (W_0 + delta_current)
                self.lora_applied = False
                self._apply_lora_to_params()
            else:
                logging.warning(f"  First checkpoint not found at {first_checkpoint}")
                logging.warning(f"  Falling back to pretrained weights - may cause inconsistency!")

                # Fallback: load from pretrained (not recommended)
                lora_path = os.path.join(checkpoint_path, 'lora_state.pt')
                lora_state = torch.load(lora_path, map_location='cpu')
                self.lora_B = {k: v.to(device) for k, v in lora_state['lora_B'].items()}
                self.lora_A = {k: v.to(device) for k, v in lora_state['lora_A'].items()}
                self.evec_a_B = {k: v.to(device) for k, v in lora_state['evec_a_B'].items()}
                self.evec_a_A = {k: v.to(device) for k, v in lora_state['evec_a_A'].items()}
                self.momentum_B = {k: v.to(device) for k, v in lora_state['momentum_B'].items()}
                self.momentum_A = {k: v.to(device) for k, v in lora_state['momentum_A'].items()}
                # Load fine-grid curvature and c_f EMA for Curvature-VMS
                if 'evec_a' in lora_state:
                    self.evec_a = {k: v.to(device=device, dtype=torch.float32) for k, v in lora_state['evec_a'].items()}
                if 'c_f_ema' in lora_state:
                    self.c_f_ema = lora_state['c_f_ema']
                if 'c_f_ema_count' in lora_state:
                    self.c_f_ema_count = lora_state['c_f_ema_count']
                elif 'c_f_ema' in lora_state:
                    self.c_f_ema_count = {k: 100 for k in self.c_f_ema}
                if 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema = lora_state['c_f_prime_ema']
                if 'c_f_prime_ema_count' in lora_state:
                    self.c_f_prime_ema_count = lora_state['c_f_prime_ema_count']
                elif 'c_f_prime_ema' in lora_state:
                    self.c_f_prime_ema_count = {k: 100 for k in self.c_f_prime_ema}
                if 'tau_ema' in lora_state:
                    self.tau_ema = lora_state['tau_ema']
                if '_lora_A_history' in lora_state and lora_state['_lora_A_history']:
                    if not hasattr(self, '_lora_A_history'):
                        self._lora_A_history = {}
                    if not self._lora_A_history:
                        self._lora_A_history = {k: v.to(device) for k, v in lora_state['_lora_A_history'].items()}
                self.lora_applied = False
                self._apply_lora_to_params()

            # Step 5: Load 1D parameters from CURRENT checkpoint
            params_1d_path = os.path.join(checkpoint_path, 'params_1d.pt')
            if os.path.exists(params_1d_path):
                params_1d = torch.load(params_1d_path, map_location='cpu')
                for name, param in self.params:
                    if name in params_1d:
                        param.data.copy_(params_1d[name].to(device))

    def measure_memory_usage(self):
        """Override to include LoRA memory."""
        return torch.cuda.max_memory_allocated() / (1024 ** 3)

    # =====================================================
    # Phase 2: Fine Grid Reconstruction (Memory Optimized)
    # =====================================================
    def _release_coarse_structures(self):
        """
        Release coarse grid structures to free GPU memory.

        This should be called AFTER:
        1. LoRA delta is applied to model params (W = W_0 + B @ A)
        2. evec_a_A is backed up for curvature reconstruction

        Memory freed (GPT-2-XL estimate):
        - lora_B, lora_A: ~90 MB
        - evec_a_B, evec_a_A: ~10 MB
        - momentum_B, momentum_A: ~180 MB
        Total: ~280 MB
        """
        # Delete coarse grid structures
        if hasattr(self, 'lora_B'):
            del self.lora_B
        if hasattr(self, 'lora_A'):
            del self.lora_A
        if hasattr(self, 'evec_a_B'):
            del self.evec_a_B
        if hasattr(self, 'evec_a_A'):
            del self.evec_a_A
        if hasattr(self, 'momentum_B'):
            del self.momentum_B
        if hasattr(self, 'momentum_A'):
            del self.momentum_A
        if hasattr(self, 'lora_param_info'):
            del self.lora_param_info

        # Force CUDA memory release
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reconstruct_fine_grid(self, model, batch, num_smoothing_steps: int = 2):
        """
        Phase 2: Reconstruct fine grid parameters from coarse grid (LoRA).

        Steps:
        1. Prolongation: θ_f = W_0 + B @ A (already applied)
        2. Release coarse grid structures
        3. Initialize fine grid momentum (zeros)
        4. Post-smoothing steps (2 LOREN steps for warm start)

        After this, the model is ready for Phase 3 (fine grid LOREN training).
        """
        from common.logging import logger

        logger.info("=" * 60)
        logger.info("Phase 2: Fine Grid Reconstruction")
        logger.info("=" * 60)

        # Step 1: Apply prolongation (W = W_0 + B @ A)
        self._apply_lora_to_params()
        logger.info("  [1/4] Prolongation: θ_f = W_0 + B @ A")

        # Step 2: Release coarse grid structures to free memory
        # Note: evec_a is preserved (already converged from Phase 1 Curvature-VMS)
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        self._release_coarse_structures()
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        logger.info(f"  [2/4] Released coarse grid structures (freed {mem_before - mem_after:.1f} MB)")
        logger.info(f"        evec_a preserved ({len(self.evec_a)} tensors, already converged)")

        # Step 3: Initialize fine grid momentum (zeros — clean start for LOREN)
        self.momentum = {}
        for name, param in self.params:
            self.momentum[name] = torch.zeros_like(param.data)
        logger.info(f"  [3/4] Fine grid momentum initialized (zeros)")

        # Step 4: Post-smoothing steps (warm start for fine grid LOREN)
        if num_smoothing_steps > 0:
            logger.info(f"  [4/4] Post-smoothing ({num_smoothing_steps} steps)...")
            dummy_log = {}
            for step in range(num_smoothing_steps):
                super().training_step(model, batch, log=dummy_log)
        else:
            logger.info("  [4/4] Post-smoothing skipped")

        mem_final = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        logger.info(f"Phase 2 Complete: Ready for fine grid training (GPU mem: {mem_final:.1f} MB)")
        logger.info("=" * 60)

        # Return current loss for logging
        with torch.no_grad():
            outputs = model(**batch)
            fine_loss = outputs.loss.item()
        return fine_loss

    @torch.no_grad()
    def absorb_lora(self):
        """
        Periodic LoRA absorption (multi V-cycle synchronization).

        Absorbs current LoRA into W_0, then re-initializes LoRA for continued
        coarse-grid training. This keeps the fine-grid base W_0 synchronized
        with the coarse-grid evolution.

        Theory (claude-vms-theory-2026-03-02-markdown.md §10.4):
        In classical VMS, the fine-grid coefficient matrix is fixed.
        In optimization, H(θ) changes with θ. Periodic absorption ensures
        the fine-grid base tracks the coarse-grid, forming multi V-cycle training.

        Steps:
        1. W_0 ← W_0 + B@A (absorption)
        2. Re-init A, B (small random)
        3. Reset momentum_B, momentum_A (zeros)
        4. Re-init evec_a_B, evec_a_A (small random)
        5. Clear VMS EMA state
        6. Preserve evec_a (fine-grid curvature continues tracking)
        """
        from common.logging import logger

        # Step 1: Absorb LoRA into base weights
        self._apply_lora_to_params()  # params now = W_0 + B@A
        # Mark as not applied: A/B will be reset to new small values
        self.lora_applied = False
        # Now params hold W_0_new, next _apply_lora_to_params() will add new B@A

        # Steps 2-4: Re-initialize LoRA structures
        for name in list(self.lora_B.keys()):
            info = self.lora_param_info[name]
            m, n, r = info['m'], info['n'], info['r']
            device = self.lora_B[name].device
            dtype = self.lora_B[name].dtype

            # Re-init A, B (same Kaiming-style init as configure_lora_params)
            std_A = 1.0 / math.sqrt(r)
            std_B = 0.1 / math.sqrt(n)
            self.lora_B[name] = torch.randn(m, r, device=device, dtype=dtype) * std_A
            self.lora_A[name] = torch.randn(r, n, device=device, dtype=dtype) * std_B

            # Reset momentum
            self.momentum_B[name] = torch.zeros_like(self.lora_B[name])
            self.momentum_A[name] = torch.zeros_like(self.lora_A[name])

            # Re-init curvature vectors
            if self.perrow_precond:
                self.evec_a_B[name] = torch.randn(m, r, device=device, dtype=dtype) * 0.01
                self.evec_a_A[name] = torch.randn(r, n, device=device, dtype=dtype) * 0.01
            else:
                self.evec_a_B[name] = torch.randn(r, device=device, dtype=dtype) * 0.01
                self.evec_a_A[name] = torch.randn(n, device=device, dtype=dtype) * 0.01

        # Step 5: Clear VMS EMA state (stale after absorption)
        if hasattr(self, 'c_f_ema'):
            self.c_f_ema = {}
        if hasattr(self, 'c_f_ema_count'):
            self.c_f_ema_count = {}
        if hasattr(self, 'c_f_prime_ema'):
            self.c_f_prime_ema = {}
        if hasattr(self, 'c_f_prime_ema_count'):
            self.c_f_prime_ema_count = {}
        if hasattr(self, 'tau_ema'):
            self.tau_ema = {}

        # Step 6: evec_a preserved — continues tracking H(W_0_new) naturally

        logger.info(f"    Absorption complete: W_0 updated, LoRA re-initialized (rank={self.lora_rank})")

    @torch.no_grad()
    def prolongation_to_fine(self):
        """
        V-cycle prolongation: coarse → fine grid transition.

        1. Absorb LoRA: W = W_0 + B @ A
        2. Momentum prolongation: M_init = m_B @ A + B @ m_A (BEFORE release)
        2.5. Curvature prolongation: evec_a_B/B → evec_a (BEFORE release)
        3. Release coarse structures (free GPU memory)
        4. Set fine-grid momentum from prolongated coarse momentum
        5. evec_a preserved (updated if curvature_multigrid enabled)
        """
        from common.logging import logger

        logger.info("  Prolongation: coarse → fine")

        # Reset per-cycle epoch counter so tau warmup restarts in each fine phase
        self._cycle_start_epoch = self._tau_current_epoch
        logger.info(f"    Per-cycle tau warmup reset: _cycle_start_epoch={self._cycle_start_epoch}")

        # Step 1: Apply LoRA to model params
        self._apply_lora_to_params()
        logger.info("    θ_f = W_0 + B @ A")

        # Step 2: Momentum prolongation (MUST be before release)
        momentum_init = {}
        if hasattr(self, 'momentum_B') and hasattr(self, 'lora_B'):
            # Reclaim fragmented GPU memory from Step 1's delta computations.
            # Without this, cached-but-unallocated blocks (~458 MiB) prevent
            # contiguous allocation for the (m, n) matmul results.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Compute M_init in param dtype (bf16/fp16) instead of fp32:
            # M_init is cast to param.dtype in Step 4, so fp32 precision is
            # wasted. For OPT-13B FFN layers, this halves peak from 400→200 MiB.
            _param_dtypes = {n: p.dtype for n, p in self.params}
            for name in self.lora_B:
                if name in self.momentum_A:
                    B = self.lora_B[name]        # (m, r) fp32
                    A = self.lora_A[name]        # (r, n) fp32
                    m_A = self.momentum_A[name]  # (r, n) fp32
                    _dt = _param_dtypes.get(name, B.dtype)

                    if self.coarse_b_only:
                        # A-only: m_B ~ 0, so M_init = B @ m_A
                        M_init = B.to(_dt) @ m_A.to(_dt)
                    else:
                        # M_init = m_B @ A + B @ m_A, computed in param dtype
                        m_B = self.momentum_B[name]  # (m, r) fp32
                        M_init = m_B.to(_dt) @ A.to(_dt)
                        M_init.add_(B.to(_dt) @ m_A.to(_dt))

                    momentum_init[name] = M_init
            if momentum_init:
                norms = [v.norm().item() for v in momentum_init.values()]
                logger.info(f"    Momentum prolongation: {len(momentum_init)} layers, "
                           f"avg ‖M_init‖={sum(norms)/len(norms):.4f}")

        # Step 2.5: Curvature prolongation — evec_a_B/B → evec_a (MUST be before release)
        if self.curvature_multigrid and hasattr(self, 'evec_a_B') and self.evec_a_B:
            beta = self.curvature_mg_beta
            prolong_count = 0
            cos_before = []
            cos_after = []
            for name in self.evec_a_B:
                if name not in self.evec_a:
                    continue
                a_B = self.evec_a_B[name]       # (r,) or (m, r)
                a_A = self.evec_a_A[name]       # (n,) or (r, n)
                a_f = self.evec_a[name]         # (n,)
                A = self.lora_A[name]           # (r, n)

                if self.perrow_precond:
                    # Per-row: a_B (m, r), a_A (r, n)
                    # Project a_B back: A^T @ a_B^T → (n, m) → mean over m → (n,)
                    a_from_B = (A.t() @ a_B.t()).mean(dim=1)  # (n,): A^T a_B
                    a_from_A = a_A.mean(dim=0)                # (n,): a_A direct
                else:
                    # Shared: a_B (r,), a_A (n,)
                    a_from_B = A.t() @ a_B                    # (n, r) @ (r,) → (n,): A^T a_B
                    a_from_A = a_A                            # (n,): a_A direct

                a_prolongated = a_from_B + a_from_A           # A^T a_B + a_A
                norm_p = a_prolongated.norm()
                if norm_p > 1e-10:
                    a_prolongated = a_prolongated / norm_p
                    # Diagnostic: cosine similarity before blend
                    a_f_norm = a_f.norm()
                    if a_f_norm > 1e-10:
                        cos_b = (torch.dot(a_f, a_prolongated) / a_f_norm).abs().item()
                    else:
                        cos_b = 0.0
                    cos_before.append(cos_b)
                    # EMA blend
                    a_new = (1.0 - beta) * a_f + beta * a_prolongated
                    a_new_norm = a_new.norm()
                    if a_new_norm > 1e-10:
                        self.evec_a[name] = a_new / a_new_norm
                    # Diagnostic: cosine similarity after blend
                    if a_f_norm > 1e-10:
                        cos_a = (torch.dot(a_f, self.evec_a[name]) / a_f_norm).abs().item()
                    else:
                        cos_a = 0.0
                    cos_after.append(cos_a)
                    prolong_count += 1

            diag_str = ""
            if cos_before:
                avg_cos_b = sum(cos_before) / len(cos_before)
                avg_cos_a = sum(cos_after) / len(cos_after)
                diag_str = (f", avg cos(old,prolongated)={avg_cos_b:.4f}, "
                           f"cos(old,blended)={avg_cos_a:.4f}")
            logger.info(f"    Curvature multigrid (prolongation): {prolong_count} layers, "
                       f"β={beta}{diag_str}")

        # Step 2.9a: Save LoRA B basis for reuse at next coarse init (basis_method dependent)
        if self.basis_method == 'reuse' and hasattr(self, 'lora_B') and self.lora_B:
            for name in self.lora_B:
                self._lora_basis_history[name] = self.lora_B[name].clone()
            logger.info(f"    basis_method='reuse': saved {len(self.lora_B)} LoRA B matrices for next cycle")

        # Step 2.9b: Save LoRA A for F2C curvature/momentum projection (ALWAYS, regardless of basis_method)
        # Without this, _release_coarse_structures() deletes lora_A,
        # and F2C's old_lora_A={} → evec_a_B projection = 0 → warm-start fails
        if hasattr(self, 'lora_A') and self.lora_A:
            if not hasattr(self, '_lora_A_history'):
                self._lora_A_history = {}
            for name in self.lora_A:
                self._lora_A_history[name] = self.lora_A[name].clone()
            logger.info(f"    Saved {len(self.lora_A)} LoRA A matrices for F2C projection")

        # Step 3: Release coarse structures
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        self._release_coarse_structures()
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        logger.info(f"    Released coarse structures (freed {mem_before - mem_after:.1f} MB)")

        # Step 4: Set fine-grid momentum from prolongated values
        self.momentum = {}
        for name, param in self.params:
            if name in momentum_init:
                M = momentum_init[name]
                if param.ndim > 2:
                    M = M.view(param.shape)
                self.momentum[name] = M.to(param.dtype)
            else:
                self.momentum[name] = torch.zeros_like(param.data)

        # Step 5: evec_a preserved (updated if curvature_multigrid enabled)
        # Log evec_a snapshot at C2F transition
        if self.evec_a:
            norms = [a.norm().item() for a in self.evec_a.values()]
            avg_norm = sum(norms) / len(norms)
            logger.info(f"    evec_a preserved ({len(self.evec_a)} tensors, "
                       f"avg_norm={avg_norm:.4f}, min={min(norms):.4f}, max={max(norms):.4f})")
        else:
            logger.info(f"    evec_a preserved ({len(self.evec_a)} tensors)")
        logger.info(f"    Fine grid ready (GPU mem: {mem_after:.1f} MB)")

    @torch.no_grad()
    def restriction_to_coarse(self):
        """
        V-cycle restriction: fine → coarse grid transition.

        1. Basis selection: evolve/select LoRA A basis from fine-grid momentum
           Dispatches based on self.basis_method: random/oja/svd/svd_hybrid/online_oja
        2. Re-init LoRA structures (configure_lora_params uses stored basis if available)
        3. Momentum injection: project fine-grid momentum to LoRA space
        4. Delete fine-grid momentum
        5. Diagnostics: kappa, SVD spectrum, retained_ratio, cosine_fidelity
        """
        from common.logging import logger
        import math

        logger.info("  Restriction: fine → coarse")

        # Reset per-cycle epoch counter so tau warmup restarts in each coarse phase
        self._cycle_start_epoch = self._tau_current_epoch
        logger.info(f"    Per-cycle tau warmup reset: _cycle_start_epoch={self._cycle_start_epoch}")

        # Step 1: Basis selection (MUST be before re-init, needs fine momentum)
        if self.basis_method != 'random' and hasattr(self, 'momentum'):
            basis_count = 0
            kappa_values = []
            svd_spectrum_samples = []

            for name, param in self.params:
                if param.ndim >= 2 and name in self.momentum:
                    M = self.momentum[name]
                    M_fp32 = M.to(torch.float32)
                    if M_fp32.ndim > 2:
                        M_fp32 = M_fp32.view(M_fp32.shape[0], -1)

                    # Dispatch basis selection
                    if self.basis_method == 'oja':
                        A_new = self.evolve_lora_basis_oja(name, M_fp32, self._oja_cycle_idx)
                    elif self.basis_method == 'svd':
                        A_new = self.select_basis_svd(name, M_fp32)
                    elif self.basis_method == 'svd_hybrid':
                        A_new = self.select_basis_svd_hybrid(name, M_fp32)
                    elif self.basis_method == 'online_oja':
                        # Online Oja: basis already updated during fine-grid training
                        # Fall back to SVD if no basis history yet (first cycle)
                        if name not in self._lora_basis_history:
                            A_new = self.select_basis_svd(name, M_fp32)
                        else:
                            A_new = self._lora_basis_history[name]
                    else:
                        continue

                    basis_count += 1

                    # Diagnostic: alignment kappa = ||A^T @ M||_F / ||M||_F
                    M_norm = M_fp32.norm().item()
                    if M_norm > 1e-8:
                        proj_norm = (A_new.T @ M_fp32).norm().item()
                        kappa = proj_norm / M_norm
                        kappa_values.append(kappa)

            if self.basis_method == 'oja':
                self._oja_cycle_idx += 1

            if kappa_values:
                avg_kappa = sum(kappa_values) / len(kappa_values)
                r = self.lora_rank
                m_typical = next((p.shape[0] for _, p in self.params if p.ndim >= 2), 768)
                random_baseline = math.sqrt(r / m_typical)
                logger.info(f"    Basis evolved ({self.basis_method}): {basis_count} layers, "
                           f"avg κ={avg_kappa:.4f} (random baseline: {random_baseline:.4f})")

            # SVD spectrum diagnostics (only for svd/svd_hybrid methods)
            if self._svd_spectrum_log:
                # Log spectrum from a representative layer (first one with full diagnostics)
                for sname, sinfo in self._svd_spectrum_log.items():
                    if 'energy_ratio' in sinfo:
                        logger.info(f"    SVD spectrum ({sname.split('.')[-2]}.{sname.split('.')[-1]}):")
                        svs = sinfo.get('top_r_sv', [])
                        logger.info(f"      top-{len(svs)} SVs: [{', '.join(f'{s:.2f}' for s in svs[:8])}]")
                        logger.info(f"      energy_ratio: {sinfo['energy_ratio']:.4f}, "
                                   f"effective_rank: {sinfo.get('effective_rank', 0):.1f}, "
                                   f"SV ratio σ₁/σᵣ: {sinfo.get('sv_ratio_1_r', 0):.1f}")
                        break  # One sample is enough for diagnostics

                # Aggregate SVD stats across all layers
                energy_ratios = [s['energy_ratio'] for s in self._svd_spectrum_log.values() if 'energy_ratio' in s]
                eff_ranks = [s['effective_rank'] for s in self._svd_spectrum_log.values() if 'effective_rank' in s]
                if energy_ratios:
                    logger.info(f"    SVD aggregate: avg energy_ratio={sum(energy_ratios)/len(energy_ratios):.4f}, "
                               f"avg eff_rank={sum(eff_ranks)/len(eff_ranks):.1f} "
                               f"(across {len(energy_ratios)} layers)")

                self._svd_spectrum_log = {}  # Clear for next cycle

        # Step 2a: FAS-A — W₀ gradient pre-correction (before LoRA re-init)
        if self.fas_correction and self.fas_w0_alpha > 0 and hasattr(self, 'momentum') and self.momentum:
            w0_norms = []
            for name, param in self.params:
                if name in self.momentum and param.ndim >= 2:
                    correction = self.momentum[name]
                    param.data.add_(correction, alpha=-self.fas_w0_alpha)
                    w0_norms.append(correction.norm().item())
            if w0_norms:
                logger.info(f"    FAS W₀ correction: α={self.fas_w0_alpha}, "
                           f"avg ‖m‖={sum(w0_norms)/len(w0_norms):.4f}")

        # §4.1 fix: get old B for evec_a_B/momentum_B projection
        # Priority: _lora_A_history (saved at C2F before release) > current lora_A > empty
        # Without this, V2+ cycles have old_lora_A={} because C2F deletes lora_A
        old_lora_A = {}
        if hasattr(self, '_lora_A_history') and self._lora_A_history:
            old_lora_A = {name: B.clone() for name, B in self._lora_A_history.items()}
        elif hasattr(self, 'lora_A') and self.lora_A:
            old_lora_A = {name: B.clone() for name, B in self.lora_A.items()}

        # Step 2b: Re-init LoRA structures
        # P4: save old v_evec (Adam second moment) before clearing, to inherit across cycles.
        # v_evec accumulates squared curvature gradients → inheriting avoids overly large
        # Adam steps at the start of a new cycle. m_evec (first moment) is direction-dependent
        # in the old coordinate system, so it is intentionally reset to zero.
        old_v_evec_B = dict(self.v_evec_B) if self.adam_cov and hasattr(self, 'v_evec_B') else {}
        old_v_evec_A = dict(self.v_evec_A) if self.adam_cov and hasattr(self, 'v_evec_A') else {}

        self.lora_B = {}
        self.lora_A = {}
        self.evec_a_B = {}
        self.evec_a_A = {}
        self.momentum_B = {}
        self.momentum_A = {}
        self.lora_param_info = {}
        if self.adam_cov:
            self.m_evec_B = {}
            self.v_evec_B = {}
            self.m_evec_A = {}
            self.v_evec_A = {}
            self.adam_cov_step = 0

        self.configure_lora_params()

        # Step 2c: Curvature multigrid — warm-start evec_a_B/B from fine-grid evec_a
        # P0 fix: use old_lora_A (from before re-init) for evec_a_B projection,
        # so that the projection direction carries real fine-grid information
        # instead of projecting onto a random direction.
        if self.curvature_multigrid and hasattr(self, 'evec_a') and self.evec_a:
            cmg_count = 0
            cmg_cos_sims = []
            for name in self.lora_B:
                if name not in self.evec_a:
                    continue
                B = self.lora_B[name]       # (m, r)
                a_f = self.evec_a[name]     # (n,)

                # Use old A for evec_a_B projection (meaningful direction)
                # Fall back to new random A if old A not available (first cycle)
                A_for_proj = old_lora_A.get(name, self.lora_A[name])  # (r, n)

                if self.perrow_precond:
                    # Per-row: evec_a_B (m, r), evec_a_A (r, n)
                    a_f_mat = a_f.view(1, -1).expand(B.shape[0], -1)  # (m, n)
                    a_B_proj = a_f_mat @ A_for_proj.t()   # (m, r): a_f·A_old^T → evec_a_B
                    a_A_proj = B.t() @ a_f_mat            # (r, n): B^T·a_f → evec_a_A
                else:
                    # Shared: evec_a_B (r,), evec_a_A (n,)
                    a_f_mat = a_f.view(1, -1).expand(B.shape[0], -1)  # (m, n)
                    a_B_proj = (a_f_mat @ A_for_proj.t()).sum(0)   # (r,): a_f·A_old^T → evec_a_B
                    a_A_proj = a_f.clone()                # (n,): direct copy → evec_a_A

                # §4.1 fix: split AND into independent checks so evec_a_A
                # can be warm-started even when evec_a_B projection fails
                norm_B = a_B_proj.norm()
                norm_A = a_A_proj.norm()
                warmed = False
                if norm_B > 1e-10:
                    self.evec_a_B[name] = a_B_proj / norm_B
                    warmed = True
                if norm_A > 1e-10:
                    self.evec_a_A[name] = a_A_proj / norm_A
                    if not self.perrow_precond:
                        cos_sim = torch.dot(a_f, self.evec_a_A[name]).item()
                        cmg_cos_sims.append(abs(cos_sim))
                    warmed = True
                if warmed:
                    cmg_count += 1

            diag_str = ""
            if cmg_cos_sims:
                avg_cos = sum(cmg_cos_sims) / len(cmg_cos_sims)
                diag_str = f", avg cos(evec_a, evec_a_A)={avg_cos:.4f}"
            logger.info(f"    Curvature multigrid (restriction): {cmg_count} layers warm-started{diag_str}")

        # Step 2d: P3 disabled — B is initialized to zero (standard LoRA convention).
        # With B=0, B@A=0 and the model starts exactly at W_fine. The P3 direction
        # (A.t()@M) is already captured by momentum injection in Step 3
        # (momentum_A = γ·A.t()@M), so no information is lost.

        # Step 2e: P4 — inherit Adam v_evec (second moment) across cycles.
        # evec_a_B/B are already warm-started (Step 2c), so v_evec from the previous cycle
        # is still a valid variance estimate for the new coordinate system.
        # Only inherit where shapes match; zero otherwise (e.g., first cycle or rank change).
        if self.adam_cov and (old_v_evec_B or old_v_evec_A):
            p4_count = 0
            for name in self.evec_a_B:
                if (name in old_v_evec_B and
                        old_v_evec_B[name].shape == self.evec_a_B[name].shape):
                    self.v_evec_B[name] = old_v_evec_B[name]
                    p4_count += 1
                if (name in old_v_evec_A and
                        old_v_evec_A[name].shape == self.evec_a_A[name].shape):
                    self.v_evec_A[name] = old_v_evec_A[name]
            logger.info(f"    P4 Adam v_evec inherited: {p4_count} layers")

        # Step 3: FAS — A+B momentum warm-start (replaces old momentum injection)
        # P1 fix: also inject momentum_B (previously only momentum_A was injected)
        if self.fas_correction and self.fas_alpha > 0 and hasattr(self, 'momentum') and self.momentum:
            fas_kappas_B = []
            fas_kappas_A = []
            inject_count = 0
            for name in self.lora_B:
                if name in self.momentum:
                    M = self.momentum[name].to(torch.float32)
                    if M.ndim > 2:
                        M = M.view(M.shape[0], -1)
                    M_norm = M.norm().item()
                    if M_norm < 1e-8:
                        continue

                    B = self.lora_B[name]  # (m, r)
                    g_A = B.t() @ M  # (r, n)

                    self.momentum_A[name] = self.fas_alpha * g_A.to(self.momentum_A[name].dtype)
                    fas_kappas_B.append(g_A.norm().item() / (M_norm + 1e-10))

                    # P1: also inject momentum_B using old A for projection
                    A_proj = old_lora_A.get(name, self.lora_A[name])  # (r, n)
                    g_B = M @ A_proj.t()  # (m, r)
                    self.momentum_B[name] = self.fas_alpha * g_B.to(self.momentum_B[name].dtype)
                    fas_kappas_A.append(g_B.norm().item() / (M_norm + 1e-10))

                    inject_count += 1

            if fas_kappas_B:
                avg_kappa_A = sum(fas_kappas_B) / len(fas_kappas_B)
                avg_kappa_B = sum(fas_kappas_A) / len(fas_kappas_A)
                logger.info(f"    FAS B+A warm-start: α={self.fas_alpha}, "
                           f"avg κ_A={avg_kappa_B:.4f}, κ_B={avg_kappa_A:.4f}, layers={inject_count}")
            else:
                logger.info(f"    FAS: no fine momentum to project (first cycle)")

        elif hasattr(self, 'momentum') and self.momentum:
            # Legacy momentum injection (when FAS is disabled)
            gamma = 0.5
            inject_count = 0
            diag_values = []

            for name in self.lora_B:
                if name in self.momentum:
                    M = self.momentum[name]
                    M_fp32 = M.to(torch.float32)
                    if M_fp32.ndim > 2:
                        M_fp32 = M_fp32.view(M_fp32.shape[0], -1)

                    B = self.lora_B[name]
                    A = self.lora_A[name]
                    M_norm = M_fp32.norm().item()

                    if M_norm < 1e-8:
                        continue

                    if self.basis_method != 'random':
                        m_A_raw = B.t() @ M_fp32
                        self.momentum_A[name] = gamma * m_A_raw
                        # P1 fix: also inject momentum_B using old A
                        A_proj = old_lora_A.get(name, A)
                        m_B_raw = M_fp32 @ A_proj.t()
                        self.momentum_B[name] = gamma * m_B_raw
                        inject_count += 1

                        M_raw = B @ (B.t() @ M_fp32)
                        M_raw_norm = M_raw.norm().item()
                        retained_ratio = M_raw_norm / (M_norm + 1e-8)
                        cos_fid = (M_raw * M_fp32).sum().item() / (M_raw_norm * M_norm + 1e-8)
                        kappa = (B.t() @ M_fp32).norm().item() / (M_norm + 1e-8)
                        diag_values.append((kappa, retained_ratio, cos_fid))
                    else:
                        # Use A_old for B-branch projection (same as non-random branch)
                        A_proj = old_lora_A.get(name, A)
                        m_B_raw = M_fp32 @ A_proj.t()
                        m_A_raw = B.t() @ M_fp32

                        M_raw = m_B_raw @ A_proj + B @ m_A_raw
                        M_raw_norm = M_raw.norm().item()

                        if M_raw_norm > 1e-8:
                            alpha = M_norm / (M_raw_norm + 1e-8)
                            self.momentum_B[name] = (gamma * alpha) * m_B_raw
                            self.momentum_A[name] = (gamma * alpha) * m_A_raw
                            inject_count += 1

            if diag_values:
                avg_kappa = sum(d[0] for d in diag_values) / len(diag_values)
                avg_retained = sum(d[1] for d in diag_values) / len(diag_values)
                avg_cos = sum(d[2] for d in diag_values) / len(diag_values)
                logger.info(f"    Momentum injected ({self.basis_method}): {inject_count} layers, γ={gamma}, "
                           f"avg κ={avg_kappa:.4f}, retained={avg_retained:.4f}, cos_fid={avg_cos:.4f}")
            elif inject_count > 0:
                logger.info(f"    Momentum injected: {inject_count} layers, γ={gamma}")
            else:
                logger.info(f"    No momentum to inject (first cycle or no fine momentum)")

        # Step 4: Delete fine-grid momentum (not needed in coarse phase)
        if hasattr(self, 'momentum'):
            del self.momentum

        # P2 revised: clear c_f_ema at F2C transition.
        # Fine-phase c_f was computed under fine-grid evec_a, but the new coarse phase
        # uses re-projected evec_a_B/B — a completely different coordinate system.
        # Preserving stale c_f_ema makes VMS correction ~30x oversized and misdirected
        # (c_f dominates c_B in the first ~20 steps). tau_ema is preserved because
        # tau's physical meaning (curvature discrepancy ratio) is coordinate-independent.
        if hasattr(self, 'c_f_ema'):
            self.c_f_ema = {}
        if hasattr(self, 'c_f_ema_count'):
            self.c_f_ema_count = {}
        if hasattr(self, 'c_f_prime_ema'):
            self.c_f_prime_ema = {}
        if hasattr(self, 'c_f_prime_ema_count'):
            self.c_f_prime_ema_count = {}

        # Apply initial LoRA delta to model params so incremental updates work correctly.
        # Without this, param = W_fine and lora_applied=False, so perturb_lora_parameters()
        # computes old_delta = B@A but never added B@A to params — creating a persistent
        # bias of -A_init@B_init in all subsequent evaluations and updates.
        # Critical when P3 (B direction init) or basis_method='reuse' makes B@A non-negligible.
        self.lora_applied = False
        self._apply_lora_to_params()
        bonly_str = ", B-only" if self.coarse_b_only else ""
        logger.info(f"    LoRA re-initialized (rank={self.lora_rank}, basis={self.basis_method}{bonly_str})")

    def get_phase(self):
        """Get current training phase."""
        return getattr(self, '_current_phase', 'coarse')

    def set_phase(self, phase: str):
        """Set current training phase: 'coarse', 'reconstruction', or 'fine'."""
        self._current_phase = phase

import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from model.glue_transformer_zomuon import (
    GLUETransformerZOMuon,
    zeropower_via_newtonschulz5,
)


class ToyLayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.conv = Conv1D(5, 4)
        self.extra_bias = nn.Parameter(torch.zeros(3))


def make_zomuon(
    rank=2,
    step_interval=2,
    num_queries=5,
    learning_rate=0.03,
    nonmatrix_lr=1e-4,
    model=None,
):
    zomuon = GLUETransformerZOMuon.__new__(GLUETransformerZOMuon)
    torch.nn.Module.__init__(zomuon)
    zomuon.rank = rank
    zomuon.step_interval = step_interval
    zomuon.weight_decay = 0.0
    zomuon.zomuon_nonmatrix_lr = nonmatrix_lr
    zomuon.zero_order_eps = 1e-3
    zomuon.z_std = 1.0
    zomuon.num_queries = num_queries
    zomuon.batchsize_limit = 64
    zomuon.full_parameter = True
    zomuon.model = model

    zomuon.params = []
    zomuon.parameter_specs = {}
    zomuon.projection_matrices = {}
    zomuon.basis_refresh_count = 0
    zomuon.last_basis_refresh_step = -1
    zomuon.last_basis_seed = None
    zomuon._basis_memory_gb = 0.0
    zomuon._effective_dim = 0
    zomuon._effective_rank_min = 0
    zomuon._effective_rank_max = 0
    zomuon._matrix_parameter_count = 0
    zomuon._nonmatrix_parameter_count = 0
    zomuon._rank_clamp_warned = False
    zomuon.state = SimpleNamespace(
        learning_rate=learning_rate,
        global_training_steps=0,
        num_forward_passes=0,
        query=[],
        time=[],
        proj_val=[],
    )
    return zomuon


def make_manual_zomuon(
    rank=2,
    step_interval=2,
    num_queries=5,
    learning_rate=0.03,
    nonmatrix_lr=1e-4,
):
    zomuon = make_zomuon(
        rank=rank,
        step_interval=step_interval,
        num_queries=num_queries,
        learning_rate=learning_rate,
        nonmatrix_lr=nonmatrix_lr,
    )
    zomuon.params = [
        (
            "linear.weight",
            nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(3, 4) / 10
            ),
        ),
        (
            "conv.weight",
            nn.Parameter(
                torch.arange(20, dtype=torch.float32).view(4, 5) / 20
            ),
        ),
        (
            "bias",
            nn.Parameter(torch.arange(3, dtype=torch.float32) / 10),
        ),
    ]
    linear_rank = min(rank, 3, 4)
    conv_rank = min(rank, 5, 4)
    zomuon.parameter_specs = {
        "linear.weight": {
            "kind": "matrix",
            "layout": "normal",
            "logical_shape": (3, 4),
            "effective_rank": linear_rank,
        },
        "conv.weight": {
            "kind": "matrix",
            "layout": "conv1d_transposed",
            "logical_shape": (5, 4),
            "effective_rank": conv_rank,
        },
        "bias": {"kind": "nonmatrix"},
    }
    zomuon._effective_rank_min = min(linear_rank, conv_rank)
    zomuon._effective_rank_max = max(linear_rank, conv_rank)
    zomuon._effective_dim = linear_rank * 4 + conv_rank * 4 + 3
    zomuon._matrix_parameter_count = 2
    zomuon._nonmatrix_parameter_count = 1
    return zomuon


def official_newton_schulz_reference(matrix):
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix
    if matrix.size(-2) > matrix.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    if matrix.size(-2) > matrix.size(-1):
        x = x.mT
    return x


class ZOMuonCoreTest(unittest.TestCase):
    def test_invalid_hyperparameters_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "num_queries"):
            GLUETransformerZOMuon(
                model_name_or_path="not-loaded",
                num_labels=2,
                num_queries=2,
            )
        with self.assertRaisesRegex(ValueError, "rank"):
            GLUETransformerZOMuon(
                model_name_or_path="not-loaded",
                num_labels=2,
                rank=0,
            )
        with self.assertRaisesRegex(ValueError, "nonmatrix_lr"):
            GLUETransformerZOMuon(
                model_name_or_path="not-loaded",
                num_labels=2,
                zomuon_nonmatrix_lr=0,
            )
        with self.assertRaisesRegex(NotImplementedError, "full-parameter"):
            GLUETransformerZOMuon(
                model_name_or_path="not-loaded",
                num_labels=2,
                full_parameter=False,
            )

    def test_newton_schulz_matches_official_five_step_routine(self):
        torch.manual_seed(0)
        for shape in ((3, 7), (7, 3), (4, 4)):
            matrix = torch.randn(shape)
            actual = zeropower_via_newtonschulz5(matrix)
            expected = official_newton_schulz_reference(matrix)
            self.assertEqual(tuple(actual.shape), shape)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.isfinite(actual).all())

        zeros = torch.zeros(3, 5)
        self.assertTrue(
            torch.equal(zeropower_via_newtonschulz5(zeros), zeros)
        )

    def test_configure_params_detects_conv1d_logical_layout(self):
        model = ToyLayoutModel()
        zomuon = make_zomuon(rank=8, model=model)
        zomuon.configure_params()

        linear = zomuon.parameter_specs["linear.weight"]
        conv = zomuon.parameter_specs["conv.weight"]
        bias = zomuon.parameter_specs["extra_bias"]

        self.assertEqual(linear["layout"], "normal")
        self.assertEqual(linear["logical_shape"], (3, 4))
        self.assertEqual(linear["effective_rank"], 3)
        self.assertEqual(conv["layout"], "conv1d_transposed")
        self.assertEqual(conv["logical_shape"], (5, 4))
        self.assertEqual(conv["effective_rank"], 4)
        self.assertEqual(bias["kind"], "nonmatrix")
        self.assertEqual(zomuon._matrix_parameter_count, 2)
        self.assertEqual(zomuon._nonmatrix_parameter_count, 2)

    def test_projection_rank_clamp_orthogonality_and_seed(self):
        zomuon = make_manual_zomuon(rank=8)
        global_rng_before = torch.random.get_rng_state().clone()
        zomuon._refresh_projection_matrices(refresh_seed=17)
        global_rng_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(global_rng_before, global_rng_after))

        linear_projection = zomuon.projection_matrices["linear.weight"]
        conv_projection = zomuon.projection_matrices["conv.weight"]
        self.assertEqual(tuple(linear_projection.shape), (3, 3))
        self.assertEqual(tuple(conv_projection.shape), (5, 4))
        self.assertTrue(
            torch.allclose(
                linear_projection.t() @ linear_projection,
                torch.eye(3),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                conv_projection.t() @ conv_projection,
                torch.eye(4),
                atol=1e-5,
                rtol=1e-5,
            )
        )

        first = {
            name: projection.clone()
            for name, projection in zomuon.projection_matrices.items()
        }
        zomuon._refresh_projection_matrices(refresh_seed=17)
        for name, projection in first.items():
            self.assertTrue(
                torch.equal(projection, zomuon.projection_matrices[name])
            )

    def test_direction_formula_conv1d_transpose_and_reconstruction(self):
        zomuon = make_manual_zomuon()
        zomuon._refresh_projection_matrices(refresh_seed=23)
        seed = 101

        for param_index, (name, param) in enumerate(zomuon.params):
            actual = zomuon._make_direction(
                name,
                param,
                param_index,
                seed,
            )
            spec = zomuon.parameter_specs[name]
            if spec["kind"] == "matrix":
                _, n = spec["logical_shape"]
                psi = zomuon._sample_matrix_psi(
                    param,
                    param_index,
                    seed,
                    spec["effective_rank"],
                    n,
                )
                logical = zomuon.projection_matrices[name] @ psi
                expected = (
                    logical.t().contiguous()
                    if spec["layout"] == "conv1d_transposed"
                    else logical
                )
            else:
                expected = zomuon._sample_nonmatrix_z(
                    param,
                    param_index,
                    seed,
                )

            self.assertEqual(tuple(actual.shape), tuple(param.shape))
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(
                torch.equal(
                    actual,
                    zomuon._make_direction(
                        name,
                        param,
                        param_index,
                        seed,
                    ),
                )
            )

    def test_one_sided_scalar_and_parameter_restoration(self):
        zomuon = make_manual_zomuon()
        zomuon._refresh_projection_matrices(refresh_seed=29)

        def toy_forward(this, model, batch):
            del model, batch
            return sum(
                param.float().square().sum()
                for _, param in this.params
            )

        zomuon.zo_forward_memory_eff = MethodType(toy_forward, zomuon)
        baseline = toy_forward(zomuon, None, None)
        initial = [param.detach().clone() for _, param in zomuon.params]
        seed = 41

        expected_loss = 0.0
        for param_index, ((name, param), reference) in enumerate(
            zip(zomuon.params, initial)
        ):
            direction = zomuon._make_direction(
                name,
                param,
                param_index,
                seed,
            )
            expected_loss += (
                reference + zomuon.zero_order_eps * direction
            ).float().square().sum()

        loss_plus, scalar = zomuon._evaluate_direction(
            None,
            {},
            baseline,
            seed,
        )
        self.assertAlmostEqual(loss_plus, expected_loss.item(), places=5)
        self.assertAlmostEqual(
            scalar,
            ((expected_loss - baseline) / zomuon.zero_order_eps).item(),
            places=4,
        )
        for (_, param), reference in zip(zomuon.params, initial):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )
            self.assertIsNone(param.grad)

    def test_update_is_lowdim_rge_then_matrix_sign_then_lift(self):
        zomuon = make_manual_zomuon(
            learning_rate=0.02,
            nonmatrix_lr=3e-4,
        )
        zomuon._refresh_projection_matrices(refresh_seed=31)
        seeds = [11, 22, 33, 44]
        scalars = [0.4, -0.2, 0.1, 0.3]
        initial = [param.detach().clone() for _, param in zomuon.params]
        expected = []

        for param_index, ((name, param), reference) in enumerate(
            zip(zomuon.params, initial)
        ):
            spec = zomuon.parameter_specs[name]
            if spec["kind"] == "matrix":
                _, n = spec["logical_shape"]
                lowdim_rge = torch.zeros(spec["effective_rank"], n)
                for seed, scalar in zip(seeds, scalars):
                    lowdim_rge.add_(
                        zomuon._sample_matrix_psi(
                            param,
                            param_index,
                            seed,
                            spec["effective_rank"],
                            n,
                        ),
                        alpha=scalar / len(seeds),
                    )
                logical_update = (
                    zomuon.projection_matrices[name]
                    @ zeropower_via_newtonschulz5(lowdim_rge)
                )
                update = (
                    logical_update.t().contiguous()
                    if spec["layout"] == "conv1d_transposed"
                    else logical_update
                )
                expected.append(
                    reference - zomuon.state.learning_rate * update
                )
            else:
                grad_estimate = torch.zeros_like(param)
                for seed, scalar in zip(seeds, scalars):
                    grad_estimate.add_(
                        zomuon._sample_nonmatrix_z(
                            param,
                            param_index,
                            seed,
                        ),
                        alpha=scalar / len(seeds),
                    )
                expected.append(
                    reference
                    - zomuon.zomuon_nonmatrix_lr * grad_estimate
                )

        zomuon._apply_zomuon_update(seeds, scalars)
        for (_, param), reference in zip(zomuon.params, expected):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )
            self.assertIsNone(param.grad)

    def test_training_step_uses_one_basis_and_exact_query_budget(self):
        zomuon = make_manual_zomuon(
            step_interval=2,
            num_queries=5,
        )

        forward_calls = 0

        def toy_forward(this, model, batch):
            nonlocal forward_calls
            del model, batch
            forward_calls += 1
            return sum(
                param.float().square().sum()
                for _, param in this.params
            )

        zomuon.zo_forward_memory_eff = MethodType(toy_forward, zomuon)
        original_evaluate = zomuon._evaluate_direction
        estimation_points = []
        query_bases = []

        def recorded_evaluate(this, model, batch, baseline, seed):
            estimation_points.append(
                [param.detach().clone() for _, param in this.params]
            )
            query_bases.append(
                {
                    name: projection.clone()
                    for name, projection in this.projection_matrices.items()
                }
            )
            return original_evaluate(model, batch, baseline, seed)

        zomuon._evaluate_direction = MethodType(recorded_evaluate, zomuon)
        initial = [param.detach().clone() for _, param in zomuon.params]
        log = {}
        np.random.seed(0)
        zomuon.training_step(None, {"input_ids": torch.zeros(2, 1)}, log=log)

        self.assertEqual(forward_calls, 5)
        self.assertEqual(len(estimation_points), 4)
        for point in estimation_points:
            for value, reference in zip(point, initial):
                self.assertTrue(
                    torch.allclose(value, reference, atol=1e-6, rtol=1e-6)
                )
        for query_basis in query_bases[1:]:
            for name in query_bases[0]:
                self.assertTrue(
                    torch.equal(query_basis[name], query_bases[0][name])
                )
        self.assertEqual(zomuon.state.query[-1], 5)
        self.assertEqual(zomuon.state.num_forward_passes, 5)
        self.assertEqual(log["zomuon_num_directions"], 4)
        self.assertEqual(log["zomuon_baseline_forward_evals"], 1)
        self.assertEqual(log["zomuon_perturbed_forward_evals"], 4)
        self.assertEqual(log["zomuon_total_forward_evals"], 5)
        self.assertEqual(zomuon.basis_refresh_count, 1)

        first_basis = {
            name: projection.clone()
            for name, projection in zomuon.projection_matrices.items()
        }
        zomuon.training_step(None, {"input_ids": torch.zeros(2, 1)}, log=log)
        self.assertEqual(zomuon.basis_refresh_count, 1)
        for name, projection in first_basis.items():
            self.assertTrue(
                torch.equal(projection, zomuon.projection_matrices[name])
            )

        zomuon.training_step(None, {"input_ids": torch.zeros(2, 1)}, log=log)
        self.assertEqual(zomuon.basis_refresh_count, 2)
        self.assertTrue(
            any(
                not torch.equal(
                    projection,
                    zomuon.projection_matrices[name],
                )
                for name, projection in first_basis.items()
            )
        )

    def test_forward_exception_restores_parameters(self):
        zomuon = make_manual_zomuon()
        zomuon._refresh_projection_matrices(refresh_seed=37)
        initial = [param.detach().clone() for _, param in zomuon.params]

        def failing_forward(this, model, batch):
            del this, model, batch
            raise RuntimeError("synthetic forward failure")

        zomuon.zo_forward_memory_eff = MethodType(failing_forward, zomuon)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            zomuon._evaluate_direction(
                None,
                {},
                torch.tensor(1.0),
                direction_seed=43,
            )

        for (_, param), reference in zip(zomuon.params, initial):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()

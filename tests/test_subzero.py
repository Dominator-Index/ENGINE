import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from model.glue_transformer_subzero import GLUETransformerSubZero


def make_subzero(rank=2, step_interval=2, num_queries=2, learning_rate=0.01):
    subzero = GLUETransformerSubZero.__new__(GLUETransformerSubZero)
    torch.nn.Module.__init__(subzero)
    subzero.rank = rank
    subzero.basis_upd_interval = step_interval
    subzero.weight_decay = 0.0
    subzero.zero_order_eps = 1e-3
    subzero.z_std = 1.0
    subzero.num_queries = num_queries
    subzero.params = [
        ("weight", torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 4) / 10)),
        ("bias", torch.nn.Parameter(torch.arange(4, dtype=torch.float32) / 10)),
    ]
    subzero.subspace_basis = {}
    subzero.basis_refresh_count = 0
    subzero.last_basis_refresh_step = -1
    subzero.last_basis_seed = None
    subzero._basis_memory_gb = 0.0
    subzero._effective_dim = 0
    subzero._effective_rank_min = 0
    subzero._effective_rank_max = 0
    subzero._rank_clamp_warned = False
    subzero.state = SimpleNamespace(
        learning_rate=learning_rate,
        global_training_steps=0,
        num_forward_passes=0,
        query=[],
        time=[],
        proj_val=[],
    )
    return subzero


class SubZeroCoreTest(unittest.TestCase):
    def test_invalid_shared_hyperparameters_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "num_queries"):
            GLUETransformerSubZero(
                model_name_or_path="not-loaded",
                num_labels=2,
                num_queries=3,
            )
        with self.assertRaisesRegex(NotImplementedError, "full-parameter"):
            GLUETransformerSubZero(
                model_name_or_path="not-loaded",
                num_labels=2,
                full_parameter=False,
            )

    def test_direction_formula_seed_and_rank_clamp(self):
        subzero = make_subzero(rank=5)
        subzero._refresh_subspace_basis(refresh_seed=17)

        basis = subzero.subspace_basis["weight"]
        self.assertEqual(basis["effective_rank"], 3)
        self.assertEqual(tuple(basis["U"].shape), (3, 3))
        self.assertEqual(tuple(basis["Vt"].shape), (3, 4))
        self.assertTrue(
            torch.allclose(
                basis["U"].t() @ basis["U"],
                torch.eye(3),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                basis["Vt"] @ basis["Vt"].t(),
                torch.eye(3),
                atol=1e-5,
                rtol=1e-5,
            )
        )

        seed = 123
        direction = subzero._make_direction(
            "weight", subzero.params[0][1], param_index=0, direction_seed=seed
        )
        generator = subzero._make_generator(
            subzero.params[0][1].device, seed, param_index=0, stream=2
        )
        core = torch.randn((3, 3), generator=generator)
        expected = basis["U"] @ core @ basis["Vt"]
        expected *= np.sqrt(12 / 9)
        self.assertTrue(torch.allclose(direction, expected))
        self.assertTrue(
            torch.equal(
                direction,
                subzero._make_direction(
                    "weight",
                    subzero.params[0][1],
                    param_index=0,
                    direction_seed=seed,
                ),
            )
        )
        self.assertFalse(
            torch.equal(
                direction,
                subzero._make_direction(
                    "weight",
                    subzero.params[0][1],
                    param_index=0,
                    direction_seed=seed + 1,
                ),
            )
        )

    def test_perturbation_restores_parameters(self):
        subzero = make_subzero()
        subzero._refresh_subspace_basis(refresh_seed=23)
        initial = [param.detach().clone() for _, param in subzero.params]

        subzero._perturb_parameters(101, scaling_factor=1.0)
        subzero._perturb_parameters(101, scaling_factor=-2.0)
        subzero._perturb_parameters(101, scaling_factor=1.0)

        for (_, param), expected in zip(subzero.params, initial):
            self.assertTrue(torch.allclose(param, expected, atol=1e-6, rtol=1e-6))
            self.assertIsNone(param.grad)

    def test_multi_direction_update_is_mean(self):
        subzero = make_subzero(learning_rate=0.03)
        subzero._refresh_subspace_basis(refresh_seed=29)
        seeds = [11, 22, 33]
        projected_grads = [0.4, -0.2, 0.1]
        initial = [param.detach().clone() for _, param in subzero.params]

        expected = [param.clone() for param in initial]
        for seed, projected_grad in zip(seeds, projected_grads):
            for param_index, (name, param) in enumerate(subzero.params):
                direction = subzero._make_direction(
                    name, param, param_index, seed
                )
                expected[param_index].add_(
                    direction,
                    alpha=-subzero.state.learning_rate
                    * projected_grad
                    / len(seeds),
                )

        subzero._apply_subzero_update(seeds, projected_grads)

        for (_, param), reference in zip(subzero.params, expected):
            self.assertTrue(torch.allclose(param, reference, atol=1e-6, rtol=1e-6))
            self.assertIsNone(param.grad)

    def test_training_step_uses_same_point_and_counts_queries(self):
        subzero = make_subzero(num_queries=6, step_interval=2)

        def toy_forward(this, model, batch):
            del model, batch
            return sum(param.float().square().sum() for _, param in this.params)

        subzero.zo_forward_memory_eff = MethodType(toy_forward, subzero)
        original_central_difference = subzero._central_difference
        estimation_points = []

        def recorded_central_difference(this, model, batch, direction_seed):
            estimation_points.append(
                [param.detach().clone() for _, param in this.params]
            )
            return original_central_difference(model, batch, direction_seed)

        subzero._central_difference = MethodType(
            recorded_central_difference, subzero
        )
        initial = [param.detach().clone() for _, param in subzero.params]
        log = {}
        np.random.seed(0)
        subzero.training_step(
            model=None,
            batch={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
            log=log,
        )

        self.assertEqual(len(estimation_points), 3)
        for point in estimation_points:
            for value, reference in zip(point, initial):
                self.assertTrue(
                    torch.allclose(value, reference, atol=1e-6, rtol=1e-6)
                )
        self.assertEqual(subzero.state.query[-1], 6)
        self.assertEqual(subzero.state.num_forward_passes, 6)
        self.assertEqual(log["num_forward_passes"], 6)
        self.assertEqual(log["subzero_num_directions"], 3)
        self.assertEqual(subzero.basis_refresh_count, 1)

        first_basis = subzero.subspace_basis["weight"]["U"].clone()
        subzero.training_step(
            model=None,
            batch={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
            log=log,
        )
        self.assertEqual(subzero.basis_refresh_count, 1)
        self.assertTrue(
            torch.equal(first_basis, subzero.subspace_basis["weight"]["U"])
        )

        subzero.training_step(
            model=None,
            batch={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
            log=log,
        )
        self.assertEqual(subzero.basis_refresh_count, 2)
        self.assertFalse(
            torch.equal(first_basis, subzero.subspace_basis["weight"]["U"])
        )


if __name__ == "__main__":
    unittest.main()

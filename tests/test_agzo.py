import math
import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

from model.glue_transformer_agzo import GLUETransformerAGZO


class ToyAGZOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.linear = nn.Linear(4, 4)
        self.conv = Conv1D(5, 4)
        self.norm = nn.LayerNorm(5)
        self.head = nn.Linear(5, 2)

    def forward(self, input_ids, labels):
        hidden = self.embedding(input_ids)
        hidden = torch.tanh(self.linear(hidden))
        hidden = self.conv(hidden)
        hidden = self.norm(hidden)
        logits = self.head(hidden.mean(dim=1))
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def make_agzo(rank=1, num_queries=2, learning_rate=0.01, model=None):
    agzo = GLUETransformerAGZO.__new__(GLUETransformerAGZO)
    torch.nn.Module.__init__(agzo)
    agzo.rank = rank
    agzo.weight_decay = 0.0
    agzo.agzo_power_iter_steps = 5
    agzo.zero_order_eps = 1e-3
    agzo.z_std = 1.0
    agzo.num_queries = num_queries
    agzo.batchsize_limit = 64
    agzo.full_parameter = True
    agzo.model = model

    agzo.params = []
    agzo._param_index = {}
    agzo._params_by_name = {}
    agzo._agzo_module_specs = {}
    agzo._agzo_basis = {}
    agzo._agzo_hook_handles = []
    agzo._agzo_hooks_registered = False
    agzo._agzo_collecting = False
    agzo._agzo_basis_seed = None
    agzo._basis_failure_warned = False
    agzo._no_guided_module_warned = False
    agzo._basis_collection_count = 0
    agzo._basis_memory_gb = 0.0
    agzo._effective_rank_min = 0
    agzo._effective_rank_max = 0
    agzo._guided_parameter_count = 0
    agzo._fallback_parameter_count = 0
    agzo._guided_parameter_numel = 0
    agzo._fallback_parameter_numel = 0
    agzo._guided_parameter_ratio = 0.0
    agzo.state = SimpleNamespace(
        learning_rate=learning_rate,
        global_training_steps=0,
        num_forward_passes=0,
        query=[],
        time=[],
        proj_val=[],
    )
    return agzo


def make_manual_agzo(rank=2, num_queries=2, learning_rate=0.01):
    agzo = make_agzo(
        rank=rank,
        num_queries=num_queries,
        learning_rate=learning_rate,
    )
    agzo.params = [
        (
            "linear.weight",
            nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 4) / 10),
        ),
        (
            "conv.weight",
            nn.Parameter(torch.arange(20, dtype=torch.float32).view(4, 5) / 20),
        ),
        ("bias", nn.Parameter(torch.arange(3, dtype=torch.float32) / 10)),
    ]
    agzo._param_index = {
        name: index for index, (name, _) in enumerate(agzo.params)
    }
    agzo._params_by_name = dict(agzo.params)
    basis_t = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    agzo._agzo_basis = {
        "linear.weight": {
            "basis_t": basis_t,
            "layout": "out_in",
            "effective_rank": 2,
            "input_dim": 4,
            "output_dim": 3,
        },
        "conv.weight": {
            "basis_t": basis_t,
            "layout": "in_out",
            "effective_rank": 2,
            "input_dim": 4,
            "output_dim": 5,
        },
    }
    return agzo


class AGZOCoreTest(unittest.TestCase):
    def test_invalid_hyperparameters_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "num_queries"):
            GLUETransformerAGZO(
                model_name_or_path="not-loaded",
                num_labels=2,
                num_queries=3,
            )
        with self.assertRaisesRegex(ValueError, "power_iter"):
            GLUETransformerAGZO(
                model_name_or_path="not-loaded",
                num_labels=2,
                agzo_power_iter_steps=0,
            )
        with self.assertRaisesRegex(NotImplementedError, "full-parameter"):
            GLUETransformerAGZO(
                model_name_or_path="not-loaded",
                num_labels=2,
                full_parameter=False,
            )

    def test_power_iteration_and_activation_rank_clamp(self):
        agzo = make_agzo(rank=5)
        h_2d = torch.diag(torch.tensor([9.0, 3.0, 1.0]))
        generator = agzo._make_generator(
            h_2d.device,
            seed=13,
            param_index=0,
            stream=0,
        )
        basis_t = agzo._block_power_iteration(
            h_2d,
            rank=1,
            num_steps=20,
            generator=generator,
        )
        self.assertEqual(tuple(basis_t.shape), (1, 3))
        self.assertGreater(abs(float(basis_t[0, 0])), 0.999)

        activation = torch.randn(1, 2, 4)
        param = nn.Parameter(torch.randn(3, 4))
        spec = agzo._compute_activation_basis(
            activation=activation,
            param=param,
            param_index=0,
            layout="out_in",
            input_dim=4,
            output_dim=3,
            basis_seed=17,
        )
        self.assertEqual(spec["effective_rank"], 2)
        self.assertEqual(tuple(spec["basis_t"].shape), (2, 4))
        self.assertTrue(
            torch.allclose(
                spec["basis_t"] @ spec["basis_t"].t(),
                torch.eye(2),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_linear_conv1d_and_fallback_directions(self):
        agzo = make_manual_agzo()
        seed = 123

        linear_param = agzo.params[0][1]
        linear_direction = agzo._make_direction(
            "linear.weight", linear_param, 0, seed
        )
        generator = agzo._make_generator(
            linear_param.device, seed, param_index=0, stream=1
        )
        random_factor = torch.randn((3, 2), generator=generator)
        expected_linear = (
            random_factor @ agzo._agzo_basis["linear.weight"]["basis_t"]
        ) / math.sqrt(2)
        self.assertTrue(torch.equal(linear_direction, expected_linear))
        self.assertTrue(torch.equal(linear_direction[:, 2:], torch.zeros(3, 2)))

        conv_param = agzo.params[1][1]
        conv_direction = agzo._make_direction(
            "conv.weight", conv_param, 1, seed
        )
        generator = agzo._make_generator(
            conv_param.device, seed, param_index=1, stream=1
        )
        random_factor = torch.randn((5, 2), generator=generator)
        expected_conv = (
            agzo._agzo_basis["conv.weight"]["basis_t"].t()
            @ random_factor.t()
        ) / math.sqrt(2)
        self.assertTrue(torch.equal(conv_direction, expected_conv))
        self.assertTrue(torch.equal(conv_direction[2:], torch.zeros(2, 5)))

        bias_param = agzo.params[2][1]
        fallback = agzo._make_direction("bias", bias_param, 2, seed)
        self.assertEqual(tuple(fallback.shape), tuple(bias_param.shape))
        self.assertTrue(
            torch.equal(
                fallback,
                agzo._make_direction("bias", bias_param, 2, seed),
            )
        )

    def test_hooks_cover_linear_and_conv1d_only(self):
        model = ToyAGZOModel()
        agzo = make_agzo(rank=1, model=model)
        agzo.configure_params()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([0, 1]),
        }

        loss, _ = agzo._collect_activation_basis(model, batch, basis_seed=29)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("linear.weight", agzo._agzo_basis)
        self.assertIn("conv.weight", agzo._agzo_basis)
        self.assertIn("head.weight", agzo._agzo_basis)
        self.assertNotIn("embedding.weight", agzo._agzo_basis)
        self.assertNotIn("norm.weight", agzo._agzo_basis)
        self.assertGreater(agzo._guided_parameter_numel, 0)
        self.assertGreater(agzo._fallback_parameter_numel, 0)

        for handle in agzo._agzo_hook_handles:
            handle.remove()

    def test_training_step_same_point_and_forward_accounting(self):
        model = ToyAGZOModel()
        agzo = make_agzo(rank=1, num_queries=6, model=model)
        agzo.configure_params()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([0, 1]),
        }
        initial = [param.detach().clone() for _, param in agzo.params]
        original_central_difference = agzo._central_difference
        estimation_points = []

        def recorded_central_difference(this, model_arg, batch_arg, seed):
            estimation_points.append(
                [param.detach().clone() for _, param in this.params]
            )
            return original_central_difference(model_arg, batch_arg, seed)

        agzo._central_difference = MethodType(
            recorded_central_difference,
            agzo,
        )
        log = {}
        np.random.seed(0)
        agzo.training_step(model, batch, log=log)

        self.assertEqual(len(estimation_points), 3)
        for point in estimation_points:
            for value, reference in zip(point, initial):
                self.assertTrue(
                    torch.allclose(value, reference, atol=1e-6, rtol=1e-6)
                )
        self.assertEqual(agzo.state.query[-1], 14)
        self.assertEqual(agzo.state.num_forward_passes, 7)
        self.assertEqual(log["num_forward_passes"], 7)
        self.assertEqual(log["agzo_num_directions"], 3)
        self.assertEqual(log["agzo_activation_forward_evals"], 1)
        self.assertEqual(log["agzo_probe_forward_evals"], 6)
        self.assertEqual(log["agzo_total_forward_evals"], 7)
        self.assertEqual(agzo._agzo_basis, {})

        for handle in agzo._agzo_hook_handles:
            handle.remove()

    def test_forward_exception_restores_parameters(self):
        agzo = make_manual_agzo()
        initial = [param.detach().clone() for _, param in agzo.params]
        calls = 0

        def failing_forward(this, model, batch):
            nonlocal calls
            del this, model, batch
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic forward failure")
            return torch.tensor(1.0)

        agzo.zo_forward_memory_eff = MethodType(failing_forward, agzo)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            agzo._central_difference(None, {}, direction_seed=41)

        for (_, param), reference in zip(agzo.params, initial):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()

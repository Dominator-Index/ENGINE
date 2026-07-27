import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from model.glue_transformer_zogv import GLUETransformerZOGV


def make_zogv(
    num_queries=4,
    gv_split_ratio=0.5,
    learning_rate=0.01,
    weight_decay=0.0,
    z_std=1.0,
):
    zogv = GLUETransformerZOGV.__new__(GLUETransformerZOGV)
    torch.nn.Module.__init__(zogv)
    zogv.zero_order_eps = 1e-3
    zogv.z_std = z_std
    zogv.num_queries = num_queries
    zogv.weight_decay = weight_decay
    zogv.gv_split_ratio = gv_split_ratio
    zogv.candidate_count = num_queries - 2
    zogv.elite_count = int(gv_split_ratio * zogv.candidate_count)
    zogv.params = [
        (
            "weight",
            torch.nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(3, 4) / 10
            ),
        ),
        (
            "bias",
            torch.nn.Parameter(torch.arange(3, dtype=torch.float32) / 10),
        ),
    ]
    zogv.state = SimpleNamespace(
        learning_rate=learning_rate,
        global_training_steps=0,
        num_forward_passes=0,
        query=[],
        time=[],
        proj_val=[],
    )
    return zogv


class ZOGVCoreTest(unittest.TestCase):
    def test_invalid_hyperparameters_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "num_queries"):
            GLUETransformerZOGV(
                model_name_or_path="not-loaded",
                num_labels=2,
                num_queries=3,
            )
        with self.assertRaisesRegex(ValueError, "gv_split_ratio"):
            GLUETransformerZOGV(
                model_name_or_path="not-loaded",
                num_labels=2,
                gv_split_ratio=0.0,
            )
        with self.assertRaisesRegex(ValueError, "selects no candidates"):
            GLUETransformerZOGV(
                model_name_or_path="not-loaded",
                num_labels=2,
                num_queries=4,
                gv_split_ratio=0.1,
            )
        with self.assertRaisesRegex(NotImplementedError, "full-parameter"):
            GLUETransformerZOGV(
                model_name_or_path="not-loaded",
                num_labels=2,
                full_parameter=False,
            )

    def test_selection_and_guided_direction_formula(self):
        zogv = make_zogv(
            num_queries=6,
            gv_split_ratio=0.5,
            z_std=1.7,
        )
        seeds = [10, 11, 12, 13]
        losses = [3.0, 1.0, 4.0, 2.0]
        low_seeds, high_seeds = zogv._select_guiding_groups(
            seeds,
            losses,
        )
        self.assertEqual(low_seeds, [11, 13])
        self.assertEqual(high_seeds, [10, 12])

        param = zogv.params[0][1]
        direction = zogv._make_guided_direction(
            param,
            param_index=0,
            low_seeds=low_seeds,
            high_seeds=high_seeds,
        )
        expected = sum(
            zogv._make_candidate_direction(param, 0, seed)
            for seed in low_seeds
        ) / 2
        expected -= sum(
            zogv._make_candidate_direction(param, 0, seed)
            for seed in high_seeds
        ) / 2
        self.assertTrue(torch.allclose(direction, expected))

    def test_m4_special_case_is_best_minus_worst(self):
        zogv = make_zogv()
        seeds = [101, 202]
        low_seeds, high_seeds = zogv._select_guiding_groups(
            seeds,
            [1.0, 3.0],
        )
        param = zogv.params[1][1]
        direction = zogv._make_guided_direction(
            param,
            param_index=1,
            low_seeds=low_seeds,
            high_seeds=high_seeds,
        )
        expected = zogv._make_candidate_direction(param, 1, 101)
        expected -= zogv._make_candidate_direction(param, 1, 202)
        self.assertTrue(torch.equal(direction, expected))

    def test_direction_reconstruction_does_not_change_global_rng(self):
        zogv = make_zogv()
        torch.manual_seed(7)
        expected = torch.rand(5)
        torch.manual_seed(7)
        zogv._make_candidate_direction(zogv.params[0][1], 0, 123)
        actual = torch.rand(5)
        self.assertTrue(torch.equal(actual, expected))

    def test_update_formula_and_weight_decay_once(self):
        zogv = make_zogv(
            learning_rate=0.03,
            weight_decay=0.2,
        )
        low_seeds = [11]
        high_seeds = [22]
        projected_grad = 0.4
        initial = [param.detach().clone() for _, param in zogv.params]
        expected = []
        decay = 1.0 - zogv.state.learning_rate * zogv.weight_decay
        for param_index, ((_, param), reference) in enumerate(
            zip(zogv.params, initial)
        ):
            direction = zogv._make_guided_direction(
                param,
                param_index,
                low_seeds,
                high_seeds,
            )
            expected.append(
                reference * decay
                - zogv.state.learning_rate * projected_grad * direction
            )

        zogv._apply_zogv_update(
            low_seeds,
            high_seeds,
            projected_grad,
        )
        for (_, param), reference in zip(zogv.params, expected):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )

    def test_candidate_and_guided_forward_exceptions_restore_parameters(self):
        zogv = make_zogv()
        initial = [param.detach().clone() for _, param in zogv.params]

        def failing_forward(this, model, batch):
            del this, model, batch
            raise RuntimeError("synthetic candidate failure")

        zogv.zo_forward_memory_eff = MethodType(failing_forward, zogv)
        with self.assertRaisesRegex(RuntimeError, "candidate"):
            zogv._score_candidate(None, {}, candidate_seed=19)
        for (_, param), reference in zip(zogv.params, initial):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )

        calls = 0

        def second_forward_fails(this, model, batch):
            nonlocal calls
            del this, model, batch
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic guided failure")
            return torch.tensor(1.0)

        zogv.zo_forward_memory_eff = MethodType(second_forward_fails, zogv)
        with self.assertRaisesRegex(RuntimeError, "guided"):
            zogv._guided_central_difference(None, {}, [19], [23])
        for (_, param), reference in zip(zogv.params, initial):
            self.assertTrue(
                torch.allclose(param, reference, atol=1e-6, rtol=1e-6)
            )

    def test_training_step_same_point_budget_and_logging(self):
        zogv = make_zogv(
            num_queries=6,
            gv_split_ratio=0.5,
            learning_rate=0.03,
        )
        initial = [param.detach().clone() for _, param in zogv.params]
        candidate_points = []

        def toy_forward(this, model, batch):
            del model, batch
            return sum(
                param.float().square().sum()
                for _, param in this.params
            )

        original_score = zogv._score_candidate

        def recorded_score(this, model, batch, seed):
            candidate_points.append(
                [param.detach().clone() for _, param in this.params]
            )
            return original_score(model, batch, seed)

        zogv.zo_forward_memory_eff = MethodType(toy_forward, zogv)
        zogv._score_candidate = MethodType(recorded_score, zogv)
        log = {}
        np.random.seed(0)
        zogv.training_step(
            model=None,
            batch={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
            log=log,
        )

        self.assertEqual(len(candidate_points), 4)
        for point in candidate_points:
            for value, reference in zip(point, initial):
                self.assertTrue(
                    torch.allclose(value, reference, atol=1e-6, rtol=1e-6)
                )
        self.assertEqual(zogv.state.query[-1], 12)
        self.assertEqual(zogv.state.num_forward_passes, 6)
        self.assertEqual(log["zogv_candidate_forward_evals"], 4)
        self.assertEqual(log["zogv_guided_forward_evals"], 2)
        self.assertEqual(log["zogv_total_forward_evals"], 6)
        self.assertEqual(log["zogv_elite_count"], 2)
        self.assertTrue(np.isfinite(log["train_loss"]))
        self.assertTrue(np.isfinite(log["zogv_projected_grad"]))
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor) and value.numel() > 1
                for value in zogv.__dict__.values()
            )
        )


if __name__ == "__main__":
    unittest.main()

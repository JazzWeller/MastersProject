"""Milestone H's real GPU-resident model (Code/AGENT_INTERFACE_PLAN.md):
bots/torch_model.py. Needs `torch`, deliberately NOT installed anywhere the
rest of this suite runs (the engine itself must stay stdlib-only -- see
that module's own docstring on the CUDA rule) -- these tests skip
themselves out when it's absent, and only actually run from the isolated
environment it was installed into to prove this real (WSL + a venv with
torch built for this machine's actual GPU, an RTX 5060 Ti). Run with:
`~/torchenv/bin/python -m unittest tests.test_torch_model` (from a WSL
shell, Code/Non-GUI as the working directory).
"""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game


def _some_views(n=10, seed=1):
    game = Game(GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=60))
    bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
    views = []
    while not game.is_over and len(views) < n:
        d = game.pending_decision
        views.append(game.view_for(d.player))
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
    return views


@unittest.skipUnless(TORCH_AVAILABLE, "torch is intentionally not installed for the rest of this suite")
class TestFeaturize(unittest.TestCase):
    def test_produces_a_fixed_length_float_vector(self):
        from bots.torch_model import STATE_FEATURES, featurize

        for view in _some_views():
            vec = featurize(view)
            self.assertEqual(len(vec), STATE_FEATURES)
            self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_never_raises_regardless_of_which_seat_is_viewing(self):
        """The whole reason this uses PlayerPublicState's *_count fields,
        not .hand/.archive themselves: those are None for the opponent."""
        from bots.torch_model import featurize

        game = Game(GameConfig(decks=("fignor", "igor"), seed=2, max_turns=40))
        bots = {1: RandomBot(seed=2), 2: RandomBot(seed=3)}
        for _ in range(15):
            if game.is_over:
                break
            d = game.pending_decision
            featurize(game.view_for(1))
            featurize(game.view_for(2))
            game.submit(bots[d.player].decide(game.view_for(d.player), d))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is intentionally not installed for the rest of this suite")
class TestKeyForgeNet(unittest.TestCase):
    def test_build_model_runs_on_cuda_when_available(self):
        from bots.torch_model import build_model

        model = build_model(device="cuda")
        if torch.cuda.is_available():
            self.assertEqual(model.device.type, "cuda")
        else:
            self.assertEqual(model.device.type, "cpu")

    def test_single_prediction_shape_and_normalization(self):
        from bots.torch_model import POLICY_SIZE, build_model, featurize

        model = build_model(device="cpu")
        view = _some_views(n=1)[0]
        policy, value = model(featurize(view))
        self.assertEqual(len(policy), POLICY_SIZE)
        self.assertAlmostEqual(sum(policy), 1.0, places=4)
        self.assertTrue(all(p >= 0 for p in policy))
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_batched_prediction_matches_single_predictions(self):
        from bots.torch_model import build_model, featurize

        model = build_model(device="cpu")
        views = _some_views(n=6)
        obs = [featurize(v) for v in views]

        batched = model.predict_many(obs)
        singles = [model(o) for o in obs]

        self.assertEqual(len(batched), len(singles))
        for (bp, bv), (sp, sv) in zip(batched, singles):
            self.assertAlmostEqual(bv, sv, places=4)
            for a, b in zip(bp, sp):
                self.assertAlmostEqual(a, b, places=4)

    def test_predict_many_of_empty_list_is_empty(self):
        from bots.torch_model import build_model

        model = build_model(device="cpu")
        self.assertEqual(model.predict_many([]), [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is intentionally not installed for the rest of this suite")
class TestOptionPolicyIndex(unittest.TestCase):
    def test_every_option_at_a_real_choose_action_decision_has_an_index(self):
        from bots.torch_model import POLICY_SIZE, option_policy_index

        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=60))
        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        checked = 0
        while not game.is_over:
            d = game.pending_decision
            if d.kind.name == "CHOOSE_ACTION":
                for option in d.options:
                    idx = option_policy_index(d, option)
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, POLICY_SIZE)
                    checked += 1
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertGreater(checked, 0)

    def test_indices_are_unique_per_distinct_card(self):
        from bots.torch_model import option_policy_index
        from keyforge.actions import EndTurn

        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=60))
        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        while not game.is_over:
            d = game.pending_decision
            if d.kind.name == "CHOOSE_ACTION" and len(d.options) > 2:
                indices = {}
                for option in d.options:
                    idx = option_policy_index(d, option)
                    name = "EndTurn" if isinstance(option, EndTurn) else option.card.name
                    if idx in indices and indices[idx] != name:
                        self.fail(f"index {idx} maps to both {indices[idx]!r} and {name!r}")
                    indices[idx] = name
                return
            game.submit(bots[d.player].decide(game.view_for(d.player), d))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is intentionally not installed for the rest of this suite")
class TestEndToEndWithRealInferenceServer(unittest.TestCase):
    """The full pipeline Milestone H describes: a self-play-style worker
    talking to a real InferenceServer, with real dynamic batching, backed
    by a real GPU-resident model -- not three separately-tested pieces
    taken on faith to compose."""

    def test_concurrent_clients_get_correct_batched_gpu_predictions(self):
        import threading
        import time

        from bots.inference_client import InferenceServer, RemoteInferenceClient
        from bots.torch_model import build_model, featurize

        model = build_model(device="cuda")
        server = InferenceServer(
            ("localhost", 0), authkey=b"e2e", model=model,
            max_batch_size=64, batch_window_seconds=0.2,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 2.0
            while server.address == ("localhost", 0) and time.monotonic() < deadline:
                time.sleep(0.01)

            views = _some_views(n=8)
            observations = [featurize(v) for v in views]
            expected = model.predict_many(observations)

            results = [None] * len(observations)

            def fire(i):
                client = RemoteInferenceClient(server.address, authkey=b"e2e")
                results[i] = client.predict(observations[i])

            workers = [threading.Thread(target=fire, args=(i,)) for i in range(len(observations))]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=10)

            for (rp, rv), (ep, ev) in zip(results, expected):
                self.assertAlmostEqual(rv, ev, places=4)
                for a, b in zip(rp, ep):
                    self.assertAlmostEqual(a, b, places=4)
        finally:
            server.stop()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

"""After Tier 2 round 4: the configurable promotion rule, M9's evaluation
options (per-player regime / leaf / simulations / no search), and M11's
random-deck self-play and held-out-deck evaluation.
"""

import random
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - torch-free environments
    torch = None


class _Args:
    deck_set = 0
    deck_seed = 2
    first_seed = 3_000_000
    unit = 4


class TestPromotionRule(unittest.TestCase):
    def test_the_rule_comes_from_the_config(self):
        from ml.selfplay_gate import GATE_FALLBACK_Z, GATE_SPRT, gate_rule

        self.assertEqual(gate_rule({"selfplay": {}}), (GATE_SPRT, GATE_FALLBACK_Z))
        self.assertEqual(gate_rule({"selfplay": {"gate_sprt": [0, 15], "gate_fallback_z": 1.0}}), ((0.0, 15.0), 1.0))

    def test_one_standard_error_promotes_what_two_did_not(self):
        from ml.selfplay_gate import gate_verdict
        from sim.paired import PairedReport

        rep = PairedReport(games=400, a_wins=212, b_wins=188, draws=0, forfeits=0)  # 0.53, round 4's typical gate
        self.assertIsNone(gate_verdict(rep, (0.0, 35.0), 2.0))
        self.assertEqual(gate_verdict(rep, (0.0, 15.0), 1.0), "H1")


class TestDecks(unittest.TestCase):
    def test_a_gate_unit_plays_the_run_decks_or_a_seeded_random_pair(self):
        from ml.selfplay_gate import unit_decks

        self.assertEqual(unit_decks({"deck_source": "presets", "decks": ["fignor", "igor"]}, 5), ("fignor", "igor"))
        a = unit_decks({"deck_source": "random"}, 5)
        self.assertEqual([d.pods for d in a], [d.pods for d in unit_decks({"deck_source": "random"}, 5)])
        self.assertNotEqual([d.pods for d in a], [d.pods for d in unit_decks({"deck_source": "random"}, 9)])

    def test_held_out_deck_pairs_depend_on_the_unit_only(self):
        from tools.eval_v2 import out_name, unit_decks

        args = _Args()
        self.assertEqual(unit_decks(args, 3_000_000), ("fignor", "igor"))
        args.deck_set = 8
        first = unit_decks(args, 3_000_000)
        self.assertEqual([d.pods for d in first], [d.pods for d in unit_decks(args, 3_000_000)])
        pairs = {tuple(d.name for d in unit_decks(args, 3_000_000 + 4 * k)) for k in range(16)}
        self.assertGreater(len(pairs), 8)
        from sim.bc_corpus import fixed_deck_set

        training = {str(d.pods) for d in fixed_deck_set(64, deck_seed=1)}
        self.assertFalse({str(d.pods) for d in fixed_deck_set(8, deck_seed=2)} & training)
        self.assertEqual(out_name("v1:x", "heuristic", 100, 64, 3_000_000, deck_set=32),
                         "v1-x__heuristic__100sims_64x3000000_decks32s2.json")
        self.assertEqual(out_name("v1:x", "heuristic", 100, 64, 3_000_000), "v1-x__heuristic__100sims_64x3000000.json")


class TestPlayerOptions(unittest.TestCase):
    def test_options_parse_and_unknown_ones_are_refused(self):
        from tools.eval_v2 import player_options

        self.assertEqual(player_options("v1:m9-fg-1#regime=within_turn,leaf=belief_oracle"),
                         ("v1:m9-fg-1", {"regime": "within_turn", "leaf": "belief_oracle"}))
        self.assertEqual(player_options("v1:tier2-v1-s3@p2"), ("v1:tier2-v1-s3@p2", {}))
        with self.assertRaises(ValueError):
            player_options("v1:x#simulations=10")

    def test_the_ladder_checks_a_run_without_its_options(self):
        import os
        import tempfile
        from unittest import mock

        from tools import ladder

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"KEYFORGE_DATA": root}):
            self.assertFalse(ladder._available("v1:no-run#sims=10"))
            self.assertTrue(ladder._available("heuristic"))


@unittest.skipIf(torch is None, "needs torch")
class TestRandomDeckSelfPlay(unittest.TestCase):
    """M11: actors draw each game's decks from its seed; v2 records replay
    exactly; v1 records train."""

    def _play(self, version):
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient

        if version == 2:
            from ml.infer_v2 import TorchModelV2
            from ml.model_v2 import KeyForgeNetV2

            net = KeyForgeNetV2({"d_model": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0, "history_arch": "none",
                                 "enumerate_cap": 1024, "options_in_trunk": False})
            model = TorchModelV2(net, "cpu", compile=False)
            extra = {"observation_version": 2, "history": "none"}
        else:
            from ml.infer_server import TorchModel
            from ml.model import KeyForgeNet
            from tests.test_agent_training_ml import _net_cfg

            net = KeyForgeNet(_net_cfg())
            model = TorchModel(net, "cpu")
            extra = {}
        s = SelfPlaySettings(leaf="student", sims_full=4, sims_small=2, full_fraction=0.5, concurrency=2, max_turns=6,
                             leaves_in_flight=2, pool={"deck_source": "random"}, **extra)
        recs = []
        Actor(InProcessInferenceClient(model), s, run_seed=4, worker=0).play(range(2), recs.append)
        return net, recs

    def test_v2_games_use_random_decks_and_replay(self):
        from ml.selfplay_v2 import encode_selfplay_game

        torch.manual_seed(0)
        _net, recs = self._play(2)
        names = [tuple(r["meta"]["decks"]) for r in recs]
        self.assertTrue(all(n == ("R1", "R2") for n in names))
        self.assertNotEqual(recs[0]["config"]["decks"], recs[1]["config"]["decks"])
        for r in recs:
            self.assertEqual(encode_selfplay_game(r)["n"], len(r["positions"]))

    def test_v1_random_deck_positions_train(self):
        from ml.selfplay_train import Buffer, losses_for

        torch.manual_seed(0)
        net, recs = self._play(1)
        buf = Buffer(10_000, 100)
        for r in recs:
            buf.add(r)
        total, L, _st = losses_for(net, buf.sample(32, random.Random(0)), "cpu", {}, "search", False, random.Random(0))
        self.assertTrue(torch.isfinite(total))
        self.assertIn("value", L)


if __name__ == "__main__":
    unittest.main()

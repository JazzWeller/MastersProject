"""SAS-surprise weighting of the learner's outcome targets (ml/sas_weight.py, `selfplay.sas_weighting`)."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest

import run
from ml.sas_weight import SasWeigher, expected
from sim.decklists import decklist, sas_of

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _record(d1, d2, z1, n=3):
    return {"meta": {"decks": [d1, d2], "seed": 1, "index": 0}, "outcome": {1: z1, 2: -z1},
            "positions": [{"seat": 1 + k % 2, "target": None, "value_only": True} for k in range(n)]}


def _decks(gap_lo, gap_hi):
    ds = decklist("phase3", "train")
    for a in ds:
        for b in ds:
            if gap_lo <= sas_of("phase3", a.name) - sas_of("phase3", b.name) < gap_hi:
                return a.name, b.name
    raise AssertionError("no pair")


class Weights(unittest.TestCase):
    def setUp(self):
        self.w = SasWeigher("phase3", 10.0)

    def test_even_decks_weigh_one_whatever_happened(self):
        a, b = _decks(0, 0.5)
        for z in (1, -1, 0):
            self.assertAlmostEqual(self.w.game_weight(_record(a, b, z)), 1.0)

    def test_the_expected_result_weighs_less_and_an_upset_more(self):
        hi, lo = _decks(30, 40)  # seat 1 holds the much stronger deck
        expected_win = self.w.game_weight(_record(hi, lo, 1))
        upset = self.w.game_weight(_record(hi, lo, -1))
        self.assertLess(expected_win, 0.6)
        self.assertGreater(upset, 1.4)
        self.assertAlmostEqual(expected_win + upset, 2.0)  # |1 - E| + |0 - E|, doubled
        self.assertAlmostEqual(self.w.game_weight(_record(lo, hi, -1)), expected_win)  # the same game, seats swapped

    def test_expected_score(self):
        self.assertAlmostEqual(expected(0, 10), 0.5)
        self.assertAlmostEqual(expected(40, 10), 1 / (1 + 10 ** -1))


@unittest.skipIf(torch is None, "needs torch")
class Learner(unittest.TestCase):
    def test_the_buffer_tags_positions_and_validation_stays_unweighted(self):
        from ml.selfplay_train import Buffer

        hi, lo = _decks(30, 40)
        train = Buffer(1000, 10, weigher=SasWeigher("phase3", 10.0))
        val = Buffer(1000, 10)
        train.add(_record(hi, lo, -1))
        val.add(_record(hi, lo, -1))
        ws = {p["w"] for g in train.games for p in g}
        self.assertEqual(len(ws), 1)
        self.assertGreater(ws.pop(), 1.4)
        self.assertFalse(any("w" in p for g in val.games for p in g))

    def test_weights_make_the_value_loss_a_weighted_mean(self):
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet
        from ml.selfplay_train import Buffer, losses_for
        from tests.test_agent_training_ml import _net_cfg

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg())
        s = SelfPlaySettings(mode="search", leaf="student", sims_full=4, sims_small=2, full_fraction=0.5, concurrency=2,
                             max_turns=10, leaves_in_flight=2)
        recs = []
        Actor(InProcessInferenceClient(TorchModel(net, "cpu")), s, run_seed=1).play(range(2), recs.append)
        buf = Buffer(10_000, 100)
        for r in recs:
            buf.add(r)
        pos = buf.sample(32, random.Random(0))
        dev = torch.device("cpu")
        wts = {"policy": 1, "value": 1, "belief": 0.25, "oracle": 0.25}
        with torch.no_grad():
            _t, plain, _s = losses_for(net, pos, dev, wts, "search", False, random.Random(1))
            ones = [dict(p, w=1.0) for p in pos]
            _t, same, _s = losses_for(net, ones, dev, wts, "search", False, random.Random(1))
            self.assertAlmostEqual(float(plain["value"]), float(same["value"]), places=5)
            self.assertAlmostEqual(float(plain["oracle"]), float(same["oracle"]), places=5)
            # all the weight on one position: the value loss is that position's error alone
            only = [dict(p, w=(1.0 if i == 0 else 0.0)) for i, p in enumerate(pos)]
            _t, one, _s = losses_for(net, only, dev, wts, "search", False, random.Random(1))
            _t, alone, _s = losses_for(net, pos[:1], dev, wts, "search", False, random.Random(1))
            self.assertAlmostEqual(float(one["value"]), float(alone["value"]), places=4)
            self.assertAlmostEqual(float(one["policy" if "policy" in one else "multi"]),
                                   float(plain["policy" if "policy" in plain else "multi"]), places=5)


class Fit(unittest.TestCase):
    def test_the_slope_is_recovered_from_shards_and_cached(self):
        from agent.selfplay import write_frame
        from ml.sas_weight import resolve

        rng = random.Random(4)
        ds = [d.name for d in decklist("phase3", "train")]
        with tempfile.TemporaryDirectory() as root:
            shards = os.path.join(root, "src", "shards")
            os.makedirs(shards)
            for w in range(3):
                for _ in range(1500):
                    a, b = rng.sample(ds, 2)
                    p1 = expected(sas_of("phase3", a) - sas_of("phase3", b), 8.0)
                    write_frame(os.path.join(shards, f"actor_{w:03d}.bin"),
                                _record(a, b, 1 if rng.random() < p1 else -1, n=0))
            weigher = resolve({"deck_list": "phase3", "elo_per_sas": "fit:src"}, root)
            self.assertAlmostEqual(weigher.k, 8.0, delta=1.5)
            with open(os.path.join(root, "src", "sas_elo.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["elo_per_sas"], round(weigher.k, 3))
            self.assertIsNone(resolve(None, root))


class Jobs(unittest.TestCase):
    def test_each_weighted_arm_is_its_plateau_arm_plus_the_weighting(self):
        for arm in run.PLATEAU_ARMS:
            plain, sasw = (run.METHODS[m]["steps"][0] for m in (f"plateau-{arm}", f"plateau-sasw-{arm}"))
            po = json.loads(plain[plain.index("--override") + 1])
            so = json.loads(sasw[sasw.index("--override") + 1])
            self.assertEqual(so["pool"], po["pool"])
            self.assertEqual(so["selfplay"], {**po["selfplay"], **run.SASW})
            self.assertEqual(sasw[sasw.index("--init-run") + 1], "real-v1")
        from agent.config import DEFAULTS

        self.assertIsNone(DEFAULTS["selfplay"]["sas_weighting"])
        ladder = run.METHODS["eval-plateau-sasw"]["steps"][0]
        for arm in run.PLATEAU_ARMS:
            self.assertIn(f"v1:plateau-sasw-{arm}@final", ladder)
            self.assertIn(f"v1:plateau-{arm}@final", ladder)


if __name__ == "__main__":
    unittest.main()

"""The paper's evaluation additions: gate checkpoints (`@g<N>`), search-only and DMC players, anchored ladders, names."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


class PaperToolsTest(unittest.TestCase):
    def test_anchored_pairs(self):
        from tools.ladder import anchored_pairs

        self.assertEqual(len(anchored_pairs(5)), 10)
        self.assertEqual(anchored_pairs(4, 2), [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])

    def test_gate_checkpoint(self):
        from tools.eval_v2 import _checkpoint

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"KEYFORGE_DATA": d}):
            root = os.path.join(d, "runs", "r")
            os.makedirs(os.path.join(root, "checkpoints"))
            for h in ("aa11", "bb22", "cc33"):
                open(os.path.join(root, "checkpoints", h + ".kfc"), "w").close()
            with open(os.path.join(root, "journal.jsonl"), "w") as f:
                for e in ({"event": "promoted", "checkpoint": "aa11", "reason": "warm start"},
                          {"event": "gate", "number": 1, "candidate": "bb22", "score": 0.4},
                          {"event": "gate", "number": 2, "candidate": "cc33", "score": 0.6},
                          {"event": "promoted", "checkpoint": "cc33"}):
                    f.write(json.dumps(e) + "\n")
            self.assertTrue(_checkpoint("r@g1").endswith("bb22.kfc"))
            self.assertTrue(_checkpoint("r@g2").endswith("cc33.kfc"))
            self.assertTrue(_checkpoint("r@p0").endswith("aa11.kfc"))
            with self.assertRaises(FileNotFoundError):
                _checkpoint("r@g3")

    def test_search_only_player(self):
        from tools.eval_v2 import _player

        args = SimpleNamespace(regime="within_turn", sims=100, a_exact=False)
        p = _player("search:heuristic#regime=full_game,sims=20", args, "a")
        self.assertEqual((p.kind, p.model, p.regime, p.simulations), ("search", None, "full_game", 20))
        for bad in ("search:other", "search:heuristic#leaf=student", "search:heuristic#mode=q", "v2:x#mode=q"):
            with self.assertRaises(ValueError):
                _player(bad, args, "a")

    def test_ladder_availability(self):
        from tools.ladder import _available

        self.assertTrue(_available("search:heuristic#regime=full_game"))
        self.assertTrue(_available("random"))

    def test_names(self):
        from tools.report_names import name

        self.assertEqual(name("search:heuristic"), "Search, no network")
        self.assertEqual(name("search:heuristic#regime=full_game"), "Search, no network · full-game search")
        self.assertEqual(name("v1:dmc-1#mode=q"), "Deep Monte Carlo (no search), 50k games")
        self.assertEqual(name("v1:real-v1@g3"), "Real-deck agent, gate 3")
        self.assertEqual(name("random"), "RandomBot")

    def test_run_methods(self):
        import run

        for m in ("dmc-1", "dmc-long", "real-v1-s1", "real-v1-s2", "eval-paper", "eval-curves", "export-paper", "paper-stats"):
            self.assertIn(m, run.METHODS)
        step = run.METHODS["dmc-1"]["steps"][0]
        self.assertIn("--mode", step)
        self.assertEqual(json.loads(step[step.index("--override") + 1])["selfplay"]["games"], 50_000)
        curve = run.METHODS["eval-curves"]["steps"][0]
        self.assertEqual(curve[curve.index("--anchors") + 1], "2")


if __name__ == "__main__":
    unittest.main()

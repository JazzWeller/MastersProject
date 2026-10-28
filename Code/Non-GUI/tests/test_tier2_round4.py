"""After Tier 2 round 3: the rating ladder (`tools.ladder`), a run's N-th
promoted network as an eval player (`<run>@p<N>`), the monitor's G5 / G6
reads, and gates played over several processes (`selfplay.gate_workers`).
"""

import json
import math
import os
import tempfile
import time
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - torch-free environments
    torch = None


def _write_eval(root, a, b, a_wins, b_wins, draws=0, sims=100, seeds=100, first=3_000_000):
    d = os.path.join(root, "evals")
    os.makedirs(d, exist_ok=True)
    name = f"{a}__{b}__{sims}sims_{seeds}x{first}".replace(":", "-")
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump({"a": a, "b": b, "games": a_wins + b_wins + draws, "a_wins": a_wins, "b_wins": b_wins, "draws": draws}, f)


class TestNetworkSearchAgent(unittest.TestCase):
    def test_search_net_is_registered_torch_free_and_reads_port_and_simulations(self):
        import agent.agents.registry_entries  # noqa: F401
        from bots.registry import make_agent, privilege_of
        from keyforge.enums import PrivilegeLevel

        a = make_agent("search-net@6150/40", seed=1)  # the client connects on first use, not here
        self.assertEqual(a.search.settings.simulations, 40)
        self.assertEqual(a.search.settings.determinization, "chance_exact")
        self.assertFalse(a.search.settings.reuse)
        self.assertEqual(a.client._address, ("127.0.0.1", 6150))
        self.assertEqual(make_agent("search-net@otherhost:7000").client._address, ("otherhost", 7000))
        self.assertEqual(make_agent("search-net@7000").search.settings.simulations, 100)
        self.assertEqual(privilege_of("search-net@6150"), PrivilegeLevel.SEARCH)


class TestLadder(unittest.TestCase):
    def test_the_fit_recovers_elo_differences_with_the_anchor_at_zero(self):
        from tools.ladder import fit

        # 76% is +200 Elo; B is 200 above A, C 200 above B (so 400 above A).
        p = 1 / (1 + 10 ** (-200 / 400))
        q = 1 / (1 + 10 ** (-400 / 400))
        n = 4000
        elo, se = fit(["A", "B", "C"], {(0, 1): ((1 - p) * n, n), (1, 2): ((1 - p) * n, n), (0, 2): ((1 - q) * n, n)})
        self.assertEqual(elo[0], 0.0)
        self.assertAlmostEqual(elo[1], 200.0, delta=1.0)
        self.assertAlmostEqual(elo[2], 400.0, delta=1.0)
        self.assertEqual(se[0], 0.0)
        self.assertTrue(0 < se[1] < 20 and 0 < se[2] < 20)

    def test_a_clean_sweep_stays_finite(self):
        from tools.ladder import fit

        elo, _se = fit(["A", "B"], {(0, 1): (0.0, 400)})
        self.assertTrue(math.isfinite(elo[1]) and elo[1] > 400)

    def test_matches_are_read_in_either_order_and_missing_runs_are_skipped(self):
        from tools import ladder

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"KEYFORGE_DATA": root}):
            _write_eval(root, "v1:x", "heuristic", 300, 100)
            self.assertEqual(ladder.result_for("v1:x", "heuristic", 100, 100, 3_000_000), (300, 400))
            self.assertEqual(ladder.result_for("heuristic", "v1:x", 100, 100, 3_000_000), (100, 400))
            self.assertIsNone(ladder.result_for("v1:y", "heuristic", 100, 100, 3_000_000))
            self.assertTrue(ladder._available("heuristic"))
            self.assertFalse(ladder._available("v1:no-such-run"))

    def test_a_promoted_checkpoint_by_number(self):
        from tools.eval_v2 import _checkpoint

        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"KEYFORGE_DATA": root}):
            run = os.path.join(root, "runs", "sp")
            os.makedirs(os.path.join(run, "checkpoints"))
            for h in ("aaa", "bbb", "ccc"):
                open(os.path.join(run, "checkpoints", h + "0.kfc"), "w").close()
            with open(os.path.join(run, "journal.jsonl"), "w") as f:
                for h in ("aaa", "bbb", "ccc"):
                    f.write(json.dumps({"event": "promoted", "checkpoint": h}) + "\n")
            self.assertTrue(_checkpoint("sp@p0").endswith("aaa0.kfc"))
            self.assertTrue(_checkpoint("sp@p2").endswith("ccc0.kfc"))
            with self.assertRaises(FileNotFoundError):
                _checkpoint("sp@p3")


class TestMonitorReads(unittest.TestCase):
    def _run(self, d, gates, finished=False, actor_rows=()):
        from agent.telemetry import Run

        run = Run.create("mon", None, root=d)
        for i, (score, verdict) in enumerate(gates):
            run.journal("gate", game_index=30_000 + 1000 * i, score=score, verdict=verdict)
        if finished:
            run.journal("run_finished", game_index=50_000, step=10, final="x")
        now = time.time()
        rows = [{"t": now - 60, "source": "learner", "interval": 60, "counts": {}, "rates": {},
                 "gauges": {"step": 10, "games": 50_000}, "means": {"loss_value": 0.6}, "maxes": {}}]
        rows += list(actor_rows)
        with open(run.metrics_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(dict(r, config_hash=run.config_hash, host="h", pid=1)) + "\n")
        return run

    def test_g6_reads_promotions_not_falling_scores(self):
        from tools.monitor import summarize

        with tempfile.TemporaryDirectory() as d:
            s = summarize(self._run(d, [(0.70, "H1"), (0.60, "H1"), (0.56, "H1")]))
            self.assertFalse(any(f.startswith("G6") for f in s["flags"]), s["flags"])
        with tempfile.TemporaryDirectory() as d:
            s = summarize(self._run(d, [(0.70, "H1"), (0.48, "H0"), (0.52, None), (0.47, "H0")]))
            self.assertTrue(any(f.startswith("G6") for f in s["flags"]), s["flags"])

    def test_a_finished_run_reports_its_whole_run_rate(self):
        from tools.monitor import summarize

        now = time.time()
        actors = [{"t": now - 7200 + 600 * k, "source": "actor_000", "interval": 600, "counts": {"games": 1000},
                   "rates": {}, "gauges": {}, "means": {}, "maxes": {}} for k in range(6)]
        with tempfile.TemporaryDirectory() as d:
            s = summarize(self._run(d, [], finished=True, actor_rows=actors))
            self.assertAlmostEqual(s["games_per_hour"], 6000.0, delta=1.0)  # 6,000 games over one hour of actor records
            self.assertIsNone(s["hours_to_next_gate"])


@unittest.skipIf(torch is None, "needs torch")
class TestParallelGate(unittest.TestCase):
    def test_merged_reports_add_up(self):
        from ml.selfplay_gate import merge_reports
        from sim.paired import PairedReport

        a = PairedReport(4, 3, 1, 0, 0, {1: [2, 2], 2: [1, 2]}, {"fignor": [2, 2]}, [10, 11, 12, 13])
        b = PairedReport(4, 1, 2, 1, 0, {1: [1, 2], 2: [0, 2]}, {"igor": [1, 4]}, [9, 9, 9, 9])
        merge_reports(a, b)
        self.assertEqual((a.games, a.a_wins, a.b_wins, a.draws), (8, 4, 3, 1))
        self.assertEqual(a.by_seat, {1: [3, 4], 2: [1, 4]})
        self.assertEqual(a.by_deck, {"fignor": [2, 2], "igor": [1, 4]})
        self.assertEqual(len(a.turns), 8)

    def test_a_gate_over_two_processes_plays_the_same_seeds(self):
        from agent import config as config_mod
        from ml.checkpoints import save_model
        from ml.model import KeyForgeNet
        from ml.selfplay_gate import run_gate
        from tests.test_agent_training_ml import _net_cfg

        cfg = config_mod.resolve("smoke_selfplay.json", {"selfplay": {"gate_games": 32}})
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for seed in (0, 1):
                torch.manual_seed(seed)
                p = os.path.join(d, f"net{seed}.kfc")
                save_model(KeyForgeNet(_net_cfg()), p, config_hash="test")
                paths.append(p)
            one = run_gate(paths[0], paths[1], cfg, sims=4, mode="search", number=1, device="cpu", workers=1)[1]
            two = run_gate(paths[0], paths[1], cfg, sims=4, mode="search", number=1, device="cpu", workers=2)[1]
        self.assertEqual(one.games, 32)
        self.assertEqual(two.games, 32)  # under 40 games the SPRT never stops early: every seed is played
        self.assertEqual(sorted(one.turns), sorted(two.turns))
        self.assertEqual((one.a_wins, one.b_wins, one.draws), (two.a_wins, two.b_wins, two.draws))


if __name__ == "__main__":
    unittest.main()

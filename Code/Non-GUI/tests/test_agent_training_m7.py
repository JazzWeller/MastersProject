"""Agent Training Plan, Milestones M7/M8 (Code/AGENT_TRAINING_PLAN.md): the
torch-free half of the self-play loop -- the actor and its shards. (The
learner, gating and the DMC loss are in test_agent_training_ml.py.)

The plan's acceptance for the loop: resumable after a kill at any point
with no duplicated or lost shards, and every game reproducible from
`(run seed, worker, game index)`.
"""

import os
import tempfile
import unittest

from agent.selfplay import Actor, SelfPlaySettings, game_seed, read_frames, repair_shard, write_frame


def _settings(**over):
    s = SelfPlaySettings(leaf="heuristic", sims_full=6, sims_small=3, full_fraction=0.5, concurrency=3, max_turns=16,
                         leaves_in_flight=2)
    for k, v in over.items():
        setattr(s, k, v)
    return s


class TestActor(unittest.TestCase):
    def test_plays_records_and_labels_every_position(self):
        records = []
        actor = Actor(None, _settings(), run_seed=1, worker=0)
        n = actor.play(range(3), records.append)
        self.assertEqual(n, 3)
        self.assertEqual(sorted(r["meta"]["index"] for r in records), [0, 1, 2])
        for rec in records:
            self.assertEqual(set(rec["outcome"]), {1, 2})
            self.assertTrue(rec["positions"])
            full = [p for p in rec["positions"] if p["full"]]
            cheap = [p for p in rec["positions"] if not p["full"] and not p["value_only"]]
            self.assertTrue(full and cheap, "playout cap randomization should produce both kinds")
            for p in rec["positions"]:
                if p["target"] is not None:
                    self.assertAlmostEqual(sum(p["target"]), 1.0, places=6)
                    if p["candidates"] is not None:
                        self.assertEqual(len(p["target"]), len(p["candidates"]))
                    else:
                        self.assertEqual(len(p["target"]), p["enc"].n_options)
                self.assertIn("opp_hand", p["priv"])
            for p in cheap:
                self.assertIsNone(p["target"])  # cheap searches are value-only targets

    def test_a_game_reproduces_from_its_seed(self):
        a, b = [], []
        Actor(None, _settings(), run_seed=4, worker=2).play([5], a.append)
        Actor(None, _settings(), run_seed=4, worker=2).play([5], b.append)
        self.assertEqual(a[0]["choice_record"], b[0]["choice_record"])
        self.assertEqual(a[0]["meta"]["seed"], game_seed(4, 2, 5))

    def test_dmc_mode_records_actions_taken(self):
        class Client:
            def predict_many(self, reqs):
                return [([0.1 * (i % 3) for i in range(r.enc.n_options)], 0.0) for r in reqs]

        records = []
        Actor(Client(), _settings(mode="dmc"), run_seed=2).play(range(2), records.append)
        for rec in records:
            for p in rec["positions"]:
                self.assertTrue(p["target"])
                self.assertTrue(all(0 <= a < p["enc"].n_options for a in p["target"]))


class TestResignation(unittest.TestCase):
    def test_resigns_on_a_run_of_low_values_except_in_exempt_games(self):
        # A threshold every real value is below: the first seat to search
        # three times in a row resigns, unless the game is exempt.
        records = []
        actor = Actor(None, _settings(resign_threshold=-2.0, resign_consecutive=3, resign_exempt_fraction=0.5), run_seed=6)
        actor.play(range(8), records.append)
        resigned = [r for r in records if r["meta"]["resigned_by"] is not None]
        exempt = [r for r in records if r["meta"]["resign_exempt"]]
        self.assertTrue(resigned and exempt)
        for r in resigned:
            seat = r["meta"]["resigned_by"]
            self.assertEqual(r["outcome"][seat], -1)
            self.assertEqual(r["outcome"][3 - seat], 1)
        for r in exempt:
            self.assertIsNone(r["meta"]["resigned_by"])
            self.assertIsNotNone(r["meta"]["would_resign"])


class TestShards(unittest.TestCase):
    def test_a_torn_last_frame_is_truncated_and_resume_skips_exactly_the_finished_games(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "actor_000.bin")
            actor = Actor(None, _settings(), run_seed=3, worker=0)
            actor.play([0, 1, 2, 3], lambda rec: write_frame(path, rec))
            good_size = os.path.getsize(path)
            with open(path, "ab") as f:
                f.write(b"\x40\x00\x00\x00partial")  # a crash mid-write
            done = repair_shard(path)
            self.assertEqual(done, {0, 1, 2, 3})
            self.assertEqual(os.path.getsize(path), good_size)
            todo = [i for i in range(6) if i not in done]
            actor.play(todo, lambda rec: write_frame(path, rec))
            indices = [rec["meta"]["index"] for _end, rec in read_frames(path)]
            self.assertEqual(sorted(indices), list(range(6)))  # nothing lost, nothing duplicated

    def test_a_shard_has_exactly_one_live_writer(self):
        from agent.lifecycle import ShardBusy, exclusive

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "actor_000.bin")
            first = exclusive(path)
            if first is None:
                self.skipTest("no advisory locks on this platform (runs are WSL-only)")
            with self.assertRaises(ShardBusy):
                exclusive(path)
            first.close()
            exclusive(path).close()  # released with its holder


class TestFallbackPositions(unittest.TestCase):
    def test_a_multi_select_over_the_enumeration_cap_records_no_policy_target(self):
        # enumerate_cap=1: every multi-select with more than one legal
        # submission falls back to the fixed policy inside the search.
        records = []
        Actor(None, _settings(enumerate_cap=1, full_fraction=1.0), run_seed=5).play(range(4), records.append)
        multi = [p for r in records for p in r["positions"] if p["kind"] in ("CHOOSE_CARDS", "ORDER_EFFECTS") and not p["value_only"]]
        if not multi:
            self.skipTest("no multi-select decision came up")
        for p in multi:
            self.assertIsNone(p["target"])
            self.assertIsNone(p["candidates"])


class TestNetAgentOverTheCap(unittest.TestCase):
    def test_a_multi_select_past_the_enumeration_cap_still_gets_a_legal_answer(self):
        from agent.agents.net_agent import NetAgent
        from keyforge.config import GameConfig
        from sim.driver import run_games

        heads = []

        class Stub:
            def predict_many(self, reqs):
                out = []
                for r in reqs:
                    heads.append(r.head)
                    n = len(r.candidates) if r.head == "subset" else r.enc.n_options + (r.head == "sequential")
                    out.append(([1.0 / max(n, 1)] * n, 0.0))
                return out

        agent = NetAgent(Stub(), enumerate_cap=1, seed=3)  # every real multi-select is over the cap
        results = run_games(4, lambda i: GameConfig(decks=("fignor", "igor"), seed=40 + i, max_turns=20),
                            lambda i: {1: agent, 2: agent}, concurrency=4)
        self.assertTrue(all(r.forfeit is None for r in results))
        if "topk" not in heads:
            self.skipTest("no multi-select decision with more than one legal answer came up")


class TestConfigsAreHonest(unittest.TestCase):
    def test_every_shipped_config_resamples_everything(self):
        """Plan status, departure 4: own_deck leaves the opponent's true
        hand in every world, opponent_private my true draws. Only the
        hidden-information diagnostic may use them, and it sets them in
        code, never through a config."""
        from agent import config as config_mod

        self.assertEqual(config_mod.DEFAULTS["search"]["resample"], "all")
        for name in sorted(os.listdir(config_mod.CONFIG_DIR)):
            if name.endswith(".json"):
                with self.subTest(config=name):
                    self.assertEqual(config_mod.resolve(name)["search"]["resample"], "all")


class TestGatePairing(unittest.TestCase):
    def test_an_early_sprt_stop_reads_only_complete_paired_seeds(self):
        from ml.arena import Player, play_paired
        from ml.selfplay_gate import GATE_SPRT

        rep = play_paired(Player("heuristic", "bot", bot="heuristic"), Player("random", "bot", bot="random"),
                          range(700, 764), concurrency=24, sprt=GATE_SPRT, min_games=8)
        self.assertEqual(rep.sprt(*GATE_SPRT), "H1")
        self.assertLess(rep.games, 256, "should stop early")
        self.assertEqual(rep.games % 4, 0)
        self.assertEqual(rep.by_seat[1][1], rep.by_seat[2][1])
        self.assertEqual(len(set(g for _w, g in rep.by_deck.values())), 1)

    def test_gate_seeds_depend_on_the_run_and_gate_number_only(self):
        from ml.selfplay_gate import gate_seeds

        self.assertEqual(gate_seeds(0, 3, 100), gate_seeds(0, 3, 100))
        self.assertNotEqual(gate_seeds(0, 3, 100), gate_seeds(0, 4, 100))
        self.assertNotEqual(gate_seeds(0, 3, 100), gate_seeds(1, 3, 100))


class TestMonitor(unittest.TestCase):
    def test_an_idle_or_blocked_learner_reads_as_such_not_as_step_zero(self):
        import json
        import time

        from agent.telemetry import Run
        from tools.monitor import summarize

        with tempfile.TemporaryDirectory() as d:
            run = Run.create("mon", None, root=d)
            now = time.time()
            rows = [
                {"t": now - 900, "source": "learner", "interval": 60, "counts": {"gradient_steps": 30, "positions_produced": 360},
                 "rates": {}, "gauges": {"step": 50, "games": 10}, "means": {"loss_value": 0.3}, "maxes": {}},
                {"t": now - 840, "source": "learner", "interval": 60, "counts": {}, "rates": {}, "gauges": {}, "means": {}, "maxes": {}},
            ]
            with open(run.metrics_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(dict(r, config_hash=run.config_hash, host="h", pid=1)) + "\n")
            s = summarize(run)
            self.assertEqual((s["games_done"], s["gradient_steps"]), (10, 50))
            self.assertEqual(s["losses"], {"loss_value": 0.3})
            self.assertTrue(any("learner silent" in f for f in s["flags"]))


if __name__ == "__main__":
    unittest.main()

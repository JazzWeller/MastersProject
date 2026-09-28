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


if __name__ == "__main__":
    unittest.main()

"""sim/actor_pool.py (Agent Interface Plan, Milestone H): the fork-based
multiprocessing.Pool alternative to sim.actor.run_self_play's subprocess
workers. POSIX/WSL/Linux only -- skipped on Windows, run for real under
WSL (this project's own choice of self-play platform).
"""

import os
import tempfile
import unittest

from sim.actor import run_self_play
from sim.actor_pool import available, run_self_play_pool
from sim.generate import read_shard


@unittest.skipUnless(available(), "needs the 'fork' start method (POSIX/WSL/Linux)")
class TestActorPool(unittest.TestCase):
    def test_spawns_one_worker_per_task_each_writing_its_own_shard(self):
        with tempfile.TemporaryDirectory() as d:
            shard_paths = run_self_play_pool(
                n_workers=3, games_per_worker=1, shard_dir=d, run_seed=0,
                decks=("fignor", "igor"), agent1="random", agent2="random", max_turns=40,
            )
            self.assertEqual(len(shard_paths), 3)
            for path in shard_paths:
                self.assertTrue(os.path.exists(path))
                trajectories = read_shard(path)
                self.assertEqual(len(trajectories), 1)
                self.assertGreater(len(trajectories[0].decisions), 0)
            games = [read_shard(p)[0].choice_record for p in shard_paths]
            self.assertEqual(len({tuple(map(str, g)) for g in games}), 3, "different workers must play different games")

    def test_produces_byte_identical_shards_to_the_subprocess_backend(self):
        """Same game_seed derivation, same agents -- a pool worker and a
        subprocess worker for the same (run_seed, worker_id, game_index)
        must produce the exact same trajectory."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            pool_paths = run_self_play_pool(
                n_workers=2, games_per_worker=1, shard_dir=d1, run_seed=7,
                decks=("fignor", "igor"), agent1="random", agent2="random", max_turns=40,
            )
            subprocess_paths = run_self_play(
                n_workers=2, games_per_worker=1, shard_dir=d2, run_seed=7,
                decks=("fignor", "igor"), agent1="random", agent2="random", max_turns=40,
            )
            for pp, sp in zip(pool_paths, subprocess_paths):
                pool_traj = read_shard(pp)[0]
                sub_traj = read_shard(sp)[0]
                self.assertEqual(pool_traj.choice_record, sub_traj.choice_record)
                self.assertEqual(pool_traj.outcome, sub_traj.outcome)

    def test_resumable_like_the_subprocess_backend(self):
        with tempfile.TemporaryDirectory() as d:
            run_self_play_pool(n_workers=1, games_per_worker=2, shard_dir=d, run_seed=1, max_turns=40)
            shard_path = os.path.join(d, "worker_0000.jsonl")
            first_pass = read_shard(shard_path)
            self.assertEqual(len(first_pass), 2)

            run_self_play_pool(n_workers=1, games_per_worker=3, shard_dir=d, run_seed=1, max_turns=40)
            second_pass = read_shard(shard_path)
            self.assertEqual(len(second_pass), 3)
            self.assertEqual(second_pass[0].choice_record, first_pass[0].choice_record)
            self.assertEqual(second_pass[1].choice_record, first_pass[1].choice_record)


class TestAvailabilityGuard(unittest.TestCase):
    def test_matches_multiprocessing_start_methods(self):
        import multiprocessing

        self.assertEqual(available(), "fork" in multiprocessing.get_all_start_methods())

    def test_raises_clearly_when_unavailable(self):
        if available():
            self.skipTest("fork is available on this platform")
        with self.assertRaises(RuntimeError):
            run_self_play_pool(n_workers=1, games_per_worker=1, shard_dir=tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()

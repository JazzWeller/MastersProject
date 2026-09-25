"""`sim.data_root` (Agent Interface Plan, Milestone 0): pipeline output goes
under `$KEYFORGE_DATA`; relative paths resolve there, absolute ones don't."""

import os
import tempfile
import unittest
from unittest import mock

from sim import data_root
from sim.actor import run_self_play


class TestDataRoot(unittest.TestCase):
    def test_env_var_sets_the_root(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"KEYFORGE_DATA": d}):
                self.assertEqual(data_root.data_root(), os.path.abspath(d))
                self.assertEqual(data_root.resolve("a/b"), os.path.join(os.path.abspath(d), "a", "b"))

    def test_default_root_is_in_the_home_directory(self):
        env = {k: v for k, v in os.environ.items() if k != "KEYFORGE_DATA"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(data_root.data_root(), os.path.join(os.path.expanduser("~"), "keyforge-data"))

    def test_absolute_paths_are_left_alone(self):
        absolute = os.path.abspath("somewhere")
        self.assertEqual(data_root.resolve(absolute), absolute)

    def test_self_play_writes_a_relative_shard_dir_under_the_root(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"KEYFORGE_DATA": d}):
                paths = run_self_play(n_workers=1, games_per_worker=1, shard_dir="run/shards", max_turns=20)
            self.assertEqual(os.path.dirname(paths[0]), os.path.join(d, "run", "shards"))
            self.assertTrue(os.path.exists(paths[0]))


if __name__ == "__main__":
    unittest.main()

"""Milestone H (Code/AGENT_INTERFACE_PLAN.md): parallelism and inference.

Scoped to what's testable without a GPU, torch, or WSL in this environment:
the InferenceClient abstraction (in-process and a real stdlib-socket remote
server), the checkpoint registry, and the self-play actor loop's real
subprocess-based parallelism and resumability. The CUDA rule and an actual
GPU-resident model are deployment concerns for whatever real model plugs
into `InferenceServer` later -- see bots/inference_client.py's docstring.
"""

import os
import tempfile
import threading
import time
import unittest

from bots.checkpoint import CheckpointRegistry
from bots.inference_client import InferenceServer, InProcessInferenceClient, RemoteInferenceClient
from sim.actor import game_seed, run_actor, run_self_play
from sim.generate import read_shard


class TestInProcessInferenceClient(unittest.TestCase):
    def test_predict_many_calls_the_model_once_per_observation_in_order(self):
        calls = []

        def model(obs):
            calls.append(obs)
            return ([obs / 2.0, 1 - obs / 2.0], float(obs))

        client = InProcessInferenceClient(model)
        results = client.predict_many([0, 1, 2])
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(results, [([0.0, 1.0], 0.0), ([0.5, 0.5], 1.0), ([1.0, 0.0], 2.0)])

    def test_predict_is_predict_many_of_one(self):
        client = InProcessInferenceClient(lambda obs: ([1.0], obs))
        self.assertEqual(client.predict(7), ([1.0], 7))


class TestRemoteInferenceClient(unittest.TestCase):
    def test_round_trips_through_a_real_socket_server(self):
        server = InferenceServer(("localhost", 0), authkey=b"secret", model=lambda obs: ([obs, -obs], float(obs)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                if server.address != ("localhost", 0):
                    break
                time.sleep(0.01)
            client = RemoteInferenceClient(server.address, authkey=b"secret")
            results = client.predict_many([1, 2, 3])
            self.assertEqual(results, [([1, -1], 1.0), ([2, -2], 2.0), ([3, -3], 3.0)])
            # A second, independent call over a fresh connection works too.
            self.assertEqual(client.predict(9), ([9, -9], 9.0))
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_wrong_authkey_is_refused_without_taking_the_server_down(self):
        server = InferenceServer(("localhost", 0), authkey=b"secret", model=lambda obs: ([], float(obs)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                if server.address != ("localhost", 0):
                    break
                time.sleep(0.01)
            bad_client = RemoteInferenceClient(server.address, authkey=b"wrong")
            with self.assertRaises(Exception):
                bad_client.predict(1)
            # A single bad-auth attempt must not kill the server for
            # everyone else.
            good_client = RemoteInferenceClient(server.address, authkey=b"secret")
            self.assertEqual(good_client.predict(5), ([], 5.0))
        finally:
            server.stop()
            thread.join(timeout=5)


class TestCheckpointRegistry(unittest.TestCase):
    def test_save_load_round_trips_and_versions_increase(self):
        with tempfile.TemporaryDirectory() as d:
            reg = CheckpointRegistry(d)
            self.assertIsNone(reg.latest())
            c1 = reg.save({"w": [1, 2, 3]})
            c2 = reg.save({"w": [4, 5, 6]})
            self.assertEqual(c1.version, 1)
            self.assertEqual(c2.version, 2)
            self.assertNotEqual(c1.content_hash, c2.content_hash)
            self.assertEqual(reg.load(c1), {"w": [1, 2, 3]})
            self.assertEqual(reg.load(c2), {"w": [4, 5, 6]})
            self.assertEqual(reg.latest().version, 2)
            self.assertEqual([c.version for c in reg.list()], [1, 2])

    def test_identical_weights_get_distinct_versions_but_matching_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            reg = CheckpointRegistry(d)
            c1 = reg.save({"w": 1})
            c2 = reg.save({"w": 1})
            self.assertNotEqual(c1.version, c2.version)
            self.assertEqual(c1.content_hash, c2.content_hash)

    def test_a_new_registry_on_the_same_directory_sees_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            CheckpointRegistry(d).save({"w": 1})
            reg2 = CheckpointRegistry(d)
            self.assertEqual(reg2.latest_version(), 1)

    def test_corrupted_checkpoint_file_is_detected_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            reg = CheckpointRegistry(d)
            c = reg.save({"w": 1})
            with open(c.path, "ab") as f:
                f.write(b"corruption")
            with self.assertRaises(ValueError):
                reg.load(c)


class TestGameSeed(unittest.TestCase):
    def test_deterministic_and_sensitive_to_every_key_component(self):
        self.assertEqual(game_seed(0, 0, 0), game_seed(0, 0, 0))
        self.assertNotEqual(game_seed(0, 0, 0), game_seed(0, 0, 1))
        self.assertNotEqual(game_seed(0, 0, 0), game_seed(0, 1, 0))
        self.assertNotEqual(game_seed(0, 0, 0), game_seed(1, 0, 0))


class TestRunActor(unittest.TestCase):
    def test_plays_the_requested_games_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            shard_path = os.path.join(d, "shard.jsonl")
            played = run_actor(
                shard_path, n_games=2, decks=("fignor", "igor"), agent1="random", agent2="random",
                run_seed=0, worker_id=0, max_turns=40,
            )
            self.assertEqual(played, 2)
            first_pass = read_shard(shard_path)
            self.assertEqual(len(first_pass), 2)

            # "Restarting" with a bigger target only plays the delta, and
            # never rewrites the games already on disk.
            played_more = run_actor(
                shard_path, n_games=3, decks=("fignor", "igor"), agent1="random", agent2="random",
                run_seed=0, worker_id=0, max_turns=40,
            )
            self.assertEqual(played_more, 1)
            second_pass = read_shard(shard_path)
            self.assertEqual(len(second_pass), 3)
            self.assertEqual(second_pass[0].choice_record, first_pass[0].choice_record)
            self.assertEqual(second_pass[1].choice_record, first_pass[1].choice_record)

            # Calling again at the same target replays nothing.
            played_none = run_actor(
                shard_path, n_games=3, decks=("fignor", "igor"), agent1="random", agent2="random",
                run_seed=0, worker_id=0, max_turns=40,
            )
            self.assertEqual(played_none, 0)


class TestRunSelfPlay(unittest.TestCase):
    def test_spawns_one_process_per_worker_each_writing_its_own_shard(self):
        with tempfile.TemporaryDirectory() as d:
            shard_paths = run_self_play(
                n_workers=2, games_per_worker=1, shard_dir=d, run_seed=0,
                decks=("fignor", "igor"), agent1="random", agent2="random", max_turns=40,
            )
            self.assertEqual(len(shard_paths), 2)
            for path in shard_paths:
                self.assertTrue(os.path.exists(path))
                trajectories = read_shard(path)
                self.assertEqual(len(trajectories), 1)
                self.assertGreater(len(trajectories[0].decisions), 0)
            # Different workers play different games (distinct seeds).
            t0 = read_shard(shard_paths[0])[0]
            t1 = read_shard(shard_paths[1])[0]
            self.assertNotEqual(t0.choice_record, t1.choice_record)


if __name__ == "__main__":
    unittest.main()

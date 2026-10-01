"""Milestone H (Code/AGENT_INTERFACE_PLAN.md): parallelism and inference.

The InferenceClient abstraction (in-process, and a real stdlib-socket
remote server with real cross-request dynamic batching) -- testable
without a GPU or torch. The real network behind it (`ml/infer_server.py`)
is tested in tests/test_agent_training_ml.py.
"""

import threading
import time
import unittest

from bots.inference_client import InferenceServer, InProcessInferenceClient, RemoteInferenceClient


def _wait_for_bind(server, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while server.address == ("localhost", 0) and time.monotonic() < deadline:
        time.sleep(0.01)


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


class TestDynamicBatching(unittest.TestCase):
    """Real cross-request batching: concurrent requests from independent
    connections must land in ONE call to the model, not one per connection."""

    def test_concurrent_requests_are_coalesced_into_one_model_call(self):
        calls = []
        call_lock = threading.Lock()

        def model_predict_many(observations):
            with call_lock:
                calls.append(len(observations))
            return [([o], float(o)) for o in observations]

        class _BatchModel:
            predict_many = staticmethod(model_predict_many)

        server = InferenceServer(
            ("localhost", 0), authkey=b"k", model=_BatchModel(),
            max_batch_size=100, batch_window_seconds=0.2,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_bind(server)
            results = [None] * 5

            def fire(i):
                client = RemoteInferenceClient(server.address, authkey=b"k")
                results[i] = client.predict_many([i, i + 100])

            workers = [threading.Thread(target=fire, args=(i,)) for i in range(5)]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=5)

            for i in range(5):
                self.assertEqual(results[i], [([i], float(i)), ([i + 100], float(i + 100))])
            # 5 connections x 2 observations each, fired concurrently and
            # well within the 0.2s batch window -- must be ONE model call,
            # not 5.
            self.assertEqual(calls, [10])
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_a_model_without_predict_many_still_works_without_batching(self):
        """Backward compatible with a plain observation -> (policy, value)
        callable (every model in this test file except this class) -- no
        batching win, but correct, via a per-item loop."""
        server = InferenceServer(("localhost", 0), authkey=b"k", model=lambda o: ([o], float(o)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_bind(server)
            client = RemoteInferenceClient(server.address, authkey=b"k")
            self.assertEqual(client.predict_many([1, 2, 3]), [([1], 1.0), ([2], 2.0), ([3], 3.0)])
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_a_batch_wide_model_error_is_reported_to_every_waiting_connection(self):
        def boom(observations):
            raise ValueError("model exploded")

        class _BoomModel:
            predict_many = staticmethod(boom)

        server = InferenceServer(("localhost", 0), authkey=b"k", model=_BoomModel(), batch_window_seconds=0.05)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_bind(server)
            client = RemoteInferenceClient(server.address, authkey=b"k")
            with self.assertRaises(RuntimeError) as ctx:
                client.predict_many([1])
            self.assertIn("model exploded", str(ctx.exception))
            # The server must survive its own model raising and keep serving.
            server2_client = RemoteInferenceClient(server.address, authkey=b"k")
            with self.assertRaises(RuntimeError):
                server2_client.predict_many([2])
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_respects_max_batch_size(self):
        batch_sizes = []
        lock = threading.Lock()

        def model_predict_many(observations):
            with lock:
                batch_sizes.append(len(observations))
            return [([o], float(o)) for o in observations]

        class _BatchModel:
            predict_many = staticmethod(model_predict_many)

        server = InferenceServer(
            ("localhost", 0), authkey=b"k", model=_BatchModel(),
            max_batch_size=3, batch_window_seconds=0.3,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_bind(server)
            results = [None] * 6

            def fire(i):
                client = RemoteInferenceClient(server.address, authkey=b"k")
                results[i] = client.predict_many([i])

            workers = [threading.Thread(target=fire, args=(i,)) for i in range(6)]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=5)

            for i in range(6):
                self.assertEqual(results[i], [([i], float(i))])
            self.assertTrue(all(size <= 3 for size in batch_sizes))
            self.assertEqual(sum(batch_sizes), 6)
        finally:
            server.stop()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

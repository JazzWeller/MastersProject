"""Milestone H (Code/AGENT_INTERFACE_PLAN.md): parallelism and inference.

The InferenceClient abstraction (in-process, and a real stdlib-socket
remote server with real cross-request dynamic batching) -- testable
without a GPU or torch. The real network behind it (`ml/infer_server.py`)
is tested in tests/test_agent_training_ml.py.
"""

import os
import pickle
import socket
import threading
import time
import unittest
from unittest import mock

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


class TestPersistentConnection(unittest.TestCase):
    """A RemoteInferenceClient connects once and reuses the connection
    (Agent Observation Plan, O8) -- but never shares it with another
    process, and reopens it when it breaks."""

    def setUp(self):
        self.server = InferenceServer(("localhost", 0), authkey=b"k", model=lambda o: ([o], float(o)), batch_window_seconds=0.001)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        _wait_for_bind(self.server)

    def tearDown(self):
        self.server.stop()
        self.thread.join(timeout=5)

    def test_one_connection_serves_every_call(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k")
        for i in range(20):
            self.assertEqual(client.predict(i), ([i], float(i)))
        self.assertEqual(self.server.connections_accepted, 1)
        client.close()

    def test_a_broken_connection_is_reopened_and_the_request_sent_again(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k")
        self.assertEqual(client.predict(1), ([1], 1.0))
        client._conn.close()  # the connection breaks under the client
        self.assertEqual(client.predict(2), ([2], 2.0))
        self.assertEqual(self.server.connections_accepted, 2)
        client.close()

    def test_a_copy_in_another_process_opens_its_own_connection(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k")
        client.predict(1)
        clone = pickle.loads(pickle.dumps(client))  # how a client reaches a spawned process
        self.assertEqual(clone.predict(2), ([2], 2.0))
        self.assertEqual(self.server.connections_accepted, 2)
        with mock.patch("bots.inference_client.os.getpid", return_value=os.getpid() + 1):  # as after a fork
            self.assertEqual(client.predict(3), ([3], 3.0))
        self.assertEqual(self.server.connections_accepted, 3)
        clone.close()
        client.close()

    def test_the_first_connection_failure_is_raised_at_once(self):
        """A caller waiting for the server to come up (the self-play actors
        do) needs to see the refusal, not a silent retry loop."""
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
        client = RemoteInferenceClient(("localhost", port), authkey=b"k", reconnect_seconds=60)
        t0 = time.monotonic()
        with self.assertRaises(OSError):
            client.predict(1)
        self.assertLess(time.monotonic() - t0, 30)

    def test_a_server_that_stays_gone_raises_a_connection_error(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k", reconnect_seconds=1.0)
        self.assertEqual(client.predict(1), ([1], 1.0))
        self.server.stop()
        self.thread.join(timeout=5)
        with self.assertRaises(ConnectionError):
            client.predict(2)

    def test_the_connection_turns_nagle_off(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k")
        client.predict(1)
        s = socket.socket(fileno=client._conn.fileno())
        try:
            self.assertTrue(s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
        finally:
            s.detach()
        client.close()

    def test_large_messages_on_one_connection_do_not_stall(self):
        """A message over 16 KB goes out as two writes (header, payload);
        with Nagle on, a long-lived connection waited ~40 ms (Linux) or
        ~200 ms (Windows) for a delayed ACK on every such message."""
        server = InferenceServer(("localhost", 0), authkey=b"k", model=lambda o: ([], 0.0), batch_window_seconds=0.0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_bind(server)
            client = RemoteInferenceClient(server.address, authkey=b"k")
            payload = [bytes(70_000)]
            client.predict_many(payload)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                client.predict_many(payload)
                times.append(time.perf_counter() - t0)
            client.close()
        finally:
            server.stop()
            thread.join(timeout=5)
        self.assertLess(sorted(times)[len(times) // 2], 0.030)

    def test_one_client_shared_by_threads_gives_each_its_own_answers(self):
        client = RemoteInferenceClient(self.server.address, authkey=b"k")
        errors = []

        def fire(t):
            try:
                for i in range(25):
                    x = t * 1000 + i
                    if client.predict(x) != ([x], float(x)):
                        errors.append(x)
            except Exception as e:  # noqa: BLE001 -- reported below
                errors.append(e)

        threads = [threading.Thread(target=fire, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(self.server.connections_accepted, 1)
        client.close()


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

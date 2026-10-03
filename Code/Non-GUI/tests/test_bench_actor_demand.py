"""tools/bench_actor_demand.py (Agent Observation Plan, O10): one
actor-shaped worker, against a stub inference server (no torch, no GPU),
searches its games and reports counts over its measured window only."""

import argparse
import threading
import time
import unittest

from bots.inference_client import InferenceServer
from tools import bench_actor_demand


def _uniform(request):
    """Uniform priors and a zero value, for any request the search makes."""
    if request.candidates is not None:
        return [1.0] * len(request.candidates), 0.0
    return [1.0] * request.enc.n_options, 0.0


class TestActorDemandWorker(unittest.TestCase):
    def test_a_worker_searches_and_counts_its_measured_window(self):
        server = InferenceServer(("localhost", 0), authkey=b"bench", model=_uniform)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                if server.address != ("localhost", 0):
                    break
                time.sleep(0.01)
            host, port = server.address
            args = argparse.Namespace(
                config="tier2_selfplay_within_turn.json", games_per_worker=2, warmup=0.5, measure=2.0, seed=0,
                worker=0, host=host, port=port, authkey="bench",
            )
            out = bench_actor_demand.run_worker(args)
        finally:
            server.stop()
            thread.join(timeout=5)
        self.assertEqual(out["concurrency"], 2)
        self.assertGreater(out["searches"], 0)
        self.assertGreater(out["simulations"], 0)
        self.assertGreater(out["requests"], 0)
        self.assertGreaterEqual(out["requests"], out["calls"])
        self.assertGreater(out["seconds"], 1.0)
        self.assertLessEqual(out["wait_seconds"], out["seconds"])


if __name__ == "__main__":
    unittest.main()

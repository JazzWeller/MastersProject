"""The self-play learner's reuse limits and offline replay (after Tier 2
round 2): `selfplay.policy_reuse` / `value_reuse` cap how often one position
trains each group of targets, and `--replay-from` trains a new learner on an
earlier run's games in the order they arrived, so learner settings can be
compared on the same games.
"""

import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - torch-free environments (PyPy has no numpy either)
    torch = None

V2_NET = {"d_model": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0, "history_arch": "none",
          "enumerate_cap": 1024, "options_in_trunk": False}


def _reseeded(records, copies):
    """`copies` copies of each record under fresh game seeds (the learner
    reads the seed for the held-out split and the game id only)."""
    out = []
    for c in range(copies):
        for i, r in enumerate(records):
            out.append(dict(r, meta=dict(r["meta"], seed=1000 * (c + 1) + i, index=c * len(records) + i)))
    return out


@unittest.skipIf(torch is None, "needs torch")
class TestUseCounts(unittest.TestCase):
    def test_limits_of_one_take_every_position_exactly_once(self):
        from ml.selfplay_train import UseCounts

        u = UseCounts(1, 1)
        has = [np.array([1, 0, 1, 1]), np.array([0, 1, 1]), np.array([1, 1])]
        for h in has:
            u.add(h)
        seen = []
        rng = random.Random(0)
        while u.trainable():
            seen.extend((g, j) for g, j, _p, _v in u.pick(3, rng))
        self.assertEqual(sorted(seen), sorted((g, j) for g, h in enumerate(has) for j in range(len(h))))

    def test_a_value_limit_spends_the_outcome_only(self):
        from ml.selfplay_train import UseCounts

        u = UseCounts(None, 1)
        u.add(np.array([1, 1, 0]))
        first = {(g, j): (p, v) for g, j, p, v in u.pick(3, random.Random(1))}
        self.assertEqual(first, {(0, 0): (True, True), (0, 1): (True, True), (0, 2): (False, True)})
        # Policy positions stay; the value-only one is spent.
        self.assertEqual(u.trainable(), 2)
        second = {(g, j): (p, v) for g, j, p, v in u.pick(3, random.Random(2))}
        self.assertEqual(second, {(0, 0): (True, False), (0, 1): (True, False)})

    def test_dropping_the_oldest_game_keeps_the_rest_aligned(self):
        from ml.selfplay_train import UseCounts

        u = UseCounts(1, 1)
        u.add(np.array([1, 1]))
        u.add(np.array([1, 1, 1]))
        u.pick(1, random.Random(0))
        u.drop_oldest()
        picked = u.pick(10, random.Random(0))
        self.assertTrue(all(g == 0 and 0 <= j < 3 for g, j, _p, _v in picked))
        self.assertEqual(u.trainable(), 0)


@unittest.skipIf(torch is None, "needs torch")
class TestReuseLimitsInTheLosses(unittest.TestCase):
    """v1 and v2: a spent outcome group leaves the value and oracle losses,
    training in pieces still gives the whole batch's gradient, and no
    position's value trains twice under value_reuse 1."""

    @classmethod
    def setUpClass(cls):
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.infer_v2 import TorchModelV2
        from ml.model import KeyForgeNet
        from ml.model_v2 import KeyForgeNetV2
        from tests.test_agent_training_ml import _net_cfg

        torch.manual_seed(0)
        cls.net = KeyForgeNet(_net_cfg())
        s = SelfPlaySettings(leaf="student", sims_full=6, sims_small=3, full_fraction=0.5, concurrency=2, max_turns=10,
                             leaves_in_flight=2)
        cls.v1 = []
        Actor(InProcessInferenceClient(TorchModel(cls.net, "cpu")), s, run_seed=1).play(range(2), cls.v1.append)
        cls.net2 = KeyForgeNetV2(dict(V2_NET))
        s2 = SelfPlaySettings(leaf="student", sims_full=4, sims_small=2, full_fraction=0.5, concurrency=2, max_turns=6,
                              leaves_in_flight=2, observation_version=2, history="none")
        cls.v2 = []
        Actor(InProcessInferenceClient(TorchModelV2(cls.net2, "cpu", compile=False)), s2, run_seed=3, worker=0).play(
            range(2), cls.v2.append)

    def test_v1_value_reuse_one_trains_each_outcome_once(self):
        from ml.selfplay_train import Buffer, UseCounts, loss_counts, losses_for

        buf = Buffer(10_000, 100, UseCounts(None, 1))
        for r in self.v1:
            buf.add(r)
        total = buf.size
        seen_value = 0
        for i in range(6):
            pos = buf.sample(32, random.Random(i))
            on = sum(1 for p in pos if p.get("value_on", True))
            seen_value += on
            n = loss_counts(pos, {}, "search")
            self.assertEqual(n["value"], on)
            _tot, L, _st = losses_for(self.net, pos, "cpu", {}, "search", False, random.Random(0))
            self.assertEqual("value" in L, on > 0)
        self.assertLessEqual(seen_value, total)

    def test_v1_pieces_match_the_whole_batch_with_rows_off(self):
        from ml.selfplay_train import Buffer, UseCounts, backward_step

        buf = Buffer(10_000, 100, UseCounts(None, 1))
        for r in self.v1:
            buf.add(r)
        buf.sample(40, random.Random(5))  # spend some outcomes, so the next batch mixes rows on and off
        pos = buf.sample(40, random.Random(6))
        self.assertTrue(any(p.get("value_on", True) for p in pos) and not all(p.get("value_on", True) for p in pos))
        grads = []
        for micro in (None, 7):
            self.net.zero_grad(set_to_none=True)
            backward_step(self.net, pos, "cpu", {"policy": 1.0, "value": 1.0, "belief": 0.25, "oracle": 0.25}, "search",
                          False, random.Random(0), micro, None)
            grads.append([p.grad.clone() for p in self.net.parameters() if p.grad is not None])
        for a, b in zip(*grads):
            self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    def _v2_buffer(self, limits):
        from ml.selfplay_v2 import BufferV2, EncodingPool

        pool = EncodingPool(0)
        pool.submit(self.v2)
        buf = BufferV2(10_000, 100, "none", limits)
        for arrays, _info in pool.wait_all():
            buf.add(arrays)
        return buf

    def test_v2_value_reuse_one_masks_spent_rows(self):
        from contextlib import nullcontext

        from ml.selfplay_train import UseCounts
        from ml.selfplay_v2 import DEFAULT_WEIGHTS, active_heads, backward_step_v2, loss_counts, losses_v2

        buf = self._v2_buffer(UseCounts(None, 1))
        buf.sample(30, random.Random(0))
        sample = buf.sample(30, random.Random(1))
        _b, tg, pis, cands = sample
        self.assertIsNotNone(tg.value_on)
        on = int(tg.value_on.sum())
        self.assertTrue(0 < on < len(pis))
        heads = active_heads(DEFAULT_WEIGHTS)
        self.assertEqual(loss_counts(*sample, heads)["value"], on)
        _tot, L, _st = losses_v2(self.net2, *sample, DEFAULT_WEIGHTS, heads)
        rows = tg.value_on.nonzero().squeeze(-1)
        with torch.no_grad():
            v = self.net2.value(self.net2.encode_state(sample[0])).float()
        self.assertAlmostEqual(float(L["value"].detach()),float(torch.nn.functional.mse_loss(v[rows], tg.z[rows])), places=5)
        grads = []
        for max_tokens in (None, sample[0].tokens() * 6):
            self.net2.zero_grad(set_to_none=True)
            backward_step_v2(self.net2, sample, weights=DEFAULT_WEIGHTS, max_tokens=max_tokens, autocast=nullcontext)
            grads.append([p.grad.clone() for p in self.net2.parameters() if p.grad is not None])
        for a, b in zip(*grads):
            self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    def test_v2_without_limits_samples_as_before(self):
        a = self._v2_buffer(None).sample(20, random.Random(4))
        self.assertIsNone(a[1].value_on)
        self.assertTrue(any(len(p) for p in a[2]))


@unittest.skipIf(torch is None, "needs torch")
class TestReplay(unittest.TestCase):
    """`--replay-from`: shards merged in arrival order, then a whole learner
    run on them (v1 and v2) that ends with the held-out report."""

    @classmethod
    def setUpClass(cls):
        TestReuseLimitsInTheLosses.setUpClass.__func__(cls)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_count_frames_skips_a_half_written_frame_and_the_merge_interleaves(self):
        from agent.selfplay import write_frame
        from ml.selfplay_train import ReplayReader, count_frames

        d = os.path.join(self.tmp, "merge")
        os.makedirs(d)
        for name, n in (("a.bin", 4), ("b.bin", 2)):
            for k in range(n):
                write_frame(os.path.join(d, name), {"meta": {"shard": name, "k": k}})
        with open(os.path.join(d, "b.bin"), "ab") as f:
            f.write(b"\x10\x00\x00\x00abc")  # a frame cut off mid-write
        self.assertEqual(count_frames(os.path.join(d, "b.bin")), 2)
        r = ReplayReader(d)
        self.assertEqual(r.total, 6)
        order = [(x["meta"]["shard"], x["meta"]["k"]) for x in r.poll(4) + r.poll(10)]
        self.assertTrue(r.done)
        self.assertEqual(order, [("a.bin", 0), ("b.bin", 0), ("a.bin", 1), ("a.bin", 2), ("b.bin", 1), ("a.bin", 3)])

    def _replay(self, version):
        from agent.config import held_out
        from agent.selfplay import write_frame
        from ml import selfplay_train

        root = os.path.join(self.tmp, f"data{version}")
        records = _reseeded(self.v1 if version == 1 else self.v2, 6)
        val = [held_out(r["meta"]["seed"], 0.25) for r in records]
        self.assertTrue(any(val) and not all(val))
        shards = os.path.join(root, "runs", "source", "shards")
        os.makedirs(shards)
        for i, r in enumerate(records):
            write_frame(os.path.join(shards, f"actor_{i % 3:03d}.bin"), r)
        sp = {"batch": 16, "positions_per_step": None, "sample_reuse": 1, "policy_reuse": 1, "value_reuse": 1,
              "validation_fraction": 0.25, "compile": False, "precision": "fp32", "encode_workers": 0}
        override = {"selfplay": sp}
        cfg = "smoke_selfplay.json"
        if version == 2:
            cfg = "smoke_selfplay_v2.json"
            override["network"] = dict(V2_NET, attention_kernel="efficient")
            sp["max_tokens"] = None
        argv = ["selfplay_train", "--run", f"replay{version}", "--config", cfg, "--replay-from", "source", "--device", "cpu",
                "--override", json.dumps(override)]
        with mock.patch.dict(os.environ, {"KEYFORGE_DATA": root}), mock.patch.object(sys, "argv", argv):
            selfplay_train.main()
            from tools.export_results import export_run

            done = export_run(f"replay{version}", os.path.join(root, "export"))  # the monitor, metrics, final network
            self.assertIn("monitor", done)
            self.assertIn("final checkpoint", done)
        with open(os.path.join(root, "runs", f"replay{version}", "journal.jsonl")) as f:
            journal = [json.loads(line) for line in f]
        events = {j["event"]: j for j in journal}
        train_positions = sum(len(r["positions"]) for r, v in zip(records, val) if not v)
        return events, train_positions

    def test_a_v1_replay_trains_every_position_at_most_once_and_reports_held_out_loss(self):
        events, train_positions = self._replay(1)
        self.assertEqual(events["replay"]["games"], 12)
        done = events["run_finished"]
        self.assertGreater(done["step"], 0)
        self.assertLessEqual(done["step"] * 16, train_positions)
        self.assertIn("val_loss_value", events["final_validation"])
        self.assertNotIn("gate", events)

    def test_a_v2_replay_runs_end_to_end(self):
        events, train_positions = self._replay(2)
        self.assertGreater(events["run_finished"]["step"], 0)
        self.assertLessEqual(events["run_finished"]["step"] * 16, train_positions)
        self.assertIn("val_loss_value", events["final_validation"])


if __name__ == "__main__":
    unittest.main()

"""Training bigger networks on the 8 GiB GPU: the SDPA encoder layer, the
activation-checkpointing switch and `micro_batch` gradient accumulation
(ml/layers.py, ml/accumulate.py). Each one changes cost, never the
function, and these tests hold them to that: the same outputs, losses and
gradients as the plain path.

Needs `torch`; runs from the WSL torch environment:
`~/torchenv/bin/python -m unittest tests.test_agent_training_memory`
(from Code/Non-GUI).
"""

import json
import os
import random
import tempfile
import unittest

try:
    import numpy as np
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH = False

from agent import config as config_mod
from keyforge.config import GameConfig

TOL = dict(rtol=1e-4, atol=1e-5)


def _net_cfg(**over):
    cfg = dict(config_mod.DEFAULTS["network"])
    cfg.update({"dropout": 0.0, **over})
    return cfg


def _grads(net):
    return {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestLayers(unittest.TestCase):
    def test_the_sdpa_layer_is_nn_transformer_encoder_layer(self):
        from torch import nn

        from ml.layers import FastEncoderLayer

        torch.manual_seed(0)
        ref = nn.TransformerEncoderLayer(64, 4, 128, 0.0, activation="gelu", batch_first=True, norm_first=True)
        fast = FastEncoderLayer(64, 4, 128, 0.0)
        self.assertEqual(set(ref.state_dict()), set(fast.state_dict()))
        fast.load_state_dict(ref.state_dict())
        x = torch.randn(5, 73, 64)
        for train in (False, True):
            ref.train(train)
            fast.train(train)
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            a, b = ref(xa), fast(xb)
            torch.testing.assert_close(a, b, **TOL)
            a.square().sum().backward()
            b.square().sum().backward()
            torch.testing.assert_close(xa.grad, xb.grad, **TOL)

    def test_a_checkpoint_moves_between_the_two_attention_implementations(self):
        from ml.encode import collate
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        old = KeyForgeNet(_net_cfg(attention="torch")).eval()
        new = KeyForgeNet(_net_cfg(attention="sdpa")).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        batch = collate(_encodings(), torch.device("cpu"))
        with torch.no_grad():
            a, b = old.encode_state(batch), new.encode_state(batch)
            torch.testing.assert_close(old.value(a), new.value(b), **TOL)
            torch.testing.assert_close(old.policy_logits(a), new.policy_logits(b), **TOL)

    def test_the_inference_server_graphs_the_sdpa_trunk(self):
        """CUDA-graph replay of an sdpa network answers like the same
        weights in the torch layer, run eagerly."""
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from agent.agents.requests import HEAD_VALUE, Request
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        old = KeyForgeNet(_net_cfg(attention="torch"))
        new = KeyForgeNet(_net_cfg(attention="sdpa"))
        new.load_state_dict(old.state_dict())
        eager, graphed = TorchModel(old, "cuda", cuda_graphs=False), TorchModel(new, "cuda")
        reqs = [Request(e) for e in _encodings()] + [Request(e, HEAD_VALUE) for e in _encodings()]
        a, b = eager.predict_many(reqs), graphed.predict_many(reqs)
        self.assertTrue(graphed._graphs, "the graph path ran")
        for (sa, va), (sb, vb) in zip(a, b):
            self.assertAlmostEqual(va, vb, delta=5e-3)
            for x, y in zip(sa, sb):
                self.assertAlmostEqual(x, y, delta=5e-3)

    def test_activation_checkpointing_keeps_the_loss_and_the_gradients(self):
        from ml.encode import collate
        from ml.layers import set_activation_checkpointing
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg(attention="sdpa", dropout=0.1)).train()
        batch = collate(_encodings(), torch.device("cpu"))
        results = []
        for on in (False, True):
            set_activation_checkpointing(net.trunk, on)
            net.zero_grad()
            torch.manual_seed(1)  # the same dropout masks both times
            loss = net.value(net.encode_state(batch)).square().mean()
            loss.backward()
            results.append((loss.detach(), _grads(net)))
        torch.testing.assert_close(results[0][0], results[1][0], **TOL)
        torch.testing.assert_close(results[0][1], results[1][1], **TOL)

    def test_activation_checkpointing_needs_the_sdpa_trunk(self):
        from ml.layers import set_activation_checkpointing
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg(attention="torch"))
        set_activation_checkpointing(net.trunk, False)
        with self.assertRaisesRegex(ValueError, "sdpa"):
            set_activation_checkpointing(net.trunk, True)

    def test_slices(self):
        from ml.accumulate import slices

        self.assertEqual(slices(10, None), [(0, 10)])
        self.assertEqual(slices(10, 10), [(0, 10)])
        self.assertEqual(slices(10, 4), [(0, 4), (4, 8), (8, 10)])
        with self.assertRaises(ValueError):
            slices(10, 0)

    def test_sdpa_trunk_compiles_with_checkpointing_under_bf16(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from ml.encode import collate
        from ml.layers import set_activation_checkpointing
        from ml.model import KeyForgeNet, compile_trunk

        device = torch.device("cuda")
        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg(attention="sdpa")).to(device).train()
        eager = {k: v.clone() for k, v in net.state_dict().items()}
        set_activation_checkpointing(net.trunk, True)
        compile_trunk(net, True, device)
        self.assertEqual(set(net.state_dict()), set(eager))
        batch = collate(_encodings() * 8, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net.value(net.encode_state(batch)).float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(g).all() for g in _grads(net).values()))


_ENC = None


def _encodings():
    """A few real encodings from one game, decisions of several kinds."""
    global _ENC
    if _ENC is None:
        from agent.features import encode
        from bots.random_bot import RandomBot
        from keyforge.game import Game
        from keyforge.infoset import build_infoset

        game = Game(GameConfig(decks=("fignor", "igor"), seed=5, max_turns=20))
        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        out = []
        while not game.is_over and len(out) < 24:
            dec = game.pending_decision
            out.append(encode(build_infoset(game, dec.player)))
            game.submit(bots[dec.player].decide(game.view_for(dec.player), dec))
        _ENC = out
    return _ENC


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestMicroBatchBC(unittest.TestCase):
    """`bc.micro_batch`: the same losses and gradients as the whole batch,
    on a small real corpus with every BC head on."""

    FIELDS = ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "opp_hand", "next_draws")

    @classmethod
    def setUpClass(cls):
        from ml.dataset import Corpus, encode_records
        from sim.bc_corpus import play_labelled

        cls.tmp = tempfile.TemporaryDirectory()
        records = os.path.join(cls.tmp.name, "records.jsonl")
        with open(records, "w", encoding="utf-8") as f:
            for i, source in enumerate(("heuristic", "random", "heuristic", "random")):
                decks = ("fignor", "igor") if i % 2 == 0 else ("igor", "fignor")
                rec = play_labelled(GameConfig(decks=decks, seed=300 + i, max_turns=40), source, 300 + i)
                rec["index"] = i
                f.write(json.dumps(rec) + "\n")
        shard = os.path.join(cls.tmp.name, "encoded", "s0")
        encode_records(records, shard)
        cls.corpus = Corpus([shard])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pieces_give_the_whole_batch_losses_and_gradients(self):
        from ml.bc_train import DEFAULT_WEIGHTS, backward_step
        from ml.model import KeyForgeNet

        batch, tg = self.corpus.batch(np.arange(min(self.corpus.size, 500)))
        heads = tuple(DEFAULT_WEIGHTS)
        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg()).train()
        results = {}
        for micro in (None, 128, 37):
            net.zero_grad()
            losses = backward_step(net, batch, tg, micro=micro, amp=None, device="cpu", weights=dict(DEFAULT_WEIGHTS),
                                   policy_head="pointer", cap=1024, heads=heads)
            results[micro] = ({k: v.detach().float() for k, v in losses.items()}, _grads(net))
        self.assertTrue({"policy", "value", "enumerate", "sequential", "topk", "belief", "oracle"} <= set(results[None][0]))
        for micro in (128, 37):
            torch.testing.assert_close(results[None][0], results[micro][0], **TOL)
            torch.testing.assert_close(results[None][1], results[micro][1], **TOL)


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestMicroBatchLearner(unittest.TestCase):
    """`selfplay.micro_batch`: the same losses, stats and gradients as the
    whole batch, on real actor output, in both learner modes."""

    @classmethod
    def setUpClass(cls):
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet
        from ml.selfplay_train import Buffer

        torch.manual_seed(0)
        cls.net = KeyForgeNet(_net_cfg())
        client = InProcessInferenceClient(TorchModel(cls.net, "cpu"))
        cls.positions = {}
        for mode in ("search", "dmc"):
            s = SelfPlaySettings(mode=mode, leaf="student", sims_full=6, sims_small=3, full_fraction=0.5, concurrency=2,
                                 max_turns=14, leaves_in_flight=2)
            buf = Buffer(10_000, 100)
            Actor(client, s, run_seed=3).play(range(3), buf.add)
            cls.positions[mode] = buf.sample(160, random.Random(0))

    def _check(self, mode, weights):
        from ml.selfplay_train import backward_step

        self.net.train()
        results = {}
        for micro in (None, 50):
            self.net.zero_grad()
            L, st = backward_step(self.net, self.positions[mode], torch.device("cpu"), weights, mode, True, random.Random(1), micro, None)
            results[micro] = ({k: v.detach().float() for k, v in L.items()}, st, _grads(self.net))
        torch.testing.assert_close(results[None][0], results[50][0], **TOL)
        torch.testing.assert_close(results[None][2], results[50][2], **TOL)
        self.assertEqual(set(results[None][1]), set(results[50][1]))
        for k, v in results[None][1].items():
            self.assertAlmostEqual(v, results[50][1][k], places=4)
        return results[None][0]

    def test_search_mode(self):
        L = self._check("search", {"policy": 1, "value": 1, "belief": 0.25, "oracle": 0.25, "distill": 0.5})
        self.assertTrue({"value", "oracle", "distill"} <= set(L))
        self.assertTrue({"policy", "multi"} & set(L))

    def test_dmc_mode(self):
        L = self._check("dmc", {"q": 1.0})
        self.assertIn("q", L)


if __name__ == "__main__":
    unittest.main()

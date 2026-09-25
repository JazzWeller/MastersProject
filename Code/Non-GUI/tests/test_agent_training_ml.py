"""Agent Training Plan, Milestone M2 (and the torch side of M3/M11): the
network, batching, checkpoints, the inference model and the search-free
network agent.

Needs `torch` (+ numpy), deliberately not installed where the rest of the
suite runs: these tests skip themselves there and run from the WSL torch
environment: `~/torchenv/bin/python -m unittest tests.test_agent_training_ml`
(from Code/Non-GUI).
"""

import os
import random
import tempfile
import time
import unittest

try:
    import numpy as np
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH = False

from agent import config as config_mod
from agent import spec
from agent.features import encode
from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.infoset import build_infoset, infoset_for
from keyforge.match import Match, MatchConfig


def _net_cfg(**over):
    cfg = dict(config_mod.DEFAULTS["network"])
    cfg.update(dropout=0.0, **over)
    return cfg


def _samples_by_kind(max_games=300):
    """{DecisionKind: [(Encoded, decision)]} across random decks and
    Adaptive matches, until every kind has a few samples."""
    from bots.heuristic_bot import HeuristicBot

    out = {}
    rng = random.Random(4)
    for g in range(max_games):
        if all(len(out.get(k, [])) >= 3 for k in DecisionKind):
            break
        d1, d2 = random_deck(rng, "R1"), random_deck(rng, "R2")
        if g % 3 == 0:
            obj = Match(MatchConfig(format="adaptive", decks=(d1, d2), seed=g, max_turns=80))
            bots = {1: HeuristicBot(seed=g), 2: HeuristicBot(seed=g + 1)}
        else:
            obj = Game(GameConfig(decks=(d1, d2), seed=g, max_turns=40))
            bots = {1: RandomBot(seed=g), 2: RandomBot(seed=g + 1)}
        while not obj.is_over:
            dec = obj.pending_decision
            bucket = out.setdefault(dec.kind, [])
            if len(bucket) < 3:
                bucket.append((encode(infoset_for(obj, dec.player)), dec))
            obj.submit(bots[dec.player].decide(obj.view_for(dec.player), dec))
    return out


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = _samples_by_kind()

    def test_every_kind_forwards_and_the_policy_sums_to_one_over_offered_options(self):
        from ml.encode import collate
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg()).eval()
        for kind in DecisionKind:
            with self.subTest(kind=kind.name):
                items = self.samples[kind]
                batch = collate([e for e, _d in items])
                out = net.encode_state(batch)
                probs = torch.softmax(net.policy_logits(out), dim=-1)
                for b, (e, d) in enumerate(items):
                    self.assertEqual(e.n_options, len(d.options))
                    self.assertAlmostEqual(float(probs[b, : e.n_options].sum()), 1.0, places=5)
                    self.assertEqual(float(probs[b, e.n_options :].sum()), 0.0)
                v = net.value(out)
                self.assertTrue(bool(((v >= -1) & (v <= 1)).all()))

    def test_collate_matches_the_dense_form(self):
        from ml.encode import collate, entity_block

        items = [e for k in (DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_CARDS) for e, _d in self.samples[k]]
        batch = collate(items)
        ent = entity_block(batch)
        for b, e in enumerate(items):
            dense = np.frombuffer(e.entity_rows().tobytes(), dtype=np.float32).reshape(72, spec.ENTITY.width)
            np.testing.assert_allclose(ent[b].numpy(), dense, atol=0)
            np.testing.assert_allclose(batch.globals[b].numpy(), np.frombuffer(e.globals.tobytes(), dtype=np.float32))

    def test_identity_ablations_and_intent_ablation_build_and_run(self):
        from ml.encode import collate
        from ml.model import KeyForgeNet

        batch = collate([e for e, _d in self.samples[DecisionKind.CHOOSE_ACTION]])
        for over in ({"identity": "id"}, {"identity": "attr"}, {"layers": 2}, {"ablate_globals": ["intent"]}, {"policy_head": "fixed"}):
            with self.subTest(**{k: str(v) for k, v in over.items()}):
                net = KeyForgeNet(_net_cfg(**over)).eval()
                out = net.encode_state(batch)
                logits = net.fixed_logits(out, batch) if over.get("policy_head") == "fixed" else net.policy_logits(out)
                self.assertEqual(logits.shape, batch.option_mask.shape)

    def test_multi_select_heads(self):
        from agent.multiselect import enumerate_candidates
        from ml.encode import collate
        from ml.model import KeyForgeNet, candidates_tensor

        net = KeyForgeNet(_net_cfg()).eval()
        for kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            e, d = self.samples[kind][0]
            batch = collate([e])
            out = net.encode_state(batch)
            ordered = kind == DecisionKind.ORDER_EFFECTS
            cands = enumerate_candidates(ordered, len(d.options), d.min_n, d.max_n, 1024)
            scores = net.subset_scores(out, torch.zeros(len(cands), dtype=torch.int64), candidates_tensor(cands), torch.full((len(cands),), ordered))
            self.assertEqual(scores.shape, (len(cands),))
            K = batch.options.shape[1]
            legal = torch.ones(1, K + 1, dtype=torch.bool)
            seq = net.sequential_logits(out, torch.zeros(1, dtype=torch.int64), torch.zeros(1, K), legal)
            self.assertEqual(seq.shape, (1, K + 1))


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestCheckpoints(unittest.TestCase):
    def test_round_trip_digest_and_stamps(self):
        from ml.checkpoints import CheckpointStore, load_model, save_model
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg())
        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(d)
            h, digest = save_model(net, store, config_hash="abc")
            loaded, meta, _ = load_model((store, h[:16]))
            self.assertEqual(meta["weights_digest"], digest)
            self.assertEqual(meta["stamp"]["feature_version"], spec.FEATURE_VERSION)
            for k, v in net.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.state_dict()[k]), k)
            # Same weights -> same weights digest, regardless of metadata.
            _h2, digest2 = save_model(loaded, store, config_hash="other")
            self.assertEqual(digest, digest2)

    def test_major_feature_mismatch_is_refused(self):
        from ml.checkpoints import deserialize, load_model, save_model, serialize
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.kfc")
            save_model(net, path)
            with open(path, "rb") as f:
                tensors, meta = deserialize(f.read())
            meta["stamp"]["feature_version"] = spec.FEATURE_VERSION + 1
            data, _ = serialize(tensors, {k: v for k, v in meta.items() if k != "weights_digest"})
            with self.assertRaises(spec.FeatureVersionMismatch):
                load_model(data)

    def test_optimizer_state_round_trips(self):
        from ml.checkpoints import load_model, save_model
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg())
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        net.value_head[0].weight.grad = torch.ones_like(net.value_head[0].weight)
        opt.step()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.kfc")
            save_model(net, path, optimizer=opt)
            net2, _meta, opt_state = load_model(path)
            opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
            opt2.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
            self.assertEqual(len(opt2.state_dict()["state"]), len(opt.state_dict()["state"]))

    def test_vocabulary_growth_keeps_old_rows_and_attribute_initializes_new_ones(self):
        from ml.checkpoints import grow_vocabulary
        from ml.encode import static_table_tensor
        from ml.model import KeyForgeNet

        old_v = 300
        torch.manual_seed(1)
        small = KeyForgeNet(_net_cfg(), vocab_size=old_v)
        # Make the old embeddings an exact linear function of the
        # attributes, so the attribute-path initialization is checkable.
        table = static_table_tensor()
        W = torch.randn(table.shape[1], small.card_embedding.weight.shape[1]) * 0.1
        with torch.no_grad():
            small.card_embedding.weight.copy_(table[:old_v] @ W)
        sd = grow_vocabulary(small.state_dict(), old_v, 370)
        big = KeyForgeNet(_net_cfg(), vocab_size=370)
        big.load_state_dict(sd)
        emb = big.card_embedding.weight.detach()
        self.assertTrue(torch.equal(emb[:old_v], small.card_embedding.weight.detach()))
        # The attribute path can only carry features some old card had: a
        # new card with an attribute no trained card ever showed gets no
        # signal from that column. Check the ones fully spanned.
        X = table[:old_v].double()
        coef = torch.linalg.lstsq(X.T, table[old_v:370].double().T).solution  # new rows as combinations of old
        residual = (X.T @ coef - table[old_v:370].double().T).norm(dim=0)
        spanned = [old_v + i for i in range(370 - old_v) if float(residual[i]) < 1e-6]
        self.assertGreater(len(spanned), 50)
        rows = torch.tensor(spanned)
        np.testing.assert_allclose(emb[rows].numpy(), (table[rows] @ W).numpy(), atol=2e-2)


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestInferenceAndNetAgent(unittest.TestCase):
    def _model(self, **over):
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        return TorchModel(KeyForgeNet(_net_cfg(**over)), device="cpu")

    def test_every_head_answers(self):
        from agent.agents.requests import (
            HEAD_BELIEF, HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET, HEAD_TOPK, HEAD_VALUE, Request,
        )

        model = self._model()
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3))
        e = encode(build_infoset(game, game.pending_decision.player))
        k = e.n_options
        reqs = [
            Request(e), Request(e, HEAD_VALUE), Request(e, HEAD_Q), Request(e, HEAD_TOPK), Request(e, HEAD_BELIEF),
            Request(e, HEAD_SUBSET, candidates=[(), (0,), (1,)]), Request(e, HEAD_SEQUENTIAL, prefix=[], legal=[0, -1]),
        ]
        res = model.predict_many(reqs)
        self.assertAlmostEqual(sum(res[0][0]), 1.0, places=4)
        self.assertEqual(res[1][0], [])
        self.assertEqual(len(res[2][0]), k)
        self.assertEqual(len(res[4][0]), 72)
        self.assertAlmostEqual(sum(res[5][0]), 1.0, places=4)
        self.assertEqual(len(res[6][0]), k + 1)
        values = {round(v, 5) for _s, v in res}
        self.assertEqual(len(values), 1, "one state, one value")

    def test_net_agent_plays_complete_games_for_every_treatment(self):
        from agent.agents.net_agent import NetAgent
        from bots.heuristic_bot import HeuristicBot
        from bots.inference_client import InProcessInferenceClient
        from sim.paired import run_paired

        model = self._model()
        client = InProcessInferenceClient(model)
        client.predict_many = model.predict_many  # batched, as the server would
        for treatment in ("enumerate", "sequential", "topk"):
            with self.subTest(treatment=treatment):
                agent = NetAgent(client, multi_select=treatment, seed=1)
                rep, results = run_paired(lambda i: agent, lambda i: HeuristicBot(seed=i), [0, 1], max_turns=40, concurrency=8)
                self.assertEqual(rep.games, 8)
                self.assertEqual(rep.forfeits, 0, [r.forfeit for r in results if r.forfeit])

    def test_dmc_mode_plays(self):
        from agent.agents.net_agent import NetAgent
        from bots.inference_client import InProcessInferenceClient
        from sim.driver import run_games

        model = self._model()
        client = InProcessInferenceClient(model)
        agent = NetAgent(client, mode="q", epsilon=0.1, seed=2)
        results = run_games(2, lambda i: GameConfig(decks=("fignor", "igor"), seed=i, max_turns=30), lambda i: {1: agent, 2: RandomBot(seed=i)})
        self.assertTrue(all(r.forfeit is None for r in results))

    def test_throughput_report(self):
        """M2's target is >= 20,000 single-position evaluations/sec at batch
        512 in fp16 on the 5060 Ti -- measured and printed, asserted only
        loosely (it's a shared machine)."""
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from ml.encode import collate
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg()).cuda().eval()
        game = Game(GameConfig(decks=("fignor", "igor"), seed=3))
        e = encode(build_infoset(game, game.pending_decision.player))
        batch = collate([e] * 512, torch.device("cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(5):
                net.value(net.encode_state(batch))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n = 40
            for _ in range(n):
                out = net.encode_state(batch)
                net.policy_logits(out)
                net.value(out)
            torch.cuda.synchronize()
        rate = n * 512 / (time.perf_counter() - t0)
        print(f"\n[M2 throughput] {rate:,.0f} evaluations/sec at batch 512, fp16")
        self.assertGreater(rate, 2000)


if __name__ == "__main__":
    unittest.main()

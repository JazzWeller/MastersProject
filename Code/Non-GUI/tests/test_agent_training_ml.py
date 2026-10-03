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

    def test_an_older_minor_checkpoint_loads_with_new_inputs_inert(self):
        from ml.checkpoints import deserialize, load_model, save_model, serialize
        from ml.model import KeyForgeNet

        net = KeyForgeNet(_net_cfg())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.kfc")
            save_model(net, path)
            with open(path, "rb") as f:
                tensors, meta = deserialize(f.read())
            meta["stamp"]["feature_minor"] = 0  # pretend it predates house_context
            data, _ = serialize(tensors, {k: v for k, v in meta.items() if k != "weights_digest"})
            old, _meta, _ = load_model(data)
            off, w = spec.OPTION.span("house_context")
            self.assertEqual(float(old.option_in.weight[:, off : off + w].abs().sum()), 0.0)
            self.assertGreater(float(net.option_in.weight[:, off : off + w].abs().sum()), 0.0)
            # Everything else is untouched.
            self.assertTrue(torch.equal(old.option_in.weight[:, :off], net.option_in.weight[:, :off]))

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

    def test_cuda_graphs_answer_like_eager_for_every_head(self):
        """The server's captured graphs (padded shapes) against the eager
        path: same answers within fp16 noise, for batch sizes that land in
        different row buckets; swapping the net drops the graphs."""
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from agent.agents.requests import (
            HEAD_BELIEF, HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET, HEAD_TOPK, HEAD_VALUE, Request,
        )
        from bots.heuristic_bot import HeuristicBot
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg())
        eager = TorchModel(net, "cuda", cuda_graphs=False)
        graphed = TorchModel(net, "cuda")
        encs = []
        game = Game(GameConfig(decks=("fignor", "igor"), seed=5))
        bot = HeuristicBot(seed=5)
        while not game.is_over and len(encs) < 40:
            d = game.pending_decision
            encs.append(encode(build_infoset(game, d.player)))
            game.submit(bot.decide(game.view_for(d.player), d))
        reqs = []
        for e in encs:
            k = e.n_options
            reqs += [Request(e), Request(e, HEAD_VALUE), Request(e, HEAD_Q), Request(e, HEAD_TOPK), Request(e, HEAD_BELIEF)]
            if k >= 2:
                reqs += [Request(e, HEAD_SUBSET, candidates=[(), (0,), (1,), (0, 1)]),
                         Request(e, HEAD_SEQUENTIAL, prefix=[0], legal=[1, -1])]
        for n in (3, 20, len(reqs)):
            with self.subTest(n=n):
                a, b = eager.predict_many(reqs[:n]), graphed.predict_many(reqs[:n])
                for r, (sa, va), (sb, vb) in zip(reqs, a, b):
                    self.assertEqual(len(sa), len(sb), r.head)
                    self.assertAlmostEqual(va, vb, delta=5e-3)
                    for x, y in zip(sa, sb):
                        self.assertAlmostEqual(x, y, delta=2e-2 if r.head == HEAD_Q else 5e-3)
        self.assertTrue(graphed._graphs, "the graph path ran")
        graphed.net = KeyForgeNet(_net_cfg()).cuda().eval()  # what ReloadingTorchModel does on a promotion
        graphed.predict_many(reqs[:3])
        self.assertIs(graphed._graph_net, graphed.net)
        self.assertEqual(len(graphed._graphs), 1)

    def test_net_agent_plays_complete_games_for_every_treatment(self):
        from agent.agents.net_agent import NetAgent
        from bots.heuristic_bot import HeuristicBot
        from bots.inference_client import InProcessInferenceClient
        from sim.paired import run_paired

        model = self._model()
        client = InProcessInferenceClient(model)  # uses model.predict_many: batched, as the server would
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


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestSelfPlayLearner(unittest.TestCase):
    """M7/M8: the learner's losses on real actor output, mirror
    augmentation, and the gate's verdict rule."""

    @classmethod
    def setUpClass(cls):
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        cls.net = KeyForgeNet(_net_cfg())
        client = InProcessInferenceClient(TorchModel(cls.net, "cpu"))
        cls.records = {}
        for mode in ("search", "dmc"):
            s = SelfPlaySettings(mode=mode, leaf="student", sims_full=6, sims_small=3, full_fraction=0.5, concurrency=2,
                                 max_turns=14, leaves_in_flight=2)
            recs = []
            Actor(client, s, run_seed=1).play(range(2), recs.append)
            cls.records[mode] = recs

    def _positions(self, mode):
        from ml.selfplay_train import Buffer

        buf = Buffer(10_000, 100)
        for r in self.records[mode]:
            buf.add(r)
        return buf.sample(48, random.Random(0))

    def test_search_mode_losses_are_finite_and_train(self):
        from ml.selfplay_train import losses_for

        pos = self._positions("search")
        total, L, stats = losses_for(self.net, pos, torch.device("cpu"), {"policy": 1, "value": 1, "belief": 0.25, "oracle": 0.25}, "search", True, random.Random(1))
        self.assertTrue(torch.isfinite(total))
        self.assertIn("value", L)
        self.assertTrue({"policy", "multi"} & set(L))
        total.backward()

    def test_dmc_mode_regresses_q(self):
        from ml.selfplay_train import losses_for

        total, L, _ = losses_for(self.net, self._positions("dmc"), torch.device("cpu"), {"q": 1.0}, "dmc", False, random.Random(1))
        self.assertIn("q", L)
        self.assertTrue(torch.isfinite(total))

    def test_mirror_is_an_involution_that_swaps_left_and_right(self):
        from ml.encode import collate
        from ml.selfplay_train import _FLANK_OFF, mirror

        encs = [p["enc"] for p in self._positions("search")[:16]]
        b0, b1 = collate(encs), collate(encs)
        which = torch.ones(len(encs), dtype=torch.bool)
        mirror(b1, which)
        self.assertTrue(torch.equal(b1.inplay[..., _FLANK_OFF], b0.inplay[..., _FLANK_OFF + 1]))
        mirror(b1, which)
        self.assertTrue(torch.allclose(b1.inplay, b0.inplay))
        self.assertTrue(torch.equal(b1.options, b0.options))

    def test_a_resumed_learner_continues_the_same_learning_rate_schedule(self):
        from ml.checkpoints import load_model, save_model
        from ml.model import KeyForgeNet
        from ml.selfplay_train import cosine_schedule

        def fresh():
            torch.manual_seed(0)
            net = KeyForgeNet(_net_cfg(layers=1))
            return net, torch.optim.AdamW(net.parameters(), lr=2e-3)

        net, opt = fresh()
        sched = cosine_schedule(opt, 100, 1e-5, 0, 2e-3)
        lrs = []
        for _ in range(60):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        with tempfile.TemporaryDirectory() as d:
            # Save at step 30 (as the learner does), resume, replay 30 more.
            net2, opt2 = fresh()
            s2 = cosine_schedule(opt2, 100, 1e-5, 0, 2e-3)
            for _ in range(30):
                opt2.step()
                s2.step()
            path = os.path.join(d, "state.kfc")
            save_model(net2, path, optimizer=opt2, extra={"step": 30})
            net3, meta, opt_state = load_model(path)
            opt3 = torch.optim.AdamW(net3.parameters(), lr=2e-3)
            opt3.load_state_dict({"state": opt_state["state"], "param_groups": opt_state["param_groups"]})
            s3 = cosine_schedule(opt3, 100, 1e-5, int(meta["extra"]["step"]), 2e-3)
            self.assertAlmostEqual(s3.get_last_lr()[0], lrs[29], places=12)
            for i in range(30, 60):
                opt3.step()
                s3.step()
                self.assertAlmostEqual(s3.get_last_lr()[0], lrs[i], places=12)

    def test_fallback_multi_select_positions_train_value_only(self):
        from ml.selfplay_train import Buffer, losses_for

        buf = Buffer(10_000, 100)
        for r in self.records["search"]:
            buf.add(r)
        pos = [dict(p) for g in buf.games for p in g]
        # What an older actor wrote for a multi-select that fell back to the
        # fixed policy: kind CHOOSE_CARDS, candidates [None], target [1.0].
        for p in pos:
            if p["target"] is not None and not p["value_only"]:
                p["kind"], p["candidates"], p["target"] = "CHOOSE_CARDS", [None], [1.0]
        total, L, _ = losses_for(self.net, pos, torch.device("cpu"), {"policy": 1, "value": 1}, "search", False, random.Random(1))
        self.assertTrue(torch.isfinite(total))
        self.assertNotIn("multi", L)

    def test_gate_verdict_uses_sprt_then_a_fixed_n_fallback(self):
        from ml.selfplay_train import gate_verdict
        from sim.paired import PairedReport

        self.assertEqual(gate_verdict(PairedReport(games=400, a_wins=300, b_wins=100, draws=0, forfeits=0)), "H1")
        self.assertEqual(gate_verdict(PairedReport(games=40, a_wins=36, b_wins=4, draws=0, forfeits=0)), "H1")  # fixed-N: far past 2 SE
        self.assertIsNone(gate_verdict(PairedReport(games=40, a_wins=21, b_wins=19, draws=0, forfeits=0)))


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestM11NetworkAgent(unittest.TestCase):
    def test_a_network_plays_legal_keyforge_on_other_pools_and_formats(self):
        from agent.agents.net_agent import NetAgent
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet
        from sim.driver import run_games

        torch.manual_seed(2)
        agent = NetAgent(InProcessInferenceClient(TorchModel(KeyForgeNet(_net_cfg()), "cpu")), seed=1)
        rng = random.Random(9)
        configs = [
            GameConfig(decks=("stonewall", "starfall"), seed=1, max_turns=30),
            GameConfig(decks=(random_deck(rng, "R1"), random_deck(rng, "R2")), seed=2, max_turns=30),
            MatchConfig(format="adaptive", decks=("vigil", "thornwood"), seed=3, max_turns=30),
            MatchConfig(format="reversal", decks=("fignor", "igor"), seed=4, max_turns=30),
        ]
        results = run_games(len(configs), lambda i: configs[i], lambda i: {1: agent, 2: RandomBot(seed=i)}, concurrency=4)
        for r in results:
            self.assertIsNone(r.forfeit, r.forfeit)


@unittest.skipUnless(TORCH, "torch is intentionally not installed for the rest of this suite")
class TestBCTrainer(unittest.TestCase):
    """The BC trainer's losses on a small real corpus: the vectorized
    multi-select losses equal straightforward per-row references (what the
    loops computed before they were vectorized), and a training step runs
    under bf16 autocast."""

    FIELDS = ("kind", "target", "forced", "z", "turn", "source", "min_n", "max_n", "n_opt", "opp_hand", "next_draws")

    @classmethod
    def setUpClass(cls):
        import json

        from ml.dataset import Corpus, encode_records
        from sim.bc_corpus import play_labelled

        cls.tmp = tempfile.TemporaryDirectory()
        records = os.path.join(cls.tmp.name, "records.jsonl")
        with open(records, "w", encoding="utf-8") as f:
            for i, source in enumerate(("heuristic", "random", "random", "heuristic", "random")):
                decks = ("fignor", "igor") if i % 2 == 0 else ("igor", "fignor")
                rec = play_labelled(GameConfig(decks=decks, seed=200 + i, max_turns=40), source, 200 + i)
                rec["index"] = i
                f.write(json.dumps(rec) + "\n")
        shard = os.path.join(cls.tmp.name, "encoded", "s0")
        encode_records(records, shard)
        cls.corpus = Corpus([shard])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _batch(self, device):
        batch, tg = self.corpus.batch(np.arange(min(self.corpus.size, 600)))
        batch = batch.to(device)
        for f in self.FIELDS:
            setattr(tg, f, getattr(tg, f).to(device))
        return batch, tg

    def test_vectorized_multi_select_losses_match_per_row_references(self):
        from torch.nn import functional as F

        from ml.bc_train import K_ORDER, MultiPrep
        from ml.model import KeyForgeNet

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg()).eval()
        batch, tg = self._batch(torch.device("cpu"))
        out = net.encode_state(batch)
        prep = MultiPrep(tg, 1024, torch.device("cpu"))
        self.assertTrue(prep.rows and prep.spans and prep.s_rows, "the corpus needs multi-select decisions")
        n_opt, kinds = tg.n_opt.tolist(), tg.kind.tolist()

        loss, _ = prep.enumerate_loss(net, out)
        scores = net.subset_scores(out, prep.c_rows, prep.c_members, prep.c_ordered)
        ref = torch.full((len(prep.spans), max(n for _b, _s, n, _t, _c in prep.spans)), float("-inf"))
        targets = torch.empty(len(prep.spans), dtype=torch.int64)
        for r, (_b, start, n, t, _c) in enumerate(prep.spans):
            ref[r, :n] = scores[start : start + n]
            targets[r] = t
        self.assertTrue(torch.allclose(loss, F.cross_entropy(ref, targets)))

        K = out.e_opt.shape[1]
        _rows, prefix, legal = prep.sequential_tensors(K)
        for i in range(len(prep.s_rows)):
            self.assertEqual(sorted(prefix[i].nonzero().flatten().tolist()), sorted(prep.s_prefix[i]))
            self.assertEqual(sorted(legal[i].nonzero().flatten().tolist()), sorted(K if l == -1 else l for l in prep.s_legal[i]))

        loss = prep.topk_loss(net, out, n_opt, kinds)
        logits = net.topk_logits(out)
        total, count = 0.0, 0
        for b in prep.rows:
            n, s, key = n_opt[b], logits[b, : n_opt[b]], prep.keys[b]
            if kinds[b] == K_ORDER:
                ss = s[torch.tensor(list(key), dtype=torch.int64)]
                total = total + sum(torch.logsumexp(ss[t:], 0) - ss[t] for t in range(len(key)))
            else:
                y = torch.zeros(n)
                if key:
                    y[list(key)] = 1.0
                total = total + F.binary_cross_entropy_with_logits(s, y, reduction="sum")
            count += n
        self.assertTrue(torch.allclose(loss, total / count, rtol=1e-5, atol=1e-6))

    def test_train_runs_on_cpu_and_gpu(self):
        """The whole loop, on each device: every step booked, finite losses
        (on the GPU this covers the fused optimizer too)."""
        import contextlib
        import io
        from unittest import mock

        from ml import bc_train

        class Metrics:
            def __init__(self):
                self.counts, self.seen = {}, []

            def count(self, k, n=1):
                self.counts[k] = self.counts.get(k, 0) + n

            def observe(self, k, v):
                self.seen.append((k, v))

            def gauge(self, k, v):
                pass

            def tick(self):
                pass

        devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device.type):
                cfg = {"seed": 0, "network": _net_cfg(), "bc": dict(config_mod.DEFAULTS["bc"], batch=64, epochs=2)}
                m = Metrics()
                with mock.patch.object(bc_train, "evaluate", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                    _net, report = bc_train.train(cfg, self.corpus, device=device, metrics=m, log_every=3)
                self.assertEqual(m.counts["gradient_steps"], report["training"]["steps"])
                self.assertEqual(m.counts["positions"], 2 * len(self.corpus.indices(split=0)))
                self.assertTrue(all(np.isfinite(v) for _k, v in m.seen))
                self.assertTrue(report["training"]["history"])

    def test_the_prefetcher_raises_what_its_thread_raised(self):
        from ml.bc_train import Prefetcher

        def items():
            yield 1
            raise RuntimeError("shard unreadable")

        got = []
        with self.assertRaisesRegex(RuntimeError, "shard unreadable"):
            for x in Prefetcher(items()):
                got.append(x)
        self.assertEqual(got, [1])

    def test_a_training_step_runs_under_bf16_autocast(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from ml.bc_train import DEFAULT_WEIGHTS, compute_losses
        from ml.model import KeyForgeNet, amp_dtype

        device = torch.device("cuda")
        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg()).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        batch, tg = self._batch(device)
        before = [p.detach().clone() for p in net.parameters()]
        with torch.autocast("cuda", dtype=amp_dtype("bf16", device)):
            total, losses, _o, _p = compute_losses(net, batch, tg, weights=dict(DEFAULT_WEIGHTS), policy_head="pointer", cap=1024,
                                                   heads=tuple(DEFAULT_WEIGHTS))
        self.assertTrue(torch.isfinite(total))
        self.assertTrue({"enumerate", "sequential", "topk"} <= set(losses))
        total.backward()
        opt.step()
        self.assertTrue(any(not torch.equal(a, p.detach()) for a, p in zip(before, net.parameters())))

    def test_the_learner_trains_under_bf16_autocast(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from agent.selfplay import Actor, SelfPlaySettings
        from bots.inference_client import InProcessInferenceClient
        from ml.infer_server import TorchModel
        from ml.model import KeyForgeNet, amp_dtype
        from ml.selfplay_train import Buffer, losses_for

        torch.manual_seed(0)
        net = KeyForgeNet(_net_cfg())
        settings = SelfPlaySettings(mode="search", leaf="student", sims_full=6, sims_small=3, full_fraction=0.5, concurrency=2,
                                    max_turns=20, leaves_in_flight=2)
        records = []
        Actor(InProcessInferenceClient(TorchModel(net, "cpu")), settings, run_seed=2).play(range(4), records.append)
        buf = Buffer(10_000, 100)
        for r in records:
            buf.add(r)
        positions = buf.sample(200, random.Random(0))
        device = torch.device("cuda")
        net = net.to(device).train()
        with torch.autocast("cuda", dtype=amp_dtype("bf16", device)):
            total, L, _ = losses_for(net, positions, device, {"policy": 1, "value": 1, "belief": 0.25, "oracle": 0.25}, "search",
                                     True, random.Random(1))
        self.assertTrue(torch.isfinite(total))
        if any(p["kind"] in ("CHOOSE_CARDS", "ORDER_EFFECTS") and p["target"] is not None and not p["value_only"] and p["candidates"]
               for p in positions):
            self.assertIn("multi", L)  # the padded multi-select path ran under bf16
        total.backward()

    def test_a_compiled_trunk_keeps_the_checkpoint_and_the_outputs(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        from ml.model import KeyForgeNet, compile_trunk

        torch.manual_seed(0)
        eager = KeyForgeNet(_net_cfg()).cuda().eval()
        compiled = KeyForgeNet(_net_cfg()).cuda().eval()
        compiled.load_state_dict(eager.state_dict())
        compile_trunk(compiled, True, "cuda")
        self.assertEqual(list(compiled.state_dict()), list(eager.state_dict()), "checkpoint keys must not change")
        batch, _tg = self._batch(torch.device("cuda"))
        with torch.no_grad():
            a, b = eager.encode_state(batch), compiled.encode_state(batch)
        self.assertTrue(torch.allclose(a.h, b.h, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(eager.value(a), compiled.value(b), atol=1e-4, rtol=1e-4))

    def test_unknown_precision_is_an_error(self):
        from ml.model import amp_dtype

        self.assertIsNone(amp_dtype("fp32", "cuda"))
        self.assertIsNone(amp_dtype("bf16", "cpu"))
        with self.assertRaises(ValueError):
            amp_dtype("fp8", "cuda")


if __name__ == "__main__":
    unittest.main()

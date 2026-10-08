"""Agent Observation Plan, Milestone O8: inference and search plumbing.

- `stream`'s cached prefix: a search's prefix computed once, then each
  leaf's state and suffix, gives the outputs the whole sequence gives.
- The v2 attention (key-padding masks, a separate causal call for
  `stream`'s history) equals attention under the dense [B, T, T] mask it
  replaced.
"""

import random
import unittest

try:
    import torch
except ImportError:  # the torch half runs in WSL (see tests/test_agent_training_ml.py)
    torch = None

CFG = {"d_model": 64, "heads": 4, "ff": 128, "layers": 2, "dropout": 0.0, "attention_kernel": "math"}


def _worlds(n_games=2, every=11, steps=6):
    """(prefix, [(encoding, full history, suffix rows, turn)] in worlds
    played on from it) at decisions of fuzz games."""
    from agent.features_v2 import encode_v2
    from agent.history import history_for
    from keyforge.enums import Resample
    from keyforge.game import Game
    from tools.o1_acceptance import POOLS, bots_for, game_config

    out = []
    for pool in list(POOLS)[:2]:
        for i in range(n_games):
            config = game_config(pool, i)
            game = Game(config)
            bots = bots_for(i, config.seed)
            n = 0
            while not game.is_over:
                d = game.pending_decision
                if n % every == 5 and game.copy_anywhere and n > 20:
                    v = d.player
                    h = history_for(game, v)
                    h.freeze()
                    leaves = []
                    rng = random.Random(n)
                    for _ in range(3):
                        world = game.fork_determinized(v, rng, Resample.ALL, backend="copy", sampler="chance_exact")
                        for _ in range(rng.randrange(steps)):
                            if world.is_over:
                                break
                            wd = world.pending_decision
                            world.submit(bots[wd.player].decide(world.view_for(wd.player), wd))
                        if world.is_over:
                            continue
                        wh = history_for(world, v)
                        leaves.append((encode_v2(world, v), wh.copy(), wh.suffix_rows(), world.turn_number))
                    if leaves:
                        out.append((h.prefix_rows(), leaves))
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                n += 1
    return out


@unittest.skipIf(torch is None, "needs torch")
class TestPrefixCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _worlds()

    def _net(self, **kw):
        from ml.model_v2 import KeyForgeNetV2

        torch.manual_seed(0)
        return KeyForgeNetV2(dict(CFG, history_arch="stream", **kw)).eval()

    def _prefix(self, net, prefixes, card_ids):
        """Prefix rows -> PrefixPast for a batch (one prefix per item)."""
        from agent.history import N_RF, N_RI
        import numpy as np

        B = len(prefixes)
        P = max(1, max(p.n for p in prefixes))
        Q = max(1, max(len(p.pointers) // 3 for p in prefixes))
        ints = np.zeros((B, P, N_RI), np.int64)
        floats = np.zeros((B, P, N_RF), np.float32)
        ptr = np.full((B, Q, 3), -1, np.int64)
        mask = np.zeros((B, P), bool)
        for b, p in enumerate(prefixes):
            ints[b, :p.n] = np.frombuffer(p.ints.tobytes(), np.int64).reshape(p.n, N_RI)
            floats[b, :p.n] = np.frombuffer(p.floats.tobytes(), np.float32).reshape(p.n, N_RF)
            q = np.frombuffer(p.pointers.tobytes(), np.int64).reshape(-1, 3)
            ptr[b, :len(q)] = q
            mask[b, :p.n] = True
        t = torch.from_numpy
        return net.history_prefix(card_ids, t(ints), t(floats), t(ptr), t(mask))

    def test_prefix_then_suffix_equals_the_whole(self):
        from agent import spec_v2 as S
        from ml.encode_v2 import collate_v2

        self.assertGreater(len(self.cases), 3)
        net = self._net()
        worst = 0.0
        for prefix, leaves in self.cases:
            whole = collate_v2([(e, h, t) for e, h, _s, t in leaves])
            part = collate_v2([(e, s, t) for e, _h, s, t in leaves])
            card_ids = part.blocks["entity"].ints[..., S.ENTITY.i["card"]]
            with torch.no_grad():
                a = net.encode_state(whole)
                past = self._prefix(net, [prefix] * len(leaves), card_ids)
                b = net.encode_state(part, past)
                va, vb = net.value(a), net.value(b)
                pa, pb = net.policy_logits(a), net.policy_logits(b)
            real = torch.isfinite(pa)
            self.assertTrue(torch.equal(real, torch.isfinite(pb)))
            worst = max(worst, (va - vb).abs().max().item())
            if real.any():
                worst = max(worst, (pa[real] - pb[real]).abs().max().item())
            self.assertTrue(torch.allclose(a.h, b.h, atol=1e-5))
        self.assertLess(worst, 1e-5)

    def test_masks_equal_the_dense_mask(self):
        """The key-padding / causal-call attention against the [B, T, T]
        mask it replaced, for joint and stream."""
        from ml.encode_v2 import collate_v2
        from ml.layers import AttnSpec

        leaves = [leaf for _p, ls in self.cases[:3] for leaf in ls]
        batch = collate_v2([(e, h, t) for e, h, _s, t in leaves])
        for arch in ("joint", "stream"):
            with self.subTest(arch=arch):
                net = self._net() if arch == "stream" else self._dense_twin(arch)
                captured = {}
                orig = net.trunk.forward

                def spy(x, spec=None, past=None, want_kv=False):
                    captured["x"], captured["spec"] = x, spec
                    return orig(x, spec, past, want_kv)

                net.trunk.forward = spy
                with torch.no_grad():
                    net.encode_state(batch)
                    x, spec = captured["x"], captured["spec"]
                    y = orig(x, spec)
                    B, T, _ = x.shape
                    dense = spec.key_valid[:, None, :].expand(B, T, T).clone()
                    if spec.hist_from is not None:
                        h0 = spec.hist_from
                        H = T - h0
                        dense[:, h0:, :h0] = False
                        dense[:, h0:, h0:] = torch.tril(torch.ones(H, H, dtype=torch.bool))
                    ref = x
                    for layer in net.trunk.layers:
                        ref = _dense_layer(layer, ref, dense)
                valid = spec.key_valid
                self.assertTrue(torch.allclose(y[valid], ref[valid], atol=1e-5))

    def _dense_twin(self, arch):
        from ml.model_v2 import KeyForgeNetV2

        torch.manual_seed(0)
        return KeyForgeNetV2(dict(CFG, history_arch=arch)).eval()


def _dense_layer(layer, x, dense):
    from torch.nn import functional as F

    a = layer.self_attn
    B, T, d = x.shape
    qkv = F.linear(layer.norm1(x), a.in_proj_weight, a.in_proj_bias)
    q, k, v = qkv.view(B, T, 3, a.heads, d // a.heads).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=dense.unsqueeze(1))
    x = x + a.out_proj(o.transpose(1, 2).reshape(B, T, d))
    return x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))


class _Checked:
    """The server, checked: every answer to a prefix request is compared
    with the same request sent whole (the naive path) to a second server
    holding the same network; pipe bytes are counted both ways."""

    def __init__(self, net, drop_after=None):
        import pickle

        from ml.infer_v2 import TorchModelV2

        self.pickle = pickle
        self.model = TorchModelV2(net, "cpu")
        self.reference = TorchModelV2(net, "cpu")
        self.compared = 0
        self.worst = 0.0
        self.bytes = {"prefix": 0, "whole": 0, "requests": 0}
        self.calls = 0
        self.drop_after = drop_after

    def predict_many(self, reqs):
        import dataclasses

        from agent.agents.requests import PREFIX_MISSING, HistoryRef
        from agent.history import HistoryRows
        from ml.infer_v2 import _concat

        self.calls += 1
        if self.drop_after is not None and self.calls == self.drop_after:
            self.model.prefixes.data.clear()  # a restarted server
            self.model.prefixes.bytes = 0
        answers = self.model.predict_many(reqs)
        whole = []
        for r, a in zip(reqs, answers):
            self.bytes["requests"] += 1
            self.bytes["prefix"] += len(self.pickle.dumps(r))
            h = r.history
            if isinstance(h, HistoryRef):
                e = self.model.prefixes.data.get(h.key)
                if a == PREFIX_MISSING or e is None:
                    continue
                rows = _concat(e.rows, HistoryRows.from_bytes(h.suffix))
                w = dataclasses.replace(r, history=rows.to_bytes())
                self.bytes["whole"] += len(self.pickle.dumps(w))
                whole.append((w, a))
            else:
                self.bytes["whole"] += len(self.pickle.dumps(r))
        if whole:
            ref = self.reference.predict_many([w for w, _a in whole])
            for (w, a), b in zip(whole, ref):
                self.compared += 1
                d = abs(a[1] - b[1])
                if a[0]:
                    d = max(d, max(abs(x - y) for x, y in zip(a[0], b[0])))
                self.worst = max(self.worst, d)
        return answers


def _searches(server, regime="within_turn", sims=48, n_decisions=4, history="rows", seed=3):
    """Searches at decisions of one fuzz game; returns the evaluators."""
    from agent.search.policies import make_policy
    from agent.search.core import Search, SearchSettings, drive
    from agent.search.full_game import FullGame
    from agent.search.leaf_v2 import NetworkEvaluatorV2
    from agent.search.within_turn import WithinTurn
    from agent.selfplay import _Cap
    from keyforge.game import Game
    from tools.o1_acceptance import POOLS, bots_for, game_config

    config = game_config(list(POOLS)[0], seed)
    game = Game(config)
    bots = bots_for(seed, config.seed)
    evaluators, n, done = [], 0, 0
    while not game.is_over and done < n_decisions:
        d = game.pending_decision
        if n > 15 and n % 9 == 0 and len(d.options) > 1 and game.copy_anywhere:
            policy = make_policy("heuristic", seed=n)
            reg = WithinTurn(policy) if regime == "within_turn" else FullGame(policy)
            ev = NetworkEvaluatorV2(history)
            search = Search(reg, ev, policy, SearchSettings(simulations=sims, reuse=False, fork_backend="copy",
                                                           determinization="chance_exact"), seed=n)
            drive(search.search_gen(_Cap(game, d.player), d), server)
            evaluators.append(ev)
            done += 1
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        n += 1
    return evaluators


@unittest.skipIf(torch is None, "needs torch")
class TestPrefixProtocol(unittest.TestCase):
    def _net(self, arch):
        from ml.model_v2 import KeyForgeNetV2

        torch.manual_seed(0)
        return KeyForgeNetV2(dict(CFG, history_arch=arch)).eval()

    def test_cached_answers_equal_whole_history_answers(self):
        for arch, regime in (("stream", "within_turn"), ("stream", "full_game"), ("joint", "within_turn")):
            with self.subTest(arch=arch, regime=regime):
                server = _Checked(self._net(arch))
                evs = _searches(server, regime)
                self.assertGreater(server.compared, 20)
                self.assertLess(server.worst, 1e-5)
                # within a search the prefix goes out once: every other
                # request finds it cached
                st = server.model.prefixes
                self.assertGreaterEqual(st.hits / max(1, st.hits + st.stored), 0.95, (st.hits, st.stored))
                self.assertLess(server.bytes["prefix"], server.bytes["whole"])
                self.assertTrue(all(e.stats["resent"] == 0 for e in evs))
                print(f"\n{arch}/{regime}: {server.compared} compared, worst {server.worst:.2e}; "
                      f"bytes/request {server.bytes['prefix'] / server.bytes['requests']:.0f} with the prefix "
                      f"cached, {server.bytes['whole'] / server.bytes['requests']:.0f} whole; "
                      f"hit rate {st.hits / (st.hits + st.stored):.3f}")

    def test_a_lost_prefix_is_sent_again(self):
        server = _Checked(self._net("stream"), drop_after=3)
        evs = _searches(server, n_decisions=1)
        self.assertGreater(sum(e.stats["resent"] for e in evs), 0)
        self.assertLess(server.worst, 1e-5)

    def test_requests_pickle(self):
        import pickle

        from agent.agents.requests import HEAD_VALUE, HistoryRef, RequestV2
        from agent.history import HistoryRows

        enc, h, suffix, turn = _worlds(n_games=1)[0][1][0]
        r = RequestV2(enc=enc, head=HEAD_VALUE, history=HistoryRef("k", 3, suffix.to_bytes()), turn=turn)
        back = pickle.loads(pickle.dumps(r))
        self.assertEqual(back.enc.to_bytes(), enc.to_bytes())
        self.assertEqual(HistoryRows.from_bytes(back.history.suffix).to_bytes(), suffix.to_bytes())


if __name__ == "__main__":
    unittest.main()

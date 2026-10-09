"""Agent Observation Plan, Milestone O9: v2 data and training.

- A round trip through the v2 shards equals live encoding: every token
  block, the event history up to each record's cursor, the turn tokens and
  summaries (exact, or to float16 where a column is stored as float16).
- The privileged labels are the true game's: each entity's zone, the
  opponent's deck order, my next draws.
- History dropout keeps every pointer row on a row that was kept.
- (WSL, torch) a few training steps of every history architecture give
  finite losses, and the evaluation reports belief v2 against O3's prior.
"""

import os
import shutil
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:  # the torch half runs in WSL (see tests/test_agent_training_ml.py); PyPy has neither
    np = torch = None

from keyforge.config import GameConfig

_TMP = None


def _corpus_dir():
    """Three short games (heuristic, epsilon, random) encoded once per test
    run."""
    global _TMP
    if _TMP is None:
        from ml.dataset_v2 import _Cols, encode_game_v2, write_shard_v2
        from sim.bc_corpus import play_labelled

        _TMP = tempfile.mkdtemp(prefix="o9_")
        recs = [play_labelled(GameConfig(decks=("fignor", "igor"), seed=s, max_turns=24), src, s)
                for s, src in ((3, "heuristic"), (4, "epsilon"), (5, "random"))]
        c = _Cols()
        for r in recs:
            encode_game_v2(r, c, value_only_fraction=1.0)
        write_shard_v2(os.path.join(_TMP, "s0"), c, {"games": len(recs)})
        _corpus_dir.recs = recs
    return _TMP


def _live(rec):
    """Every record's live (encoding, history copy, turn, game) in the shard's
    order: the decider, then the other seat."""
    from agent.features_v2 import encode_v2
    from agent.history import history_for
    from keyforge.game import Game
    from keyforge.replay import config_from_dict

    game = Game(config_from_dict(rec["config"]))
    out = []
    for enc_choice in rec["record"]:
        pid = game.pending_decision.player
        for v in (pid, 3 - pid):
            out.append((encode_v2(game, v), history_for(game, v).copy(), game.turn_number, v, _truth(game, v)))
        game.submit_index(enc_choice)
    return out


def _truth(game, v):
    me, them = game.players[v], game.players[3 - v]
    order = [c.instance_id for c in me.all_cards] + [c.instance_id for c in them.all_cards]
    zone = {}
    for p in (me, them):
        for cls, zone_obj in ((1, p.hand), (2, p.archive), (3, p.deck)):
            for c in zone_obj.cards():
                zone[c.instance_id] = cls
    return ([zone.get(i, 0) for i in order], [order.index(c.instance_id) for c in them.deck.cards()][:40],
            [order.index(c.instance_id) for c in me.deck.cards()][:5])


def _records(batches):
    """Each record of a batch stream as plain data: every real row of every
    block, its history rows, pointers, turn tokens, summaries and labels."""
    out = []
    for b, t in batches:
        for i in range(b.size):
            rec = []
            for k, blk in sorted(b.blocks.items()):
                m = blk.mask[i]
                rec.append((k, blk.ints[i][m].tolist(), blk.floats[i][m].tolist()))
            hm = b.hist_mask[i]
            rec.append(("hist", b.hist_ints[i][hm].tolist(), b.hist_floats[i][hm].tolist(),
                        [p for p in b.hist_ptr[i].tolist() if p[0] >= 0]))
            tm = b.turn_mask[i]
            rec.append(("turns", b.turn_ints[i][tm].tolist(), b.turn_floats[i][tm].tolist(),
                        b.turn_sets[i][tm].nonzero().tolist()))
            rec.append(("summary", b.summary_entity[i].tolist(), b.summary_global[i].tolist(), int(b.turn_now[i])))
            rec.append(("labels", int(t.kind[i]), int(t.target[i]), float(t.z[i]), t.chosen[i], t.zone[i].tolist(),
                        t.opp_deck[i].tolist(), t.my_draws[i].tolist()))
            out.append(rec)
    return out


def _f16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


@unittest.skipIf(np is None, "needs numpy")
class TestShards(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        global _TMP
        if _TMP is not None:
            shutil.rmtree(_TMP, ignore_errors=True)
            _TMP = None

    def test_round_trip_equals_live_encoding(self):
        if torch is None:
            self.skipTest("needs torch (ml/encode_v2.py)")
        from agent import spec_v2 as S
        from ml.dataset_v2 import CorpusV2
        from ml.encode_v2 import collate_v2

        root = _corpus_dir()
        live = [x for r in _corpus_dir.recs for x in _live(r)]
        rows = CorpusV2.from_dir(root, history="rows")
        folded = CorpusV2.from_dir(root, history="folded")
        self.assertEqual(rows.size, len(live))
        for start in range(0, len(live), 64):
            idx = np.arange(start, min(start + 64, len(live)))
            got, tg = rows.batch(idx)
            got_f, _tg = folded.batch(idx)
            want = collate_v2([(e, h, t) for e, h, t, _v, _tr in live[start:idx[-1] + 1]])
            for b in S.BLOCKS:
                g, w = got.blocks[b.name], want.blocks[b.name]
                R = min(g.mask.shape[1], w.mask.shape[1])
                self.assertTrue(torch.equal(g.mask[:, :R], w.mask[:, :R]), b.name)
                self.assertFalse(g.mask[:, R:].any() or w.mask[:, R:].any(), b.name)
                self.assertTrue(torch.equal(g.ints[:, :R][g.mask[:, :R]], w.ints[:, :R][w.mask[:, :R]]), b.name)
                self.assertTrue(np.array_equal(g.floats[:, :R][g.mask[:, :R]].numpy(),
                                               _f16(w.floats[:, :R][w.mask[:, :R]].numpy())), b.name)
            H = want.hist_mask.shape[1]
            self.assertTrue(torch.equal(got.hist_mask[:, :H], want.hist_mask))
            self.assertTrue(torch.equal(got.hist_ints[:, :H][want.hist_mask], want.hist_ints[want.hist_mask]))
            self.assertTrue(torch.equal(got.hist_floats[:, :H][want.hist_mask], want.hist_floats[want.hist_mask]))
            Q = want.hist_ptr.shape[1]
            self.assertTrue(torch.equal(got.hist_ptr[:, :Q], want.hist_ptr))
            U = want.turn_mask.shape[1]
            self.assertTrue(torch.equal(got_f.turn_mask[:, :U], want.turn_mask))
            self.assertTrue(torch.equal(got_f.turn_ints[:, :U][want.turn_mask], want.turn_ints[want.turn_mask]))
            self.assertTrue(np.array_equal(got_f.turn_floats[:, :U][want.turn_mask].numpy(),
                                           _f16(want.turn_floats[want.turn_mask].numpy())))
            self.assertTrue(torch.equal(got_f.turn_sets[:, :U], want.turn_sets))
            self.assertTrue(np.array_equal(got_f.summary_entity.numpy(), _f16(want.summary_entity.numpy())))
            self.assertTrue(np.array_equal(got_f.summary_global.numpy(), _f16(want.summary_global.numpy())))
            for j, (_e, _h, turn, v, (zone, deck, draws)) in enumerate(live[start:idx[-1] + 1]):
                self.assertEqual(tg.zone[j].tolist(), zone)
                self.assertEqual([x for x in tg.opp_deck[j].tolist() if x >= 0], deck)
                self.assertEqual([x for x in tg.my_draws[j].tolist() if x >= 0], draws)
                self.assertEqual(int(tg.turn[j]), turn)

    def test_packed_corpus_gives_the_same_batches(self):
        if torch is None:
            self.skipTest("needs torch (ml/encode_v2.py)")
        from dataclasses import fields

        from ml.dataset_v2 import CorpusV2, PackedCorpusV2, pack_corpus_v2

        root = _corpus_dir()
        packs = os.path.join(root, "packed")
        pack_corpus_v2(root, packs, workers=1, chunk=7)
        for form in ("rows", "folded", "none"):
            plain = CorpusV2.from_dir(root, history=form)
            packed = PackedCorpusV2(packs, history=form, segment_chunks=5)
            self.assertEqual(packed.size, plain.size)
            for split in (None, 0):
                a = list(plain.iterate(plain.indices(split=split), 50, shuffle=False))
                b = list(packed.iterate(packed.indices(split=split), 50, shuffle=False))
                rows_a = sum(x[0].size for x in a)
                self.assertEqual(rows_a, sum(x[0].size for x in b))
                self.assertEqual(rows_a, len(packed.indices(split=split)))
                ga = torch.cat([x[1].z for x in a])
                gb = torch.cat([x[1].z for x in b])
                self.assertTrue(torch.equal(ga, gb))
            # record by record, unshuffled with no split: the same contents
            # (batches split differently: by shard, against by segment)
            self.assertEqual(_records(plain.iterate(plain.indices(), 50, shuffle=False)),
                             _records(packed.iterate(packed.indices(), 50, shuffle=False)), form)
            # shuffled, every record once
            seen = torch.cat([t.z for _b, t in packed.iterate(packed.indices(), 50, shuffle=True, seed=3)])
            self.assertEqual(len(seen), packed.size)

    def test_history_dropout_keeps_pointers_on_kept_rows(self):
        if torch is None:
            self.skipTest("needs torch (ml/encode_v2.py)")
        from ml.dataset_v2 import CorpusV2

        corpus = CorpusV2.from_dir(_corpus_dir(), history="rows", history_dropout=0.5)
        idx = np.arange(min(corpus.size, 128))
        full, _ = CorpusV2.from_dir(_corpus_dir(), history="rows").batch(idx)
        got, _ = corpus.batch(idx, train=True, rng=np.random.default_rng(1))
        kept = got.hist_mask.sum(1)
        self.assertTrue((kept <= full.hist_mask.sum(1)).all())
        self.assertLess(int(kept.sum()), int(full.hist_mask.sum()))
        rows = got.hist_ptr[..., 0]
        self.assertTrue(((rows < 0) | (rows < kept.unsqueeze(1))).all())


@unittest.skipIf(torch is None, "needs torch")
class TestTraining(unittest.TestCase):
    def test_every_architecture_trains_and_reports(self):
        from agent.config import resolve
        from ml.bc_train_v2 import train
        from ml.dataset_v2 import CorpusV2

        root = _corpus_dir()
        for arch, form in (("none", "none"), ("summary", "folded"), ("turn_tokens", "folded"), ("joint", "rows"),
                           ("stream", "rows")):
            with self.subTest(arch=arch):
                cfg = resolve("tier0b_bc.json", {"network": {"d_model": 32, "heads": 4, "ff": 64, "layers": 1,
                                                             "history_arch": arch, "attention_kernel": "math"},
                                                 "bc": {"batch": 32, "epochs": 1, "compile": False,
                                                        "history_dropout": 0.1}})
                corpus = CorpusV2.from_dir(root, history=form, history_dropout=0.1)
                every = np.arange(corpus.size)
                model, report = train(cfg, corpus, device=torch.device("cpu"), log_every=10 ** 9, max_steps=3,
                                      train_idx=every, val_idx=every)
                self.assertEqual(report["training"]["steps"], 3)
                self.assertTrue(np.isfinite(report["value"]["logloss"]))
                self.assertIsNotNone(report["belief"])
                self.assertTrue(np.isfinite(report["belief"]["zone_prior_logloss"]))

    def test_checkpoint_round_trip(self):
        from ml.checkpoints import CheckpointMismatch, load_model_v2, save_model
        from ml.model_v2 import KeyForgeNetV2, checkpoint_stamp

        torch.manual_seed(0)
        model = KeyForgeNetV2({"d_model": 32, "heads": 4, "ff": 64, "layers": 1, "history_arch": "stream"})
        path = os.path.join(_corpus_dir(), "m.kfc")
        save_model(model, path, extra={"kind": "bc_v2", **checkpoint_stamp()})
        back, meta = load_model_v2(path)
        self.assertEqual(back.history_arch, "stream")
        for (k, a), (_k, b) in zip(model.state_dict().items(), back.state_dict().items()):
            self.assertTrue(torch.equal(a, b), k)
        save_model(model, path, extra={"kind": "bc_v2", **dict(checkpoint_stamp(), layout_hash="other")})
        with self.assertRaises(CheckpointMismatch):
            load_model_v2(path)


if __name__ == "__main__":
    unittest.main()

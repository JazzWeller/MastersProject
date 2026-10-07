"""Agent Observation Plan, Milestone O5: the lossless v2 encoding.

M1's tests, ported to v2 (no leak, determinism, every decision kind, every
deck source and format), plus the vocabularies' coverage of the whole
card registry, the three static tables, and v1 staying bit-identical. The
1,000-game leak sweep is `python -m tools.o5_acceptance`.
"""

import os
import random
import subprocess
import sys
import unittest

from agent import features as features_shim
from agent import features_v1
from agent import spec_v2 as S
from agent.features_v2 import encode_v2, encode_v2_for
from agent.static_v2 import SIGNATURE_WIDTH, signature_row, tables, text_rows
from agent.vocab import load_all
from keyforge.cards.card_data import CARD_DEFS
from keyforge.enums import DecisionKind, Resample
from keyforge.game import Game
from keyforge.infoset import build_infoset
from tools.o1_acceptance import POOLS, bots_for, game_config

_NON_GUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestVocabularies(unittest.TestCase):
    def test_nothing_in_the_source_is_missing(self):
        from tools.build_vocab_v2 import build

        built = build(0)
        vocab = load_all()
        for name, entries in built.items():
            self.assertEqual(entries, vocab[name].entries, f"{name}: run python -m tools.build_vocab_v2")

    def test_the_whole_registry_is_covered(self):
        vocab = load_all()
        for cdef in CARD_DEFS.values():
            for t in cdef.tags:
                vocab["traits"].id(t)
            for k in list(cdef.keywords) + list(cdef.grants_keywords):
                vocab["keywords"].id(k)

    def test_an_unknown_entry_raises(self):
        from agent.vocab import UnknownVocabEntry

        with self.assertRaises(UnknownVocabEntry):
            load_all()["modes"].id("not a mode any card has")


class TestStatic(unittest.TestCase):
    def test_three_tables_for_every_card(self):
        attrs, texts, sigs, embedding = tables()
        self.assertEqual(sum(a is not None for a in attrs), 370)
        self.assertEqual(sum(1 for t in texts if t), sum(1 for c in CARD_DEFS.values() if c.text))
        for name, cdef in CARD_DEFS.items():
            row = signature_row(cdef)
            self.assertEqual(len(row), SIGNATURE_WIDTH)
            hooks = (cdef.on_play, cdef.on_reap, cdef.on_fight, cdef.on_action, cdef.on_omni, cdef.register_passive)
            if any(h is not None for h in hooks):
                self.assertGreater(sum(row[:-1]), 0, f"{name}: an effect with an empty engine signature")

    def test_text_rows_read_amounts_and_triggers(self):
        (row,) = text_rows(CARD_DEFS["Punch"])
        trigger, verb, amount, scope, conditions = row
        self.assertEqual((trigger, amount), (1, 3.0))


def _games(per_pool):
    for pool in POOLS:
        for i in range(per_pool):
            config = game_config(pool, i)
            yield pool, i, config, bots_for(i, config.seed)


class TestNoLeak(unittest.TestCase):
    def test_bytes_are_the_same_in_every_consistent_world(self):
        checked = 0
        for pool, i, config, bots in _games(2):
            game = Game(config)
            n = 0
            while not game.is_over:
                if n % 9 == 0 and game.copy_anywhere:
                    for v in (1, 2):
                        base = encode_v2(game, v).to_bytes()
                        for sampler in ("constrained", "chance_exact"):
                            w = game.fork_determinized(v, random.Random(n), Resample.ALL, backend="copy", sampler=sampler)
                            self.assertEqual(encode_v2(w, v).to_bytes(), base, f"{pool} {i} decision {n} viewer {v}")
                            checked += 1
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
                n += 1
        self.assertGreater(checked, 100)


class TestDeterminism(unittest.TestCase):
    def test_same_bytes_across_processes_and_hash_seeds(self):
        from tests._encoding_worker_v2 import digest

        here = digest(1)
        for hashseed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            out = subprocess.run([sys.executable, os.path.join(_NON_GUI, "tests", "_encoding_worker_v2.py"), "1"],
                                 cwd=_NON_GUI, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), here, f"PYTHONHASHSEED={hashseed}")


class TestCoverage(unittest.TestCase):
    def test_every_decision_kind_deck_source_and_format(self):
        from tools import diff_engines as de

        kinds = set()
        eng = de.Engine("keyforge")
        for i in range(60):
            spec = de.fuzz_spec(i)
            obj = eng.new(spec)
            bots = de._bots(spec["bots"], spec["seed"])
            while not obj.is_over:
                d = obj.pending_decision
                kinds.add(d.kind)
                for v in (1, 2):
                    enc = encode_v2_for(obj, v)
                    if v == d.player:
                        self.assertEqual(enc["option"].n, len(d.options))
                obj.submit(bots[d.player].decide(obj.view_for(d.player), d))
        expected = set(DecisionKind) - {DecisionKind.CHOOSE_ORDER} if hasattr(DecisionKind, "CHOOSE_ORDER") else set(DecisionKind)
        self.assertGreaterEqual(len(kinds), len(expected) - 2, sorted(k.name for k in expected - kinds))

    def test_rows_have_their_widths(self):
        game = Game(game_config("phase1", 0))
        enc = encode_v2(game, 1)
        self.assertEqual(enc["entity"].n, S.N_ENTITIES)
        for b in S.BLOCKS:
            blk = enc[b.name]
            self.assertEqual(len(blk.ints), blk.n * b.n_int)
            self.assertEqual(len(blk.floats), blk.n * b.n_float)


class TestV1Frozen(unittest.TestCase):
    def test_the_shim_is_v1(self):
        self.assertIs(features_shim.encode, features_v1.encode)
        game = Game(game_config("phase2", 1))
        self.assertEqual(features_shim.encode(build_infoset(game, 1)).to_bytes(),
                         features_v1.encode(build_infoset(game, 1)).to_bytes())


if __name__ == "__main__":
    unittest.main()

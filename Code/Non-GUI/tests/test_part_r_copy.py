"""Agent Observation Plan, Part R, R6: copy at any decision.

In compiled execution `Game.copy()` is valid at every decision, mid-effect
included: the resolution stack's frames are copied with their locals
remapped into the copy (keyforge/copying.py). A copy must equal a replay
fork when it is taken, and keep equalling the original when both are given
the same choices; neither may change the other.
"""

import random
import unittest

from keyforge import vm
from keyforge.config import GameConfig
from keyforge.enums import BOUNDARY_KINDS, DecisionKind, Resample
from keyforge.game import Game
from tools import diff_engines as de

E = de.Engine("keyforge@compiled")


def _play_with_copies(spec, rate, rng):
    """Plays `spec` compiled, copying at a random `rate` of decisions; checks
    each copy against a replay fork, then drives every copy alongside the
    original. Returns {decision kind: copies taken}."""
    g = E.new(spec)
    bots = de._bots(spec["bots"], spec["seed"])
    copies, kinds = [], {}
    while not g.is_over:
        d = g.pending_decision
        if rng.random() < rate:
            c = g.copy()
            f = g.fork_by_replay()
            assert E.state(c) == E.state(f), (spec["name"], len(g.choice_record), d.kind)
            assert E.decision(c.pending_decision) == E.decision(f.pending_decision)
            copies.append(c)
            kinds[d.kind] = kinds.get(d.kind, 0) + 1
        g.submit(bots[d.player].decide(g.view_for(d.player), d))
        enc = g.choice_record[-1]
        for c in copies:
            c.submit_index(enc)
            assert E.decision(c.pending_decision) == E.decision(g.pending_decision)
    final = E.state(g)
    for c in copies:
        assert E.state(c) == final, spec["name"]
    return kinds


class TestCopyAnywhere(unittest.TestCase):
    def test_copies_at_every_kind_of_decision_equal_replay_and_continue_identically(self):
        rng = random.Random(11)
        kinds = {}
        for i in range(80):
            spec = de.fuzz_spec(i, 21)
            if spec["format"] != "game":
                continue
            for k, n in _play_with_copies(spec, 0.06, rng).items():
                kinds[k] = kinds.get(k, 0) + n
        mid = {k for k in kinds if k not in BOUNDARY_KINDS}
        self.assertIn(DecisionKind.CHOOSE_CARDS, mid)
        self.assertIn(DecisionKind.CHOOSE_FLANK, mid)
        self.assertGreaterEqual(len(mid), 4, kinds)

    def test_a_copy_and_its_original_are_independent(self):
        g = Game(GameConfig(seed=4, max_turns=80), execution="compiled")
        bots = de._bots("heuristic", 4)
        while g.pending_decision.kind in BOUNDARY_KINDS or len(g.choice_record) < 20:
            d = g.pending_decision
            g.submit(bots[d.player].decide(g.view_for(d.player), d))
        before = g.state_hash()
        c = g.copy()
        cc = c.copy()  # a copy of a copy
        bots_c = de._bots("random", 9)
        for _ in range(40):
            if c.is_over:
                break
            d = c.pending_decision
            c.submit(bots_c[d.player].decide(c.view_for(d.player), d))
        self.assertEqual(g.state_hash(), before)
        self.assertEqual(cc.state_hash(), before)
        for a, b in ((g, c), (g, cc)):
            ids = {id(x) for x in a._cards_by_id.values()}
            self.assertFalse(ids & {id(x) for x in b._cards_by_id.values()})

    def test_frames_point_into_the_copy(self):
        """Every card, player and game a copied frame holds is the copy's."""
        g = Game(GameConfig(seed=8, max_turns=80), execution="compiled")
        bots = de._bots("heuristic", 8)
        while g.pending_decision.kind in BOUNDARY_KINDS or len(g.choice_record) < 30:
            d = g.pending_decision
            g.submit(bots[d.player].decide(g.view_for(d.player), d))
        c = g.copy()
        mine = {id(x) for x in c._cards_by_id.values()} | {id(p) for p in c.players.values()} | {id(c)}
        theirs = {id(x) for x in g._cards_by_id.values()} | {id(p) for p in g.players.values()} | {id(g)}

        def walk(x, seen):
            if id(x) in seen:
                return
            seen.add(id(x))
            self.assertNotIn(id(x), theirs)
            if isinstance(x, (list, tuple, set, frozenset)):
                for v in x:
                    walk(v, seen)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v, seen)

        held = 0
        for fr in c._driver.machine.stack:
            walk(fr.L, set())
            held += sum(1 for v in fr.L if id(v) in mine)
        self.assertGreater(held, 0)

    def test_determinized_forks_copy_off_a_boundary(self):
        g = Game(GameConfig(seed=12, max_turns=80), execution="compiled")
        bots = de._bots("heuristic", 12)
        while g.pending_decision.kind in BOUNDARY_KINDS or len(g.choice_record) < 25:
            d = g.pending_decision
            g.submit(bots[d.player].decide(g.view_for(d.player), d))
        self.assertTrue(g.copy_anywhere)
        a = g.fork_determinized(1, random.Random(3), Resample.ALL, backend="copy")
        b = g.fork_determinized(1, random.Random(3), Resample.ALL, backend="replay")
        self.assertEqual(a.state_hash(), b.state_hash())

    def test_a_suspended_native_frame_cannot_be_copied(self):
        with vm.disabled():
            g = Game(GameConfig(seed=3), execution="compiled")
            self.assertFalse(g.copy_anywhere)
            with self.assertRaises(ValueError):
                g.copy()


if __name__ == "__main__":
    unittest.main()

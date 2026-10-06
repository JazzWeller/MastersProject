"""Agent Observation Plan, Part R, R7: the resolution stack as information.

`Game.resolution_view(viewer)` lists what is still resolving, frame by
frame. These tests hold it to the plan:

- I1 with frames: a viewer's view is the same in the true game and in every
  determinized fork for that viewer (the re-deal relabels hidden cards held
  in frames, so a world stays consistent with what it shows);
- no instance id of a card hidden from the viewer appears in it;
- it is the same after a copy and after a snapshot round trip;
- every routine has its table of remaining operations.
"""

import random
import unittest

from keyforge import vm
from keyforge.enums import BOUNDARY_KINDS, Resample
from keyforge.resolution import card_zones, visible_to
from tools import diff_engines as de

E = de.Engine("keyforge@compiled")


def _ids(view):
    out = set()

    def walk(v):
        if isinstance(v, tuple):
            if len(v) == 2 and v[0] == "card":
                out.add(v[1])
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(view)
    return out


class TestResolutionView(unittest.TestCase):
    def _positions(self, n_games, rate, seed):
        rng = random.Random(seed)
        for i in range(n_games):
            spec = de.fuzz_spec(i, seed)
            if spec["format"] != "game":
                continue
            g = E.new(spec)
            bots = de._bots(spec["bots"], spec["seed"])
            while not g.is_over:
                if rng.random() < rate:
                    yield g
                d = g.pending_decision
                g.submit(bots[d.player].decide(g.view_for(d.player), d))

    def test_invariant_under_determinization_and_never_names_a_hidden_card(self):
        n = mid = 0
        for g in self._positions(40, 0.08, 41):
            zones = card_zones(g)
            for viewer in (1, 2):
                view = g.resolution_view(viewer)
                hidden = {iid for iid, z in zones.items() if not visible_to(g, z, viewer)}
                self.assertFalse(_ids(view) & hidden)
                for s in range(2):
                    w = g.fork_determinized(viewer, random.Random(s), Resample.ALL)
                    self.assertEqual(w.resolution_view(viewer), view)
            n += 1
            mid += g.pending_decision.kind not in BOUNDARY_KINDS
        self.assertGreater(mid, 5)

    def test_same_after_copy_and_snapshot(self):
        for g in self._positions(15, 0.1, 42):
            for viewer in (1, 2):
                v = g.resolution_view(viewer)
                self.assertEqual(g.copy().resolution_view(viewer), v)
                self.assertEqual(type(g).restore(g.snapshot()).resolution_view(viewer), v)

    def test_entries_name_the_ability_and_what_remains(self):
        kinds = set()
        for g in self._positions(30, 0.2, 43):
            for e in g.resolution_view(1):
                kinds.add(e["kind"])
                self.assertIn(e["routine"], vm.ROUTINES_BY_ID)
                self.assertIsNotNone(e["site"])
        self.assertTrue({"turn", "kernel", "play"} <= kinds, kinds)

    def test_every_suspension_point_has_its_tables(self):
        vm.load_compiled()
        for r in vm.ROUTINES_BY_ID.values():
            self.assertEqual(set(r.site_of_pc), set(r.line_of_pc), r.rid)
            self.assertEqual(set(r.ops_of_pc), set(r.line_of_pc), r.rid)


if __name__ == "__main__":
    unittest.main()

"""Agent Observation Plan, Part R, R6: snapshot and restore.

`Game.snapshot()` serializes a game at any decision (compiled execution) into
canonical, versioned, pickle-free bytes; `Game.restore()` gives back a game
that equals it and continues identically. The same state always gives the
same bytes.
"""

import json
import random
import unittest

from keyforge.config import GameConfig
from keyforge.enums import BOUNDARY_KINDS
from keyforge.game import Game
from keyforge.snapshot import SnapshotError
from tools import diff_engines as de

E = de.Engine("keyforge@compiled")


class TestSnapshot(unittest.TestCase):
    def test_round_trips_at_every_kind_of_decision(self):
        rng = random.Random(3)
        kinds = set()
        for i in range(30):
            spec = de.fuzz_spec(i, 31)
            if spec["format"] != "game":
                continue
            g = E.new(spec)
            bots = de._bots(spec["bots"], spec["seed"])
            restored = []
            while not g.is_over:
                d = g.pending_decision
                if rng.random() < 0.04:
                    data = g.snapshot()
                    r = Game.restore(data)
                    self.assertEqual(E.state(r), E.state(g))
                    self.assertEqual(r.snapshot(), data)  # canonical
                    restored.append(r)
                    kinds.add(d.kind)
                g.submit(bots[d.player].decide(g.view_for(d.player), d))
                for r in restored:
                    r.submit_index(g.choice_record[-1])
            for r in restored:
                self.assertEqual(E.state(r), E.state(g))
        self.assertTrue(kinds - set(BOUNDARY_KINDS), "no mid-resolution snapshot was taken")

    def test_the_same_state_gives_the_same_bytes(self):
        a = Game(GameConfig(seed=5, max_turns=80), execution="compiled")
        b = Game(GameConfig(seed=5, max_turns=80), execution="compiled")
        bots_a, bots_b = de._bots("heuristic", 5), de._bots("heuristic", 5)
        for g, bots in ((a, bots_a), (b, bots_b)):
            for _ in range(50):
                d = g.pending_decision
                g.submit(bots[d.player].decide(g.view_for(d.player), d))
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertEqual(a.snapshot(), a.copy().snapshot())

    def test_native_games_snapshot_only_at_a_boundary(self):
        g = Game(GameConfig(seed=1, max_turns=80), execution="native")
        bots = de._bots("random", 1)
        while not g.is_over:
            d = g.pending_decision
            if d.kind in BOUNDARY_KINDS and len(g.choice_record) > 10:
                r = Game.restore(g.snapshot())
                self.assertEqual(r.state_hash(), g.state_hash())
                break
            g.submit(bots[d.player].decide(g.view_for(d.player), d))
        while g.pending_decision.kind in BOUNDARY_KINDS:
            d = g.pending_decision
            g.submit(bots[d.player].decide(g.view_for(d.player), d))
        with self.assertRaises(SnapshotError):
            g.snapshot()

    def test_a_snapshot_from_another_engine_is_refused(self):
        data = json.loads(Game(GameConfig(seed=2), execution="compiled").snapshot())
        data["rules_hash"] = "0" * 64
        with self.assertRaises(SnapshotError):
            Game.restore(json.dumps(data).encode())


if __name__ == "__main__":
    unittest.main()

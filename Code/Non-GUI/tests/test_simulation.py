import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.simulate import check_invariants, run_one, run_one_match

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import random_deck


class TestSimulation(unittest.TestCase):
    def test_random_vs_random_all_pairings(self):
        for p1_deck, p2_deck in (("fignor", "igor"), ("igor", "fignor"), ("fignor", "fignor"), ("igor", "igor")):
            for seed in range(5):
                result, turns = run_one(p1_deck, p2_deck, None, seed, 150, True)
                self.assertIn(result["reason"], ("3 keys", "turn limit"))
                self.assertGreater(turns, 0)


class TestPhase3Invariants(unittest.TestCase):
    def test_invariants_hold_across_random_7_house_decks(self):
        # fignor/igor (the other tests here) are Phase 1/2, Dis/Logos/
        # Shadows only -- random 7-house decks are needed to actually
        # exercise armor, hazardous, assault, and captured/placed Æmber
        # (Phase 3's additions to check_invariants).
        deck_rng = random.Random(3)
        for seed in range(40):
            p1_deck = random_deck(deck_rng, name="p1")
            p2_deck = random_deck(deck_rng, name="p2")
            result, turns = run_one(p1_deck, p2_deck, None, seed, 60, True)
            self.assertIn(result["reason"], ("3 keys", "turn limit"))
            self.assertGreater(turns, 0)


class TestCardCoverage(unittest.TestCase):
    def test_every_card_gets_played_or_used_across_a_long_random_run(self):
        """Milestone F: across enough random-house games, every one of the
        370 cards should turn up at least once (played, reaped, fought
        with, or had its Action/Omni used). Deliberately the slowest test
        in this file (~10-15s) -- it's the one thing that actually proves
        out Phase 3's whole card pool end to end, not just a sample of it."""
        deck_rng = random.Random(11)
        usage = set()
        for i in range(1800):
            p1_deck = random_deck(deck_rng, name="p1")
            p2_deck = random_deck(deck_rng, name="p2")
            run_one(p1_deck, p2_deck, None, i, 60, False, usage)
        missing = sorted(set(CARD_DEFS) - usage)
        self.assertEqual(missing, [], f"{len(missing)} card(s) never played/used: {missing}")


class TestMatchSimulation(unittest.TestCase):
    def test_random_vs_random_all_formats_with_invariants(self):
        for fmt in ("archon", "reversal", "adaptive"):
            for seed in range(8):
                match = run_one_match("fignor", "igor", fmt, None, seed, 150, True)
                self.assertTrue(match.is_over)
                self.assertIn(match.result["winner"], (1, 2, None))
                self.assertGreaterEqual(len(match.games), 1)
                if fmt != "adaptive":
                    self.assertEqual(len(match.games), 1)
                else:
                    self.assertIn(len(match.games), (2, 3))
                    if len(match.games) == 3:
                        self.assertIsNotNone(match.bid)
                        self.assertGreaterEqual(match.bid.amount, 0)
                        self.assertLessEqual(match.bid.amount, 24)


if __name__ == "__main__":
    unittest.main()

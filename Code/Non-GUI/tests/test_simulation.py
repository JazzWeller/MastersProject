import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.simulate import check_invariants, run_one, run_one_match


class TestSimulation(unittest.TestCase):
    def test_random_vs_random_all_pairings(self):
        for p1_deck, p2_deck in (("fignor", "igor"), ("igor", "fignor"), ("fignor", "fignor"), ("igor", "igor")):
            for seed in range(5):
                result, turns = run_one(p1_deck, p2_deck, None, seed, 150, True)
                self.assertIn(result["reason"], ("3 keys", "turn limit"))
                self.assertGreater(turns, 0)


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

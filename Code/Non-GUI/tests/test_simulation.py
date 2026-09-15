import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.simulate import check_invariants, run_one


class TestSimulation(unittest.TestCase):
    def test_random_vs_random_all_pairings(self):
        for p1_deck, p2_deck in (("fignor", "igor"), ("igor", "fignor"), ("fignor", "fignor"), ("igor", "igor")):
            for seed in range(5):
                result, turns = run_one(p1_deck, p2_deck, None, seed, 150, True)
                self.assertIn(result["reason"], ("3 keys", "turn limit"))
                self.assertGreater(turns, 0)


if __name__ == "__main__":
    unittest.main()

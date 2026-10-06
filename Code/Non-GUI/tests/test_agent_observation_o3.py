"""Agent Observation Plan, Milestone O3: knowledge-consistent
determinization and exact priors.

The fuzz-scale checks are tools/o3_acceptance.py; these run its checks on a
few games, compare `chance_exact` with a brute-force enumeration of the
generative process (tests/_chance_brute.py), test that `constrained` is
uniform, and check the search's plumbing.
"""

import itertools
import math
import random
import unittest

from keyforge.determinize import ChanceFilter, _assign, split_constrained
from keyforge.enums import Resample
from tests import _chance_brute as brute
from tools import o3_acceptance
from tools.o1_acceptance import POOLS, game_config


class TestChanceExact(unittest.TestCase):
    SCENARIOS = [
        [("deal", 4), ("draw",), ("draw",)],
        [("deal", 5), ("draw",), ("draw",), ("play", 0), ("archive",)],
        [("deal", 5), ("draw",), ("draw",), ("draw",), ("archive",), ("play", 1)],
        [("deal", 5), ("draw",), ("draw",), ("play", 0), ("play", 0), ("reshuffle",), ("draw",)],
        [("deal", 5), ("draw",), ("draw",), ("play", 0), ("put_on_top", 0), ("draw",), ("archive",)],
        [("deal", 5), ("reveal_top",), ("draw",), ("draw",), ("archive",)],
        [("deal", 5), ("draw",), ("draw",), ("play", 0), ("return", 0), ("archive",), ("draw",)],
        [("deal", 6), ("draw",), ("draw",), ("draw",), ("play", 0), ("play", 0), ("reshuffle",), ("draw",),
         ("draw",), ("archive",)],
    ]

    def test_probabilities_equal_the_enumeration(self):
        for sc in self.SCENARIOS:
            for seed in range(4):
                with self.subTest(scenario=sc, seed=seed):
                    items, _ = brute.items(sc, random.Random(seed))
                    expected = brute.posterior(sc, items)
                    f = ChanceFilter(1, {1: [], 2: list(range(sc[0][1]))})
                    f.update(items)
                    for x, (ph, pa, pd, pn) in expected.items():
                        got = f.p_zone(x)
                        self.assertIsNotNone(got)
                        for a, b in zip(got[:3], (ph, pa, pd)):
                            self.assertAlmostEqual(a, float(b), places=12)
                        self.assertAlmostEqual(f.p_next_draw(x), float(pn), places=12)


def _chi2_p(stat: float, dof: int) -> float:
    """Upper-tail p of a chi-square (Wilson-Hilferty)."""
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


class TestConstrainedIsUniform(unittest.TestCase):
    def test_chi_square_over_an_enumerable_scenario(self):
        # cards a0 a1 may be in hand or deck; b0 b1 anywhere; c0 in archive or
        # deck; zones: hand 2, archive 1, deck 2
        classes = [("a", 2, frozenset({0, 2})), ("b", 2, frozenset({0, 1, 2})), ("c", 1, frozenset({1, 2}))]
        caps = (2, 1, 2)
        cards = [f"{name}{k}" for name, size, _ in classes for k in range(size)]
        allowed = {f"{name}{k}": z for name, size, z in classes for k in range(size)}
        worlds = []
        for zs in itertools.product(range(3), repeat=len(cards)):
            if all(z in allowed[c] for c, z in zip(cards, zs)) and all(zs.count(k) == caps[k] for k in range(3)):
                worlds.append(zs)
        self.assertGreater(len(worlds), 5)
        index = {w: i for i, w in enumerate(worlds)}
        counts = [0] * len(worlds)
        rng = random.Random(1)
        n = 30000
        for _ in range(n):
            splits = split_constrained([(size, z) for _, size, z in classes], caps, rng)
            where = {}
            for (name, size, _), s in zip(classes, splits):
                for z, group in enumerate(_assign([f"{name}{k}" for k in range(size)], s, rng)):
                    for c in group:
                        where[c] = z
            counts[index[tuple(where[c] for c in cards)]] += 1
        expected = n / len(worlds)
        stat = sum((c - expected) ** 2 / expected for c in counts)
        self.assertGreater(_chi2_p(stat, len(worlds) - 1), 1e-3, (len(worlds), counts))


class TestAcceptanceSample(unittest.TestCase):
    def test_worlds_respect_what_the_viewer_knows(self):
        for pool in POOLS:
            with self.subTest(pool=pool):
                self.assertEqual(o3_acceptance.check_game(game_config(pool, 0), 0, every=20), [])

    def test_chance_exact_beats_the_uniform_prior(self):
        n = ll_c = ll_u = 0
        for i in range(3):
            dn, dc, du = o3_acceptance.calibration_game(i)
            n, ll_c, ll_u = n + dn, ll_c + dc, ll_u + du
        self.assertLess(ll_c / n, ll_u / n)


class TestSearchPlumbing(unittest.TestCase):
    def test_the_sampler_reaches_the_capability(self):
        from agent.search.core import Search, SearchSettings

        seen = {}

        class Cap:
            def fork_determinized(self, rng, resample, *, backend, sampler="uniform"):
                seen["sampler"] = sampler
                return None

        search = Search.__new__(Search)
        search.settings = SearchSettings(determinization="chance_exact")
        search._world(Cap(), random.Random(0))
        self.assertEqual(seen["sampler"], "chance_exact")

    def test_the_config_default_is_chance_exact(self):
        from agent.config import DEFAULTS

        self.assertEqual(DEFAULTS["search"]["determinization"], "chance_exact")


if __name__ == "__main__":
    unittest.main()

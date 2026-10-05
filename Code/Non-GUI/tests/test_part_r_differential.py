"""Agent Observation Plan, Part R: the permanent differential test (I8).

The same source runs natively and compiled; after every decision both
engines must show the same pending decision, log, RNG counters and choice
record, and periodically the same whole canonical state
(tools/diff_engines.py). The engine frozen before Part R (`keyforge_ref`)
checks the source itself the same way.

The full corpora (100,000 fuzz games) run from the command line; this keeps
a slice of every one in the suite.
"""

import unittest

from tools import diff_engines as de


class TestDifferential(unittest.TestCase):
    def _pair(self, a, b, specs):
        ea, eb = de.Engine(a), de.Engine(b)
        n = 0
        for spec in specs:
            with self.subTest(game=spec.get("name")):
                n += de.play(ea, eb, spec, hash_every=10)
        self.assertGreater(n, 0)

    def test_compiled_equals_native_on_the_golden_corpus(self):
        self._pair("keyforge@compiled", "keyforge@native", de.golden_specs())

    def test_compiled_equals_native_on_a_fuzz_slice(self):
        # every bot, deck source and format recurs within 60 consecutive specs
        self._pair("keyforge@compiled", "keyforge@native", [de.fuzz_spec(i, 7) for i in range(60)])

    def test_native_equals_the_frozen_reference(self):
        self._pair("keyforge@native", "keyforge_ref", de.golden_specs() + [de.fuzz_spec(i, 8) for i in range(30)])

    def test_the_harness_catches_a_divergence(self):
        """A one-Æmber change to one rule in one engine is caught."""
        ea, eb = de.Engine("keyforge@native"), de.Engine("keyforge_ref")
        steps_ref = eb.game_mod.steps
        original = steps_ref.gain

        def gain_plus_one(game, player, amount):
            return original(game, player, amount + 1)

        steps_ref.gain = gain_plus_one
        try:
            with self.assertRaises(de.Divergence):
                for i in range(20):
                    de.play(ea, eb, de.fuzz_spec(i, 9), hash_every=1)
        finally:
            steps_ref.gain = original


if __name__ == "__main__":
    unittest.main()

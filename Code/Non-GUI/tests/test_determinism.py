"""Cross-process determinism regression test (Agent Interface Plan,
Milestone A). A replay record is a valid unit of transfer between worker
processes only if the same seed produces byte-identical play regardless of
`PYTHONHASHSEED` -- otherwise a future `for h in some_set:` in a card effect
could quietly make outcomes depend on it. Verified once by hand (30 seeded
games, byte-identical under PYTHONHASHSEED 0/1/2/12345); this keeps it true.
"""

import os
import subprocess
import sys
import unittest

_NON_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_determinism_worker.py")
_HASHSEED_VARIANTS = ("0", "1", "2", "12345")


class TestCrossProcessDeterminism(unittest.TestCase):
    def test_same_games_digest_identically_across_hashseed_variants(self):
        digests = {}
        for hashseed in _HASHSEED_VARIANTS:
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            result = subprocess.run(
                [sys.executable, _WORKER],
                env=env, cwd=_NON_GUI_DIR, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, f"worker failed under PYTHONHASHSEED={hashseed}:\n{result.stderr}")
            digests[hashseed] = result.stdout.strip()
        unique = set(digests.values())
        self.assertEqual(len(unique), 1, f"digests differ across PYTHONHASHSEED variants: {digests}")


if __name__ == "__main__":
    unittest.main()

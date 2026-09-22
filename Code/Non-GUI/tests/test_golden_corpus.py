"""The golden seed corpus safety net (Code/AGENT_INTERFACE_PLAN.md,
Milestone L): every entry in golden_corpus.json (built by
tools/build_golden_corpus.py) must replay to the exact same outcome, turn
count, and state_hash it was recorded with. A pure-performance change to
the engine must leave every single one byte-identical -- this is the test
that says so; if it fails, either the change wasn't pure, or the corpus
needs deliberately regenerating (tools/build_golden_corpus.py) because the
change was meant to alter play.
"""

import json
import os
import unittest

from keyforge.config import GameConfig
from keyforge.game import Game
from keyforge.replay import decode_choice

_CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_corpus.json")

with open(_CORPUS_PATH, "r", encoding="utf-8") as _f:
    _CORPUS = json.load(_f)


def _replay_raw(config: GameConfig, record):
    """Like keyforge.replay.replay, but for a plain GameConfig (no
    version-stamp dict) -- the golden corpus intentionally bypasses that
    stamp; see tools/build_golden_corpus.py's docstring for why."""
    game = Game(config)
    for encoded in record:
        game.submit(decode_choice(game.pending_decision, encoded))
    return game


class TestGoldenCorpus(unittest.TestCase):
    def test_corpus_is_not_empty(self):
        self.assertGreater(len(_CORPUS), 0)

    def test_every_entry_replays_byte_identically(self):
        for entry in _CORPUS:
            with self.subTest(name=entry["name"]):
                config = GameConfig(
                    decks=tuple(entry["decks"]), seed=entry["seed"], max_turns=entry["max_turns"],
                    first_player=entry["first_player"], starting_chains=entry["starting_chains"],
                    setup_script=entry["setup_script"],
                )
                game = _replay_raw(config, entry["choice_record"])
                self.assertTrue(game.is_over, f"{entry['name']}: game did not finish")
                self.assertEqual(game.turn_number, entry["turns"], f"{entry['name']}: turn count changed")
                self.assertEqual(game.result, entry["outcome"], f"{entry['name']}: outcome changed")
                self.assertEqual(game.state_hash(), entry["state_hash"], f"{entry['name']}: state_hash changed")


if __name__ == "__main__":
    unittest.main()

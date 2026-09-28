"""Agent Training Plan, Milestone M11 (Code/AGENT_TRAINING_PLAN.md): the
extension story is mostly *verification that earlier choices held*.

- The search agent passes the conformance checks on random legal decks,
  alliance decks, and the Reversal and Adaptive match formats (the network
  agent's version is in test_agent_training_ml.py).
- Mirror augmentation is exact for Fignor/Igor and is switched off for any
  pool containing a card that breaks the left/right symmetry.
- Held-out deck sets are disjoint by construction.
- No FEATURE_VERSION bump was needed for any of it
  (test_agent_training_m1.TestReservedSlots pins the widths).
"""

import random
import unittest

from agent import spec
from agent.agents.search_agent import SearchAgent
from agent.augment import MIRROR_UNSAFE_CARDS, card_breaks_mirror, mirror_safe
from agent.search.core import SearchSettings
from bots.heuristic_bot import HeuristicBot
from keyforge.cards.decks import Deck, build_alliance_deck, random_deck, resolve_deck
from keyforge.config import GameConfig
from keyforge.enums import House, PrivilegeLevel
from keyforge.match import MatchConfig
from sim.bc_corpus import fixed_deck_set
from sim.driver import run_games


def _search_agent(seed):
    return SearchAgent("within_turn", settings=SearchSettings(simulations=6), seed=seed)


class TestConformanceAcrossPoolsAndFormats(unittest.TestCase):
    def _check(self, make_config, n=2):
        results = run_games(
            n, make_config, lambda i: {1: _search_agent(i), 2: HeuristicBot(seed=i)},
            privilege={1: PrivilegeLevel.SEARCH}, concurrency=1,
        )
        for r in results:
            self.assertIsNone(r.forfeit, r.forfeit)

    def test_random_legal_decks(self):
        rng = random.Random(3)
        decks = [(random_deck(rng, "R1"), random_deck(rng, "R2")) for _ in range(2)]
        self._check(lambda i: GameConfig(decks=decks[i], seed=i, max_turns=30))

    def test_alliance_decks(self):
        alliance = build_alliance_deck("A", {House.BROBNAR: "stonewall", House.MARS: "starfall", House.DIS: "vigil"})
        self._check(lambda i: GameConfig(decks=(alliance, "igor"), seed=i, max_turns=30))

    def test_reversal_and_adaptive_matches(self):
        for fmt in ("reversal", "adaptive"):
            with self.subTest(format=fmt):
                self._check(lambda i, fmt=fmt: MatchConfig(format=fmt, decks=("fignor", "igor"), seed=i, max_turns=30), n=1)


class TestMirrorAugmentation(unittest.TestCase):
    def test_exact_for_the_phase_1_1_decks(self):
        self.assertTrue(mirror_safe(["fignor", "igor"]))

    def test_every_left_forcing_card_breaks_it(self):
        for name in MIRROR_UNSAFE_CARDS:
            self.assertTrue(card_breaks_mirror(name), name)
        igor = resolve_deck("igor")
        pods = {h: list(v) for h, v in igor.pods.items()}
        pods[House.LOGOS][0] = "Spangler Box"  # a Logos artifact that returns creatures to the left flank
        self.assertFalse(mirror_safe([Deck(name="IgorSB", pods=pods), "fignor"]))


class TestHeldOutDecks(unittest.TestCase):
    def test_training_and_evaluation_deck_sets_are_disjoint(self):
        train = fixed_deck_set(64, deck_seed=1)
        held = fixed_deck_set(16, deck_seed=2)
        train_keys = {tuple(sorted(d.all_card_names())) for d in train}
        self.assertFalse(any(tuple(sorted(d.all_card_names())) in train_keys for d in held))
        self.assertEqual([d.pods for d in fixed_deck_set(4, 1)], [d.pods for d in train[:4]])


class TestNoVersionBump(unittest.TestCase):
    def test_the_whole_registry_fits_the_version_1_layout(self):
        self.assertEqual(spec.FEATURE_VERSION, 1)
        self.assertGreaterEqual(spec.VOCAB_CAPACITY, 370)


if __name__ == "__main__":
    unittest.main()

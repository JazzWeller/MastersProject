"""Registry test for the full 159-card CotA pool (Code/PHASE_2_PLAN.md
Milestone A): every pool card has a definition, real art, and canonical
text, and the pool's house counts match Call of the Archons exactly."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card_data import CARD_DEFS, POOL_JSON_PATH
from keyforge.enums import CardType, House

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_EXPECTED_HOUSE_COUNTS = {"Dis": 54, "Logos": 53, "Shadows": 52}


class TestCardRegistry(unittest.TestCase):
    def setUp(self):
        with open(POOL_JSON_PATH, "r", encoding="utf-8") as f:
            self.pool = json.load(f)

    def test_pool_has_exactly_159_cards(self):
        self.assertEqual(len(self.pool), 159)
        self.assertEqual(len(CARD_DEFS), 159)

    def test_house_counts_match_call_of_the_archons(self):
        counts = {}
        for entry in self.pool:
            counts[entry["house"]] = counts.get(entry["house"], 0) + 1
        self.assertEqual(counts, _EXPECTED_HOUSE_COUNTS)

    def test_no_duplicate_names(self):
        names = [e["name"] for e in self.pool]
        self.assertEqual(len(names), len(set(names)), "duplicate card names in the pool")

    def test_every_pool_card_has_a_definition(self):
        for entry in self.pool:
            self.assertIn(entry["name"], CARD_DEFS, f"{entry['name']} has no CardDef")

    def test_every_definition_has_a_real_house_and_type(self):
        for name, cd in CARD_DEFS.items():
            self.assertIsInstance(cd.house, House, name)
            self.assertIsInstance(cd.type, CardType, name)

    def test_every_definition_has_canonical_text(self):
        for name, cd in CARD_DEFS.items():
            self.assertTrue(cd.text, f"{name} has no canonical text")

    def test_every_definition_has_art_that_exists_on_disk(self):
        for name, cd in CARD_DEFS.items():
            self.assertIsNotNone(cd.image, name)
            full = os.path.join(_PROJECT_DIR, cd.image)
            self.assertTrue(os.path.isfile(full), f"{name}: art file not found at {full}")

    def test_no_pool_card_has_armor(self):
        # Confirmed in PHASE_2_CARD_POOL.md's notes: no CotA card in Dis,
        # Logos or Shadows has printed armor.
        for name, cd in CARD_DEFS.items():
            self.assertEqual(cd.armor, 0, name)

    def test_creatures_have_positive_power_and_non_creatures_dont(self):
        for name, cd in CARD_DEFS.items():
            if cd.type == CardType.CREATURE:
                self.assertGreater(cd.power, 0, name)
            else:
                self.assertEqual(cd.power, 0, name)

    def test_ids_are_unique(self):
        ids = [cd.id for cd in CARD_DEFS.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_49_phase1_cards_are_flagged_and_present(self):
        phase1_names = {e["name"] for e in self.pool if e["phase1"]}
        self.assertEqual(len(phase1_names), 49)
        for name in phase1_names:
            self.assertIn(name, CARD_DEFS)


if __name__ == "__main__":
    unittest.main()

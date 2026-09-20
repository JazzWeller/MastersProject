"""Registry test for the full 370-card CotA pool, all 7 houses
(Code/PHASE_2_PLAN.md Milestone A, Code/PHASE_3_PLAN.md Milestone A): every
pool card has a definition, real art, and canonical text, the pool's house
counts match Call of the Archons exactly, and armor is parsed correctly for
the 17 printed-armor creatures (Brobnar 1, Mars 3, Sanctum 13)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card_data import CARD_DEFS, POOL_JSON_PATH
from keyforge.enums import CardType, House

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_EXPECTED_HOUSE_COUNTS = {
    "Brobnar": 52,
    "Dis": 54,
    "Logos": 53,
    "Mars": 52,
    "Sanctum": 55,
    "Shadows": 52,
    "Untamed": 52,
}
# Every CotA creature with printed armor (Code/PHASE_3_CARD_POOL.md), cross-
# checked against keyteki's CotA.json. No Dis/Logos/Shadows card has armor.
_EXPECTED_ARMOR = {
    "Firespitter": 1,
    "Grabber Jammer": 1,
    "Tunk": 1,
    "Yxilx Dominator": 1,
    "Bulwark": 2,
    "Champion Anaphiel": 1,
    "Champion Tabris": 2,
    "Francus": 1,
    "Gatekeeper": 1,
    "Lady Maxena": 2,
    "Lord Golgotha": 2,
    "Raiding Knight": 2,
    "Sanctum Guardian": 1,
    "Sequis": 2,
    "Sergeant Zakiel": 1,
    "Staunch Knight": 2,
    "The Vaultkeeper": 1,
}


class TestCardRegistry(unittest.TestCase):
    def setUp(self):
        with open(POOL_JSON_PATH, "r", encoding="utf-8") as f:
            self.pool = json.load(f)

    def test_pool_has_exactly_370_cards(self):
        self.assertEqual(len(self.pool), 370)
        self.assertEqual(len(CARD_DEFS), 370)

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

    def test_armor_is_parsed_correctly(self):
        for name, cd in CARD_DEFS.items():
            self.assertEqual(cd.armor, _EXPECTED_ARMOR.get(name, 0), name)
        self.assertEqual(
            sum(1 for cd in CARD_DEFS.values() if cd.armor),
            len(_EXPECTED_ARMOR),
            "armored creature count changed -- update _EXPECTED_ARMOR",
        )

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

"""Card sprite chips for the Milestone B/C mechanics (taunt, poison,
hazardous, versatile, stunned, power counters, stored Æmber, under-cards,
owner != controller) must render without crashing."""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui.assets import AssetCache
from gui.snapshot import CardState
from gui.sprites.card_sprite import CardSprite


def _state(**overrides) -> CardState:
    base = dict(
        iid=1, name="Test Card", house="Shadows", type="Creature", image=None,
        owner=1, controller=1, zone="play_creature", index=0, zone_count=1,
        power=3, armor=0, damage=0,
    )
    base.update(overrides)
    return CardState(**base)


class TestCardSpriteChips(unittest.TestCase):
    def setUp(self):
        self.assets = AssetCache()
        self.surface = pygame.Surface((400, 400))

    def _draw(self, cs: CardState) -> None:
        sprite = CardSprite(cs.iid, self.assets)
        sprite.set_size(160, 220)
        sprite.snap_to(200, 200)
        sprite.apply_state(cs)
        sprite.draw(self.surface)  # must not raise

    def test_creature_with_every_new_keyword_chip(self):
        self._draw(_state(
            taunt=True, poison=True, hazardous=2, versatile=True, elusive=True, skirmish=True,
        ))

    def test_creature_with_power_counters_and_under_cards(self):
        self._draw(_state(power_counters=2, under_count=1, armor=1, aember_captured=1))

    def test_stunned_and_exhausted_creature(self):
        self._draw(_state(stunned=True, exhausted=True))

    def test_owner_differs_from_controller(self):
        self._draw(_state(owner=2, controller=1))

    def test_artifact_with_stored_aember_and_under_cards(self):
        self._draw(_state(zone="play_artifact", type="Artifact", aember_stored=3, under_count=1, owner=1, controller=2))

    def test_creature_with_phase_3_chips_assault_and_shield(self):
        self._draw(_state(assault=2, cannot_be_dealt_damage=True))

    def test_creature_with_partially_used_armor(self):
        self._draw(_state(armor=2, armor_used=1))


if __name__ == "__main__":
    unittest.main()

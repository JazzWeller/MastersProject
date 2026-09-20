"""Every card in the full 159-card CotA pool must resolve to a real, loadable
image file, at every size the app actually uses; the procedural card back
and icons must render without crashing."""

import unittest

import tests.helpers  # noqa: F401

from keyforge.cards.card_data import CARD_DEFS

from gui import settings as S
from gui.assets import AssetCache

SIZES = [
    (S.BOARD_CARD_W, S.BOARD_CARD_H),
    (S.HAND_CARD_W, S.HAND_CARD_H),
    (S.ZOOM_CARD_W, S.ZOOM_CARD_H),
    (S.PILE_CARD_W, S.PILE_CARD_H),
    (S.PILE_ICON_W, S.PILE_ICON_H),
]


class TestAssets(unittest.TestCase):
    def setUp(self):
        self.assets = AssetCache()

    def test_every_decklist_card_has_real_art_at_every_used_size(self):
        self.assertEqual(len(CARD_DEFS), 159)
        for name, card_def in CARD_DEFS.items():
            self.assertIsNotNone(card_def.image, name)
            # the raw file must actually exist and decode
            raw = self.assets._raw(card_def.image)
            self.assertIsNotNone(raw, f"{name}: {card_def.image} failed to load")
            for size in SIZES:
                face = self.assets.card_face(card_def.image, size)
                self.assertEqual(face.get_size(), size)

    def test_missing_art_falls_back_to_a_placeholder_without_crashing(self):
        face = self.assets.card_face("Nope/does-not-exist.png", (100, 140))
        self.assertEqual(face.get_size(), (100, 140))

    def test_card_back_renders_at_every_size(self):
        for size in SIZES:
            back = self.assets.card_back(size)
            self.assertEqual(back.get_size(), size)

    def test_icons_render(self):
        for house in ("Dis", "Logos", "Shadows"):
            img = self.assets.house_emblem(house, 32)
            self.assertEqual(img.get_size(), (32, 32))
        for forged in (True, False):
            img = self.assets.key_icon(20, forged)
            self.assertEqual(img.get_size(), (20, 20))
        self.assertEqual(self.assets.chain_icon(18).get_size(), (18, 18))
        self.assertEqual(self.assets.aember_gem(16).get_size(), (16, 16))

    def test_sound_effects_load_and_play_without_crashing(self):
        expected = {"click", "card_move", "damage", "destroy", "key_forge", "gain"}
        self.assertTrue(expected.issubset(self.assets._sounds.keys()), self.assets._sounds.keys())
        for name in expected:
            self.assets.play(name)  # must not raise, even under a dummy audio driver
        self.assets.play("does-not-exist")  # silently ignored

        self.assets.muted = True
        self.assets.play("click")  # muted: still must not raise
        self.assets.muted = False

    def test_fonts_load(self):
        cinzel = self.assets.font("cinzel", 24)
        inter = self.assets.font("inter", 14)
        self.assertGreater(cinzel.size("KEYFORGE")[0], 0)
        self.assertGreater(inter.size("test")[0], 0)


if __name__ == "__main__":
    unittest.main()

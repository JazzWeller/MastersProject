"""Alliance Builder scene tests (Code/PHASE_2_PLAN.md v2.1 Milestone F)."""

import os
import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.scenes.alliance_builder_scene import AllianceBuilderScene
from gui.scenes.deck_builder_scene import HOUSES
from gui.scenes.menu_scene import MenuScene

from keyforge.cards.decks import AllianceDeck, load_deck_json, validate_deck
from keyforge.enums import House


def _click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def _right_click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=3))


def _scene():
    app = App(window_size=(1600, 900))
    app.push(AllianceBuilderScene())
    return app, app.scenes[-1]


class TestAllianceBuilderScene(unittest.TestCase):
    def tearDown(self):
        for f in os.listdir(S.USER_DECKS_DIR):
            os.remove(os.path.join(S.USER_DECKS_DIR, f))

    def test_starts_with_a_legal_deck_from_default_sources(self):
        app, g = _scene()
        self.assertEqual(g._errors(), [])
        for house in HOUSES:
            self.assertEqual(len(g._pod(house)), 12)

    def test_picking_a_different_source_changes_that_houses_pod_only(self):
        app, g = _scene()
        before_logos = list(g._pod(House.LOGOS))
        g.picker_open = House.DIS
        options = g._source_options()
        # Pick a preset different from the current Dis source.
        current = g.house_sources[House.DIS]
        kind, ref, label = next((k, r, l) for k, r, l in options if r != current)
        row = g._picker_row_rect([o[1] for o in options].index(ref))
        _click(g, row.center)
        self.assertEqual(g.house_sources[House.DIS], ref)
        self.assertIsNone(g.picker_open)
        self.assertEqual(g._pod(House.LOGOS), before_logos)  # untouched

    def test_save_writes_an_alliance_deck_with_sources_recorded(self):
        app, g = _scene()
        g.name = "My Alliance"
        g._save()
        self.assertIsNotNone(g.saved_path)
        loaded = load_deck_json(g.saved_path)
        self.assertIsInstance(loaded, AllianceDeck)
        self.assertEqual(loaded.name, "My Alliance")
        self.assertEqual(validate_deck(loaded), [])
        self.assertEqual(set(loaded.sources.keys()), set(HOUSES))

    def test_load_restores_name_and_per_house_sources(self):
        app, g = _scene()
        g.name = "Reloadable"
        original_sources = dict(g.house_sources)
        g._save()
        path = g.saved_path

        g2 = AllianceBuilderScene()
        g2.app = app
        g2._load(path)
        self.assertEqual(g2.name, "Reloadable")
        for house in HOUSES:
            self.assertEqual(g2._pod(house), g._pod(house))

    def test_delete_removes_the_saved_file(self):
        app, g = _scene()
        g._save()
        path = g.saved_path
        self.assertTrue(os.path.exists(path))
        g._delete(path)
        self.assertFalse(os.path.exists(path))
        self.assertIsNone(g.saved_path)

    def test_right_click_a_pod_card_opens_the_inspector(self):
        app, g = _scene()
        house = HOUSES[0]
        row = g._pod_row_rects(house)[0]
        name = g._pod(house)[0]
        _right_click(g, row.center)
        self.assertIsNotNone(g.inspect_info)
        self.assertEqual(g.inspect_info.name, name)
        app.draw_scenes()  # must not crash

    def test_menu_alliance_button_opens_the_scene_and_back_returns(self):
        app = App(window_size=(1600, 900))
        app.push(MenuScene())
        menu = app.scenes[-1]
        button, _cb = menu.buttons["alliance_builder"]
        _click(menu, button.rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene", "AllianceBuilderScene"])
        builder = app.scenes[-1]
        builder.name = "From Menu Alliance"
        builder._save()
        builder._back()
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene"])
        self.assertIn(builder.saved_path, menu.decks)

    def test_deck_label_on_a_saved_alliance_deck_is_tagged(self):
        from keyforge.cards.decks import deck_label, resolve_deck

        app, g = _scene()
        g.name = "Tagged"
        g._save()
        self.assertEqual(deck_label(resolve_deck(g.saved_path)), "Tagged (Alliance)")


if __name__ == "__main__":
    unittest.main()

"""Deck Builder scene tests (Code/PHASE_2_PLAN.md Milestone E)."""

import os
import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.scenes.deck_builder_scene import DeckBuilderScene
from gui.scenes.menu_scene import MenuScene

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import CARDS_PER_POD, load_deck_json, validate_deck
from keyforge.enums import House


def _click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def _right_click(scene, pos):
    scene.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=3))


def _pick_default_houses(scene):
    """Resolve the mandatory house-picker step with the classic Dis/Logos/
    Shadows triple, so existing tests keep exercising the same houses."""
    scene._on_houses_confirmed([House.DIS, House.LOGOS, House.SHADOWS])
    scene.house_picker.is_open = False


def _scene():
    app = App(window_size=(1600, 900))
    scene = DeckBuilderScene()
    _pick_default_houses(scene)
    app.push(scene)
    return app, app.scenes[-1]


class TestDeckBuilderScene(unittest.TestCase):
    def tearDown(self):
        for f in os.listdir(S.USER_DECKS_DIR):
            os.remove(os.path.join(S.USER_DECKS_DIR, f))

    def test_clicking_a_card_adds_it_to_the_active_house_pod(self):
        app, g = _scene()
        self.assertEqual(g.active_house, House.DIS)
        cell, name = g._grid_cells()[0]
        _click(g, cell.center)
        self.assertEqual(g.pods[House.DIS], [name])

    def test_pod_caps_at_twelve_cards(self):
        app, g = _scene()
        cell, name = g._grid_cells()[0]
        for _ in range(CARDS_PER_POD + 3):
            _click(g, cell.center)
        self.assertEqual(len(g.pods[House.DIS]), CARDS_PER_POD)
        self.assertTrue(g.message_is_error)

    def test_switching_house_tabs_changes_the_grid_and_pod(self):
        app, g = _scene()
        logos_tab = g._house_tab_rects()[House.LOGOS]
        _click(g, logos_tab.center)
        self.assertEqual(g.active_house, House.LOGOS)
        for name in g._house_cards(House.LOGOS)[:3]:
            self.assertEqual(CARD_DEFS[name].house, House.LOGOS)

    def test_type_filter_narrows_the_grid_to_that_type(self):
        from keyforge.enums import CardType

        app, g = _scene()
        creature_filter_rect = g._type_filter_rects()[CardType.CREATURE]
        _click(g, creature_filter_rect.center)
        self.assertEqual(g.type_filter, CardType.CREATURE)
        for name in g._house_cards(House.DIS):
            self.assertEqual(CARD_DEFS[name].type, CardType.CREATURE)

    def test_search_filters_grid_by_name_substring_case_insensitively(self):
        app, g = _scene()
        g.search_query = "SNUDGE"
        self.assertEqual(g._house_cards(House.DIS), ["Snudge"])

    def test_search_combines_with_type_filter(self):
        from keyforge.enums import CardType

        app, g = _scene()
        g.search_query = "snudge"  # Snudge is a Creature
        g.type_filter = CardType.ACTION
        self.assertEqual(g._house_cards(House.DIS), [])

    def test_clicking_search_box_activates_typing_and_filters_live(self):
        app, g = _scene()
        _click(g, g._search_rect().center)
        self.assertTrue(g.search_active)
        for ch in "snudge":
            g.handle_event(pygame.event.Event(pygame.TEXTINPUT, text=ch))
        self.assertEqual(g.search_query, "snudge")
        self.assertEqual(g._house_cards(House.DIS), ["Snudge"])
        g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertFalse(g.search_active)
        self.assertEqual(g.search_query, "snudge")  # Escape unfocuses, doesn't clear the filter

    def test_search_backspace_removes_last_character(self):
        app, g = _scene()
        _click(g, g._search_rect().center)
        for ch in "snudge":
            g.handle_event(pygame.event.Event(pygame.TEXTINPUT, text=ch))
        g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
        self.assertEqual(g.search_query, "snudg")

    def test_search_clear_button_resets_query(self):
        app, g = _scene()
        g.search_query = "snudge"
        _click(g, g._search_clear_rect().center)
        self.assertEqual(g.search_query, "")

    def test_removing_a_card_from_the_pod(self):
        app, g = _scene()
        cell, name = g._grid_cells()[0]
        _click(g, cell.center)
        _click(g, cell.center)
        self.assertEqual(g.pods[House.DIS].count(name), 2)
        row_rect, row_name, count = g._pod_rows()[0]
        self.assertEqual(row_name, name)
        self.assertEqual(count, 2)
        remove_rect = pygame.Rect(row_rect.right - 28, row_rect.top + 4, 22, row_rect.height - 8)
        _click(g, remove_rect.center)
        self.assertEqual(g.pods[House.DIS].count(name), 1)

    def test_random_fill_produces_a_legal_deck(self):
        app, g = _scene()
        g._random_fill()
        self.assertEqual(g._errors(), [])
        for house in (House.DIS, House.LOGOS, House.SHADOWS):
            self.assertEqual(len(g.pods[house]), CARDS_PER_POD)

    def test_save_is_disabled_until_the_deck_is_legal_then_writes_a_file(self):
        app, g = _scene()
        self.assertNotEqual(g._errors(), [])
        self.assertFalse(g._top_buttons()["save"].enabled)
        g._random_fill()
        self.assertTrue(g._top_buttons()["save"].enabled)
        g.name = "My Test Deck"
        g._save()
        self.assertIsNotNone(g.saved_path)
        self.assertTrue(os.path.exists(g.saved_path))
        loaded = load_deck_json(g.saved_path)
        self.assertEqual(loaded.name, "My Test Deck")
        self.assertEqual(validate_deck(loaded), [])

    def test_load_picker_lists_presets_and_saved_decks(self):
        app, g = _scene()
        g._random_fill()
        g.name = "Loadable"
        g._save()
        saved_path = g.saved_path

        g2 = DeckBuilderScene()
        g2.app = app
        g2.picker_open = True
        rows = g2._picker_rows()
        self.assertTrue(any(kind == "preset" for kind, ref, label in rows))
        self.assertTrue(any(kind == "user" and ref == saved_path for kind, ref, label in rows))

        g2._load(saved_path)
        self.assertEqual(g2.name, "Loadable")
        self.assertEqual(g2.pods, g.pods)
        self.assertEqual(len(g2.pods[House.DIS]), CARDS_PER_POD)

    def test_delete_removes_the_saved_file(self):
        app, g = _scene()
        g._random_fill()
        g.name = "Deletable"
        g._save()
        path = g.saved_path
        self.assertTrue(os.path.exists(path))
        g._delete(path)
        self.assertFalse(os.path.exists(path))
        self.assertIsNone(g.saved_path)

    def test_renaming_via_text_input(self):
        app, g = _scene()
        name_rect = g._name_rect()
        _click(g, name_rect.center)
        self.assertTrue(g.editing_name)
        self.assertEqual(g.name_buffer, "New Deck")  # pre-filled with the current name
        for _ in range(len(g.name_buffer)):
            g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
        for ch in "Wraith II":
            g.handle_event(pygame.event.Event(pygame.TEXTINPUT, text=ch))
        g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        self.assertFalse(g.editing_name)
        self.assertEqual(g.name, "Wraith II")

    def test_right_click_opens_the_inspector_with_official_text(self):
        app, g = _scene()
        cell, name = g._grid_cells()[0]
        _right_click(g, cell.center)
        self.assertIsNotNone(g.inspect_info)
        self.assertEqual(g.inspect_info.name, name)
        self.assertTrue(g.inspect_info.text)
        app.draw_scenes()  # must not crash

    def test_menu_deck_builder_button_opens_the_scene_and_back_returns(self):
        app = App(window_size=(1600, 900))
        app.push(MenuScene())
        menu = app.scenes[-1]
        button, _cb = menu.buttons["deck_builder"]
        _click(menu, button.rect.center)
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene", "DeckBuilderScene"])
        builder = app.scenes[-1]
        _pick_default_houses(builder)
        builder._random_fill()
        builder.name = "From Menu"
        builder._save()
        builder._back()
        self.assertEqual([type(s).__name__ for s in app.scenes], ["MenuScene"])
        self.assertIn(builder.saved_path, menu.decks)

    def test_new_deck_forces_the_house_picker_open_and_blocks_interaction(self):
        app = App(window_size=(1600, 900))
        scene = DeckBuilderScene()
        app.push(scene)
        self.assertTrue(scene.house_picker.is_open)
        self.assertFalse(scene.house_picker.dismissable)
        self.assertEqual(scene.chosen_houses, [])
        self.assertIsNone(scene.active_house)
        # Interaction is blocked while it's open: clicking a top button does nothing.
        _click(scene, scene._panel().topleft)
        self.assertEqual(scene.chosen_houses, [])

    def test_picking_three_houses_closes_the_picker_and_enables_the_builder(self):
        app = App(window_size=(1600, 900))
        scene = DeckBuilderScene()
        app.push(scene)
        rects = scene.house_picker.button_rects()
        for house in (House.BROBNAR, House.MARS, House.SANCTUM):
            _click(scene, rects[house].center)
        self.assertEqual(scene.house_picker.selection, [House.BROBNAR, House.MARS, House.SANCTUM])
        _click(scene, scene.house_picker.confirm_rect().center)
        self.assertFalse(scene.house_picker.is_open)
        self.assertEqual(set(scene.chosen_houses), {House.BROBNAR, House.MARS, House.SANCTUM})
        self.assertIn(scene.active_house, scene.chosen_houses)
        # The ordinary builder UI now works, scoped to the chosen houses.
        cell, name = scene._grid_cells()[0]
        _click(scene, cell.center)
        self.assertEqual(CARD_DEFS[name].house, scene.active_house)

    def test_change_houses_button_reopens_a_dismissable_picker(self):
        app, g = _scene()
        hb = g._houses_button_rect()
        _click(g, hb.center)
        self.assertTrue(g.house_picker.is_open)
        self.assertTrue(g.house_picker.dismissable)
        self.assertEqual(set(g.house_picker.selection), {House.DIS, House.LOGOS, House.SHADOWS})
        g.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertFalse(g.house_picker.is_open)
        self.assertEqual(set(g.chosen_houses), {House.DIS, House.LOGOS, House.SHADOWS})


if __name__ == "__main__":
    unittest.main()

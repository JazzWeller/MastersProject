"""Tests for the Deck model, storage, and random decks (Code/PHASE_2_PLAN.md Milestone D)."""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import (
    CARDS_PER_POD,
    DECKS,
    AllianceDeck,
    Deck,
    build_alliance_deck,
    build_deck,
    deck_from_dict,
    deck_label,
    deck_to_dict,
    load_deck_json,
    random_deck,
    resolve_deck,
    save_deck_json,
    validate_deck,
)
from keyforge.config import GameConfig
from keyforge.enums import House
from keyforge.game import Game
from keyforge.replay import config_from_dict, config_to_dict, replay


class TestDeckValidation(unittest.TestCase):
    def _valid_pods(self):
        return {
            House.DIS: ["Charette"] * CARDS_PER_POD,
            House.LOGOS: ["Doc Bookton"] * CARDS_PER_POD,
            House.SHADOWS: ["Urchin"] * CARDS_PER_POD,
        }

    def test_valid_deck_has_no_errors(self):
        deck = Deck(name="Test", pods=self._valid_pods())
        self.assertEqual(validate_deck(deck), [])
        deck.validate()  # doesn't raise

    def test_missing_house_is_an_error(self):
        pods = self._valid_pods()
        del pods[House.SHADOWS]
        deck = Deck(name="Test", pods=pods)
        errors = validate_deck(deck)
        self.assertTrue(any("houses" in e for e in errors))
        with self.assertRaises(ValueError):
            deck.validate()

    def test_wrong_pod_size_is_an_error(self):
        pods = self._valid_pods()
        pods[House.DIS] = pods[House.DIS][:11]
        deck = Deck(name="Test", pods=pods)
        errors = validate_deck(deck)
        self.assertTrue(any("12" in e for e in errors))

    def test_unknown_card_name_is_an_error(self):
        pods = self._valid_pods()
        pods[House.DIS][0] = "Not A Real Card"
        deck = Deck(name="Test", pods=pods)
        errors = validate_deck(deck)
        self.assertTrue(any("Not A Real Card" in e for e in errors))

    def test_card_in_wrong_house_pod_is_an_error(self):
        pods = self._valid_pods()
        pods[House.DIS][0] = "Doc Bookton"  # a Logos card
        deck = Deck(name="Test", pods=pods)
        errors = validate_deck(deck)
        self.assertTrue(any("Doc Bookton" in e and "not Dis" in e for e in errors))

    def test_duplicate_card_names_within_a_pod_are_legal(self):
        deck = Deck(name="Test", pods=self._valid_pods())  # already all duplicates
        self.assertEqual(validate_deck(deck), [])


class TestRandomDeck(unittest.TestCase):
    def test_random_deck_is_always_legal(self):
        rng = random.Random(1)
        for _ in range(20):
            deck = random_deck(rng)
            self.assertEqual(validate_deck(deck), [])

    def test_random_deck_cards_are_all_from_the_right_house(self):
        rng = random.Random(2)
        deck = random_deck(rng)
        for house, names in deck.pods.items():
            for name in names:
                self.assertEqual(CARD_DEFS[name].house, house)

    def test_random_deck_is_deterministic_for_a_given_rng_state(self):
        deck1 = random_deck(random.Random(42))
        deck2 = random_deck(random.Random(42))
        self.assertEqual(deck1.pods, deck2.pods)


class TestDeckJson(unittest.TestCase):
    def test_dict_round_trip_preserves_name_and_pods(self):
        deck = random_deck(random.Random(3), name="Roundtrip")
        restored = deck_from_dict(deck_to_dict(deck))
        self.assertEqual(restored.name, deck.name)
        self.assertEqual(restored.pods, deck.pods)

    def test_file_round_trip_preserves_name_and_pods(self):
        deck = random_deck(random.Random(4), name="FileRoundtrip")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "deck.json")
            save_deck_json(deck, path)
            self.assertTrue(os.path.exists(path))
            restored = load_deck_json(path)
        self.assertEqual(restored.name, deck.name)
        self.assertEqual(restored.pods, deck.pods)


class TestResolveAndBuild(unittest.TestCase):
    def test_all_bundled_presets_are_valid(self):
        self.assertGreaterEqual(len(DECKS), 6)
        for key, deck in DECKS.items():
            self.assertEqual(validate_deck(deck), [], f"preset {key!r} is invalid")

    def test_resolve_deck_by_preset_name_is_case_insensitive(self):
        self.assertIs(resolve_deck("fignor"), resolve_deck("FIGNOR"))

    def test_resolve_deck_object_passthrough(self):
        deck = random_deck(random.Random(5))
        self.assertIs(resolve_deck(deck), deck)

    def test_resolve_deck_by_file_path(self):
        deck = random_deck(random.Random(6), name="PathDeck")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deck.json")
            save_deck_json(deck, path)
            resolved = resolve_deck(path)
        self.assertEqual(resolved.name, "PathDeck")
        self.assertEqual(resolved.pods, deck.pods)

    def test_build_deck_returns_36_cards_matching_the_pods(self):
        cards = build_deck("fignor", owner=1)
        self.assertEqual(len(cards), 36)
        self.assertTrue(all(c.owner == 1 for c in cards))
        names = sorted(c.name for c in cards)
        self.assertEqual(names, sorted(DECKS["fignor"].all_card_names()))

    def test_build_deck_from_a_deck_object(self):
        deck = random_deck(random.Random(7))
        cards = build_deck(deck, owner=2)
        self.assertEqual(sorted(c.name for c in cards), sorted(deck.all_card_names()))

    def test_deck_label(self):
        self.assertEqual(deck_label("fignor"), "fignor")
        deck = random_deck(random.Random(8), name="Labelled")
        self.assertEqual(deck_label(deck), "Labelled")


class TestGameWithDeckSources(unittest.TestCase):
    def test_game_setup_accepts_a_deck_object_directly(self):
        deck1 = random_deck(random.Random(9), name="One")
        deck2 = random_deck(random.Random(10), name="Two")
        game = Game(GameConfig(decks=(deck1, deck2), first_player=1, seed=1))
        self.assertEqual(len(game.players[1].all_cards), 36)
        self.assertEqual(len(game.players[2].all_cards), 36)

    def test_game_setup_accepts_a_deck_file_path(self):
        deck = random_deck(random.Random(11), name="PathPlayed")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deck.json")
            save_deck_json(deck, path)
            game = Game(GameConfig(decks=(path, "igor"), first_player=1, seed=2))
        self.assertEqual(len(game.players[1].all_cards), 36)


class TestAllianceDeck(unittest.TestCase):
    """Code/PHASE_2_PLAN.md v2.1 Milestone F."""

    def test_build_takes_one_pod_per_house_from_different_sources(self):
        alliance = build_alliance_deck("Mixed", {
            House.DIS: "fignor", House.LOGOS: "igor", House.SHADOWS: "wraith",
        })
        self.assertIsInstance(alliance, AllianceDeck)
        self.assertEqual(alliance.pods[House.DIS], DECKS["fignor"].pods[House.DIS])
        self.assertEqual(alliance.pods[House.LOGOS], DECKS["igor"].pods[House.LOGOS])
        self.assertEqual(alliance.pods[House.SHADOWS], DECKS["wraith"].pods[House.SHADOWS])
        self.assertEqual(alliance.sources, {House.DIS: "Fignor", House.LOGOS: "Igor", House.SHADOWS: "Wraith"})

    def test_build_result_is_a_legal_deck(self):
        alliance = build_alliance_deck("Mixed", {
            House.DIS: "cinder", House.LOGOS: "riftwalker", House.SHADOWS: "gambit",
        })
        self.assertEqual(validate_deck(alliance), [])

    def test_pods_are_copied_by_value_not_by_reference(self):
        source = random_deck(random.Random(20), name="Mutable Source")
        alliance = build_alliance_deck("Copy Test", {
            House.DIS: source, House.LOGOS: "igor", House.SHADOWS: "wraith",
        })
        before = list(alliance.pods[House.DIS])
        source.pods[House.DIS][0] = "Charette"  # mutate the source after building
        self.assertEqual(alliance.pods[House.DIS], before)

    def test_dict_round_trip_preserves_alliance_type_and_sources(self):
        alliance = build_alliance_deck("Mixed", {
            House.DIS: "fignor", House.LOGOS: "igor", House.SHADOWS: "wraith",
        })
        restored = deck_from_dict(deck_to_dict(alliance))
        self.assertIsInstance(restored, AllianceDeck)
        self.assertEqual(restored.pods, alliance.pods)
        self.assertEqual(restored.sources, alliance.sources)

    def test_file_round_trip_preserves_alliance_type_and_sources(self):
        alliance = build_alliance_deck("Mixed", {
            House.DIS: "cinder", House.LOGOS: "riftwalker", House.SHADOWS: "gambit",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alliance.json")
            save_deck_json(alliance, path)
            restored = load_deck_json(path)
        self.assertIsInstance(restored, AllianceDeck)
        self.assertEqual(restored.pods, alliance.pods)
        self.assertEqual(restored.sources, alliance.sources)

    def test_plain_deck_still_round_trips_as_a_plain_deck(self):
        deck = random_deck(random.Random(21), name="Plain")
        restored = deck_from_dict(deck_to_dict(deck))
        self.assertIsInstance(restored, Deck)
        self.assertNotIsInstance(restored, AllianceDeck)

    def test_deck_label_tags_an_alliance_deck(self):
        alliance = build_alliance_deck("Mixed", {
            House.DIS: "fignor", House.LOGOS: "igor", House.SHADOWS: "wraith",
        })
        self.assertEqual(deck_label(alliance), "Mixed (Alliance)")
        self.assertEqual(deck_label(DECKS["fignor"]), "Fignor")

    def test_alliance_deck_plays_a_full_game_and_replays_identically(self):
        p1_alliance = build_alliance_deck("P1 Mixed", {
            House.DIS: "fignor", House.LOGOS: "cinder", House.SHADOWS: "riftwalker",
        })
        config = GameConfig(decks=(p1_alliance, "igor"), first_player=1, seed=77, max_turns=60)
        game = Game(config)
        from bots.random_bot import RandomBot

        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        while not game.is_over:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertEqual(len(game.players[1].all_cards), 36)

        data = config_to_dict(config)
        restored_config = config_from_dict(data)
        self.assertIsInstance(restored_config.decks[0], AllianceDeck)
        replayed = replay(restored_config, list(game.choice_record))
        self.assertEqual(replayed.result, game.result)
        self.assertEqual(replayed.turn_number, game.turn_number)


if __name__ == "__main__":
    unittest.main()

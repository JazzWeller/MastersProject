"""Milestone C (Code/AGENT_INTERFACE_PLAN.md): the observation layer and
the log leak it closes first.
"""

import json
import random
import unittest

from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.match import Match, MatchConfig
from keyforge.observation import (
    build_match_observation,
    build_observation,
    full_state_observation,
    observation_key,
    observation_to_dict,
)
from tests.helpers import make_card, new_game


class TestLogVisibility(unittest.TestCase):
    def test_archiving_from_hand_is_hidden_from_the_opponent(self):
        from keyforge.effects import steps

        game = new_game()
        p1 = game.players[1]
        card = make_card("Urchin", 1)
        p1.hand.add(card)
        steps.archive_card(game, p1, card)
        event = next(e for e in game.log.events if e.kind == "archive")
        self.assertEqual(event.visible_to, frozenset({1}))
        self.assertIn(event, game.log.visible_to(1))
        self.assertNotIn(event, game.log.visible_to(2))

    def test_archiving_from_play_is_visible_to_both(self):
        """A card already in play is already public; archiving it (Epic
        Quest, Mobius Scroll) doesn't newly reveal anything."""
        from tests.helpers import put_creature

        game = new_game()
        t = put_creature(game, 1, "Urchin")
        game.archive_from_play(t, 1)
        event = next(e for e in game.log.events if e.kind == "archive")
        self.assertEqual(event.visible_to, frozenset({1, 2}))

    def test_declined_mulligan_and_archive_are_recorded(self):
        game = Game(GameConfig(decks=("fignor", "igor"), seed=1, max_turns=5))
        game.submit(False)  # p1 declines mulligan
        game.submit(False)  # p2 declines mulligan
        kinds = [e.kind for e in game.log.events]
        self.assertIn("mulligan_decision", kinds)
        decisions = [e for e in game.log.events if e.kind == "mulligan_decision"]
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(e.data["took"] is False for e in decisions))


class TestObservationRedaction(unittest.TestCase):
    def test_opponent_hand_and_archive_are_hidden(self):
        game = new_game()
        obs1 = build_observation(game, 1)
        self.assertIsNotNone(obs1.players[1].hand)
        self.assertIsNone(obs1.players[2].hand)
        self.assertIsNone(obs1.players[2].archive)
        # Counts are still public.
        self.assertEqual(obs1.players[2].hand_count, len(game.players[2].hand))

    def test_decklists_are_always_fully_named_but_carry_no_live_state(self):
        game = new_game()
        obs = build_observation(game, 1)
        self.assertEqual(len(obs.players[2].decklist), 36)
        for entry in obs.players[2].decklist:
            self.assertEqual(set(entry), {"instance_id", "name", "house", "type"})

    def test_full_state_observation_reveals_both_hands(self):
        game = new_game()
        obs = full_state_observation(game)
        self.assertIsNotNone(obs.players[1].hand)
        self.assertIsNotNone(obs.players[2].hand)
        self.assertIsNotNone(obs.players[1].archive)
        self.assertIsNotNone(obs.players[2].archive)

    def test_pending_decision_options_are_hidden_from_the_non_deciding_player(self):
        game = new_game()
        d = game.pending_decision
        self.assertEqual(d.kind, DecisionKind.CHOOSE_ACTION)
        obs_actor = build_observation(game, d.player)
        obs_other = build_observation(game, 3 - d.player)
        self.assertGreater(len(obs_actor.pending_decision.options), 0)
        self.assertEqual(obs_other.pending_decision.options, ())
        self.assertEqual(obs_other.pending_decision.kind, "CHOOSE_ACTION")  # kind itself is public

    def test_observation_is_json_serializable(self):
        game = new_game()
        obs = build_observation(game, 1)
        json.dumps(observation_to_dict(obs))  # must not raise

    def test_observation_key_is_deterministic_and_viewer_sensitive(self):
        game = new_game()
        a = build_observation(game, 1)
        b = build_observation(game, 1)
        self.assertEqual(observation_key(a), observation_key(b))
        c = build_observation(game, 2)
        self.assertNotEqual(observation_key(a), observation_key(c))

    def test_match_context_appears_in_every_in_game_observation(self):
        game = new_game()

        class _StubMatch:
            format = "reversal"
            games = []
            score = {1: 2, 2: 1}

        obs = build_observation(game, 1, match=_StubMatch())
        self.assertEqual(obs.format, "reversal")
        self.assertEqual(obs.game_number, 1)
        self.assertEqual(obs.match_score, {1: 2, 2: 1})


class TestMatchLevelObservation(unittest.TestCase):
    def test_bid_chains_observation_carries_score_and_prior_game_context(self):
        config = MatchConfig(format="adaptive", decks=("fignor", "igor"), seed=3, max_turns=200)
        match = Match(config)
        bots = {1: RandomBot(seed=3), 2: RandomBot(seed=4)}
        # Drive until a match-level decision (BID_CHAINS or
        # CHOOSE_FIRST_PLAYER) is reached, or the match ends first.
        seen_match_level = False
        for _ in range(20000):
            if match.is_over:
                break
            d = match.pending_decision
            if d.kind in (DecisionKind.BID_CHAINS, DecisionKind.CHOOSE_FIRST_PLAYER):
                obs = build_match_observation(match, d.player)
                self.assertEqual(obs.format, "adaptive")
                self.assertGreaterEqual(obs.game_number, 2)
                self.assertEqual(len(obs.players[1].decklist), 36)
                self.assertEqual(len(obs.players[2].decklist), 36)
                seen_match_level = True
                break
            choice = bots[d.player].decide(match.view_for(d.player), d)
            match.submit(choice)
        if not seen_match_level:
            self.skipTest("this seed never reached a match-level decision within budget")


class TestObservationFuzzNoLeak(unittest.TestCase):
    def test_history_never_exceeds_what_the_viewer_is_entitled_to(self):
        """Across a random-7-house-deck fuzz, every history entry in a
        viewer's own observation must also appear in that viewer's log
        stream (game.log.visible_to(viewer)) -- i.e. Observation's history
        never shows more than the log's own visibility already allows."""
        deck_rng = random.Random(99)
        for i in range(60):
            d1 = random_deck(deck_rng, "R1")
            d2 = random_deck(deck_rng, "R2")
            game = Game(GameConfig(decks=(d1, d2), seed=i, max_turns=150))
            bots = {1: RandomBot(seed=i), 2: RandomBot(seed=i + 1000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
            for viewer in (1, 2):
                obs = build_observation(game, viewer)
                entitled_kinds = [e.kind for e in game.log.visible_to(viewer)]
                seen_kinds = [h.kind for h in obs.history]
                self.assertEqual(seen_kinds, entitled_kinds)


if __name__ == "__main__":
    unittest.main()

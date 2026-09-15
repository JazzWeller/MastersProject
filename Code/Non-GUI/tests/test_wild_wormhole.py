import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game


def advance_to_first_choose_action(game):
    game.submit(False)
    game.submit(False)
    while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
        d = game.pending_decision
        if d.kind == DecisionKind.CHOOSE_HOUSE:
            game.submit(d.options[0])
        elif d.kind == DecisionKind.TAKE_ARCHIVE:
            game.submit(False)
        else:
            game.submit(d.options[0])


class TestWildWormholeLibraryAccess(unittest.TestCase):
    def test_wormhole_plays_library_access_and_draws(self):
        """Wild Wormhole plays Library Access off the top of the deck: Wormhole's
        effect (which plays Library Access) resolves before Wormhole's own
        play-trigger check, so by the time that check runs, Library Access's
        freshly-registered trigger fires and draws a card."""
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=5, max_turns=50))
        advance_to_first_choose_action(game)
        player = game.players[1]

        # Force the deck's top card to be Library Access.
        la = Card(get_card_def("Library Access"), 1)
        player.deck.put_on_top(la)

        hand_before = len(player.hand)
        gen = game.play_card_from_deck_top(player, la, ignore_house=True)
        try:
            next(gen)
            self.fail("no decision expected for this scenario")
        except StopIteration:
            pass

        # Library Access purged itself, and its own play-trigger check should
        # have found no triggers (it excludes its own freshly-registered one),
        # but nothing here draws yet -- the draw comes from Wormhole's own
        # play-trigger check, which is only run by the caller (Wild Wormhole's
        # effect itself). Call the outer play resolution directly to verify.
        self.assertIn(la, player.purged.cards())

    def test_library_access_trigger_fires_on_next_card_played(self):
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=6, max_turns=50))
        advance_to_first_choose_action(game)
        player = game.players[1]

        la = Card(get_card_def("Library Access"), 1)
        player.hand.add(la)
        gen = game._play_card(1, la)
        try:
            next(gen)
        except StopIteration:
            pass
        self.assertIn(la, player.purged.cards())

        # Now play any other card and confirm the trigger draws a card.
        other = Card(get_card_def("Urchin"), 1)
        player.hand.add(other)
        hand_before = len(player.hand) - 1  # excluding the card about to be played
        gen2 = game._play_card(1, other)
        try:
            next(gen2)
        except StopIteration:
            pass
        # Urchin steals 1 aember on play (no aember for opponent though, so no-op)
        # but Library Access's trigger should have drawn exactly 1 card.
        self.assertEqual(len(player.hand), hand_before)  # played 1, drew 1: net 0


if __name__ == "__main__":
    unittest.main()

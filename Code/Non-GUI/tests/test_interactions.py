import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game


def advance_to_choose_action(game, pid, mulligans=(False, False)):
    game.submit(mulligans[0])
    game.submit(mulligans[1])
    while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION or game.pending_decision.player != pid:
        d = game.pending_decision
        if d.kind == DecisionKind.CHOOSE_HOUSE:
            game.submit(d.options[0])
        elif d.kind == DecisionKind.TAKE_ARCHIVE:
            game.submit(False)
        elif d.kind == DecisionKind.CHOOSE_ACTION:
            game.submit(EndTurn())
        else:
            game.submit(d.options[0])


def find_action(game, cls, name=None):
    for opt in game.pending_decision.options:
        if isinstance(opt, cls) and (name is None or opt.card.name == name):
            return opt
    return None


class TestDustImpDestroyed(unittest.TestCase):
    def test_dust_imp_gains_aember_on_destroy(self):
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=42, max_turns=50))
        advance_to_choose_action(game, 1)
        # force a Dust Imp into player 1's play area directly for a controlled test
        player = game.players[1]
        card = Card(get_card_def("Dust Imp"), 1)
        player.play_area.add_creature(card)
        card.Exhausted = False
        card.CanBeUsed = True
        before = player.aember
        # destroy it directly via the engine's pipeline
        gen = game.destroy_cards([card])
        try:
            next(gen)
            self.fail("destroy of a single creature with no other ties should not need a decision")
        except StopIteration:
            pass
        self.assertEqual(player.aember, before + 2)
        self.assertIn(card, player.discard.cards())


class TestBadPennyGoesToHand(unittest.TestCase):
    def test_bad_penny_destroyed_goes_to_hand(self):
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=7, max_turns=50))
        advance_to_choose_action(game, 1)
        player = game.players[1]
        card = Card(get_card_def("Bad Penny"), 1)
        player.play_area.add_creature(card)
        gen = game.destroy_cards([card])
        try:
            next(gen)
            self.fail("no decision expected")
        except StopIteration:
            pass
        self.assertIn(card, player.hand.cards())
        self.assertNotIn(card, player.discard.cards())


class TestElusiveAndSkirmish(unittest.TestCase):
    def test_elusive_skips_first_fight_only(self):
        game = Game(GameConfig(decks=("igor", "igor"), first_player=1, seed=11, max_turns=50))
        advance_to_choose_action(game, 1)
        p1, p2 = game.players[1], game.players[2]
        attacker = Card(get_card_def("Titan Mechanic"), 1)  # power 6, no keyword
        p1.play_area.add_creature(attacker)
        attacker.Exhausted = False
        attacker.CanBeUsed = True

        target = Card(get_card_def("Urchin"), 2)  # power 1
        p2.play_area.add_creature(target)
        get_card_def("Urchin").register_passive(game, target)  # grant Elusive directly

        list(game.destroy_cards([]))  # no-op, sanity that generator protocol works

        gen = game._fight(1, attacker)
        try:
            next(gen)
        except StopIteration:
            pass
        # Elusive skipped the fight: target takes no damage, attacker takes no damage
        self.assertEqual(target.type_object.damage, 0)
        self.assertEqual(attacker.type_object.damage, 0)
        self.assertTrue(target.fought_this_turn)

        # Second fight this turn: Elusive no longer applies. The target (power 1)
        # takes the attacker's power 6 damage and is destroyed; the attacker
        # (power 6) takes the target's power 1 and survives.
        attacker.Exhausted = False
        gen2 = game._fight(1, attacker)
        try:
            next(gen2)
        except StopIteration:
            pass
        self.assertIn(target, p2.discard.cards())
        self.assertNotIn(target, p2.play_area.creatures)
        self.assertEqual(attacker.type_object.damage, 1)  # took target's power, survived


if __name__ == "__main__":
    unittest.main()

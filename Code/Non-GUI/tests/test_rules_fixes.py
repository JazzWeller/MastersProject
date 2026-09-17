"""Regression tests for the rules pass in Code/PLAYTEST_FIX_PLAN.md (R1-R9),
plus the move record used for game history and replays."""

import random
import unittest

from tests.helpers import drive, hand_card, make_card, new_game, put_creature, run_hook

from bots.random_bot import RandomBot
from keyforge.actions import DiscardCard, EndTurn, PlayCard, Reap, UseAction
from keyforge.config import GameConfig
from keyforge.effects import named
from keyforge.enums import DecisionKind, House
from keyforge.game import Game
from keyforge.replay import config_from_dict, config_to_dict, replay


def _mid_game(house):
    """A game on player 1's CHOOSE_ACTION, past the first-turn limit, with
    `house` selected."""
    game = new_game()
    game.turn_number = 3
    game.players[1].selected_house = house
    return game


def _plays(game, pid=1):
    return [a.card for a in game._legal_actions(pid) if isinstance(a, PlayCard)]


class TestScramblerStorm(unittest.TestCase):  # R1
    def test_blocks_playing_actions_from_hand(self):
        game = _mid_game(House.LOGOS)
        storm = make_card("Scrambler Storm", 2)
        run_hook(game, storm.card_def.on_play, storm)
        labwork = hand_card(game, 1, "Labwork")
        mother = hand_card(game, 1, "Mother")
        plays = _plays(game)
        self.assertNotIn(labwork, plays)
        self.assertIn(mother, plays)
        self.assertIn("cannot play actions", game.why_not_playable(1, labwork))

    def test_blocks_actions_played_by_wild_wormhole(self):
        game = _mid_game(House.LOGOS)
        storm = make_card("Scrambler Storm", 2)
        run_hook(game, storm.card_def.on_play, storm)
        p1 = game.players[1]
        top = make_card("Labwork", 1)
        p1.deck.put_on_top(top)
        drive(game.play_card_from_deck_top(p1, top))
        self.assertIs(p1.deck.peek_top(), top)  # refused, returned to the top


class TestCardsEnteringPlayCanBeUsed(unittest.TestCase):  # R2
    def test_silvertooth_can_reap_the_turn_it_is_played(self):
        game = _mid_game(House.SHADOWS)
        tooth = hand_card(game, 1, "Silvertooth")
        drive(game._play_card(1, tooth))
        self.assertFalse(tooth.Exhausted)
        self.assertTrue(tooth.CanBeUsed)
        self.assertTrue(any(isinstance(a, Reap) and a.card is tooth for a in game._legal_actions(1)))

    def test_off_house_card_entering_play_is_not_usable(self):
        game = _mid_game(House.LOGOS)
        tooth = hand_card(game, 1, "Silvertooth")
        game.players[1].NonLogosCardsPlayable = 1
        drive(game._play_card(1, tooth))
        self.assertFalse(tooth.CanBeUsed)


class TestDuskrunner(unittest.TestCase):  # R3
    def test_playable_onto_an_enemy_creature_with_no_friendly_creatures(self):
        game = _mid_game(House.SHADOWS)
        enemy = put_creature(game, 2, "Mother")
        dusk = hand_card(game, 1, "Duskrunner")
        self.assertIn(dusk, _plays(game))
        drive(game._play_card(1, dusk), answers=[[enemy]])
        self.assertIn(dusk, enemy.type_object.upgrades)

    def test_not_playable_with_no_creatures_anywhere(self):
        game = _mid_game(House.SHADOWS)
        dusk = hand_card(game, 1, "Duskrunner")
        self.assertNotIn(dusk, _plays(game))
        self.assertEqual(game.why_not_playable(1, dusk), "No creature to attach it to")


class TestLightsOut(unittest.TestCase):  # R4
    def test_must_return_exactly_two(self):
        game = _mid_game(House.SHADOWS)
        for name in ("Mother", "Doc Bookton", "Titan Mechanic"):
            put_creature(game, 2, name)
        card = make_card("Lights Out", 1)
        gen = named.lights_out(game, card)
        d = next(gen)
        self.assertEqual((d.min_n, d.max_n), (2, 2))

    def test_single_enemy_creature_is_returned_without_a_choice(self):
        game = _mid_game(House.SHADOWS)
        mother = put_creature(game, 2, "Mother")
        run_hook(game, named.lights_out, make_card("Lights Out", 1))
        self.assertIn(mother, game.players[2].hand.cards())


class TestGuardianDemon(unittest.TestCase):  # R5
    def test_never_damages_the_healed_creature(self):
        game = _mid_game(House.DIS)
        demon = put_creature(game, 1, "Guardian Demon")
        demon.type_object.damage = 2
        run_hook(game, named.guardian_demon, demon, answers=[[demon]])
        self.assertEqual(demon.type_object.damage, 0)  # healed, and nothing else to damage

    def test_only_damaged_creatures_are_offered_as_heal_targets(self):
        game = _mid_game(House.DIS)
        demon = put_creature(game, 1, "Guardian Demon")
        hurt = put_creature(game, 2, "Mother")
        hurt.type_object.damage = 1
        also_hurt = put_creature(game, 2, "Doc Bookton")
        also_hurt.type_object.damage = 3
        put_creature(game, 2, "Titan Mechanic")  # undamaged
        d = next(named.guardian_demon(game, demon))
        self.assertEqual(d.options, [hurt, also_hurt])

    def test_no_damaged_creatures_means_no_decision(self):
        game = _mid_game(House.DIS)
        demon = put_creature(game, 1, "Guardian Demon")
        put_creature(game, 2, "Mother")
        with self.assertRaises(StopIteration):
            next(named.guardian_demon(game, demon))


class TestEmberImpFollowsSpec(unittest.TestCase):  # R7 (deliberate)
    def test_limits_hand_plays_but_not_effect_plays(self):
        game = _mid_game(House.LOGOS)
        put_creature(game, 2, "Ember Imp")
        p1 = game.players[1]
        p1.hand_plays_this_turn = 2
        labwork = hand_card(game, 1, "Labwork")
        self.assertNotIn(labwork, _plays(game))
        top = make_card("Mother", 1)
        p1.deck.put_on_top(top)
        drive(game.play_card_from_deck_top(p1, top), answers=["left"])
        self.assertIn(top, p1.play_area.creatures)


class TestDominatorBauble(unittest.TestCase):  # R8
    def test_only_ready_creatures_are_offered(self):
        game = _mid_game(House.DIS)
        ready = put_creature(game, 1, "Mother", exhausted=False)
        put_creature(game, 1, "Doc Bookton", exhausted=True)
        also_ready = put_creature(game, 1, "Titan Mechanic", exhausted=False)
        bauble = make_card("Dominator Bauble", 1)
        d = next(named.dominator_bauble(game, bauble))
        self.assertEqual(d.options, [ready, also_ready])


class TestSilentEffectsAreLogged(unittest.TestCase):  # R9
    def _to_player_2_turn(self, game):
        while game.pending_decision.player != 2 or game.pending_decision.kind not in (
            DecisionKind.CHOOSE_ACTION,
            DecisionKind.TAKE_ARCHIVE,
        ):
            d = game.pending_decision
            if d.kind == DecisionKind.CHOOSE_ACTION:
                game.submit(next(o for o in d.options if isinstance(o, EndTurn)))
            elif d.kind == DecisionKind.CHOOSE_CARDS:
                game.submit(list(d.options[: d.min_n]))
            else:
                game.submit(d.options[0])

    def test_miasma_logs_a_skipped_forge(self):
        game = new_game()
        miasma = make_card("Miasma", 1)
        run_hook(game, miasma.card_def.on_play, miasma)
        game.players[2].aember = 7
        self._to_player_2_turn(game)
        skipped = [e for e in game.log.events if e.kind == "forge_skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].data["source"], "Miasma")
        self.assertEqual(game.players[2].keys, 0)

    def test_control_the_weak_logs_a_forced_house(self):
        game = new_game()
        ctw = make_card("Control the Weak", 1)
        run_hook(game, named.control_the_weak, ctw, answers=[House.SHADOWS])
        self._to_player_2_turn(game)
        forced = [e for e in game.log.events if e.kind == "house_forced"]
        self.assertEqual(forced[-1].data, {"player": 2, "house": "Shadows", "source": "Control the Weak"})

    def test_forge_key_log_records_turn_and_cost(self):
        game = new_game()
        game.players[2].aember = 6
        self._to_player_2_turn(game)
        ev = [e for e in game.log.events if e.kind == "forge_key"][-1]
        self.assertEqual((ev.data["turn"], ev.data["cost"]), (2, 6))


class TestFirstTurnRule(unittest.TestCase):
    def test_only_the_first_player_is_limited(self):
        for seed in range(20):
            game = Game(GameConfig(decks=("fignor", "igor"), seed=seed))
            bot = RandomBot(seed)
            while not (game.pending_decision.kind == DecisionKind.CHOOSE_ACTION and game.turn_number == 2):
                d = game.pending_decision
                if d.kind == DecisionKind.CHOOSE_ACTION:
                    game.submit(next(o for o in d.options if isinstance(o, EndTurn)))
                else:
                    game.submit(bot.decide(game.view_for(d.player), d))
            second = game.pending_decision.player
            self.assertNotEqual(second, game.first_player)
            p = game.players[second]
            same_house = [c for c in p.hand.cards() if c.house == p.selected_house]
            played = 0
            while True:
                plays = [o for o in game.pending_decision.options if isinstance(o, (PlayCard, DiscardCard))]
                if not plays or game.pending_decision.player != second:
                    break
                game.submit(next(o for o in plays if isinstance(o, DiscardCard)))
                played += 1
                while game.pending_decision.kind != DecisionKind.CHOOSE_ACTION:
                    game.submit(bot.decide(game.view_for(game.pending_decision.player), game.pending_decision))
            self.assertEqual(played, len(same_house), seed)

    def test_why_not_playable_explains_the_first_turn_limit(self):
        game = new_game()  # player 1 is first, on turn 1
        p1 = game.players[1]
        card = next(c for c in p1.hand.cards() if c.house == p1.selected_house)
        game.submit(DiscardCard(card))
        other = hand_card(game, 1, "Arise")
        p1.selected_house = House.DIS
        self.assertIn("First turn", game.why_not_playable(1, other))


class TestReplayRecord(unittest.TestCase):
    def test_replaying_a_record_reproduces_the_game_exactly(self):
        for seed in range(15):
            config = GameConfig(decks=("fignor", "igor"), seed=seed * 7919)
            game = Game(config)
            bots = {1: RandomBot(seed), 2: RandomBot(seed + 1)}
            while not game.is_over:
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
            again = replay(config_from_dict(config_to_dict(config)), list(game.choice_record))
            self.assertTrue(again.is_over)
            self.assertEqual(again.result, game.result)
            self.assertEqual(
                [(e.kind, {k: v for k, v in e.data.items() if "iid" not in k}) for e in again.log.events],
                [(e.kind, {k: v for k, v in e.data.items() if "iid" not in k}) for e in game.log.events],
            )

    def test_partial_replay_stops_at_the_requested_step(self):
        config = GameConfig(seed=5)
        game = Game(config)
        bot = RandomBot(5)
        for _ in range(30):
            d = game.pending_decision
            game.submit(bot.decide(game.view_for(d.player), d))
        partial = replay(config, game.choice_record, upto=12)
        self.assertEqual(len(partial.choice_record), 12)
        self.assertFalse(partial.is_over)


if __name__ == "__main__":
    unittest.main()

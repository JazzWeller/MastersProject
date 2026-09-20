"""Targeted tests for HeuristicBot's handling of the newer decision kinds
(Code/PHASE_2_PLAN.md Milestone C/E: bots must handle every decision kind)
and, below, its Phase 3 armor/assault/hazardous/mode awareness
(Code/PHASE_3_PLAN.md Milestone F)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import new_game, put_creature

from bots.heuristic_bot import HeuristicBot
from keyforge.decision import Decision
from keyforge.enums import DecisionKind, House


class TestHeuristicBotChooseNumber(unittest.TestCase):
    def test_prefers_the_power_that_destroys_more_enemies_than_friendlies(self):
        game = new_game()
        put_creature(game, 1, "Charette")  # power 4, friendly
        put_creature(game, 2, "Doc Bookton")  # power 5, enemy
        put_creature(game, 2, "Truebaru")  # power 7, enemy
        bot = HeuristicBot(seed=1)
        view = game.view_for(1)
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_NUMBER, prompt="Dance of Doom: choose a number", options=[4, 5, 7])
        choice = bot.decide(view, decision)
        # 4 kills 1 friendly for 0 enemies (score -1); 5 and 7 each kill 1
        # enemy for 0 friendlies (score +1) -- either is an equally good pick.
        self.assertIn(choice, (5, 7))

    def test_does_not_crash_with_no_matching_creatures_in_play(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        view = game.view_for(1)
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_NUMBER, prompt="Dance of Doom: choose a number", options=[3, 6])
        choice = bot.decide(view, decision)
        self.assertIn(choice, (3, 6))


class TestHeuristicBotFightValue(unittest.TestCase):
    """Phase 3: _fight_value must account for armor, assault, hazardous,
    and damage prevention, not just raw power."""

    def test_armor_makes_a_fight_that_would_otherwise_kill_unfavorable(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        attacker = put_creature(game, 1, "Doc Bookton")  # power 5, survives Bulwark's counter-hit either way
        bulwark = put_creature(game, 2, "Bulwark")  # power 4, armor 2 -> needs 6 damage to kill, not 4
        # Ignoring armor, 5 power looks like a safe, guaranteed kill against
        # a power-4 creature (a clearly good, positive-value fight); with
        # its 2 armor counted, 5 power doesn't kill it at all.
        value = bot._fight_value(attacker, bulwark)
        self.assertLessEqual(value, -1.0)

    def test_assault_can_win_a_fight_the_main_exchange_alone_would_lose(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        bear = put_creature(game, 1, "Ancient Bear")  # power 5, assault 2
        target = put_creature(game, 2, "Charette")  # power 4
        target.type_object.damage = 2  # 2 remaining HP: assault 2 alone finishes it off
        value = bot._fight_value(bear, target)
        self.assertGreater(value, 0)

    def test_hazardous_defender_that_would_kill_the_attacker_is_avoided(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        attacker = put_creature(game, 1, "Charette")  # power 4, 4 HP
        grubbling = put_creature(game, 2, "Briar Grubbling")  # power 2, hazardous 5
        value = bot._fight_value(attacker, grubbling)
        self.assertLess(value, -1.0)  # worse than an ordinary bad trade

    def test_damage_prevented_target_is_never_worth_fighting(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        attacker = put_creature(game, 1, "Charette")
        target = put_creature(game, 2, "Snudge")
        target.damage_prevented = True
        value = bot._fight_value(attacker, target)
        self.assertLess(value, 0)


class TestHeuristicBotHouseScore(unittest.TestCase):
    def test_cannot_reap_creature_with_no_action_scores_lower_than_a_normal_one(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        put_creature(game, 1, "Tireless Crocag")  # Brobnar, cannot_reap, no action
        score_with_crocag = bot._house_score(game.view_for(1), House.BROBNAR)
        game2 = new_game()
        put_creature(game2, 1, "Krump")  # Brobnar, ordinary vanilla creature
        score_with_krump = bot._house_score(game2.view_for(1), House.BROBNAR)
        self.assertLess(score_with_crocag, score_with_krump)


class TestHeuristicBotChooseMode(unittest.TestCase):
    def test_begone_destroys_dis_creatures_when_the_opponent_has_one(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        put_creature(game, 2, "Charette")  # Dis
        decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_MODE, prompt="Begone!: choose one",
            options=["Destroy each Dis creature", "Gain 1Æ"],
        )
        self.assertEqual(bot.decide(game.view_for(1), decision), "Destroy each Dis creature")

    def test_begone_takes_the_aember_when_the_opponent_has_no_dis_creature(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        put_creature(game, 2, "Krump")  # Brobnar, not Dis
        decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_MODE, prompt="Begone!: choose one",
            options=["Destroy each Dis creature", "Gain 1Æ"],
        )
        self.assertEqual(bot.decide(game.view_for(1), decision), "Gain 1Æ")

    def test_ozmo_heals_a_hurt_friendly_mars_creature_then_targets_it_not_the_enemy(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        hurt = put_creature(game, 1, "Blypyp")  # a Mars creature
        hurt.type_object.damage = 1
        enemy_mars = put_creature(game, 2, "Blypyp")
        mode_decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_MODE, prompt="Ozmo: heal or stun a Mars creature",
            options=["Heal 3", "Stun"],
        )
        self.assertEqual(bot.decide(game.view_for(1), mode_decision), "Heal 3")
        target_decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_CARDS, prompt="Ozmo: choose a Mars creature",
            options=[hurt, enemy_mars], min_n=1, max_n=1,
        )
        self.assertEqual(bot.decide(game.view_for(1), target_decision), [hurt])

    def test_ozmo_stuns_the_enemy_mars_creature_when_no_friendly_one_is_hurt(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        healthy = put_creature(game, 1, "Blypyp")
        enemy_mars = put_creature(game, 2, "Blypyp")
        mode_decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_MODE, prompt="Ozmo: heal or stun a Mars creature",
            options=["Heal 3", "Stun"],
        )
        self.assertEqual(bot.decide(game.view_for(1), mode_decision), "Stun")
        target_decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_CARDS, prompt="Ozmo: choose a Mars creature",
            options=[healthy, enemy_mars], min_n=1, max_n=1,
        )
        self.assertEqual(bot.decide(game.view_for(1), target_decision), [enemy_mars])


class TestHeuristicBotChooseCards(unittest.TestCase):
    def test_ready_and_fight_prefers_the_best_fighter_not_the_weakest(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        weak = put_creature(game, 1, "Blypyp")  # power 2, loses this fight
        strong = put_creature(game, 1, "Krump")  # power 6, safely kills the target below
        target = put_creature(game, 2, "Snudge")  # power 4
        decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_CARDS,
            prompt="Anger: choose a friendly creature to ready and fight with",
            options=[weak, strong], min_n=1, max_n=1,
        )
        self.assertEqual(bot.decide(game.view_for(1), decision), [strong])

    def test_choose_a_creature_to_stun_avoids_an_already_stunned_enemy(self):
        game = new_game()
        bot = HeuristicBot(seed=1)
        stunned_enemy = put_creature(game, 2, "Charette")
        stunned_enemy.stunned = True
        other_enemy = put_creature(game, 2, "Snudge")
        decision = Decision(
            player=1, kind=DecisionKind.CHOOSE_CARDS, prompt="Smaaash: choose a creature to stun",
            options=[stunned_enemy, other_enemy], min_n=1, max_n=1,
        )
        self.assertEqual(bot.decide(game.view_for(1), decision), [other_enemy])


if __name__ == "__main__":
    unittest.main()

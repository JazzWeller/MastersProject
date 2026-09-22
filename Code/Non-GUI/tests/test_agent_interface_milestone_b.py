"""Milestone B (Code/AGENT_INTERFACE_PLAN.md): decision provenance.

Every CHOOSE_CARDS/YES_NO/CHOOSE_MODE/CHOOSE_NUMBER/ORDER_EFFECTS decision
must carry a `source_card`/`intent` tag derived from the actual card, not
from `decision.prompt` -- HeuristicBot's `_choose_cards` (Ozmo's motivating
example) proves the tags carry enough meaning to play on, and a fuzz run
across the full 7-house pool is the mechanical check that the 167-call-site
audit didn't miss a spot: any untagged decision is a hard, loud failure
here, not a silently blind agent later.
"""

import random
import unittest

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.cards.decks import random_deck
from keyforge.config import GameConfig
from keyforge.decision import Decision
from keyforge.enums import Affects, CardType, DecisionIntent, DecisionKind, House
from keyforge.game import Game

_PROVENANCE_KINDS = (
    DecisionKind.CHOOSE_CARDS,
    DecisionKind.YES_NO,
    DecisionKind.CHOOSE_MODE,
    DecisionKind.CHOOSE_NUMBER,
    DecisionKind.ORDER_EFFECTS,
)


class TestDecisionProvenanceCoverage(unittest.TestCase):
    def test_every_provenance_decision_across_the_full_pool_carries_an_intent(self):
        deck_rng = random.Random(2026)
        untagged = []
        for i in range(300):
            d1 = random_deck(deck_rng, "R1")
            d2 = random_deck(deck_rng, "R2")
            config = GameConfig(decks=(d1, d2), seed=i, max_turns=200)
            game = Game(config)
            bots = {1: RandomBot(seed=i), 2: RandomBot(seed=i + 100_000)}
            while not game.is_over:
                d = game.pending_decision
                if d.kind in _PROVENANCE_KINDS and d.intent is None:
                    untagged.append((d.kind.name, d.prompt))
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)
        self.assertEqual(untagged, [], f"{len(untagged)} untagged decision(s), e.g. {untagged[:5]}")

    def test_heuristic_bot_never_raises_across_the_same_fuzz(self):
        """HeuristicBot._choose_cards hard-errors on a missing/unrecognized
        intent (Milestone B's acceptance bar) -- running it as both seats
        across the same fuzz means any gap the coverage test above would
        catch also can't hide behind RandomBot never happening to ask."""
        deck_rng = random.Random(4041)
        for i in range(150):
            d1 = random_deck(deck_rng, "R1")
            d2 = random_deck(deck_rng, "R2")
            config = GameConfig(decks=(d1, d2), seed=i, max_turns=200)
            game = Game(config)
            bots = {1: HeuristicBot(seed=i), 2: HeuristicBot(seed=i + 100_000)}
            while not game.is_over:
                d = game.pending_decision
                choice = bots[d.player].decide(game.view_for(d.player), d)
                game.submit(choice)


class TestHeuristicBotHardErrors(unittest.TestCase):
    def test_raises_on_a_choose_cards_decision_with_no_intent(self):
        bot = HeuristicBot(seed=1)
        from tests.helpers import make_card

        options = [make_card("Urchin", 1), make_card("Old Bruno", 1)]
        decision = Decision(player=1, kind=DecisionKind.CHOOSE_CARDS, prompt="synthetic", options=options, min_n=1, max_n=1)

        class _StubView:
            def me(self):
                return _Side()

            def opponent(self):
                return _Side()

        class _Side:
            creatures = []
            artifacts = []
            hand = []
            discard = []
            archive = []

        with self.assertRaises(ValueError):
            bot._choose_cards(_StubView(), decision, options)


class TestOzmoIntentDisambiguation(unittest.TestCase):
    def test_source_card_and_intent_flow_from_the_real_engine_not_a_shared_prompt(self):
        """The engine itself (logos.py's `ozmo`) resolves the target
        decision's intent from the mode chosen a decision earlier -- this
        checks the *engine's* tagging, not just the bot's handling of a
        hand-built Decision (see test_heuristic_bot.py for that side).

        `ozmo`'s own target filter checks `"Mars" in c.tags` (a literal tag
        no card in this pool actually carries -- see
        test_cards_logos_phase2.py's test_ozmo_is_elusive_and_has_no_valid_
        target_in_this_pool, which documents this as existing, accepted
        behavior). That's a pool/rules question outside this plan's scope,
        so this test satisfies the filter with synthetic Mars-tagged
        creatures rather than changing it.
        """
        from tests.helpers import new_game, put_creature
        from keyforge.cards.card import Card, CardDef
        from keyforge.effects import named

        def mars_tagged(name):
            return CardDef(id=9500, name=name, house=House.MARS, type=CardType.CREATURE, power=3, tags=("Mars",))

        game = new_game()
        hurt = Card(mars_tagged("Synthetic Mars A"), 1)
        game.players[1].play_area.add_creature(hurt)
        hurt.type_object.damage = 1
        enemy = Card(mars_tagged("Synthetic Mars B"), 2)
        game.players[2].play_area.add_creature(enemy)
        ozmo = put_creature(game, 1, "Ozmo, Martianologist", exhausted=False, can_be_used=True)
        gen = named.ozmo(game, ozmo)
        mode_decision = next(gen)
        self.assertEqual(mode_decision.kind, DecisionKind.CHOOSE_MODE)
        self.assertEqual(mode_decision.source_card, ozmo)
        target_decision = gen.send("Heal 3")
        self.assertEqual(target_decision.kind, DecisionKind.CHOOSE_CARDS)
        self.assertEqual(target_decision.intent, DecisionIntent.HEAL)
        self.assertEqual(target_decision.affects, Affects.ANY)
        self.assertEqual(target_decision.source_card, ozmo)


if __name__ == "__main__":
    unittest.main()

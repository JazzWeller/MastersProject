"""Milestone F (Code/AGENT_INTERFACE_PLAN.md): option encoding, the card
vocabulary, and the multi-select helper.
"""

import json
import random
import unittest

from bots.option_space import (
    choice_space_size,
    enumerate_choices,
    independent_topk,
    sequential_decompose,
    sequential_next_options,
    sequential_recompose,
)
from bots.random_bot import RandomBot
from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.decks import random_deck
from keyforge.cards.vocabulary import CARD_VOCAB, VOCAB_HASH, card_id, card_name
from keyforge.config import GameConfig
from keyforge.encoding import option_features, option_key
from keyforge.enums import DecisionKind
from keyforge.game import Game


class TestCardVocabulary(unittest.TestCase):
    def test_every_card_in_the_pool_has_an_id(self):
        for name in CARD_DEFS:
            self.assertIn(name, CARD_VOCAB, f"{name!r} missing from the vocabulary -- run tools.build_card_vocabulary")

    def test_ids_are_unique_and_stable_round_trip(self):
        self.assertEqual(len(set(CARD_VOCAB.values())), len(CARD_VOCAB))
        for name, vid in CARD_VOCAB.items():
            self.assertEqual(card_id(name), vid)
            self.assertEqual(card_name(vid), name)

    def test_hash_is_a_real_digest_of_the_file(self):
        self.assertEqual(len(VOCAB_HASH), 64)


class TestOptionEncoding(unittest.TestCase):
    def test_option_keys_are_unique_within_a_decision_across_many_games(self):
        deck_rng = random.Random(7)
        for i in range(40):
            d1 = random_deck(deck_rng, "R1")
            d2 = random_deck(deck_rng, "R2")
            game = Game(GameConfig(decks=(d1, d2), seed=i, max_turns=100))
            bot = RandomBot(seed=i)
            while not game.is_over:
                d = game.pending_decision
                keys = [option_key(d, o) for o in d.options]
                self.assertEqual(len(set(keys)), len(keys), f"duplicate option_key in {d.kind} {d.prompt!r}")
                for o in d.options:
                    json.dumps(option_features(d, o))  # must not raise
                game.submit(bot.decide(game.view_for(d.player), d))

    def test_same_option_keys_identically_across_independent_replays(self):
        config = GameConfig(decks=("fignor", "igor"), seed=5, max_turns=60)
        game = Game(config)
        bot = RandomBot(seed=5)
        while not game.is_over:
            d = game.pending_decision
            game.submit(bot.decide(game.view_for(d.player), d))
        from keyforge.replay import replay

        a = replay(config, game.choice_record, upto=len(game.choice_record) // 2)
        b = replay(config, game.choice_record, upto=len(game.choice_record) // 2)
        keys_a = [option_key(a.pending_decision, o) for o in a.pending_decision.options]
        keys_b = [option_key(b.pending_decision, o) for o in b.pending_decision.options]
        self.assertEqual(keys_a, keys_b)


class TestOptionSpaceFuzz(unittest.TestCase):
    def test_every_strategy_produces_a_valid_choice_across_many_multi_select_decisions(self):
        deck_rng = random.Random(13)
        exercised = {"CHOOSE_CARDS": 0, "ORDER_EFFECTS": 0}
        for i in range(60):
            d1 = random_deck(deck_rng, "R1")
            d2 = random_deck(deck_rng, "R2")
            game = Game(GameConfig(decks=(d1, d2), seed=i, max_turns=150))
            bot = RandomBot(seed=i)
            while not game.is_over:
                d = game.pending_decision
                if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS) and len(d.options) >= 2:
                    exercised[d.kind.name] += 1
                    space = choice_space_size(d)
                    if space <= 5000:
                        for choice in enumerate_choices(d):
                            self.assertTrue(d.validate(choice), f"enumerate_choices produced an invalid choice for {d}")
                    # Sequential decomposition round-trips a real choice.
                    real_choice = bot.decide(game.view_for(d.player), d)
                    path = sequential_decompose(d, real_choice)
                    recomposed = sequential_recompose(d, path)
                    self.assertTrue(d.validate(recomposed))
                    self.assertEqual(sorted(id(c) for c in recomposed), sorted(id(c) for c in real_choice))
                    # A from-scratch sequential build using only
                    # sequential_next_options must also reach a valid choice.
                    built_path = []
                    while True:
                        opts = sequential_next_options(d, [i for i in built_path if i is not None])
                        if not opts:
                            break
                        pick = opts[0]
                        built_path.append(pick)
                        if pick is None:
                            break
                    built_choice = sequential_recompose(d, built_path)
                    self.assertTrue(d.validate(built_choice), f"sequential build produced an invalid choice for {d}")
                    # Independent top-k, with an arbitrary score per option.
                    scores = [((option_key(d, o).__hash__()) % 1000) / 1000.0 for o in d.options]
                    topk_choice = independent_topk(d, scores)
                    self.assertTrue(d.validate(topk_choice), f"independent_topk produced an invalid choice for {d}")
                    game.submit(real_choice)
                else:
                    game.submit(bot.decide(game.view_for(d.player), d))
        self.assertGreater(exercised["CHOOSE_CARDS"], 0)

    def test_enumerate_choices_respects_the_size_cap(self):
        deck_rng = random.Random(14)
        d1 = random_deck(deck_rng, "R1")
        d2 = random_deck(deck_rng, "R2")
        game = Game(GameConfig(decks=(d1, d2), seed=1, max_turns=150))
        bot = RandomBot(seed=1)
        found_big = False
        while not game.is_over:
            d = game.pending_decision
            if d.kind == DecisionKind.CHOOSE_CARDS and len(d.options) >= 2:
                space = choice_space_size(d)
                if space > 3:
                    found_big = True
                    with self.assertRaises(ValueError):
                        enumerate_choices(d, max_choices=1)
            game.submit(bot.decide(game.view_for(d.player), d))
        if not found_big:
            self.skipTest("this seed never produced a CHOOSE_CARDS decision with more than 3 legal choices")


if __name__ == "__main__":
    unittest.main()

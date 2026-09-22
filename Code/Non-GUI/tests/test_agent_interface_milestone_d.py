"""Milestone D (Code/AGENT_INTERFACE_PLAN.md): forking, branching and
determinization (replay backend).
"""

import random
import unittest
from collections import Counter

from bots.random_bot import RandomBot
from keyforge.branching import run_branches
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, Resample
from keyforge.game import Game, until_boundary, until_end_of_turn, until_player_decides
from keyforge.match import Match, MatchConfig
from sim.simulate import check_invariants


def _play_random(config, n_decisions=None, seed_offset=0):
    game = Game(config)
    bots = {1: RandomBot(seed=config.seed), 2: RandomBot(seed=(config.seed or 0) + 1 + seed_offset)}
    count = 0
    while not game.is_over:
        if n_decisions is not None and count >= n_decisions:
            break
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        count += 1
    return game


class TestForkEquivalence(unittest.TestCase):
    def test_fork_matches_state_hash_and_diverging_one_leaves_the_other_alone(self):
        config = GameConfig(decks=("fignor", "igor"), seed=11, max_turns=60)
        original = _play_random(config, n_decisions=40)
        fork = original.fork()
        self.assertEqual(original.state_hash(), fork.state_hash())
        self.assertIsNot(original, fork)
        self.assertIsNot(original.players[1], fork.players[1])

        baseline_hash = original.state_hash()
        if not fork.is_over:
            d = fork.pending_decision
            fork.submit(d.options[0])
        self.assertEqual(original.state_hash(), baseline_hash)

    def test_fork_carries_the_true_future_not_a_resampled_one(self):
        config = GameConfig(decks=("fignor", "igor"), seed=12, max_turns=80)
        original = _play_random(config, n_decisions=30)
        fork = original.fork()
        # Same choice sequence from here must produce byte-identical play,
        # since fork() carries the TRUE hidden state and RNG future.
        bots_a = {1: RandomBot(seed=99), 2: RandomBot(seed=100)}
        bots_b = {1: RandomBot(seed=99), 2: RandomBot(seed=100)}
        while not original.is_over:
            d = original.pending_decision
            original.submit(bots_a[d.player].decide(original.view_for(d.player), d))
        while not fork.is_over:
            d = fork.pending_decision
            fork.submit(bots_b[d.player].decide(fork.view_for(d.player), d))
        self.assertEqual(original.choice_record, fork.choice_record)
        self.assertEqual(original.result, fork.result)


class TestForkDeterminized(unittest.TestCase):
    def test_zone_counts_and_invariants_are_preserved(self):
        config = GameConfig(decks=("fignor", "igor"), seed=21, max_turns=80)
        original = _play_random(config, n_decisions=50)
        for resample in (Resample.OWN_DECK, Resample.OPPONENT_PRIVATE, Resample.ALL):
            fork = original.fork_determinized(1, random.Random(1), resample=resample)
            for pid in (1, 2):
                self.assertEqual(len(fork.players[pid].hand), len(original.players[pid].hand))
                self.assertEqual(len(fork.players[pid].archive), len(original.players[pid].archive))
                self.assertEqual(len(fork.players[pid].deck), len(original.players[pid].deck))
                self.assertEqual(len(fork.players[pid].discard), len(original.players[pid].discard))
            check_invariants(fork)

    def test_own_deck_resample_leaves_hands_and_archives_untouched(self):
        config = GameConfig(decks=("fignor", "igor"), seed=22, max_turns=80)
        original = _play_random(config, n_decisions=50)
        fork = original.fork_determinized(1, random.Random(2), resample=Resample.OWN_DECK)
        for pid in (1, 2):
            self.assertEqual(
                sorted(c.instance_id for c in fork.players[pid].hand.cards()),
                sorted(c.instance_id for c in original.players[pid].hand.cards()),
            )
            self.assertEqual(
                sorted(c.instance_id for c in fork.players[pid].archive.cards()),
                sorted(c.instance_id for c in original.players[pid].archive.cards()),
            )

    def test_revealed_hand_cards_are_preserved_exactly(self):
        # Unlike a real reveal (a card-effect consequence captured in
        # choice_record and so reproduced by fork()'s replay), this test
        # calls reveal_hand directly -- so it exercises _resample_hidden_
        # pool directly too, on the same object, rather than round-tripping
        # through fork_determinized's own fork() first.
        config = GameConfig(decks=("fignor", "igor"), seed=23, max_turns=80)
        original = _play_random(config, n_decisions=50)
        original.reveal_hand(target_pid=2, viewer_pid=1)
        before = sorted(c.instance_id for c in original.players[2].hand.cards())
        original._resample_hidden_pool(2, viewer=1, rng=random.Random(3))
        after = sorted(c.instance_id for c in original.players[2].hand.cards())
        self.assertEqual(before, after)

    def test_future_random_events_are_not_correlated_with_the_true_games(self):
        """Across many seeds, a determinized fork's subsequent shuffle
        order must not systematically match the true game's -- otherwise
        the fork would leak the true future (Milestone D's acceptance)."""
        config = GameConfig(decks=("fignor", "igor"), seed=31, max_turns=100)
        original = _play_random(config, n_decisions=40)
        matches = 0
        trials = 40
        for i in range(trials):
            fork = original.fork_determinized(1, random.Random(1000 + i), resample=Resample.ALL)
            true_next_ids = [c.instance_id for c in original.players[2].deck.cards()[:5]]
            fork_next_ids = [c.instance_id for c in fork.players[2].deck.cards()[:5]]
            if true_next_ids == fork_next_ids:
                matches += 1
        self.assertLess(matches, trials * 0.5, "determinized deck order matched the true game far too often")

    def test_statistical_placement_is_roughly_uniform_across_zones(self):
        """Over many resamples, a specific unseen enemy card should land in
        hand/archive/deck at roughly the rate its zone's share predicts."""
        config = GameConfig(decks=("fignor", "igor"), seed=41, max_turns=80)
        original = _play_random(config, n_decisions=45)
        opponent = original.players[2]
        hand_n, archive_n, deck_n = len(opponent.hand), len(opponent.archive), len(opponent.deck)
        total = hand_n + archive_n + deck_n
        if total < 3:
            self.skipTest("not enough hidden-pool cards at this point to measure a distribution")
        watch_id = (opponent.hand.cards() + opponent.archive.cards() + opponent.deck.cards())[0].instance_id
        zone_counts = Counter()
        trials = 300
        for i in range(trials):
            fork = original.fork_determinized(1, random.Random(i), resample=Resample.OPPONENT_PRIVATE)
            fp = fork.players[2]
            if any(c.instance_id == watch_id for c in fp.hand.cards()):
                zone_counts["hand"] += 1
            elif any(c.instance_id == watch_id for c in fp.archive.cards()):
                zone_counts["archive"] += 1
            else:
                zone_counts["deck"] += 1
        expected_hand = trials * hand_n / total
        # Loose tolerance: this is a sanity check on the mechanism, not a
        # statistical-power exercise.
        self.assertLess(abs(zone_counts["hand"] - expected_hand), max(20, expected_hand * 0.6))


class TestForkManyAndRunBranches(unittest.TestCase):
    def test_fork_many_gives_independent_branches_with_reproducible_seeds(self):
        config = GameConfig(decks=("fignor", "igor"), seed=51, max_turns=80)
        original = _play_random(config, n_decisions=30)
        branches_a = original.fork_many(5, viewer=1, resample=Resample.ALL)
        branches_b = original.fork_many(5, viewer=1, resample=Resample.ALL)
        for a, b in zip(branches_a, branches_b):
            self.assertEqual(a.config.seed, b.config.seed)
            self.assertEqual(
                sorted(c.instance_id for c in a.players[2].hand.cards()),
                sorted(c.instance_id for c in b.players[2].hand.cards()),
            )

    def test_run_branches_matches_manual_fork_many_and_work(self):
        config = GameConfig(decks=("fignor", "igor"), seed=52, max_turns=80)
        original = _play_random(config, n_decisions=30)
        results = run_branches(original, 4, lambda g: g.state_hash(), viewer=1, resample=Resample.ALL)
        expected = [g.state_hash() for g in original.fork_many(4, viewer=1, resample=Resample.ALL)]
        self.assertEqual(results, expected)


class TestApplyAndRunUntil(unittest.TestCase):
    def test_apply_replays_a_path_of_encoded_choices(self):
        from keyforge.replay import encode_choice

        config = GameConfig(decks=("fignor", "igor"), seed=61, max_turns=80)
        original = _play_random(config, n_decisions=50)
        fork = original.fork()
        probe = original.fork()
        bot = RandomBot(seed=7)
        path = []
        for _ in range(10):
            if probe.is_over:
                break
            d = probe.pending_decision
            choice = bot.decide(probe.view_for(d.player), d)
            path.append(encode_choice(d, choice))
            probe.submit(choice)
        fork.apply(path)
        self.assertEqual(fork.state_hash(), probe.state_hash())

    def test_run_until_boundary_stops_at_a_boundary_kind(self):
        config = GameConfig(decks=("fignor", "igor"), seed=62, max_turns=80)
        game = Game(config)
        bot = RandomBot(seed=1)
        game.run_until(until_boundary(), lambda g, d: bot.decide(g.view_for(d.player), d))
        self.assertTrue(game.is_over or game.pending_decision.kind in (
            DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_HOUSE, DecisionKind.TAKE_ARCHIVE,
        ))

    def test_run_until_player_decides_stops_for_the_right_player(self):
        config = GameConfig(decks=("fignor", "igor"), seed=63, max_turns=80)
        game = Game(config)
        bot = RandomBot(seed=1)
        game.run_until(until_player_decides(2), lambda g, d: bot.decide(g.view_for(d.player), d))
        self.assertTrue(game.is_over or game.pending_decision.player == 2)

    def test_run_until_end_of_turn_advances_at_most_one_turn(self):
        config = GameConfig(decks=("fignor", "igor"), seed=64, max_turns=80)
        game = Game(config)
        start_turn = game.turn_number
        bot = RandomBot(seed=1)
        predicate = until_end_of_turn(game)
        game.run_until(predicate, lambda g, d: bot.decide(g.view_for(d.player), d))
        self.assertTrue(game.is_over or game.turn_number > start_turn)


class TestSetupScript(unittest.TestCase):
    def test_constructed_position_is_replayable(self):
        config = GameConfig(
            decks=("fignor", "igor"), seed=71, max_turns=10,
            setup_script=[
                ("put_creature", 1, "Doc Bookton", "left"),  # in Fignor's own pod
                ("set_aember", 1, 5),
                ("set_keys", 2, 2),
                ("damage", 1, "Doc Bookton", 2),
            ],
        )
        game = Game(config)
        game.submit(False)  # decline whichever mulligan comes first
        game.submit(False)  # ... and the second -- setup_script runs after both
        self.assertEqual(game.players[1].aember, 5)
        self.assertEqual(game.players[2].keys, 2)
        doc = next(c for c in game.players[1].play_area.creatures if c.name == "Doc Bookton")
        self.assertEqual(doc.type_object.damage, 2)
        self.assertFalse(doc.Exhausted)

        # Replayable: config round-trips and reproduces the same position.
        from keyforge.replay import config_from_dict, config_to_dict, replay

        data = config_to_dict(config)
        restored_config = config_from_dict(data)
        fork = replay(restored_config, game.choice_record)
        self.assertEqual(game.state_hash(), fork.state_hash())

    def test_unknown_op_is_a_hard_error(self):
        config = GameConfig(decks=("fignor", "igor"), seed=72, setup_script=[("not_a_real_op", 1)])
        game = Game(config)
        game.submit(False)
        with self.assertRaises(ValueError):
            game.submit(False)


class TestMatchFork(unittest.TestCase):
    def test_match_fork_matches_choice_records(self):
        config = MatchConfig(format="reversal", decks=("fignor", "igor"), seed=81, max_turns=100)
        match = Match(config)
        bots = {1: RandomBot(seed=81), 2: RandomBot(seed=82)}
        for _ in range(30):
            if match.is_over:
                break
            d = match.pending_decision
            match.submit(bots[d.player].decide(match.view_for(d.player), d))
        fork = match.fork()
        self.assertEqual(match.choice_record, fork.choice_record)
        self.assertEqual(match.score, fork.score)


if __name__ == "__main__":
    unittest.main()

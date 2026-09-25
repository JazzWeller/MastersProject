"""Milestone E (Code/AGENT_INTERFACE_PLAN.md): fast branching backends.

Both E1 (process-fork) and E2 (snapshot-copy). E1's tests need `os.fork()`,
which doesn't exist on Windows, so those are skipped there; run under
WSL/Linux to actually exercise them (this project's own choice of self-play
platform -- see the plan's "Platform and interpreter" section). E2 is pure
Python and runs everywhere.

The equivalence fuzz the plan calls for: at a boundary decision, a backend's
branches must match the replay backend's own `state_hash`, exactly, for the
same seeds -- and the true game must be completely untouched by branching
from it. E1 additionally supports mid-resolution (E2 raises there instead
-- see Game.copy()'s own docstring for why).
"""

import os
import unittest

from bots.heuristic_bot import HeuristicBot
from bots.random_bot import RandomBot
from keyforge.branching import run_branches
from keyforge.branching_fork import available, run_branches_forked
from keyforge.branching_snapshot import run_branches_snapshot
from keyforge.config import GameConfig
from keyforge.effects.effect_object import TriggerEffect
from keyforge.enums import BOUNDARY_KINDS as _BOUNDARY_KINDS
from keyforge.enums import Resample
from keyforge.game import Game


def _play_random(config, n_decisions=None):
    game = Game(config)
    bots = {1: RandomBot(seed=config.seed), 2: RandomBot(seed=(config.seed or 0) + 1)}
    count = 0
    while not game.is_over:
        if n_decisions is not None and count >= n_decisions:
            break
        d = game.pending_decision
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        count += 1
    return game


def _play_to_mid_resolution(config, limit=400):
    """Drives `config` forward, RandomBot on both seats, to the first
    decision whose kind is NOT a boundary kind -- guaranteed mid-resolution
    (e.g. a triggered CHOOSE_CARDS/ORDER_EFFECTS), regardless of exactly
    which cards come up under this seed."""
    game = Game(config)
    bots = {1: RandomBot(seed=config.seed), 2: RandomBot(seed=(config.seed or 0) + 1)}
    for _ in range(limit):
        if game.is_over:
            break
        d = game.pending_decision
        if d.kind not in _BOUNDARY_KINDS:
            return game
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
    raise AssertionError("never reached a mid-resolution decision within the limit")


@unittest.skipUnless(available(), "process-fork backend needs os.fork() (POSIX/WSL/Linux)")
class TestProcessForkBackend(unittest.TestCase):
    def test_exact_branches_match_the_replay_backend_at_a_boundary(self):
        config = GameConfig(decks=("fignor", "igor"), seed=21, max_turns=60)
        game = _play_random(config, n_decisions=30)
        self.assertIn(game.pending_decision.kind, _BOUNDARY_KINDS)
        before = game.state_hash()

        replayed = run_branches(game, 4, lambda b: b.state_hash())
        forked = run_branches_forked(game, 4, lambda b: b.state_hash())
        self.assertEqual(replayed, forked)
        self.assertEqual(game.state_hash(), before, "the true game must be untouched by branching from it")

    def test_exact_branches_match_the_replay_backend_mid_resolution(self):
        """The whole point of this backend: valid at any decision, mid-
        resolution included, where E2 (snapshot-copy) is not."""
        config = GameConfig(decks=("fignor", "igor"), seed=22, max_turns=60)
        game = _play_to_mid_resolution(config)
        self.assertNotIn(game.pending_decision.kind, _BOUNDARY_KINDS)
        before = game.state_hash()

        replayed = run_branches(game, 4, lambda b: b.state_hash())
        forked = run_branches_forked(game, 4, lambda b: b.state_hash())
        self.assertEqual(replayed, forked)
        self.assertEqual(game.state_hash(), before)

    def test_determinized_branches_match_the_replay_backend(self):
        config = GameConfig(decks=("fignor", "igor"), seed=23, max_turns=60)
        game = _play_random(config, n_decisions=25)
        seeds = [101, 202, 303]

        for resample in (Resample.OWN_DECK, Resample.OPPONENT_PRIVATE, Resample.ALL):
            with self.subTest(resample=resample):
                replayed = run_branches(game, 3, lambda b: b.state_hash(), viewer=1, resample=resample, seeds=seeds)
                forked = run_branches_forked(game, 3, lambda b: b.state_hash(), viewer=1, resample=resample, seeds=seeds)
                self.assertEqual(replayed, forked)

    def test_determinized_branch_future_is_decorrelated_from_the_true_game(self):
        """Same acceptance test as fork_determinized's own (Milestone D):
        without the post-resample reseed, a branch's future random draws
        would leak the true game's, since both would derive from the same
        (seed, counters). Checked here by comparing the outcome of running
        each branch to completion against the true game's own continuation
        -- they must not always agree."""
        config = GameConfig(decks=("fignor", "igor"), seed=24, max_turns=60)
        game = _play_random(config, n_decisions=20)

        def _finish(branch):
            bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
            while not branch.is_over:
                d = branch.pending_decision
                branch.submit(bots[d.player].decide(branch.view_for(d.player), d))
            return branch.result

        outcomes = run_branches_forked(
            game, 8, _finish, viewer=1, resample=Resample.ALL, seeds=list(range(8)),
        )
        self.assertGreater(len({str(o) for o in outcomes}), 1, "all 8 determinized branches finished identically")

    def test_a_raising_work_function_surfaces_in_the_parent(self):
        config = GameConfig(decks=("fignor", "igor"), seed=25, max_turns=40)
        game = _play_random(config, n_decisions=10)

        def _boom(branch):
            raise ValueError("boom")

        with self.assertRaises(RuntimeError):
            run_branches_forked(game, 2, _boom)

    def test_a_valid_choice_object_survives_the_pipe(self):
        """`work`'s return value just needs to be picklable -- not only
        primitives. A `Decision`'s own option (a `Card`, an action
        dataclass, ...) round-trips fine."""
        config = GameConfig(decks=("fignor", "igor"), seed=26, max_turns=40)
        game = _play_random(config, n_decisions=15)

        def _first_option(branch):
            d = branch.pending_decision
            return d.options[0] if d.options else None

        results = run_branches_forked(game, 2, _first_option)
        self.assertEqual(len(results), 2)

    def test_available_reports_true_on_this_platform(self):
        self.assertTrue(available())


class TestAvailabilityGuard(unittest.TestCase):
    def test_matches_hasattr_os_fork(self):
        self.assertEqual(available(), hasattr(os, "fork"))


def _play_to_boundary(config, bot_cls=RandomBot, n_decisions=None):
    """Drives `config` forward to the first (or `n_decisions`-th) boundary
    decision -- true for the vast majority of decisions (88.9%, the plan's
    own baseline), so most seeds land on one almost immediately."""
    game = Game(config)
    bots = {1: bot_cls(seed=config.seed), 2: bot_cls(seed=(config.seed or 0) + 1)}
    count = 0
    while not game.is_over:
        d = game.pending_decision
        if d.kind in _BOUNDARY_KINDS and (n_decisions is None or count >= n_decisions):
            return game
        game.submit(bots[d.player].decide(game.view_for(d.player), d))
        count += 1
    raise AssertionError("game finished before reaching the requested boundary")


class TestGameCopy(unittest.TestCase):
    def test_matches_state_hash_at_every_boundary_across_many_games(self):
        """The equivalence fuzz at scale: every boundary decision across 15
        full random games (well over a thousand copy() calls) must match
        state_hash exactly, and never mutate the true game."""
        checked = 0
        for seed in range(15):
            config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=100)
            game = Game(config)
            bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
            while not game.is_over:
                d = game.pending_decision
                if d.kind in _BOUNDARY_KINDS:
                    before = game.state_hash()
                    branch = game.copy()
                    self.assertEqual(branch.state_hash(), before)
                    self.assertEqual(game.state_hash(), before)
                    checked += 1
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertGreater(checked, 1000)

    def test_matches_state_hash_with_heuristic_bot_too(self):
        checked = 0
        for seed in range(10):
            config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=60)
            game = Game(config)
            bots = {1: HeuristicBot(seed=seed), 2: HeuristicBot(seed=seed + 1)}
            while not game.is_over:
                d = game.pending_decision
                if d.kind in _BOUNDARY_KINDS:
                    before = game.state_hash()
                    self.assertEqual(game.copy().state_hash(), before)
                    checked += 1
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
        self.assertGreater(checked, 100)

    def test_raises_off_a_boundary(self):
        config = GameConfig(decks=("fignor", "igor"), seed=1, max_turns=60)
        game = Game(config)
        bots = {1: RandomBot(seed=1), 2: RandomBot(seed=2)}
        while game.pending_decision.kind in _BOUNDARY_KINDS:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            if game.is_over:
                self.skipTest("game finished before reaching a non-boundary decision under this seed")
        with self.assertRaises(ValueError):
            game.copy()

    def test_works_on_a_finished_game(self):
        config = GameConfig(decks=("fignor", "igor"), seed=2, max_turns=60)
        game = Game(config)
        bots = {1: RandomBot(seed=2), 2: RandomBot(seed=3)}
        while not game.is_over:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
        branch = game.copy()
        self.assertTrue(branch.is_over)
        self.assertIsNone(branch.pending_decision)
        self.assertEqual(branch.state_hash(), game.state_hash())

    def test_diverging_the_copy_never_touches_the_original(self):
        config = GameConfig(decks=("fignor", "igor"), seed=3, max_turns=200)
        game = _play_to_boundary(config, n_decisions=25)
        before = game.state_hash()
        branch = game.copy()

        bots = {1: RandomBot(seed=50), 2: RandomBot(seed=51)}
        while not branch.is_over:
            d = branch.pending_decision
            branch.submit(bots[d.player].decide(branch.view_for(d.player), d))

        self.assertEqual(game.state_hash(), before)

    def test_no_mutable_player_container_is_shared_with_the_copy(self):
        """`_copy_player` starts from a shallow `__dict__` copy, so any
        dict/list/set attribute it forgets to rebuild ends up shared between
        branches (ExtraHousePlayable once was). This fails for a new
        container field the moment one is added to Player."""
        config = GameConfig(decks=("fignor", "igor"), seed=4, max_turns=60)
        game = _play_to_boundary(config, n_decisions=10)
        branch = game.copy()
        for pid, old in game.players.items():
            new = branch.players[pid]
            for name, value in vars(old).items():
                if isinstance(value, (dict, list, set)):
                    self.assertIsNot(getattr(new, name), value, f"Player.{name} is shared between the game and its copy")

    def test_end_of_turn_cleanups_resolve_independently_per_branch(self):
        """A converted cleanup (Milestone E2's own motivating example)
        looked up by instance_id against each game's OWN `_cards_by_id` --
        resolving it on one branch must never touch the other's card."""
        from tests.helpers import make_card, put_creature, run_hook

        from keyforge.effects.effect_object import resolve_cleanup
        from keyforge.effects.named import dis

        config = GameConfig(decks=("fignor", "igor"), seed=1, max_turns=50)
        game = _play_to_boundary(config)
        armored = put_creature(game, 2, "Firespitter")
        card = make_card("Red-Hot Armor", 1)
        run_hook(game, dis.red_hot_armor, card)
        self.assertEqual(len(game._end_of_turn_cleanups), 1)

        branch = game.copy()
        branch_armored = branch.card_by_id(armored.instance_id)
        operation, iid = branch._end_of_turn_cleanups[0]
        resolve_cleanup(branch, operation, iid)

        self.assertFalse(branch_armored.armor_negated)
        self.assertTrue(armored.armor_negated, "resolving the branch's cleanup must not affect the original's card")


class TestClosureRebinding(unittest.TestCase):
    """The specific bug `Game.copy()` exists to avoid: a TriggerEffect
    handler that closes over a Card must, after copy(), close over that
    branch's OWN card -- not silently keep reading the original's."""

    def _game_with_a_creature_in_play(self, seed):
        """Drives forward, one boundary at a time, until some creature is
        in play -- robust to exactly which cards a given seed's opening
        hand happens to draw, unlike checking after a fixed decision count."""
        config = GameConfig(decks=("fignor", "igor"), seed=seed, max_turns=60)
        game = _play_to_boundary(config)
        bots = {1: RandomBot(seed=seed), 2: RandomBot(seed=seed + 1)}
        for _ in range(200):
            if game.players[1].play_area.creatures or game.players[2].play_area.creatures:
                return game
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            while game.pending_decision.kind not in _BOUNDARY_KINDS and not game.is_over:
                d = game.pending_decision
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
        raise AssertionError(f"seed {seed}: no creature ever entered play")

    def _first_creature(self, game):
        for pid in (1, 2):
            if game.players[pid].play_area.creatures:
                return game.players[pid].play_area.creatures[0]
        raise AssertionError("no creature in play")

    def test_the_closure_cell_itself_points_at_the_branchs_own_card(self):
        """The most direct check of `_rebind_closure`'s own mechanism:
        inspect the rebound handler's `__closure__` cell contents."""
        game = self._game_with_a_creature_in_play(seed=1)
        creature = self._first_creature(game)

        def handler(g, event):
            return creature.controller  # closes over `creature` from the enclosing scope
            yield

        game.active_effects.add(TriggerEffect(creature, creature.controller, "probe_event", handler))

        branch = game.copy()
        branch_creature = branch.card_by_id(creature.instance_id)
        branch_handler = next(e.handler for e in branch.active_effects.trigger_effects if e.event == "probe_event")

        self.assertIsNot(branch_handler, handler, "the handler must be rebound to a new function, not shared")
        cell_values = [c.cell_contents for c in branch_handler.__closure__]
        self.assertIn(branch_creature, cell_values)
        self.assertNotIn(creature, cell_values)

    def test_controller_change_is_isolated_between_branches(self):
        """A concrete, behavioral version of the above: a trigger handler
        that reads `card.controller` via closure must, after diverging
        control of that card in only one branch, report each branch's own
        value -- never the other's."""
        game = self._game_with_a_creature_in_play(seed=4)
        creature = self._first_creature(game)
        original_controller = creature.controller

        def handler(g, event):
            event["seen"].append(creature.controller)
            return
            yield

        game.active_effects.add(TriggerEffect(creature, creature.controller, "probe", handler))

        branch = game.copy()
        branch_creature = branch.card_by_id(creature.instance_id)
        branch_creature.controller = 3 - original_controller  # diverge: only the branch's card changes

        game_seen, branch_seen = [], []
        for e in game.active_effects.trigger_effects:
            if e.event == "probe":
                list(e.handler(game, {"seen": game_seen}))
        for e in branch.active_effects.trigger_effects:
            if e.event == "probe":
                list(e.handler(branch, {"seen": branch_seen}))

        self.assertEqual(game_seen, [original_controller])
        self.assertEqual(branch_seen, [3 - original_controller])
        self.assertEqual(creature.controller, original_controller, "the original card must be untouched")


class TestSnapshotCopyBackend(unittest.TestCase):
    def test_exact_branches_match_the_replay_backend(self):
        config = GameConfig(decks=("fignor", "igor"), seed=27, max_turns=60)
        game = _play_to_boundary(config, n_decisions=30)
        before = game.state_hash()

        replayed = run_branches(game, 4, lambda b: b.state_hash())
        snapshotted = run_branches_snapshot(game, 4, lambda b: b.state_hash())
        self.assertEqual(replayed, snapshotted)
        self.assertEqual(game.state_hash(), before)

    def test_determinized_branches_match_the_replay_backend(self):
        config = GameConfig(decks=("fignor", "igor"), seed=28, max_turns=60)
        game = _play_to_boundary(config, n_decisions=25)
        seeds = [11, 22, 33]

        for resample in (Resample.OWN_DECK, Resample.OPPONENT_PRIVATE, Resample.ALL):
            with self.subTest(resample=resample):
                replayed = run_branches(game, 3, lambda b: b.state_hash(), viewer=1, resample=resample, seeds=seeds)
                snapshotted = run_branches_snapshot(game, 3, lambda b: b.state_hash(), viewer=1, resample=resample, seeds=seeds)
                self.assertEqual(replayed, snapshotted)

    def test_raises_off_a_boundary_same_as_copy_itself(self):
        config = GameConfig(decks=("fignor", "igor"), seed=29, max_turns=60)
        game = Game(config)
        bots = {1: RandomBot(seed=29), 2: RandomBot(seed=30)}
        while game.pending_decision.kind in _BOUNDARY_KINDS:
            d = game.pending_decision
            game.submit(bots[d.player].decide(game.view_for(d.player), d))
            if game.is_over:
                self.skipTest("game finished before reaching a non-boundary decision under this seed")
        with self.assertRaises(ValueError):
            run_branches_snapshot(game, 2, lambda b: b.state_hash())


if __name__ == "__main__":
    unittest.main()

"""Milestone E (Code/AGENT_INTERFACE_PLAN.md): fast branching backends.

E1 (process-fork) only -- E2 (snapshot-copy) is a separate, larger
rearchitecture tracked separately. `os.fork()` doesn't exist on Windows, so
every test here is skipped there; run under WSL/Linux to actually exercise
it (this project's own choice of self-play platform -- see the plan's
"Platform and interpreter" section).

The equivalence fuzz the plan calls for: at a boundary decision AND a
mid-resolution one, this backend's branches must match the replay
backend's own `state_hash`, exactly, for the same seeds -- and the true
game must be completely untouched by branching from it.
"""

import os
import unittest

from bots.random_bot import RandomBot
from keyforge.branching import run_branches
from keyforge.branching_fork import available, run_branches_forked
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind, Resample
from keyforge.game import Game

_BOUNDARY_KINDS = (DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_HOUSE, DecisionKind.TAKE_ARCHIVE)


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


if __name__ == "__main__":
    unittest.main()

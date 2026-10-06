"""Capability objects: what an agent may do to the live game behind a
decision, enforced by construction (Agent Interface Plan, Milestone G).

The driver hands each agent one of these, bound to the specific game (or
match) a decision came from, matching the agent's registered
`PrivilegeLevel`. An observation-only agent's object has no `fork`/
`fork_determinized` method to call at all -- not merely a policy asking it
not to -- since `ObservationCapability` simply doesn't define one.
"""

from __future__ import annotations

from typing import Any, Optional

from .branching import run_branches
from .enums import PrivilegeLevel, Resample
from .infoset import InfoSet, build_infoset, build_match_infoset
from .observation import Observation, build_match_observation, build_observation, full_state_observation


class ObservationCapability:
    """observation-only (the default): can only ask for its own redacted
    `Observation` of the live game."""

    level = PrivilegeLevel.OBSERVATION

    def __init__(self, game, viewer: int, match=None):
        self._game = game
        self._viewer = viewer
        # The match this game belongs to, if any -- supplies format/score
        # context, and is all there is for a between-games decision
        # (BID_CHAINS/CHOOSE_FIRST_PLAYER), when `game` is None.
        self._match = match

    def observation(self) -> Observation:
        if self._game is None:
            return build_match_observation(self._match, self._viewer)
        return build_observation(self._game, self._viewer, match=self._match)

    def infoset(self) -> InfoSet:
        """The fast, redacted extract a network encoder reads
        (keyforge/infoset.py) -- the same visibility rules as
        `observation()`, a small fraction of the cost."""
        if self._game is None:
            return build_match_infoset(self._match, self._viewer)
        return build_infoset(self._game, self._viewer, match=self._match)


class SearchCapability(ObservationCapability):
    """Adds forking (always determinized -- never the true hidden state)
    and the branch-and-run primitive."""

    level = PrivilegeLevel.SEARCH

    def fork_determinized(self, rng, resample: Resample = Resample.ALL, *, backend: str = "replay",
                          sampler: str = "uniform", weights=None):
        return self._game.fork_determinized(self._viewer, rng, resample=resample, backend=backend, sampler=sampler,
                                            weights=weights)

    @property
    def viewer(self) -> int:
        return self._viewer

    def fork_many(self, n: int, *, resample: Resample = Resample.ALL, seeds: Optional[Any] = None):
        return self._game.fork_many(n, viewer=self._viewer, resample=resample, seeds=seeds)

    def run_branches(self, n: int, work, *, resample: Resample = Resample.ALL, seeds: Optional[Any] = None):
        return run_branches(self._game, n, work, viewer=self._viewer, resample=resample, seeds=seeds)

    def apply(self, forked_game, choice_indices):
        return forked_game.apply(choice_indices)

    def run_until(self, forked_game, predicate, policy):
        return forked_game.run_until(predicate, policy)


class PrivilegedCapability(SearchCapability):
    """Adds the exact (non-determinized) fork and the everything-visible
    observation -- oracle baselines and the hidden-information diagnostic
    (Milestone J) only."""

    level = PrivilegeLevel.PRIVILEGED

    def fork(self):
        return self._game.fork()

    def full_state_observation(self) -> Observation:
        return full_state_observation(self._game, self._viewer)


_CAPABILITY_TYPES = {
    PrivilegeLevel.OBSERVATION: ObservationCapability,
    PrivilegeLevel.SEARCH: SearchCapability,
    PrivilegeLevel.PRIVILEGED: PrivilegedCapability,
}


def make_capability(level: PrivilegeLevel, game, viewer: int, match=None) -> ObservationCapability:
    """`game` may be None only for a between-games match decision, and then
    only an observation-level capability exists: there is no live game to
    fork."""
    if game is None:
        return ObservationCapability(None, viewer, match)
    return _CAPABILITY_TYPES[level](game, viewer, match)

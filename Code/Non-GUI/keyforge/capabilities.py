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
from .observation import Observation, build_observation, full_state_observation


class ObservationCapability:
    """observation-only (the default): can only ask for its own redacted
    `Observation` of the live game."""

    level = PrivilegeLevel.OBSERVATION

    def __init__(self, game, viewer: int):
        self._game = game
        self._viewer = viewer

    def observation(self) -> Observation:
        return build_observation(self._game, self._viewer)


class SearchCapability(ObservationCapability):
    """Adds forking (always determinized -- never the true hidden state)
    and the branch-and-run primitive."""

    level = PrivilegeLevel.SEARCH

    def fork_determinized(self, rng, resample: Resample = Resample.ALL):
        return self._game.fork_determinized(self._viewer, rng, resample=resample)

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


def make_capability(level: PrivilegeLevel, game, viewer: int) -> ObservationCapability:
    return _CAPABILITY_TYPES[level](game, viewer)

"""Regime B: full-game ISMCTS (Agent Training Plan, Milestone M5).

Standard PUCT over every decision regardless of owner; the tree crosses
turn boundaries, and the backup negates the value at every change of
acting player (the core stores each edge's value from the chooser's
perspective). The leaf estimator is applied wherever the leaf falls,
mid-turn included.

Determinization: `OPPONENT_PRIVATE` (the plan's default) or `ALL` -- the
search must invent a concrete opponent hand to let the opponent act at all.
"""

from __future__ import annotations


class FullGame:
    name = "full_game"

    def __init__(self, rollout=None):
        self.rollout = rollout  # unused for values; the fixed policy for over-cap multi-selects

    def classify(self, search, world, decision) -> str:
        return "branch"

    def value_state(self, search, world) -> None:
        return None

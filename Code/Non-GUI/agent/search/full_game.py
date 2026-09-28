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
    """`quiet_leaves` (an option, off by default): play a leaf's *own* turn
    out with the rollout policy before evaluating it, so the estimator is
    only ever asked at the end of a turn -- regime A's evaluation point on
    regime B's tree. It exists to tell a search defect apart from the cost
    of evaluating mid-turn (gate G2 with the heuristic evaluator)."""

    name = "full_game"

    def __init__(self, rollout=None, *, quiet_leaves: bool = False):
        self.rollout = rollout  # the fixed policy for over-cap multi-selects, and quiet-leaf rollouts
        self.quiet_leaves = quiet_leaves

    def classify(self, search, world, decision) -> str:
        return "branch"

    def value_state(self, search, world) -> None:
        if not self.quiet_leaves:
            return None
        start = world.turn_number
        world.run_until(lambda g: g.is_over or g.turn_number > start, self.rollout)

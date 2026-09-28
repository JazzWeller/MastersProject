"""Regime A: within-turn search (Agent Training Plan, Milestone M5).

The tree spans only the acting player's own decisions within the current
turn. A leaf is the **end of the turn**: a newly expanded node is played
out to the end of the turn with the rollout policy (`run_until` the turn
number advances), and the value estimator is applied to that quiet,
consistent point. Opponent decisions that arise mid-turn (a card asking
them to choose) are resolved by the fixed policy without branching, unless
`branch_opponent` is set (the plan's option).

Determinization: the plan names `OWN_DECK` -- what's stochastic within your
own turn is your own draws. That mode leaves the *true* opponent hand in
the world, though, and the mid-turn opponent replies above read it; the
search defaults to `ALL` (same cost) so nothing in a world is ever the true
hidden state, and `OWN_DECK` stays available for the hidden-information
diagnostic, where "knows their hand" is the point.
"""

from __future__ import annotations


class WithinTurn:
    name = "within_turn"

    def __init__(self, rollout, *, branch_opponent: bool = False):
        self.rollout = rollout
        self.branch_opponent = branch_opponent

    def classify(self, search, world, decision) -> str:
        if world.turn_number != search.root_turn:
            return "leaf"
        if decision.player == search.searcher or self.branch_opponent:
            return "branch"
        return "fixed"

    def value_state(self, search, world) -> None:
        """Plays `world` forward to the end of the root turn (or the end of
        the game)."""
        root_turn = search.root_turn
        world.run_until(lambda g: g.is_over or g.turn_number > root_turn, self.rollout)

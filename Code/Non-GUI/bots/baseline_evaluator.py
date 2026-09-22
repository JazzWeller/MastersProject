"""A no-network baseline evaluator (Agent Interface Plan, Milestone J): a
heuristic value function built from `HeuristicBot`'s own fight/aember
judgment, plus a uniform prior over legal options. Lets search mechanics
(ISMCTS availability counts, determinization, subtree reuse) be built and
debugged before any trained network exists.
"""

from __future__ import annotations

from typing import List

_POWER_KEYWORD = "base_power"


def _power(card) -> int:
    return getattr(card.type_object, _POWER_KEYWORD, 0)


def uniform_policy(decision) -> List[float]:
    """A flat prior over `decision.options` -- the search-free baseline
    a real policy head is measured against."""
    n = len(decision.options)
    return [1.0 / n] * n if n else []


def heuristic_value(view) -> float:
    """A rough estimate, in `[-1, 1]`, of `view`'s own player's advantage
    -- not a principled evaluation (it doesn't know card text, board
    position, or house synergy), just enough signal that search mechanics
    can be exercised and tested meaningfully before a trained value
    network exists. Blends keys (the actual win condition, weighted
    highest), Æmber on hand, total friendly vs. enemy board power, and
    hand size."""
    me, opp = view.me(), view.opponent()
    key_term = (me.keys - opp.keys) / 3.0
    aember_term = (me.aember - opp.aember) / 12.0
    my_power = sum(_power(c) for c in me.creatures)
    opp_power = sum(_power(c) for c in opp.creatures)
    power_term = (my_power - opp_power) / 20.0
    hand_term = (me.hand_count - opp.hand_count) / 12.0
    raw = 0.55 * key_term + 0.2 * aember_term + 0.2 * power_term + 0.05 * hand_term
    return max(-1.0, min(1.0, raw))

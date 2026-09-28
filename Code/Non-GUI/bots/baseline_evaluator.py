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


BASE_KEY_COST = 6


def key_cost(view, pid: int) -> int:
    """The current key cost for `pid`, from the view's public duration
    effects (Lash of Broken Dreams, Murmook, ...): numeric "+"/"-" effects
    only -- a conditional or computed modifier is ignored, which is the
    same "rough, not principled" contract as the rest of this module."""
    cost = BASE_KEY_COST
    for e in getattr(view, "active_effects", ()):
        if e.variable == "KeyForgeCost" and e.player_affected == pid and isinstance(e.value, int) and not isinstance(e.value, bool):
            if e.op == "+":
                cost += e.value
            elif e.op == "-":
                cost -= e.value
    return max(0, cost)


def key_progress(keys: int, aember: int, cost: int) -> float:
    """Keys, plus progress toward the next one -- a full key once Æmber
    reaches the cost, since it forges at the start of the next turn unless
    the opponent takes it away first. That threshold is what decides races
    (2 keys + enough Æmber is a win in waiting), so it's worth much more
    than the linear Æmber term alone says."""
    if cost <= 0 or aember >= cost:
        return keys + 1.0
    return keys + 0.6 * aember / cost


def heuristic_value(view) -> float:
    """A rough estimate, in `[-1, 1]`, of `view`'s own player's advantage
    -- not a principled evaluation (it doesn't know card text, board
    position, or house synergy), just enough signal that search mechanics
    can be exercised and tested meaningfully before a trained value
    network exists. Blends key progress (the actual win condition,
    weighted highest -- see `key_progress`), Æmber on hand, total friendly
    vs. enemy board power, and hand size."""
    me, opp = view.me(), view.opponent()
    key_term = (key_progress(me.keys, me.aember, key_cost(view, me.id)) - key_progress(opp.keys, opp.aember, key_cost(view, opp.id))) / 3.0
    aember_term = (me.aember - opp.aember) / 12.0
    my_power = sum(_power(c) for c in me.creatures)
    opp_power = sum(_power(c) for c in opp.creatures)
    power_term = (my_power - opp_power) / 20.0
    hand_term = (me.hand_count - opp.hand_count) / 12.0
    raw = 0.55 * key_term + 0.2 * aember_term + 0.2 * power_term + 0.05 * hand_term
    return max(-1.0, min(1.0, raw))

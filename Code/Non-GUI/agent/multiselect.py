"""Multi-select decisions at the index level (Agent Training Plan, M3
Screen 3): the candidate lists and step sequences the three treatments
are trained and decoded on. Torch-free, shared by `ml.model` (training)
and `agent.agents` (inference), so both sides always agree on what
candidate `i` or step `t` means. `bots.option_space` is the same logic at
the option-object level, for agents that don't need indices.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def enumerate_candidates(kind_is_order: bool, n: int, min_n: int, max_n: int, cap: int) -> List[Tuple[int, ...]]:
    """Every legal submission as a tuple of option indices -- subsets for
    CHOOSE_CARDS, permutations for ORDER_EFFECTS -- matching
    `bots.option_space.enumerate_choices`'s order. Raises past `cap` (a
    config value; never a silent truncation -- M11)."""
    from itertools import combinations, permutations

    if kind_is_order:
        if math.factorial(n) > cap:
            raise ValueError(f"ORDER_EFFECTS over {n} items exceeds enumerate_cap={cap}")
        return list(permutations(range(n)))
    hi = min(max_n, n)
    lo = min(min_n, hi)
    total = sum(math.comb(n, k) for k in range(lo, hi + 1))
    if total > cap:
        raise ValueError(f"CHOOSE_CARDS space {total} exceeds enumerate_cap={cap}")
    out: List[Tuple[int, ...]] = []
    for k in range(lo, hi + 1):
        out.extend(combinations(range(n), k))
    return out


def sequential_steps(kind_is_order: bool, n: int, min_n: int, max_n: int, chosen: Sequence[int]):
    """The training steps for one multi-select target: yields `(prefix
    indices, legal option indices, stop_legal, target)` with `target` = an
    option index or `n` for "stop". CHOOSE_CARDS picks in increasing index
    order (so the k! orderings of one set are one path) and stops
    explicitly when under `max_n`; ORDER_EFFECTS places every item."""
    if kind_is_order:
        remaining = list(range(n))
        for t, c in enumerate(chosen):
            yield list(chosen[:t]), list(remaining), False, c
            remaining.remove(c)
        return
    path = sorted(chosen)
    hi = min(max_n, n)
    for t in range(len(path) + 1):
        prefix = path[:t]
        if t == hi:
            return  # full: no stop decision needed
        last = prefix[-1] if prefix else -1
        need_after = max(0, min_n - t - 1)  # picks still needed after this one
        legal = [i for i in range(last + 1, n) if n - 1 - i >= need_after]
        stop_ok = t >= min_n
        target = path[t] if t < len(path) else n
        yield prefix, legal, stop_ok, target


def decode_topk(scores: Sequence[float], kind_is_order: bool, min_n: int, max_n: int) -> List[int]:
    """The independent top-k treatment's decision rule: ORDER_EFFECTS ranks
    every option by score; CHOOSE_CARDS takes every option whose logit is
    positive, then clamps the count into `[min_n, max_n]` by adding the
    best remaining / dropping the worst chosen. Returns option indices
    (sorted, for CHOOSE_CARDS)."""
    n = len(scores)
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))
    if kind_is_order:
        return ranked
    hi = min(max_n, n)
    lo = min(min_n, hi)
    picked = [i for i in ranked if scores[i] > 0.0]
    k = max(lo, min(hi, len(picked)))
    return sorted(ranked[:k])


def next_sequential_legal(kind_is_order: bool, n: int, min_n: int, max_n: int, prefix: Sequence[int]) -> Tuple[List[int], bool]:
    """(legal option indices, whether "stop" is legal) after `prefix` --
    the inference-time twin of `sequential_steps`. An empty legal list with
    stop illegal means the choice is complete."""
    if kind_is_order:
        return [i for i in range(n) if i not in prefix], False
    hi = min(max_n, n)
    if len(prefix) >= hi:
        return [], False
    last = prefix[-1] if prefix else -1
    need_after = max(0, min_n - len(prefix) - 1)
    return [i for i in range(last + 1, n) if n - 1 - i >= need_after], len(prefix) >= min_n

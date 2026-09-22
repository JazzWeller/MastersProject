"""Three treatments for a multi-select decision (`CHOOSE_CARDS` with
`max_n > 1`, or `ORDER_EFFECTS`), implemented once so an agent's choice of
treatment is a design decision, not a re-derivation (Agent Interface Plan,
Milestone F). `min_n = 0` (choosing nothing) is handled explicitly in all
three that need it.

1. **Enumerate** (`enumerate_choices`): every legal subset (`CHOOSE_CARDS`)
   or permutation (`ORDER_EFFECTS`), for a policy head over the whole
   space. Only viable under a size cap -- the worst observed space in the
   Phase 1.1 pool is 379 (`min_n=0, max_n=2` over 27 options; see
   `tools/bench_engine.py`), so this raises rather than silently building
   an enormous list.
2. **Sequential decomposition** (`sequential_next_options` /
   `sequential_decompose` / `sequential_recompose`): pick option indices
   one at a time, each restricted to indices greater than the last --
   collapses the `k!` orderings that reach the same *set* into one path,
   so a search tree doesn't duplicate work. The engine still only ever
   sees one submitted list; this is purely an agent-side (or training-data)
   representation.
3. **Independent top-k** (`independent_topk`): the `k` highest-scoring
   options, chosen independently of each other. Cheapest, ignores
   interactions between picks entirely, kept as a baseline/ablation.
"""

from __future__ import annotations

import math
from itertools import combinations, permutations
from typing import Any, List, Optional, Sequence, Tuple

from keyforge.enums import DecisionKind

_MULTI_SELECT_KINDS = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)


def choice_space_size(decision) -> int:
    """The number of distinct legal submissions for `decision` -- the same
    count `tools/bench_engine.py` reports, exposed here so a caller can
    check before calling `enumerate_choices`."""
    n = len(decision.options)
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        return math.factorial(n)
    if decision.kind == DecisionKind.CHOOSE_CARDS:
        if n == 0:
            return 1
        hi = min(decision.max_n, n)
        lo = min(decision.min_n, hi)
        return sum(math.comb(n, k) for k in range(lo, hi + 1))
    raise ValueError(f"choice_space_size: {decision.kind} has no multi-select space")


def enumerate_choices(decision, max_choices: Optional[int] = 10_000) -> List[List[Any]]:
    """Every legal submission for `decision`, each as a list ready to
    `submit()`. Raises `ValueError` if the space exceeds `max_choices`
    (pass `None` to disable the cap)."""
    if decision.kind not in _MULTI_SELECT_KINDS:
        raise ValueError(f"enumerate_choices: {decision.kind} is not a multi-select decision")
    space = choice_space_size(decision)
    if max_choices is not None and space > max_choices:
        raise ValueError(f"enumerate_choices: {space} choices exceeds max_choices={max_choices}")
    options = list(decision.options)
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        return [list(p) for p in permutations(options)]
    n = len(options)
    hi = min(decision.max_n, n)
    lo = min(decision.min_n, hi)
    out: List[List[Any]] = []
    for k in range(lo, hi + 1):
        out.extend(list(c) for c in combinations(options, k))
    return out


def sequential_next_options(decision, prefix: Sequence[int]) -> List[Optional[int]]:
    """Legal next INDEX choices (into `decision.options`) after already
    having picked `prefix` (indices, strictly increasing) -- each
    restricted to indices greater than `prefix`'s last, and including the
    "stop here" option `None` once `len(prefix)` satisfies `min_n`.
    `ORDER_EFFECTS` never stops early (it always places every option)."""
    if decision.kind not in _MULTI_SELECT_KINDS:
        raise ValueError(f"sequential_next_options: {decision.kind} is not a multi-select decision")
    n = len(decision.options)
    last = prefix[-1] if prefix else -1
    remaining = list(range(last + 1, n))
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        # ORDER_EFFECTS orders ALL options (min_n == max_n == n), but
        # "greater than the last index" doesn't apply to a re-ordering the
        # same way it does to a subset -- callers order by giving the
        # REMAINING (not-yet-placed) option indices at each step instead.
        return [i for i in range(n) if i not in prefix]
    if len(prefix) >= decision.max_n:
        return []
    if len(prefix) >= decision.min_n:
        return remaining + [None]
    return remaining


def sequential_decompose(decision, chosen: Sequence[Any]) -> List[Optional[int]]:
    """The path of index picks that produces `chosen` -- the inverse of
    `sequential_recompose`, e.g. for turning a completed choice into
    training data for a sequential policy."""
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        return [decision.options.index(c) for c in chosen]
    indices = sorted(decision.options.index(c) for c in chosen)
    if len(indices) < decision.max_n:
        indices = indices + [None]
    return indices


def sequential_recompose(decision, index_path: Sequence[Optional[int]]) -> List[Any]:
    """The inverse of `sequential_decompose`: turns a path of index picks
    (optionally ending in `None`, for `CHOOSE_CARDS`) back into the option
    list `submit()` expects."""
    return [decision.options[i] for i in index_path if i is not None]


def independent_topk(decision, scores: Sequence[float], k: Optional[int] = None) -> List[Any]:
    """The `k` highest-scoring options, chosen independently of each other
    -- ignores interactions between picks entirely (two options might
    individually score high but be redundant together), which is exactly
    why this is the cheap baseline, not the default. `k` defaults to
    `decision.max_n`, clamped to `[min_n, len(options)]`."""
    if decision.kind not in _MULTI_SELECT_KINDS:
        raise ValueError(f"independent_topk: {decision.kind} is not a multi-select decision")
    options = list(decision.options)
    if len(scores) != len(options):
        raise ValueError("independent_topk: one score per option is required")
    if decision.kind == DecisionKind.ORDER_EFFECTS:
        return [options[i] for i in sorted(range(len(options)), key=lambda i: scores[i], reverse=True)]
    if k is None:
        k = decision.max_n
    k = max(decision.min_n, min(k, len(options)))
    ranked = sorted(range(len(options)), key=lambda i: scores[i], reverse=True)
    return [options[i] for i in ranked[:k]]

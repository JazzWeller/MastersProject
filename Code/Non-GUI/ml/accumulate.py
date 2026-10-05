"""Gradient accumulation that leaves the update unchanged (`micro_batch`).

A training batch is split into pieces of at most `micro_batch` rows, each
piece is forwarded and backpropagated on its own, and only one piece's
activations are alive at a time. Training memory then scales with the
piece, not the batch. It is for networks that don't fit otherwise: every
piece repeats the heads' kernel launches and the CPU-side loss
preparation, so on real BC steps (RTX 5060 Ti, 2026-10-05) pieces of 128
cost the 1M-parameter network 60% of its samples/s, pieces of 256 cost
d=256/8 layers 11%, and at d=512/12 layers -- which doesn't fit whole --
pieces of 256 were the fastest option (1.9k samples/s, 4.3 GiB). A d=768/12
network (68M) trains at ~1k samples/s in 4 GiB with pieces of 128.

**The gradient is the whole batch's.** Each loss is a mean over its own
subset of rows -- the policy over single-choice decisions, belief over
unseen entities, the multi-select heads over their decisions -- so a
piece's share of a loss is not its share of the rows. Before any forward
pass, `counts_of(piece)` gives every loss's denominator in that piece;
each piece's loss k is then weighted by `n_k(piece) / n_k(batch)`, which
sums back to the whole batch's mean exactly. The returned losses are the
whole batch's means too.

Dropout masks are drawn per piece, so with dropout on the result equals
the whole batch's in distribution rather than bit for bit.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


def slices(n: int, micro: Optional[int]) -> List[Tuple[int, int]]:
    """Row ranges of the pieces; one piece when `micro` is unset or covers
    the batch."""
    if micro is None or int(micro) >= n:
        return [(0, n)]
    if int(micro) < 1:
        raise ValueError(f"micro_batch must be at least 1, not {micro}")
    m = int(micro)
    return [(a, min(a + m, n)) for a in range(0, n, m)]


def backward_accumulated(
    pieces: Sequence,
    counts_of: Callable[[object], Dict[str, int]],
    loss_of: Callable[[object], Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]],
    weights: Dict[str, float],
    autocast: Callable[[], contextlib.AbstractContextManager],
    stat_counts: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Backpropagates every piece into the parameters' `.grad`; the caller
    zeroes gradients before and steps after. `loss_of(piece)` returns
    `(total, losses, stats)` with each loss a mean over its denominator.
    A stat is averaged with the weights of the loss named in
    `stat_counts` (else by piece rows, `counts_of`'s "rows").
    Returns the whole batch's (losses, stats), detached."""
    counts = [counts_of(p) for p in pieces]
    totals: Dict[str, int] = defaultdict(int)
    for c in counts:
        for k, n in c.items():
            totals[k] += n
    losses: Dict[str, torch.Tensor] = {}
    stats: Dict[str, float] = defaultdict(float)
    stat_counts = stat_counts or {}
    for piece, c in zip(pieces, counts):
        with autocast():
            _total, L, st = loss_of(piece)
        expected = {k for k, n in c.items() if n > 0 and k != "rows"}
        if set(L) != expected:
            raise RuntimeError(f"losses {sorted(L)} don't match counts_of's denominators {c}")
        scaled = {k: v * (c[k] / totals[k]) for k, v in L.items()}
        total = sum(weights.get(k, 1.0) * v for k, v in scaled.items())
        if isinstance(total, torch.Tensor):
            total.backward()
        for k, v in scaled.items():
            v = v.detach().float()
            losses[k] = losses[k] + v if k in losses else v
        for k, v in st.items():
            key = stat_counts.get(k, "rows")
            if c.get(key, 0) > 0:
                stats[k] += v * c[key] / totals[key]
    return losses, dict(stats)

"""Search-free network agents (Agent Training Plan, Milestones M3 and M8):
one forward pass per decision, batched across every concurrent game the
driver hands over in one `decide_many` call.

- `mode="policy"` -- the behaviour-cloned (or self-play) policy head:
  argmax at temperature 0, else a sample at `temperature`.
- `mode="q"` -- Deep Monte-Carlo (M8): argmax of Q(s, a), epsilon-greedy
  during training.

Multi-select decisions (`CHOOSE_CARDS`/`ORDER_EFFECTS`) go through whichever
treatment the network was trained with -- `enumerate` (a distribution over
every legal submission), `sequential` (chained picks, one network call per
pick) or `topk` (independent per-option scores).

The agent reads the redacted `InfoSet` through its capability
(`capability.infoset()`), so it needs only observation-level privilege,
and talks to the network only through an `InferenceClient` -- no torch
here.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional

from bots.base import BatchController
from keyforge.enums import DecisionKind

from ..features import encode
from ..multiselect import decode_topk, enumerate_candidates, next_sequential_legal
from .requests import HEAD_FIXED, HEAD_POLICY, HEAD_Q, HEAD_SEQUENTIAL, HEAD_SUBSET, HEAD_TOPK, Request

_MULTI = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)


def _argmax(xs: List[float]) -> int:
    best, bi = -math.inf, 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


class NetAgent(BatchController):
    needs_view = False

    def __init__(
        self, client, *, mode: str = "policy", multi_select: str = "enumerate", policy_head: str = "pointer",
        temperature: float = 0.0, epsilon: float = 0.0, enumerate_cap: int = 1024, seed: Optional[int] = None,
    ):
        if mode not in ("policy", "q"):
            raise ValueError(f"NetAgent mode must be policy/q, not {mode!r}")
        if multi_select not in ("enumerate", "sequential", "topk"):
            raise ValueError(f"unknown multi_select treatment {multi_select!r}")
        self.client = client
        self.mode = mode
        self.multi_select = multi_select
        self.head = HEAD_Q if mode == "q" else (HEAD_FIXED if policy_head == "fixed" else HEAD_POLICY)
        self.temperature = temperature
        self.epsilon = epsilon
        self.enumerate_cap = enumerate_cap
        self.rng = random.Random(seed)
        self.last_values: List[float] = []

    def decide(self, view, decision, budget=None, capability=None):
        return self.decide_many([(view, decision, budget, capability)])[0]

    # ------------------------------------------------------------------
    def _pick(self, scores: List[float], is_prob: bool, allowed: Optional[List[int]] = None) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(allowed) if allowed else self.rng.randrange(len(scores))
        if self.temperature <= 0 or not is_prob:
            return _argmax(scores)
        weights = [max(p, 0.0) ** (1.0 / self.temperature) for p in scores]
        total = sum(weights)
        if total <= 0:
            return _argmax(scores)
        r = self.rng.random() * total
        for i, w in enumerate(weights):
            r -= w
            if r <= 0:
                return i
        return len(weights) - 1

    def decide_many(self, requests):
        n = len(requests)
        decisions = [d for (_v, d, _b, _c) in requests]
        encs = []
        for (_view, d, _b, cap) in requests:
            if cap is None or not hasattr(cap, "infoset"):
                raise RuntimeError("NetAgent needs an observation capability (run it through sim.driver)")
            encs.append(encode(cap.infoset()))
        choices: List[Any] = [None] * n
        reqs, slots = [], []
        seq_rows = []
        for i, (d, e) in enumerate(zip(decisions, encs)):
            if d.kind in _MULTI:
                ordered = d.kind == DecisionKind.ORDER_EFFECTS
                if self.mode == "q":
                    # DMC (M8) chooses from Q alone: per-option action
                    # values, centred, through the top-k rule.
                    reqs.append(Request(enc=e, head=HEAD_Q))
                    slots.append((i, "q_multi", None))
                elif self.multi_select == "enumerate":
                    try:
                        cands = enumerate_candidates(ordered, len(d.options), d.min_n, d.max_n, self.enumerate_cap)
                    except ValueError:
                        # Over the enumeration cap (the search falls back to
                        # its fixed policy here): per-option scores through
                        # the top-k rule -- always a legal submission.
                        reqs.append(Request(enc=e, head=HEAD_TOPK))
                        slots.append((i, "topk", None))
                        continue
                    reqs.append(Request(enc=e, head=HEAD_SUBSET, candidates=cands, ordered=ordered))
                    slots.append((i, "subset", cands))
                elif self.multi_select == "topk":
                    reqs.append(Request(enc=e, head=HEAD_TOPK))
                    slots.append((i, "topk", None))
                else:
                    seq_rows.append(i)
            else:
                reqs.append(Request(enc=e, head=self.head))
                slots.append((i, "single", None))
        results = self.client.predict_many(reqs) if reqs else []
        values = [0.0] * n
        for (i, how, extra), (scores, value) in zip(slots, results):
            d = decisions[i]
            values[i] = value
            if how == "single":
                choices[i] = d.options[self._pick(scores, self.head != HEAD_Q)]
            elif how == "subset":
                picked = extra[self._pick(scores, True)]
                choices[i] = [d.options[j] for j in picked]
            elif how == "q_multi":
                mean = sum(scores) / len(scores) if scores else 0.0
                centred = [q - mean for q in scores]
                if self.epsilon and self.rng.random() < self.epsilon:
                    centred = [self.rng.uniform(-1, 1) for _ in centred]
                idx = decode_topk(centred, d.kind == DecisionKind.ORDER_EFFECTS, d.min_n, d.max_n)
                choices[i] = [d.options[j] for j in idx]
            else:
                idx = decode_topk(scores, d.kind == DecisionKind.ORDER_EFFECTS, d.min_n, d.max_n)
                choices[i] = [d.options[j] for j in idx]
        if seq_rows:
            prefixes = {i: [] for i in seq_rows}
            active = list(seq_rows)
            while active:
                step_reqs, legal_of = [], {}
                for i in active:
                    d = decisions[i]
                    ordered = d.kind == DecisionKind.ORDER_EFFECTS
                    legal, stop_ok = next_sequential_legal(ordered, len(d.options), d.min_n, d.max_n, prefixes[i])
                    legal_of[i] = (legal, stop_ok)
                    step_reqs.append(Request(enc=encs[i], head=HEAD_SEQUENTIAL, prefix=list(prefixes[i]), legal=legal + ([-1] if stop_ok else [])))
                step_results = self.client.predict_many(step_reqs)
                still = []
                for i, (scores, value) in zip(active, step_results):
                    values[i] = value
                    legal, stop_ok = legal_of[i]
                    allowed = legal + ([len(scores) - 1] if stop_ok else [])
                    masked = [scores[j] if j in allowed else -1.0 for j in range(len(scores))]
                    j = self._pick(masked, True, allowed)
                    if j == len(scores) - 1:
                        continue  # stop
                    prefixes[i].append(j)
                    nxt, nxt_stop = next_sequential_legal(
                        decisions[i].kind == DecisionKind.ORDER_EFFECTS, len(decisions[i].options),
                        decisions[i].min_n, decisions[i].max_n, prefixes[i],
                    )
                    if nxt or nxt_stop:
                        still.append(i)
                active = still
            for i in seq_rows:
                choices[i] = [decisions[i].options[j] for j in prefixes[i]]
        self.last_values = values
        return choices

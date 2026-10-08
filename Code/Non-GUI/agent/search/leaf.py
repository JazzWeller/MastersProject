"""Leaf value estimators (Agent Training Plan, Milestone M6): what the
search asks at a leaf, and the priors for a newly expanded node.

An evaluator's `job(world, search, actor, actions, need_priors,
to_value_state)` is a **generator**: it yields lists of
`agent.agents.requests.Request` (possibly several rounds) and returns
`(priors aligned to actions or None, value from the searcher's
perspective)`. The search core batches every job of a wave -- and
`core.run_searches`, every search of a round -- into one network call.

**The value is always the searcher's**: it is computed from the searcher's
own information set in the world (`build_infoset(world, searcher)`), never
from the other seat's, so no hidden card in the world -- sampled or true --
ever reaches an evaluation (the dataset trains the value head for any
viewer, `ml/dataset.py`'s value-only records). Priors at a node come from
the *actor's* information set in the world: for the searcher's own nodes
that's their own; for an opponent node (regime B) the world's opponent hand
is a determinization sample.

- `HeuristicEvaluator` -- the no-network debug mode: uniform priors and
  `bots.baseline_evaluator.heuristic_value`. Yields nothing.
- `NetworkEvaluator` (S, "student", the default) -- one value evaluation.
- `BeliefOracleEvaluator` (B) -- samples K opponent hands from the belief
  head under the hand-count constraint and averages the oracle value head
  over them: K evaluations, no extra engine forks.
"""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple

from bots.baseline_evaluator import heuristic_value
from keyforge.enums import DecisionKind
from keyforge.infoset import ZONE, build_infoset

from ..agents.requests import HEAD_BELIEF, HEAD_ORACLE, HEAD_POLICY, HEAD_SUBSET, HEAD_VALUE, Request
from ..features import encode

_UNSEEN = ZONE["opp_unseen"]
_MY_DECK = ZONE["my_deck_unordered"]


def _uniform(actions) -> List[float]:
    n = len(actions)
    return [1.0 / n] * n


class HeuristicEvaluator:
    """Uniform priors + the hand-written value function: search mechanics
    (availability, determinization, backup signs, reuse) can be built and
    debugged before any network exists."""

    def job(self, world, search, actor, actions, need_priors: bool, to_value_state: bool):
        priors = _uniform(actions) if need_priors else None
        if to_value_state:
            search.regime.value_state(search, world)
        if world.is_over:
            return priors, float(world.outcome_for(search.searcher))
        return priors, heuristic_value(world.view_for(search.searcher))
        yield  # pragma: no cover - makes this a generator


class _Cache:
    """An LRU over network answers, keyed by the exact network input (the
    encoding's bytes) plus the head and candidates -- determinized search
    revisits the same information sets constantly."""

    def __init__(self, size: int):
        self.size = size
        self.data: "OrderedDict[bytes, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(req: Request) -> bytes:
        h = hashlib.blake2b(req.enc.to_bytes(), digest_size=16)
        h.update(req.head.encode())
        if req.candidates is not None:
            h.update(repr(req.candidates).encode())
        if req.hidden is not None:
            h.update(repr(req.hidden).encode())
        return h.digest()

    def get(self, k):
        v = self.data.get(k)
        if v is not None:
            self.data.move_to_end(k)
            self.hits += 1
        else:
            self.misses += 1
        return v

    def put(self, k, v):
        if self.size <= 0:
            return
        self.data[k] = v
        if len(self.data) > self.size:
            self.data.popitem(last=False)


class NetworkEvaluator:
    """The student estimator (S): one forward pass of the value head at the
    value state, plus the policy (or multi-select subset) head for priors."""

    def __init__(self, cache_size: int = 50_000):
        self.cache = _Cache(cache_size)

    def _cached(self, reqs: List[Request]):
        """Yields only the cache misses; returns every answer in order."""
        keys = [self.cache.key(r) for r in reqs]
        answers: List[Any] = [self.cache.get(k) for k in keys]
        missing = [i for i, a in enumerate(answers) if a is None]
        if missing:
            got = yield [reqs[i] for i in missing]
            for i, a in zip(missing, got):
                answers[i] = a
                self.cache.put(keys[i], a)
        return answers

    def _prior_request(self, world, actor, actions) -> Request:
        d = world.pending_decision
        e = encode(build_infoset(world, actor))
        if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            return Request(enc=e, head=HEAD_SUBSET, candidates=[form for _k, _c, form in actions],
                           ordered=d.kind == DecisionKind.ORDER_EFFECTS)
        return Request(enc=e, head=HEAD_POLICY)

    @staticmethod
    def _align(actions, scores: Sequence[float], multi: bool) -> List[float]:
        if multi:
            priors = list(scores)
        else:
            priors = [scores[form] for _k, _c, form in actions]
        total = sum(priors)
        return [p / total for p in priors] if total > 0 else _uniform(actions)

    def value_requests(self, world, search) -> Tuple[List[Request], Any]:
        """(requests, context) for the searcher's value at `world`. The
        context travels with this one job -- jobs of a wave interleave, so
        nothing per-job may live on the evaluator itself."""
        return [Request(enc=encode(build_infoset(world, search.searcher)), head=HEAD_VALUE)], None

    def value_from(self, answers, ctx):
        """Generator: turns the value answers into the searcher's value
        (a subclass may need further rounds)."""
        return answers[0][1]
        yield  # pragma: no cover

    def job(self, world, search, actor, actions, need_priors: bool, to_value_state: bool):
        reqs: List[Request] = []
        multi = False
        if need_priors:
            multi = world.pending_decision.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)
            reqs.append(self._prior_request(world, actor, actions))
        if to_value_state:
            search.regime.value_state(search, world)
        terminal = world.is_over
        value_reqs, ctx = ([], None) if terminal else self.value_requests(world, search)
        answers = yield from self._cached(reqs + value_reqs)
        priors = self._align(actions, answers[0][0], multi) if need_priors else None
        if terminal:
            return priors, float(world.outcome_for(search.searcher))
        value = yield from self.value_from(answers[len(reqs):], ctx)
        return priors, float(value)


def sample_hand(marginals: Sequence[float], candidates: Sequence[int], hand_count: int, rng: random.Random) -> List[int]:
    """Sequential sampling without replacement under the exact hand-count
    constraint: draw one card at a time in proportion to the remaining
    marginals, renormalizing after each draw (M6)."""
    pool = list(candidates)
    weights = [max(float(marginals[i]), 1e-6) for i in pool]
    hand: List[int] = []
    for _ in range(min(hand_count, len(pool))):
        total = sum(weights)
        r = rng.random() * total
        j = 0
        for j, w in enumerate(weights):
            r -= w
            if r <= 0:
                break
        hand.append(pool.pop(j))
        weights.pop(j)
    return hand


class BeliefOracleEvaluator(NetworkEvaluator):
    """The belief-sampled oracle (B): belief marginals over the opponent's
    unseen cards, K hands sampled under the hand-count constraint, the
    oracle value head on each (with my next draws sampled uniformly from
    my deck -- also unknown to me), averaged."""

    def __init__(self, samples: int = 8, seed: Optional[int] = None, cache_size: int = 50_000):
        super().__init__(cache_size)
        self.samples = samples
        self.rng = random.Random(seed)
        self.next_draws = 5

    def value_requests(self, world, search):
        info = build_infoset(world, search.searcher)
        e = encode(info)
        return [Request(enc=e, head=HEAD_BELIEF)], (info, e)

    def value_from(self, answers, ctx):
        info, e = ctx
        marginals = answers[0][0]
        unseen = [i for i, z in enumerate(info.zones) if z == _UNSEEN]
        my_deck = [i for i, z in enumerate(info.zones) if z == _MY_DECK]
        hand_count = info.players[1][3]  # PL_HAND of the opponent
        reqs = []
        for _ in range(self.samples):
            hand = sample_hand(marginals, unseen, hand_count, self.rng)
            draws = self.rng.sample(my_deck, min(self.next_draws, len(my_deck)))
            reqs.append(Request(enc=e, head=HEAD_ORACLE, hidden=(tuple(sorted(hand)), tuple(draws))))
        got = yield from self._cached(reqs)
        return sum(v for _s, v in got) / len(got)


def make_evaluator(name: str, *, samples: int = 8, seed: Optional[int] = None, cache_size: int = 50_000,
                   history: str = "rows"):
    """`student_v2`: the v2 network (`leaf_v2.py`); `history` is its request
    form (`leaf_v2.history_form` of the network's `history_arch`)."""
    if name == "heuristic":
        return HeuristicEvaluator()
    if name == "student":
        return NetworkEvaluator(cache_size)
    if name == "student_v2":
        from .leaf_v2 import NetworkEvaluatorV2

        return NetworkEvaluatorV2(history, cache_size)
    if name == "belief_oracle":
        return BeliefOracleEvaluator(samples, seed, cache_size)
    raise ValueError(f"unknown leaf estimator {name!r}")

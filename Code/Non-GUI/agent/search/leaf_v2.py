"""The v2 leaf evaluator (Agent Observation Plan, O8): `leaf.NetworkEvaluator`
with v2 encodings and the event history.

`history` names what the network reads, matching its `history_arch`:
- `"rows"` (`joint`, `stream`): the event rows. At a search's start
  (`prepare`) the searcher's history is frozen on the true game, so every
  world carries that prefix; a request then names the prefix by its digest
  (`HistoryRef`) and carries only its world's suffix. The prefix's rows go
  to the server once per search (again only if it answers
  `PREFIX_MISSING`). `naive=True` sends the whole history every time: the
  reference the prefix path is tested against.
- `"folded"` (`turn_tokens`, `summary`): the turn tokens and summaries.
- `"none"`: no history.

The opponent's history inside a world (their priors at their own nodes,
full-game search) is not frozen -- each world rebuilds it from the world's
projection, which O3's `history` option has already cut to what the
searcher saw -- so it always travels whole.

Answers are cached under (the encoding, the head and its arguments, the
whole history's digest).
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Tuple

from keyforge.enums import DecisionKind

from ..agents.requests import (HEAD_POLICY, HEAD_SUBSET, HEAD_VALUE, PREFIX_MISSING, HistoryRef, RequestV2)
from ..features_v2 import encode_v2
from ..history import HistoryFolded, history_for
from .leaf import NetworkEvaluator

HISTORY_FORMS = ("rows", "folded", "none")


def history_form(history_arch: str) -> str:
    """The request form a network's `history_arch` reads."""
    return {"joint": "rows", "stream": "rows", "turn_tokens": "folded", "summary": "folded", "none": "none"}[history_arch]


class NetworkEvaluatorV2(NetworkEvaluator):
    def __init__(self, history: str = "rows", cache_size: int = 50_000, *, naive: bool = False):
        super().__init__(cache_size)
        if history not in HISTORY_FORMS:
            raise ValueError(f"history must be one of {HISTORY_FORMS}, not {history!r}")
        self.history = history
        self.naive = naive
        self._sent = set()  # prefix keys already sent to the server
        self._prefix_bytes = {}  # key -> the prefix's rows (this search's)
        self.stats = {"requests": 0, "suffix_bytes": 0, "prefix_bytes": 0, "whole_bytes": 0, "resent": 0}

    # ------------------------------------------------------------ search
    def prepare(self, capability, search) -> None:
        """Called by `Search.search_gen` before the first fork: freezes the
        searcher's history prefix on the true game."""
        if self.history != "rows" or self.naive:
            return
        history = getattr(capability, "history", None)
        if history is None:
            return
        h = history()
        h.freeze()
        key = h.prefix_digest()
        if key not in self._prefix_bytes:
            self._prefix_bytes = {key: h.prefix_rows().to_bytes()}

    # ---------------------------------------------------------- requests
    def _history(self, world, viewer: int) -> Tuple[Any, str]:
        if self.history == "none":
            return None, ""
        h = history_for(world, viewer)
        if self.history == "folded":
            return HistoryFolded.of(h, world.turn_number), h.digest()
        key = h.prefix_digest()
        if self.naive or h.prefix_n() == 0 or key not in self._prefix_bytes:
            whole = h.to_rows().to_bytes()
            self.stats["whole_bytes"] += len(whole)
            return whole, h.digest()
        suffix = h.suffix_rows().to_bytes()
        self.stats["suffix_bytes"] += len(suffix)
        ref = HistoryRef(key=key, n=h.prefix_n(), suffix=suffix)
        if key not in self._sent:
            self._attach(ref)
        return ref, h.digest()

    def _attach(self, ref: HistoryRef) -> None:
        ref.prefix = self._prefix_bytes[ref.key]
        self._sent.add(ref.key)
        self.stats["prefix_bytes"] += len(ref.prefix)

    def _request(self, world, viewer: int, head: str, **kw) -> RequestV2:
        hist, digest = self._history(world, viewer)
        self.stats["requests"] += 1
        return RequestV2(enc=encode_v2(world, viewer), head=head, history=hist, turn=world.turn_number,
                         digest=digest, **kw)

    def _prior_request(self, world, actor, actions) -> RequestV2:
        d = world.pending_decision
        if d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            return self._request(world, actor, HEAD_SUBSET, candidates=[form for _k, _c, form in actions],
                                 ordered=d.kind == DecisionKind.ORDER_EFFECTS)
        return self._request(world, actor, HEAD_POLICY)

    def value_requests(self, world, search):
        return [self._request(world, search.searcher, HEAD_VALUE)], None

    # ------------------------------------------------------------- cache
    @staticmethod
    def _key(req: RequestV2) -> bytes:
        h = hashlib.blake2b(req.enc.to_bytes(), digest_size=16)
        h.update(req.head.encode())
        h.update(req.digest.encode())
        if req.candidates is not None:
            h.update(repr((req.candidates, req.ordered)).encode())
        if req.hidden is not None:
            h.update(repr(req.hidden).encode())
        return h.digest()

    def _cached(self, reqs: List[RequestV2]):
        """Yields the cache misses (and, if the server lacked a prefix, those
        requests again with it); returns every answer in order."""
        keys = [self._key(r) for r in reqs]
        answers: List[Any] = [self.cache.get(k) for k in keys]
        missing = [i for i, a in enumerate(answers) if a is None]
        if missing:
            got = yield [reqs[i] for i in missing]
            again = [i for i, a in zip(missing, got) if a == PREFIX_MISSING]
            for i, a in zip(missing, got):
                if a != PREFIX_MISSING:
                    answers[i] = a
                    self.cache.put(keys[i], a)
            if again:
                for i in again:
                    self.stats["resent"] += 1
                    self._attach(reqs[i].history)
                got = yield [reqs[i] for i in again]
                for i, a in zip(again, got):
                    if a == PREFIX_MISSING:
                        raise RuntimeError("the inference server dropped a prefix it was just sent")
                    answers[i] = a
                    self.cache.put(keys[i], a)
        return answers

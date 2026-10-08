"""What an engine-side agent asks the network for (Agent Training Plan):
plain data that pickles across the worker -> inference-server pipe, so the
worker never imports torch. `ml.infer_server.TorchModel.predict_many`
answers a list of these with one `(scores, value)` pair each:

- `HEAD_POLICY` / `HEAD_FIXED`: a probability per option (pointer head /
  fixed-vocabulary ablation).
- `HEAD_VALUE`: no scores, just the value.
- `HEAD_SUBSET`: a probability per entry of `candidates` (the enumerate
  treatment of a multi-select decision; `ordered` for ORDER_EFFECTS).
- `HEAD_SEQUENTIAL`: one step of the chained-picks treatment -- a
  probability per option plus a final "stop" entry; `prefix` is what's
  already picked, `legal` the pickable option indices now (-1 = stop).
- `HEAD_TOPK`: an independent logit per option.
- `HEAD_Q`: a Deep Monte-Carlo action value per option (M8).
- `HEAD_BELIEF`: per entity, P(in the opponent's hand) (M6).
- `HEAD_ORACLE`: the oracle value head given a hypothesized hidden state:
  `hidden` = (entity indices in the opponent's hand, entity indices of my
  next draws) -- a *sample*, from the belief head, never the truth (M6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

HEAD_POLICY = "policy"
HEAD_FIXED = "fixed"
HEAD_VALUE = "value"
HEAD_SUBSET = "subset"
HEAD_SEQUENTIAL = "sequential"
HEAD_TOPK = "topk"
HEAD_Q = "q"
HEAD_BELIEF = "belief"
HEAD_ORACLE = "oracle"


@dataclass
class Request:
    enc: Any  # agent.features.Encoded
    head: str = HEAD_POLICY
    candidates: Optional[List[Tuple[int, ...]]] = None
    ordered: bool = False
    prefix: Sequence[int] = field(default_factory=tuple)
    legal: Sequence[int] = field(default_factory=tuple)
    hidden: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None


# ------------------------------------------------- v2 (observation O8) ----


@dataclass
class HistoryRef:
    """A v2 request's event history as a reference to a cached prefix plus
    its own suffix (Agent Observation Plan, O8). `key` is the prefix's
    rolling digest (`HistoryEncoder.prefix_digest`, which starts from both
    decklists); `prefix` carries the prefix's `HistoryRows` bytes when the
    server may not have them yet, else None; `suffix` is the `HistoryRows`
    bytes of the rows after it. A server without the prefix answers
    `PREFIX_MISSING`, and the client sends it again with `prefix` set."""

    key: str
    n: int
    suffix: bytes
    prefix: Optional[bytes] = None


@dataclass
class RequestV2:
    """`Request`'s v2 form: `enc` is an `agent.features_v2.EncodedV2`;
    `history` is a `HistoryRef` (the `joint` / `stream` architectures,
    prefix cached), `HistoryRows` bytes (the whole history -- the naive path,
    kept as the reference), an `agent.history.HistoryFolded` (`turn_tokens` /
    `summary`) or None; `turn` is the current turn number.

    Heads are `Request`'s, with v2's belief and oracle: `HEAD_BELIEF` answers
    per entity (hand, archive, deck, next draw), flattened entity by entity;
    `HEAD_ORACLE`'s `hidden` is (the opponent's hand, their archive, my next
    draws) as entity indices."""

    enc: Any
    head: str = HEAD_POLICY
    history: Any = None
    turn: int = 0
    candidates: Optional[List[Tuple[int, ...]]] = None
    ordered: bool = False
    prefix: Sequence[int] = field(default_factory=tuple)
    legal: Sequence[int] = field(default_factory=tuple)
    hidden: Optional[Tuple[Tuple[int, ...], ...]] = None
    digest: str = ""  # the whole history's digest: with `enc`, what an answer is cached under


# The answer to a request whose `HistoryRef` names a prefix the server
# doesn't hold (evicted, or a restarted server): resend with `prefix` set.
PREFIX_MISSING = (None, None)

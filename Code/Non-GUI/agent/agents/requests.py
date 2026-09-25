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


@dataclass
class Request:
    enc: Any  # agent.features.Encoded
    head: str = HEAD_POLICY
    candidates: Optional[List[Tuple[int, ...]]] = None
    ordered: bool = False
    prefix: Sequence[int] = field(default_factory=tuple)
    legal: Sequence[int] = field(default_factory=tuple)

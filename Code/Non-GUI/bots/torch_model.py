"""A real (if intentionally small) GPU-resident policy/value network for
`bots.inference_client.InferenceServer` (Agent Interface Plan, Milestone H).

**The CUDA rule.** This module is the ONE place in this repository that may
import `torch` -- never `keyforge`, `bots.base`, `bots.registry`, or `sim`,
which stay stdlib-only (PyPy compatibility; the process-fork backend,
keyforge/branching_fork.py, forbids a forking process from having
initialized CUDA at all, since CUDA does not survive `fork()`). Nothing
imports this module at package load time; a caller opts in explicitly by
importing `bots.torch_model` itself, in the one process meant to own the
GPU (the plan's own inference server, never an engine worker).

Featurizing a position into a plain list of floats (`featurize`) happens
here too, deliberately kept as plain Python/stdlib -- so it could be
reused by a non-torch model (or tested) without pulling torch in.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from keyforge.cards.vocabulary import CARD_VOCAB
from keyforge.enums import House

# Fixed-vocabulary policy head (Milestone F's card_vocabulary.json) plus a
# handful of extra slots for the non-card options a decision can offer.
N_HOUSES = len(House)
EXTRA_SLOTS = ("end_turn", "yes", "no") + tuple(f"house_{h.value}" for h in House)
POLICY_SIZE = len(CARD_VOCAB) + len(EXTRA_SLOTS)
_EXTRA_INDEX = {name: len(CARD_VOCAB) + i for i, name in enumerate(EXTRA_SLOTS)}

STATE_FEATURES = 14


def option_policy_index(decision, option) -> int:
    """Where `option` lives in the fixed-vocabulary policy head's output --
    a card option (Play/Discard/UseAction/UseOmni/Reap/Fight/a bare Card)
    by its printed name's vocabulary id, everything else by one of
    `EXTRA_SLOTS`. Raises `KeyError` for a card not yet in the vocabulary
    (see vocabulary.py's own docstring: rebuild it after a pool change)."""
    from keyforge.cards.card import Card
    from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni

    if isinstance(option, (PlayCard, DiscardCard, UseAction, UseOmni, Reap, Fight)):
        return CARD_VOCAB[option.card.name]
    if isinstance(option, Card):
        return CARD_VOCAB[option.name]
    if isinstance(option, EndTurn):
        return _EXTRA_INDEX["end_turn"]
    if isinstance(option, House):
        return _EXTRA_INDEX[f"house_{option.value}"]
    if option is True:
        return _EXTRA_INDEX["yes"]
    if option is False:
        return _EXTRA_INDEX["no"]
    raise KeyError(f"no fixed policy slot for option {option!r} -- multi-select/CHOOSE_CARDS decisions need a different head")


def featurize(view) -> List[float]:
    """A small, fixed-size (`STATE_FEATURES`-long) numeric summary of one
    `PlayerView` -- own and opponent aember, chains, keys, hand/deck/
    discard/archive COUNTS (`PlayerPublicState`'s own always-visible count
    fields -- unlike `.hand`/`.archive` themselves, never `None` regardless
    of which side is being featurized), creature/artifact counts. On
    purpose simple: this is the reference featurization Milestone H's
    inference server plugs in by default, not a claim that it's a good one
    -- see bots/baseline_evaluator.py's own heuristic_value for the same
    spirit applied to a hand-written policy."""
    me = view.me()
    opp = view.opponent()
    return [
        float(me.aember), float(me.chains), float(me.keys),
        float(me.hand_count), float(me.deck_count), float(len(me.discard)), float(me.archive_count),
        float(len(me.creatures)), float(len(me.artifacts)),
        float(opp.aember), float(opp.chains), float(opp.keys),
        float(len(opp.creatures)), float(len(opp.artifacts)),
    ]


class KeyForgeNet(nn.Module):
    """input (STATE_FEATURES) -> shared trunk -> policy logits
    (POLICY_SIZE, fixed vocabulary) + a scalar value in [-1, 1]."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, POLICY_SIZE)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)


class TorchInferenceModel:
    """A callable `observation -> (policy, value)` (the shape `bots.
    inference_client.InferenceClient`/`InferenceServer` expect), backing a
    `KeyForgeNet` on `device`. `observation` here is already a plain
    feature vector (`featurize`'s own output) -- kept that way so this
    class, and everything downstream of it, stays free of any keyforge
    import beyond the vocabulary/enum modules used for `featurize` itself."""

    def __init__(self, net: "KeyForgeNet", device: str = "cuda"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, observation: List[float]) -> Tuple[List[float], float]:
        x = torch.tensor([observation], dtype=torch.float32, device=self.device)
        logits, value = self.net(x)
        policy = torch.softmax(logits[0], dim=-1).tolist()
        return policy, float(value.item())

    @torch.no_grad()
    def predict_many(self, observations: List[List[float]]) -> List[Tuple[List[float], float]]:
        """Real batched inference -- one GPU call for the whole list,
        exactly the case `InferenceServer`'s cross-request batching
        (`bots.inference_client`) exists to feed."""
        if not observations:
            return []
        x = torch.tensor(observations, dtype=torch.float32, device=self.device)
        logits, values = self.net(x)
        policies = torch.softmax(logits, dim=-1).tolist()
        return list(zip(policies, values.tolist()))


def build_model(device: str = "cuda", hidden: int = 128) -> TorchInferenceModel:
    """A freshly-initialized (untrained -- random weights) model on
    `device`, falling back to CPU if CUDA isn't actually available (so code
    written against this keeps working, just slower, on a machine without
    a GPU) -- callers that need to KNOW whether CUDA was actually used
    should check `model.device` on the result."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return TorchInferenceModel(KeyForgeNet(hidden=hidden), device=device)

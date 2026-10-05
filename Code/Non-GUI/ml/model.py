"""The network (Agent Training Plan, Milestone M2): one trunk, and every
head the plan's arms need.

**Trunk.** 73 tokens -- one global token plus the 72 card entities, always
(both decklists are public) -- through pre-LN transformer layers (`ml/layers.py`), with no
positional encoding: the entity set is unordered, and everything positional
(flank, battleline index, zone) is already an explicit feature. Each entity
enters as `[card embedding ; STATIC[card_id] ; ENTITY]`: the learned
identity embedding and the attribute row are both there (`identity` =
`both`), or either one alone for the M3 Screen 4/5 ablations.

**Heads.**
- `policy_logits` -- pointer scoring over exactly the offered options: an
  option's embedding is built from its own features plus the embedding of
  the entity it points at, and scored against a query from the global
  token. Illegal options are unrepresentable, not masked.
- `fixed_logits` -- the fixed-vocabulary ablation: one output per
  `(verb, card)` / `(verb, payload)` slot, gathered at the offered options.
- `value` -- tanh scalar, from the deciding player's perspective.
- multi-select (`CHOOSE_CARDS`/`ORDER_EFFECTS`), three treatments trained
  side by side for M3 Screen 3: `subset_scores` (enumerate), the
  `sequential_logits` chain, `topk_logits` (independent per-option).
- `belief_logits` (M6) -- per entity: in the opponent's hand vs their deck.
- `oracle_value` (M6) -- value given the true hidden state (training only;
  its extra input comes from the privileged label file).
- `q_values` (M8) -- Deep Monte-Carlo action values, the action as input.

This module never touches an engine object: its inputs are `ml.encode.Batch`
tensors and plain index lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from agent import spec

from .encode import Batch, entity_block, static_table_tensor
from .layers import build_trunk

G = spec.GLOBAL.width
O = spec.OPTION.width
S = spec.STATIC.width
E = spec.ENTITY.width
N = spec.N_ENTITIES
VERB_OFF, VERB_W = spec.OPTION.span("verb")
MAX_ORDER = 8  # ORDER_EFFECTS permutation positions with their own embedding
FIXED_PAYLOAD_BUCKETS = 32


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


@dataclass
class TrunkOut:
    g: torch.Tensor  # [B, d] global token
    h: torch.Tensor  # [B, 72, d] entity tokens
    e_opt: torch.Tensor  # [B, K, d] option embeddings
    option_mask: torch.Tensor  # [B, K]


class KeyForgeNet(nn.Module):
    def __init__(self, net_cfg: dict, vocab_size: int = spec.VOCAB_CAPACITY):
        super().__init__()
        d = int(net_cfg["d_model"])
        self.d = d
        self.net_cfg = dict(net_cfg)
        self.identity = net_cfg.get("identity", "both")
        if self.identity not in ("id", "attr", "both"):
            raise ValueError(f"network.identity must be id/attr/both, not {self.identity!r}")
        emb = int(net_cfg["card_embed"])
        self.vocab_size = vocab_size
        # M3 Screen 4: global fields zeroed on input (e.g. ["intent"] --
        # "what did interface plan B's DecisionIntent buy?").
        ablate = torch.ones(G)
        for name in net_cfg.get("ablate_globals", ()) or ():
            off, w = spec.GLOBAL.span(name)
            ablate[off : off + w] = 0.0
        self.register_buffer("global_keep", ablate, persistent=False)
        table = static_table_tensor()
        if table.shape[0] < vocab_size:
            table = torch.cat([table, torch.zeros(vocab_size - table.shape[0], S)])
        self.register_buffer("static_table", table[:vocab_size].clone(), persistent=False)
        self.card_embedding = nn.Embedding(vocab_size, emb)
        self.entity_in = nn.Linear(emb + S + E, d)
        self.entity_norm = nn.LayerNorm(d)
        self.global_in = nn.Linear(G, d)
        self.global_norm = nn.LayerNorm(d)
        # `attention` picks the layer implementation, not the function: both
        # have the same parameters and state-dict keys (ml/layers.py), so a
        # checkpoint saved before the option existed loads into the faster one.
        self.trunk = build_trunk(
            d, int(net_cfg["heads"]), int(net_cfg["ff"]), int(net_cfg["layers"]), float(net_cfg.get("dropout", 0.0)),
            net_cfg.get("attention", "sdpa"), net_cfg.get("attention_kernel", "flash"),
        )
        self.out_norm = nn.LayerNorm(d)

        # Options.
        self.option_in = nn.Linear(O, d)
        self.null_entity = nn.Parameter(torch.zeros(d))
        self.option_mlp = _mlp(2 * d, d, d)
        self.policy_query = nn.Linear(d, d)
        # Value.
        self.value_head = _mlp(d, d, 1)
        # Fixed-vocabulary ablation -- built only when asked for: at ~1M
        # parameters it would otherwise be half the network, unused.
        self.fixed_slots = VERB_W * vocab_size + VERB_W * FIXED_PAYLOAD_BUCKETS
        self.fixed_head = nn.Linear(d, self.fixed_slots) if net_cfg.get("policy_head") == "fixed" else None
        # Multi-select: enumerate.
        self.subset_attn = nn.Linear(d, 1)
        self.subset_size = nn.Embedding(64, d)
        self.order_position = nn.Embedding(MAX_ORDER, d)
        self.empty_subset = nn.Parameter(torch.zeros(d))
        self.subset_mlp = _mlp(d, d, d)
        self.subset_query = nn.Linear(d, d)
        # Multi-select: sequential.
        self.seq_query = nn.Linear(d, d)
        self.seq_prefix = nn.Linear(d, d)
        self.seq_stop = nn.Parameter(torch.zeros(d))
        # Multi-select: independent top-k.
        self.topk_head = _mlp(2 * d, d, 1)
        # M6 auxiliaries.
        self.belief_head = _mlp(d, d // 2, 1)
        self.oracle_hidden = nn.Linear(2 * vocab_size, d)
        self.oracle_head = _mlp(2 * d, d, 1)
        # M8 Deep Monte-Carlo.
        self.q_head = _mlp(2 * d, d, 1)

    # ---------------------------------------------------------------- trunk
    def encode_state(self, batch: Batch) -> TrunkOut:
        ids = batch.card_ids
        ident = self.card_embedding(ids)
        static = self.static_table[ids]
        if self.identity == "id":
            static = torch.zeros_like(static)
        elif self.identity == "attr":
            ident = torch.zeros_like(ident)
        ent = torch.cat([ident, static, entity_block(batch)], dim=-1)
        tokens = torch.cat(
            [self.global_norm(self.global_in(batch.globals * self.global_keep)).unsqueeze(1), self.entity_norm(self.entity_in(ent))], dim=1,
        )
        x = self.out_norm(self.trunk(tokens))
        g, h = x[:, 0], x[:, 1:]
        return TrunkOut(g=g, h=h, e_opt=self._options(batch, h), option_mask=batch.option_mask)

    def _options(self, batch: Batch, h: torch.Tensor) -> torch.Tensor:
        B, K = batch.pointers.shape
        ptr = batch.pointers.clamp(min=0)
        pointed = torch.gather(h, 1, ptr.unsqueeze(-1).expand(B, K, self.d))
        has = (batch.pointers >= 0).unsqueeze(-1)
        pointed = torch.where(has, pointed, self.null_entity.expand(B, K, self.d))
        return self.option_mlp(torch.cat([self.option_in(batch.options), pointed], dim=-1))

    # ---------------------------------------------------------------- heads
    def policy_logits(self, out: TrunkOut) -> torch.Tensor:
        q = self.policy_query(out.g).unsqueeze(1)
        logits = (q * out.e_opt).sum(-1) / math.sqrt(self.d)
        return logits.masked_fill(~out.option_mask, float("-inf"))

    def value(self, out: TrunkOut) -> torch.Tensor:
        return torch.tanh(self.value_head(out.g)).squeeze(-1)

    def fixed_index(self, batch: Batch) -> torch.Tensor:
        """[B, K] slot per option: `verb * vocab + card` for an option that
        points at a card, else a `(verb, payload)` bucket."""
        verb = batch.options[..., VERB_OFF : VERB_OFF + VERB_W].argmax(-1)
        card = torch.gather(batch.card_ids, 1, batch.pointers.clamp(min=0))
        payload = batch.options[..., VERB_OFF + VERB_W :]
        weights = torch.arange(1, payload.shape[-1] + 1, device=payload.device, dtype=payload.dtype)
        bucket = ((payload * weights).sum(-1) * 7).round().long().remainder(FIXED_PAYLOAD_BUCKETS)
        pointed = verb * self.vocab_size + card
        other = VERB_W * self.vocab_size + verb * FIXED_PAYLOAD_BUCKETS + bucket
        return torch.where(batch.pointers >= 0, pointed, other)

    def fixed_logits(self, out: TrunkOut, batch: Batch) -> torch.Tensor:
        if self.fixed_head is None:
            raise RuntimeError("this network was built with policy_head='pointer'; it has no fixed-vocabulary head")
        all_logits = self.fixed_head(out.g)
        logits = torch.gather(all_logits, 1, self.fixed_index(batch))
        return logits.masked_fill(~out.option_mask, float("-inf"))

    def topk_logits(self, out: TrunkOut) -> torch.Tensor:
        g = out.g.unsqueeze(1).expand_as(out.e_opt)
        return self.topk_head(torch.cat([out.e_opt, g], dim=-1)).squeeze(-1).masked_fill(~out.option_mask, float("-inf"))

    def q_values(self, out: TrunkOut) -> torch.Tensor:
        g = out.g.unsqueeze(1).expand_as(out.e_opt)
        return self.q_head(torch.cat([out.e_opt, g], dim=-1)).squeeze(-1)

    def belief_logits(self, out: TrunkOut) -> torch.Tensor:
        """[B, 72]: logit that each entity is in the opponent's HAND rather
        than their deck. Only meaningful at `opp_unseen` entities."""
        return self.belief_head(out.h).squeeze(-1)

    def oracle_value(self, out: TrunkOut, hidden_counts: torch.Tensor) -> torch.Tensor:
        """`hidden_counts` [B, 2*vocab]: the opponent's actual hand, then my
        actual next k draws, as count vectors over the vocabulary."""
        hid = F.gelu(self.oracle_hidden(hidden_counts))
        return torch.tanh(self.oracle_head(torch.cat([out.g, hid], dim=-1))).squeeze(-1)

    # ------------------------------------------------- multi-select: enumerate
    def subset_scores(
        self, out: TrunkOut, rows: torch.Tensor, members: torch.Tensor, ordered: torch.Tensor,
    ) -> torch.Tensor:
        """Scores for candidate submissions. `rows` [C]: which batch row
        each candidate belongs to; `members` [C, M]: option indices in the
        candidate (-1 padded; an all-padding row is the empty set);
        `ordered` [C] bool: position matters (ORDER_EFFECTS). Returns [C]."""
        C, M = members.shape
        e = out.e_opt[rows]  # [C, K, d]
        idx = members.clamp(min=0)
        mem = torch.gather(e, 1, idx.unsqueeze(-1).expand(C, M, self.d))  # [C, M, d]
        valid = members >= 0
        pos = self.order_position(torch.arange(M, device=mem.device).clamp(max=MAX_ORDER - 1))
        mem = mem + ordered.view(C, 1, 1).to(mem.dtype) * pos.unsqueeze(0)
        att = self.subset_attn(mem).squeeze(-1).masked_fill(~valid, float("-inf"))
        size = valid.sum(-1)
        empty = size == 0
        w = torch.softmax(att.masked_fill(empty.unsqueeze(-1), 0.0), dim=-1) * valid
        pooled = (w.unsqueeze(-1) * mem).sum(1)
        pooled = torch.where(empty.unsqueeze(-1), self.empty_subset.expand(C, self.d), pooled)
        s = self.subset_mlp(pooled + self.subset_size(size.clamp(max=63)))
        q = self.subset_query(out.g[rows])
        return (q * s).sum(-1) / math.sqrt(self.d)

    # ------------------------------------------------ multi-select: sequential
    def sequential_logits(self, out: TrunkOut, rows: torch.Tensor, prefix: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """One step of the chained-picks treatment. `rows` [S]; `prefix`
        [S, K] 0/1 -- options already picked; `legal` [S, K+1] bool -- which
        options (and, in the last column, "stop") may be picked now.
        Returns [S, K+1] logits, illegal entries -inf."""
        e = out.e_opt[rows]  # [S, K, d]
        summary = (prefix.unsqueeze(-1).to(e.dtype) * e).sum(1)
        q = self.seq_query(out.g[rows] + self.seq_prefix(summary))
        opt_logits = (q.unsqueeze(1) * e).sum(-1)
        stop_logit = (q * self.seq_stop).sum(-1, keepdim=True)
        logits = torch.cat([opt_logits, stop_logit], dim=-1) / math.sqrt(self.d)
        return logits.masked_fill(~legal, float("-inf"))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compile_trunk(model: "KeyForgeNet", enabled: bool, device) -> None:
    """`torch.compile`s the transformer trunk in place (state_dict keys
    unchanged), on CUDA only. The trunk is the fixed-shape [B, 73, d] part,
    and fusing its layer norms, activations and autocast casts cuts kernel
    launches -- the training step is launch-bound (+18% samples/s,
    2026-10-03). The heads take a different shape every batch and stay
    eager. Needs a C compiler (Triton builds its launcher)."""
    if enabled and torch.device(device).type == "cuda":
        model.trunk.compile()


def amp_dtype(precision: str, device) -> Optional[torch.dtype]:
    """The autocast dtype for a training `precision` setting; None means
    plain fp32. bf16 needs no gradient scaling, and applies on CUDA only.
    Measured on the RTX 5060 Ti with Tier 0's BC steps: 1.5x faster than
    fp32 at the same loss."""
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16 if torch.device(device).type == "cuda" else None
    raise ValueError(f"unknown training precision {precision!r} (fp32 | bf16)")


# ------------------------------------------------------ candidate helpers
# The index-level enumeration/sequencing lives torch-free in
# agent.multiselect, so engine-side agents build exactly the same candidate
# lists the heads are trained on.
from agent.multiselect import enumerate_candidates, sequential_steps  # noqa: E402,F401


def candidates_tensor(cands: Sequence[Sequence[int]], device=None) -> torch.Tensor:
    width = max((len(c) for c in cands), default=0)
    t = torch.full((len(cands), max(width, 1)), -1, dtype=torch.int64)
    for i, c in enumerate(cands):
        if c:
            t[i, : len(c)] = torch.tensor(list(c), dtype=torch.int64)
    return t.to(device) if device is not None else t

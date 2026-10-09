"""Network v2 (Agent Observation Plan, Milestone O7). v1's `ml/model.py` is
untouched.

**One trunk over typed token sets**: the global token, the 72 entities,
effects, resolution frames (with their pointer and operation rows pooled
in), cleanups, match tokens, options (`options_in_trunk`: options attend
to the state and history before they are scored), and history in the
chosen representation. Each type has its own input projection plus a type
embedding.

**Columns.** Every int column of `agent/spec_v2.py` is an id (an
embedding), a bit set (unpacked), an entity pointer, or a number (signed
log scale); float columns go in as they are.

**Pointer enrichment at input.** A pointer adds its column's role
embedding plus the pointed entity's identity (card embedding + static
meaning); NO_POINTER / HIDDEN_MINE / HIDDEN_THEIRS have learned vectors.

**Static meaning**: the attribute row, the engine signature, the parsed
text rows (a small set encoder), and the text embedding if its table
exists. `static` (default all) picks which (O10's rung A7).

**Knowledge** (`knowledge`, default on; O10's rungs A1/A2): off zeroes the
O2 tracker's knowledge beyond a card's exactly known zone -- the
possible-zone masks, deck positions, provenance groups, last sightings, how
a card left view, what the opponent knows -- and O3's priors, at the input.

**Time**: sinusoidal features -- no table, no cap. `history_time`:
`absolute` (the row's turn and sequence index, with the current turn on the
global token, so attention can still take the difference) or `relative`
(the turn offset from now). `stream` needs `absolute`: a row's features
must not change while a search runs, or its prefix couldn't be cached.

**Attention** (`ml/layers.AttnSpec`): key-padding masks only; `stream`'s
history is a separate causal attention over the history rows. A cached
prefix (`history_prefix` -> `PrefixPast`) is each layer's keys and values
for a search's prefix rows (Agent Observation Plan, O8): `encode_state(batch,
past)` then runs only the state tokens and the leaf's suffix rows, with the
same outputs as the whole sequence.

**History** (`model.history_arch`): `joint` (history tokens in the trunk),
`stream` (block-causal: history attends only to earlier history; state
attends to everything), `turn_tokens`, `summary` (entity and global
features), `none`.

**Heads**: v1's (policy, value, top-k, Q, multi-select enumerate and
sequential) with the same interfaces; belief v2 (per entity, hand /
archive / deck, plus P(next draw); O3's exact prior is an input, so the
head learns the difference); the oracle over hand, archive and next
draws; optional auxiliaries (the opponent's next house, next card).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from agent import spec_v2 as S
from agent.history import BOOLS, ROW_FLOATS, ROW_INTS
from agent.vocab import load_all
from keyforge.cards.vocabulary import CARD_VOCAB
from keyforge.enums import Affects, DecisionIntent, DecisionKind

from agent.history import SUMMARY_ENTITY, SUMMARY_GLOBAL, TURN_SCALARS

from .encode_v2 import BatchV2, static_tables_tensor
from .layers import AttnSpec, build_trunk

_V = load_all()
NZ = S.N_ZONE_CODES + 1
VOCAB = max(CARD_VOCAB.values()) + 1
HOUSES = 8

ID, BITS, PTR = "id", "bits", "ptr"


def _n(name):
    return len(_V[name])


COLUMNS: Dict[str, Dict[str, tuple]] = {
    "entity": {
        "card": (ID, VOCAB), "owner": (ID, 3), "exact": (ID, 2), "zone": (ID, NZ), "last_seen": (ID, NZ),
        "exit_op": (ID, _n("journal_ops")), "opp_exact": (ID, 2), "visible": (ID, 2), "controller": (ID, 3),
        "position": (ID, NZ), "flank": (ID, 5), "house": (ID, HOUSES), "house_override": (ID, HOUSES),
        "destined_zone": (ID, _n("destined_zones")), "temp_original": (ID, 3), "selected": (ID, 2),
        "mask": (BITS, NZ), "opp_mask": (BITS, NZ), "flags": (BITS, len(S.ENTITY_FLAGS)),
        "keywords": (BITS, _n("keywords")), "extra_triggers": (BITS, _n("events")),
        "host": (PTR,), "redirect_to": (PTR,), "purged_by": (PTR,),
    },
    "global": dict(
        {c: (ID, 2) for c in ("is_my_turn", "i_went_first", "has_max_turns", "elusive_suppressed", "is_over",
                              "optional", "swapped")},
        active_house=(ID, HOUSES), my_selected_house=(ID, HOUSES), their_selected_house=(ID, HOUSES),
        my_house_selection=(ID, HOUSES), their_house_selection=(ID, HOUSES), format=(ID, 4),
        kind=(ID, len(DecisionKind) + 1), intent=(ID, len(DecisionIntent) + 1), affects=(ID, len(Affects) + 1),
        decider=(ID, 3), my_houses=(BITS, 7), their_houses=(BITS, 7), my_extra_house_playable=(BITS, 7),
        their_extra_house_playable=(BITS, 7), decision_source=(PTR,),
        **{f"{side}_{f}": (ID, 2) for side in ("my", "their") for f in S.PLAYER_FLAGS},
    ),
    "effect": {
        "kind": (ID, 5), "variable": (ID, _n("effect_variables")), "event": (ID, _n("events")),
        "effect_kind": (ID, _n("effect_kinds")), "op": (ID, _n("effect_ops")), "value_kind": (ID, 5),
        "infinite": (ID, 2), "active": (ID, 2), "conditional": (ID, 2), "affected": (ID, 3), "controller": (ID, 3),
        "source": (PTR,),
    },
    "resolution": {
        "routine": (ID, _n("routines")), "site": (ID, _n("sites")), "kind": (ID, _n("ability_kinds")),
        "event": (ID, _n("events")), "source": (PTR,),
    },
    "res_pointer": {"role": (ID, _n("roles")), "entity": (PTR,), "zone": (ID, NZ), "frame": ("skip",)},
    "res_op": {"op": (ID, _n("op_names")), "frame": ("skip",)},
    "cleanup": {"op": (ID, _n("cleanup_ops")), "target": (PTR,)},
    "match": {"winner": (ID, 3), "first": (ID, 3), "swapped": (ID, 2)},
    "option": {
        "verb": (ID, _n("verbs")), "pointer": (PTR,), "bool": (ID, 3), "house": (ID, HOUSES), "flank": (ID, 3),
        "first_player": (ID, 3), "mode": (ID, _n("modes")), "event": (ID, _n("events")),
        "tag": (ID, _n("option_tags")),
    },
}
_KIND_MAX = max(_n("journal_ops"), _n("log_kinds"), len(DecisionKind) + 1)
HISTORY_COLUMNS = {
    "stream": (ID, 5), "kind": ("kind",), "actor": (ID, 3), "from_zone": (ID, NZ), "to_zone": (ID, NZ),
    "house": (ID, HOUSES), "flank": (ID, 5), "vis": (ID, 4), "turn": ("skip",), "seq": ("skip",),
    "position": (ID, 5), "flags": (BITS, len(BOOLS)), "intent": (ID, len(DecisionIntent) + 1),
    "affects": (ID, len(Affects) + 1), "optional": (ID, 2),
}
HISTORY_ROLES = 16


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    shifts = torch.arange(n, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def sinusoid(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous time features of `x` (any scale): [..., dim]."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * (-math.log(1000.0) / max(half, 1)))
    a = x.to(torch.float32).unsqueeze(-1) * freqs
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class ColumnEmbed(nn.Module):
    """A token block's rows -> [B, R, d], pointers excluded (the model adds
    them, it holds the entity identities).

    Vectorized across columns: every id column looks up one shared table
    (each column its own range of rows), every bit column unpacks in one
    shift, so a block costs a handful of kernel launches, not one per column
    (the eager v2 forward was launch-bound: 613 kernels for 2 ms of GPU
    work)."""

    def __init__(self, cols: S.Columns, kinds: Dict[str, tuple], d: int):
        super().__init__()
        self.cols = cols
        self.ids, self.bits, self.ptrs, self.nums = [], [], [], []
        width = cols.n_float
        sizes = []
        for j, name in enumerate(cols.int_names):
            kind = kinds.get(name, ("num",))
            if kind[0] == ID:
                self.ids.append((j, name))
                sizes.append(kind[1] + 1)
            elif kind[0] == BITS:
                self.bits.append((j, kind[1]))
                width += kind[1]
            elif kind[0] == PTR:
                self.ptrs.append((j, name))
            elif kind[0] == "num":
                self.nums.append(j)
                width += 1
        self.proj = nn.Linear(max(width, 1), d)
        self.id_table = nn.Embedding(max(1, sum(sizes)), d)
        offsets = [sum(sizes[:k]) for k in range(len(sizes))]
        self.register_buffer("id_cols", torch.tensor([j for j, _ in self.ids], dtype=torch.long), persistent=False)
        self.register_buffer("id_offset", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer("id_max", torch.tensor([n - 1 for n in sizes], dtype=torch.long), persistent=False)
        bit_cols = [j for j, n in self.bits for _ in range(n)]
        bit_shift = [k for _j, n in self.bits for k in range(n)]
        self.register_buffer("bit_cols", torch.tensor(bit_cols, dtype=torch.long), persistent=False)
        self.register_buffer("bit_shift", torch.tensor(bit_shift, dtype=torch.long), persistent=False)
        self.register_buffer("num_cols", torch.tensor(self.nums, dtype=torch.long), persistent=False)
        self.register_buffer("ptr_cols", torch.tensor([j for j, _ in self.ptrs], dtype=torch.long), persistent=False)

    def forward(self, ints: torch.Tensor, floats: torch.Tensor) -> torch.Tensor:
        parts = [floats[..., :self.cols.n_float]] if self.cols.n_float else []
        if self.bits:
            parts.append(((ints[..., self.bit_cols].clamp(min=0) >> self.bit_shift) & 1).to(torch.float32))
        if self.nums:
            parts.append(_signed_log(ints[..., self.num_cols].to(torch.float32)))
        x = torch.cat(parts, dim=-1) if parts else ints.new_zeros(ints.shape[:-1] + (1,), dtype=torch.float32)
        out = self.proj(x)
        if self.ids:
            idx = torch.minimum(ints[..., self.id_cols].clamp(min=0), self.id_max) + self.id_offset
            out = out + self.id_table(idx).sum(-2)
        return out


_KNOWLEDGE_INTS = [S.ENTITY.i[c] for c in ("mask", "deck_pos", "group", "group_size", "since_seen", "last_seen",
                                           "exit_op", "opp_exact", "opp_mask")]
_KNOWLEDGE_FLOATS = [S.ENTITY.f[c] for c in ("p_hand", "p_archive", "p_deck", "p_elsewhere", "p_next_draw")]


def _without_knowledge(batch: BatchV2) -> BatchV2:
    """The batch with the tracker's knowledge (beyond exactly known zones)
    and O3's priors zeroed in the entity block."""
    import dataclasses

    from .encode_v2 import TokenBlock

    ent = batch.blocks["entity"]
    ints = ent.ints.clone()
    ints[..., _KNOWLEDGE_INTS] = 0
    floats = ent.floats.clone()
    floats[..., _KNOWLEDGE_FLOATS] = 0
    return dataclasses.replace(batch, blocks=dict(batch.blocks, entity=TokenBlock(ints, floats, ent.mask)))


@dataclass
class PrefixPast:
    """A batch's cached history prefixes: per layer (k, v) [B, heads, P,
    d / heads], `valid` [B, P] (prefixes padded to the longest), and each
    item's prefix row count (`n`, [B]) -- its suffix's pointer rows are
    numbered after it."""

    kv: list
    valid: torch.Tensor
    n: torch.Tensor


@dataclass
class TrunkOutV2:
    g: torch.Tensor  # [B, d]
    h: torch.Tensor  # [B, 72, d]
    e_opt: torch.Tensor  # [B, K, d]
    option_mask: torch.Tensor  # [B, K]


class KeyForgeNetV2(nn.Module):
    def __init__(self, net_cfg: dict):
        super().__init__()
        d = int(net_cfg["d_model"])
        self.d = d
        self.net_cfg = dict(net_cfg)
        self.vocab_size = VOCAB
        self.history_arch = net_cfg.get("history_arch", "none")
        if self.history_arch not in ("joint", "stream", "turn_tokens", "summary", "none"):
            raise ValueError(f"model.history_arch must be joint/stream/turn_tokens/summary/none, not {self.history_arch!r}")
        self.options_in_trunk = bool(net_cfg.get("options_in_trunk", True))
        self.knowledge = bool(net_cfg.get("knowledge", True))
        self.static = tuple(net_cfg.get("static", ("attr", "text", "sig", "embedding")))
        unknown = set(self.static) - {"attr", "text", "sig", "embedding"}
        if unknown:
            raise ValueError(f"model.static: unknown tables {sorted(unknown)} (attr | text | sig | embedding)")
        default_time = "absolute" if self.history_arch == "stream" else "relative"
        self.history_time = net_cfg.get("history_time", default_time)
        if self.history_time not in ("absolute", "relative"):
            raise ValueError(f"model.history_time must be absolute or relative, not {self.history_time!r}")
        if self.history_arch == "stream" and self.history_time != "absolute":
            raise ValueError("model.history_arch 'stream' needs history_time 'absolute' (its prefix is cached)")
        at, sg, tx, tm, em = static_tables_tensor()
        for name, t in (("static_attr", at), ("static_sig", sg), ("static_text", tx), ("static_text_mask", tm)):
            self.register_buffer(name, t, persistent=False)
        self.register_buffer("static_emb", em if em is not None else torch.zeros(at.shape[0], 1), persistent=False)
        self.has_text_embedding = em is not None
        self.card_embedding = nn.Embedding(VOCAB, d)
        self.attr_in = nn.Linear(at.shape[1], d)
        self.sig_in = nn.Linear(sg.shape[1], d)
        self.text_trigger = nn.Embedding(32, d)
        self.text_verb = nn.Embedding(64, d)
        self.text_scope = nn.Embedding(32, d)
        self.text_rest = nn.Linear(1 + tx.shape[2] - 4, d)
        self.text_pool = _mlp(d, d, d)
        self.emb_in = nn.Linear(self.static_emb.shape[1], d) if self.has_text_embedding else None
        self.special_ptr = nn.Embedding(4, d)  # padding, none, hidden mine, hidden theirs
        self.embed = nn.ModuleDict({b.name: ColumnEmbed(b, COLUMNS.get(b.name, {}), d) for b in S.BLOCKS})
        self.role = nn.ModuleDict({
            b.name: nn.Embedding(len([c for c in b.int_names if COLUMNS.get(b.name, {}).get(c, ("num",))[0] == PTR]) + 1, d)
            for b in S.BLOCKS})
        self.type_embedding = nn.Embedding(12, d)
        self.norm_in = nn.LayerNorm(d)
        # history
        hist_cols = S.Columns("history", [(c, "I") for c in ROW_INTS] + [(f, "F") for f in ROW_FLOATS])
        self.hist_cols = hist_cols
        self.hist_embed = ColumnEmbed(hist_cols, HISTORY_COLUMNS, d)
        self.hist_kind = nn.Embedding(5 * (_KIND_MAX + 1), d)
        self.hist_role = nn.Embedding(HISTORY_ROLES, d)
        self.time_dim = int(net_cfg.get("time_dim", 32))
        self.hist_time = nn.Linear(2 * self.time_dim, d)
        self.now_time = nn.Linear(self.time_dim, d)
        self.turn_in = nn.Linear(len(TURN_SCALARS) + 3 * d, d)
        self.turn_actor = nn.Embedding(3, d)
        self.turn_house = nn.Embedding(HOUSES, d)
        self.turn_time = nn.Linear(self.time_dim, d)
        self.summary_entity = nn.Linear(len(SUMMARY_ENTITY), d)
        self.summary_global = nn.Linear(SUMMARY_GLOBAL, d)
        self.trunk = build_trunk(d, int(net_cfg["heads"]), int(net_cfg["ff"]), int(net_cfg["layers"]),
                                 float(net_cfg.get("dropout", 0.0)), "sdpa", net_cfg.get("attention_kernel", "flash"))
        self.out_norm = nn.LayerNorm(d)
        # heads (v1's interfaces)
        self.null_entity = nn.Parameter(torch.zeros(d))
        self.option_mlp = _mlp(2 * d, d, d)
        self.policy_query = nn.Linear(d, d)
        self.value_head = _mlp(d, d, 1)
        self.subset_attn = nn.Linear(d, 1)
        self.subset_size = nn.Linear(self.time_dim, d)  # continuous: no cap
        self.order_position = nn.Linear(self.time_dim, d)
        self.empty_subset = nn.Parameter(torch.zeros(d))
        self.subset_mlp = _mlp(d, d, d)
        self.subset_query = nn.Linear(d, d)
        self.seq_query = nn.Linear(d, d)
        self.seq_prefix = nn.Linear(d, d)
        self.seq_stop = nn.Parameter(torch.zeros(d))
        self.topk_head = _mlp(2 * d, d, 1)
        self.q_head = _mlp(2 * d, d, 1)
        self.belief_head = _mlp(d, d // 2, 4)  # hand, archive, deck, next draw
        self.oracle_hidden = nn.Linear(3 * VOCAB, d)
        self.oracle_head = _mlp(2 * d, d, 1)
        self.aux_house = nn.Linear(d, 7)
        self.aux_card = nn.Linear(d, VOCAB)

    # ------------------------------------------------------- identities
    def identities(self, card_ids: torch.Tensor) -> torch.Tensor:
        """[B, 72] card ids -> [B, 72, d]: embedding + static meaning. A
        card's identity depends on the card alone, so it is computed once per
        call for every card in the vocabulary and gathered: ~400 rows instead
        of B x 72 (at batch 512 the per-entity text rows were a third of the
        forward's GPU time)."""
        return self.identity_table()[card_ids]

    def identity_table(self) -> torch.Tensor:
        """[vocab, d]: every card's identity vector."""
        card_ids = torch.arange(self.static_attr.shape[0], device=self.static_attr.device)
        x = self.card_embedding(card_ids)
        if "attr" in self.static:
            x = x + self.attr_in(self.static_attr[card_ids])
        if "sig" in self.static:
            x = x + self.sig_in(self.static_sig[card_ids])
        if "text" not in self.static:
            if self.emb_in is not None and "embedding" in self.static:
                x = x + self.emb_in(self.static_emb[card_ids])
            return x
        tx = self.static_text[card_ids]  # [V, R, W]
        tm = self.static_text_mask[card_ids]
        t = (self.text_trigger(tx[..., 0].long().clamp(0, 31)) + self.text_verb(tx[..., 1].long().clamp(0, 63))
             + self.text_scope(tx[..., 3].long().clamp(0, 31))
             + self.text_rest(torch.cat([tx[..., 2:3], tx[..., 4:]], dim=-1)))
        w = tm.unsqueeze(-1).to(t.dtype)
        pooled = (t * w).sum(-2) / w.sum(-2).clamp(min=1.0)
        x = x + self.text_pool(pooled)
        if self.emb_in is not None and "embedding" in self.static:
            x = x + self.emb_in(self.static_emb[card_ids])
        return x

    def _pointed(self, ident: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        """ptr [B, ...] (entity index or -1/-2/-3; 0 padding handled by the
        mask) -> [B, ..., d]."""
        B = ptr.shape[0]
        flat = ptr.reshape(B, -1)
        gathered = torch.gather(ident, 1, flat.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d))
        special = self.special_ptr((-flat).clamp(min=0, max=3))
        out = torch.where((flat >= 0).unsqueeze(-1), gathered, special)
        return out.reshape(ptr.shape + (self.d,))

    def _block(self, batch: BatchV2, name: str, ident: torch.Tensor, type_id: int) -> torch.Tensor:
        blk = batch.blocks[name]
        emb = self.embed[name]
        x = emb(blk.ints, blk.floats)
        if emb.ptrs:  # every pointer column at once: its entity plus its role
            pointed = self._pointed(ident, blk.ints[..., emb.ptr_cols])  # [B, R, n_ptr, d]
            x = x + (pointed + self.role[name].weight[1:1 + len(emb.ptrs)]).sum(-2)
        return x + self.type_embedding.weight[type_id]

    # ------------------------------------------------------------ trunk
    def compile_inputs(self) -> None:
        """Compiles the input embedding (`embed_tokens`, dynamic shapes): the
        launch-heavy part of the eager forward (~300 of its ~350 kernels).
        The trunk stays eager, so a cached prefix's shapes never reach the
        compiler."""
        self._embed = torch.compile(self.embed_tokens, dynamic=True)

    def encode_state(self, batch: BatchV2, past: Optional[PrefixPast] = None) -> TrunkOutV2:
        """`past` (`stream` only): the batch's cached history prefixes; the
        batch's history rows are then each item's suffix."""
        if past is not None and self.history_arch != "stream":
            raise ValueError("a cached prefix needs history_arch 'stream'")
        embed = self.__dict__.get("_embed", self.embed_tokens)
        x, valid, opt = embed(batch, None if past is None else past.n)
        B = x.shape[0]
        blocks = batch.blocks
        n_state = 1 + S.N_ENTITIES + sum(blocks[n].mask.shape[1] for n in ("effect", "resolution", "cleanup", "match"))
        hist_len = batch.hist_mask.shape[1] if self.history_arch in ("joint", "stream") else 0
        T = x.shape[1]
        # (the global token is always a valid key, so no query is ever empty)
        if self.history_arch == "stream":
            h0 = T - hist_len
            if past is None:
                spec = AttnSpec(key_valid=valid, hist_from=h0)
            else:
                P = past.valid.shape[1]
                causal = torch.tril(torch.ones(hist_len, hist_len, dtype=torch.bool, device=x.device))
                hmask = torch.cat([past.valid[:, None, None, :].expand(B, 1, hist_len, P),
                                   causal[None, None].expand(B, 1, hist_len, hist_len)], dim=-1)
                spec = AttnSpec(key_valid=torch.cat([past.valid, valid], dim=1), hist_from=h0, past_len=P,
                                hist_mask=hmask)
            y = self.trunk(x, spec, None if past is None else past.kv)
        else:
            y = self.trunk(x, AttnSpec(key_valid=valid))
        y = self.out_norm(y)
        g_out = y[:, 0]
        h_out = y[:, 1:1 + S.N_ENTITIES]
        opt_mask = blocks["option"].mask
        if self.options_in_trunk:
            e_opt = y[:, n_state:n_state + opt.shape[1]]
        else:
            ptr = blocks["option"].ints[..., S.OPTION.i["pointer"]]
            pointed = torch.gather(h_out, 1, ptr.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d))
            pointed = torch.where((ptr >= 0).unsqueeze(-1), pointed, self.null_entity.expand_as(pointed))
            e_opt = self.option_mlp(torch.cat([opt, pointed], dim=-1))
        return TrunkOutV2(g=g_out, h=h_out, e_opt=e_opt, option_mask=opt_mask)

    def embed_tokens(self, batch: BatchV2, first=None):
        """Every token's input embedding, normalized: (x [B, T, d], valid
        [B, T], the option tokens [B, K, d]). Token order: global, entities,
        effects, frames, cleanups, match, options (if in the trunk), then
        history. `first` [B]: each item's first history row number (a
        suffix after a cached prefix)."""
        if not self.knowledge:
            batch = _without_knowledge(batch)
        ent_blk = batch.blocks["entity"]
        card_ids = ent_blk.ints[..., S.ENTITY.i["card"]].clamp(min=0)
        ident = self.identities(card_ids)
        B = card_ids.shape[0]
        g = self._block(batch, "global", ident, 0)  # [B, 1, d]
        if self.history_time == "absolute" and self.history_arch in ("joint", "stream"):
            g = g + self.now_time(sinusoid(batch.turn_now, self.time_dim)).unsqueeze(1)
        ent = ident + self._block(batch, "entity", ident, 1)
        if self.history_arch == "summary":
            ent = ent + self.summary_entity(_signed_log(batch.summary_entity))
            g = g + self.summary_global(_signed_log(batch.summary_global)).unsqueeze(1)
        eff = self._block(batch, "effect", ident, 2)
        fr = self._block(batch, "resolution", ident, 3)
        fr = fr + self._frame_extras(batch, ident, fr.shape[1])
        cl = self._block(batch, "cleanup", ident, 4)
        mt = self._block(batch, "match", ident, 5)
        parts = [g, ent, eff, fr, cl, mt]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=g.device), ent_blk.mask, batch.blocks["effect"].mask,
                 batch.blocks["resolution"].mask, batch.blocks["cleanup"].mask, batch.blocks["match"].mask]
        opt = self._block(batch, "option", ident, 6)
        if self.options_in_trunk:
            parts.append(opt)
            masks.append(batch.blocks["option"].mask)
        if self.history_arch in ("joint", "stream"):
            hx = self._history_tokens(batch.hist_ints, batch.hist_floats, batch.hist_ptr, batch.turn_now, ident, first)
            parts.append(hx)
            masks.append(batch.hist_mask)
        elif self.history_arch == "turn_tokens":
            tt = self._turn_tokens(batch, ident)
            parts.append(tt)
            masks.append(batch.turn_mask)
        return self.norm_in(torch.cat(parts, dim=1)), torch.cat(masks, dim=1), opt

    def _frame_extras(self, batch: BatchV2, ident: torch.Tensor, n_frames: int) -> torch.Tensor:
        """The resolution frames' pointer and operation rows, pooled into
        their frames."""
        B = ident.shape[0]
        out = ident.new_zeros(B, n_frames, self.d)
        for name, cols in (("res_pointer", S.RES_POINTER), ("res_op", S.RES_OP)):
            blk = batch.blocks[name]
            x = self.embed[name](blk.ints, blk.floats)
            if name == "res_pointer":
                x = x + self._pointed(ident, blk.ints[..., cols.i["entity"]])
            frame = blk.ints[..., cols.i["frame"]].clamp(min=0, max=n_frames - 1)
            x = x * blk.mask.unsqueeze(-1).to(x.dtype)
            out = out.scatter_add(1, frame.unsqueeze(-1).expand(-1, -1, self.d), x)
        return out

    def _history_tokens(self, ints, floats, hist_ptr, turn_now, ident: torch.Tensor, first=None) -> torch.Tensor:
        """History rows -> tokens. `first` [B]: the row number of each item's
        first row (a suffix after a cached prefix), so pointer rows map to
        their tokens."""
        x = self.hist_embed(ints, floats)
        stream = ints[..., ROW_INTS.index("stream")].clamp(min=0, max=4)
        kind = ints[..., ROW_INTS.index("kind")].clamp(min=0, max=_KIND_MAX)
        x = x + self.hist_kind(stream * (_KIND_MAX + 1) + kind)
        turn = ints[..., ROW_INTS.index("turn")]
        when = turn if self.history_time == "absolute" else turn_now.unsqueeze(1) - turn
        seq = ints[..., ROW_INTS.index("seq")]
        x = x + self.hist_time(torch.cat([sinusoid(when, self.time_dim), sinusoid(seq, self.time_dim)], dim=-1))
        rows, roles, ptrs = hist_ptr[..., 0], hist_ptr[..., 1], hist_ptr[..., 2]
        valid = rows >= 0
        if first is not None:
            rows = rows - first.unsqueeze(1)
        pv = (self._pointed(ident, ptrs) + self.hist_role(roles.clamp(0, HISTORY_ROLES - 1))) * valid.unsqueeze(-1)
        x = x.scatter_add(1, rows.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d), pv.to(x.dtype))
        return x + self.type_embedding.weight[7]

    def history_prefix(self, card_ids: torch.Tensor, ints, floats, hist_ptr, mask) -> PrefixPast:
        """`stream`'s cached prefix: every layer's keys and values for these
        history rows ([B, P, ...], `mask` [B, P]; `card_ids` [B, 72], the
        game's entities). A history row's token depends only on rows before
        it, so these are the same keys and values a whole sequence computes."""
        if self.history_arch != "stream":
            raise ValueError("a cached prefix needs history_arch 'stream'")
        ident = self.identities(card_ids.clamp(min=0))
        hx = self._history_tokens(ints, floats, hist_ptr, None, ident)
        _y, kvs = self.trunk(self.norm_in(hx), AttnSpec(key_valid=mask, hist_from=0), want_kv=True)
        return PrefixPast(kv=kvs, valid=mask, n=mask.sum(1))

    def _turn_tokens(self, batch: BatchV2, ident: torch.Tensor) -> torch.Tensor:
        sets = batch.turn_sets  # [B, U, 3, 72]
        pooled = torch.einsum("busn,bnd->busd", sets, ident) / sets.sum(-1, keepdim=True).clamp(min=1.0)
        B, U = sets.shape[:2]
        x = self.turn_in(torch.cat([_signed_log(batch.turn_floats), pooled.reshape(B, U, 3 * self.d)], dim=-1))
        x = x + self.turn_actor(batch.turn_ints[..., 1].clamp(0, 2)) + self.turn_house(batch.turn_ints[..., 2].clamp(0, HOUSES - 1))
        x = x + self.turn_time(sinusoid(batch.turn_ints[..., 0], self.time_dim))
        return x + self.type_embedding.weight[8]

    # ------------------------------------------------------------ heads
    def policy_logits(self, out: TrunkOutV2) -> torch.Tensor:
        q = self.policy_query(out.g).unsqueeze(1)
        return ((q * out.e_opt).sum(-1) / math.sqrt(self.d)).masked_fill(~out.option_mask, float("-inf"))

    def value(self, out: TrunkOutV2) -> torch.Tensor:
        return torch.tanh(self.value_head(out.g)).squeeze(-1)

    def topk_logits(self, out: TrunkOutV2) -> torch.Tensor:
        g = out.g.unsqueeze(1).expand_as(out.e_opt)
        return self.topk_head(torch.cat([out.e_opt, g], dim=-1)).squeeze(-1).masked_fill(~out.option_mask, float("-inf"))

    def q_values(self, out: TrunkOutV2) -> torch.Tensor:
        g = out.g.unsqueeze(1).expand_as(out.e_opt)
        return self.q_head(torch.cat([out.e_opt, g], dim=-1)).squeeze(-1)

    def belief_v2(self, out: TrunkOutV2, batch: BatchV2):
        """([B, 72, 3] logits over hand / archive / deck, [B, 72] next-draw
        logit), each the exact prior's logit plus what the head learned."""
        ent = batch.blocks["entity"]
        F_ = S.ENTITY.f
        prior = ent.floats[..., [F_["p_hand"], F_["p_archive"], F_["p_deck"]]].clamp(1e-6, 1.0)
        nd = ent.floats[..., F_["p_next_draw"]].clamp(1e-6, 1 - 1e-6)
        if not self.knowledge:  # no prior as an input: none as the starting point either
            prior, nd = torch.ones_like(prior), torch.full_like(nd, 0.5)
        d = self.belief_head(out.h)
        return torch.log(prior) + d[..., :3], torch.log(nd) - torch.log1p(-nd) + d[..., 3]

    def oracle_value(self, out: TrunkOutV2, hidden_counts: torch.Tensor) -> torch.Tensor:
        """`hidden_counts` [B, 3 * vocab]: the opponent's hand, their archive,
        my next draws, as count vectors."""
        hid = F.gelu(self.oracle_hidden(hidden_counts))
        return torch.tanh(self.oracle_head(torch.cat([out.g, hid], dim=-1))).squeeze(-1)

    def aux_logits(self, out: TrunkOutV2):
        """(opponent's next house [B, 7], next card played [B, vocab])."""
        return self.aux_house(out.g), self.aux_card(out.g)

    def subset_scores(self, out: TrunkOutV2, rows, members, ordered) -> torch.Tensor:
        C, M = members.shape
        e = out.e_opt[rows]
        mem = torch.gather(e, 1, members.clamp(min=0).unsqueeze(-1).expand(C, M, self.d))
        valid = members >= 0
        pos = self.order_position(sinusoid(torch.arange(M, device=mem.device), self.time_dim))  # continuous: no cap
        mem = mem + ordered.view(C, 1, 1).to(mem.dtype) * pos.unsqueeze(0)
        att = self.subset_attn(mem).squeeze(-1).masked_fill(~valid, float("-inf"))
        size = valid.sum(-1)
        empty = size == 0
        w = torch.softmax(att.masked_fill(empty.unsqueeze(-1), 0.0), dim=-1) * valid
        pooled = torch.where(empty.unsqueeze(-1), self.empty_subset.expand(C, self.d), (w.unsqueeze(-1) * mem).sum(1))
        s = self.subset_mlp(pooled + self.subset_size(sinusoid(size, self.time_dim)))
        return (self.subset_query(out.g[rows]) * s).sum(-1) / math.sqrt(self.d)

    def sequential_logits(self, out: TrunkOutV2, rows, prefix, legal) -> torch.Tensor:
        e = out.e_opt[rows]
        summary = (prefix.unsqueeze(-1).to(e.dtype) * e).sum(1)
        q = self.seq_query(out.g[rows] + self.seq_prefix(summary))
        logits = torch.cat([(q.unsqueeze(1) * e).sum(-1), (q * self.seq_stop).sum(-1, keepdim=True)], dim=-1) / math.sqrt(self.d)
        return logits.masked_fill(~legal, float("-inf"))


def checkpoint_stamp() -> dict:
    """What a v2 checkpoint records: the feature layout and every
    vocabulary's version."""
    return S.stamp()


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

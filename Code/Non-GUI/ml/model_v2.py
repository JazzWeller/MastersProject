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
exists.

**Time**: sinusoidal features of the turn offset and the sequence index --
no table, no cap.

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

from .encode_v2 import SUMMARY_ENTITY, SUMMARY_GLOBAL, TURN_SCALARS, BatchV2, static_tables_tensor
from .layers import build_trunk

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
    them, it holds the entity identities)."""

    def __init__(self, cols: S.Columns, kinds: Dict[str, tuple], d: int):
        super().__init__()
        self.cols = cols
        self.ids, self.bits, self.ptrs, self.nums = [], [], [], []
        width = cols.n_float
        for j, name in enumerate(cols.int_names):
            kind = kinds.get(name, ("num",))
            if kind[0] == ID:
                self.ids.append((j, name))
                self.add_module(f"emb_{name}", nn.Embedding(kind[1] + 1, d))
            elif kind[0] == BITS:
                self.bits.append((j, kind[1]))
                width += kind[1]
            elif kind[0] == PTR:
                self.ptrs.append((j, name))
            elif kind[0] == "num":
                self.nums.append(j)
                width += 1
        self.proj = nn.Linear(max(width, 1), d)

    def forward(self, ints: torch.Tensor, floats: torch.Tensor) -> torch.Tensor:
        parts = [floats[..., :self.cols.n_float]] if self.cols.n_float else []
        for j, n in self.bits:
            parts.append(_bits(ints[..., j].clamp(min=0), n))
        if self.nums:
            parts.append(_signed_log(ints[..., self.nums].to(torch.float32)))
        x = torch.cat(parts, dim=-1) if parts else ints.new_zeros(ints.shape[:-1] + (1,), dtype=torch.float32)
        out = self.proj(x)
        for j, name in self.ids:
            emb = getattr(self, f"emb_{name}")
            out = out + emb(ints[..., j].clamp(min=0, max=emb.num_embeddings - 1))
        return out


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
        self.history_arch = net_cfg.get("history_arch", "none")
        if self.history_arch not in ("joint", "stream", "turn_tokens", "summary", "none"):
            raise ValueError(f"model.history_arch must be joint/stream/turn_tokens/summary/none, not {self.history_arch!r}")
        self.options_in_trunk = bool(net_cfg.get("options_in_trunk", True))
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
        self.subset_size = nn.Embedding(64, d)
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
        """[B, 72] card ids -> [B, 72, d]: embedding + static meaning."""
        x = self.card_embedding(card_ids) + self.attr_in(self.static_attr[card_ids]) + self.sig_in(self.static_sig[card_ids])
        tx = self.static_text[card_ids]  # [B, 72, R, W]
        tm = self.static_text_mask[card_ids]
        t = (self.text_trigger(tx[..., 0].long().clamp(0, 31)) + self.text_verb(tx[..., 1].long().clamp(0, 63))
             + self.text_scope(tx[..., 3].long().clamp(0, 31))
             + self.text_rest(torch.cat([tx[..., 2:3], tx[..., 4:]], dim=-1)))
        w = tm.unsqueeze(-1).to(t.dtype)
        pooled = (t * w).sum(-2) / w.sum(-2).clamp(min=1.0)
        x = x + self.text_pool(pooled)
        if self.emb_in is not None:
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
        role = self.role[name]
        for k, (j, _) in enumerate(emb.ptrs):
            x = x + self._pointed(ident, blk.ints[..., j]) + role.weight[k + 1]
        return x + self.type_embedding.weight[type_id]

    # ------------------------------------------------------------ trunk
    def encode_state(self, batch: BatchV2) -> TrunkOutV2:
        ent_blk = batch.blocks["entity"]
        card_ids = ent_blk.ints[..., S.ENTITY.i["card"]].clamp(min=0)
        ident = self.identities(card_ids)
        B = card_ids.shape[0]
        g = self._block(batch, "global", ident, 0)  # [B, 1, d]
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
        n_state = sum(p.shape[1] for p in parts)
        if self.options_in_trunk:
            parts.append(opt)
            masks.append(batch.blocks["option"].mask)
        hist_len = 0
        if self.history_arch in ("joint", "stream"):
            hx = self._history_tokens(batch, ident)
            parts.append(hx)
            masks.append(batch.hist_mask)
            hist_len = hx.shape[1]
        elif self.history_arch == "turn_tokens":
            tt = self._turn_tokens(batch, ident)
            parts.append(tt)
            masks.append(batch.turn_mask)
        x = self.norm_in(torch.cat(parts, dim=1))
        valid = torch.cat(masks, dim=1)
        T = x.shape[1]
        attn = valid.unsqueeze(1).expand(B, T, T).clone()
        if self.history_arch == "stream" and hist_len:
            h0 = T - hist_len
            causal = torch.tril(torch.ones(hist_len, hist_len, dtype=torch.bool, device=x.device))
            attn[:, h0:, :h0] = False
            attn[:, h0:, h0:] &= causal
        # a padding query still needs one key (itself) to stay finite
        attn |= torch.eye(T, dtype=torch.bool, device=x.device).unsqueeze(0)
        y = self.out_norm(self.trunk(x, attn))
        g_out = y[:, 0]
        h_out = y[:, 1:1 + S.N_ENTITIES]
        opt_mask = batch.blocks["option"].mask
        if self.options_in_trunk:
            e_opt = y[:, n_state:n_state + opt.shape[1]]
        else:
            ptr = batch.blocks["option"].ints[..., S.OPTION.i["pointer"]]
            pointed = torch.gather(h_out, 1, ptr.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d))
            pointed = torch.where((ptr >= 0).unsqueeze(-1), pointed, self.null_entity.expand_as(pointed))
            e_opt = self.option_mlp(torch.cat([opt, pointed], dim=-1))
        return TrunkOutV2(g=g_out, h=h_out, e_opt=e_opt, option_mask=opt_mask)

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

    def _history_tokens(self, batch: BatchV2, ident: torch.Tensor) -> torch.Tensor:
        ints, floats = batch.hist_ints, batch.hist_floats
        B, H, _ = ints.shape
        x = self.hist_embed(ints, floats)
        stream = ints[..., ROW_INTS.index("stream")].clamp(min=0, max=4)
        kind = ints[..., ROW_INTS.index("kind")].clamp(min=0, max=_KIND_MAX)
        x = x + self.hist_kind(stream * (_KIND_MAX + 1) + kind)
        offset = batch.turn_now.unsqueeze(1) - ints[..., ROW_INTS.index("turn")]
        seq = ints[..., ROW_INTS.index("seq")]
        x = x + self.hist_time(torch.cat([sinusoid(offset, self.time_dim), sinusoid(seq, self.time_dim)], dim=-1))
        rows, roles, ptrs = batch.hist_ptr[..., 0], batch.hist_ptr[..., 1], batch.hist_ptr[..., 2]
        valid = rows >= 0
        pv = (self._pointed(ident, ptrs) + self.hist_role(roles.clamp(0, HISTORY_ROLES - 1))) * valid.unsqueeze(-1)
        x = x.scatter_add(1, rows.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d), pv.to(x.dtype))
        return x + self.type_embedding.weight[7]

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
        s = self.subset_mlp(pooled + self.subset_size(size.clamp(max=63)))
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

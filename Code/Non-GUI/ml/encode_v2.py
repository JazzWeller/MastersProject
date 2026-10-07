"""v2 encodings -> batched tensors (Agent Observation Plan, Milestone O7).

`collate_v2(items)` takes `(EncodedV2, history)` pairs -- history is an
`agent.history.HistoryEncoder` (or None) -- and pads every token block to
the batch's longest, with a mask (True = a real row). Blocks with no rows
anywhere in the batch get one padding row, masked, so shapes never vanish.
Sequences can be bucketed by length (`bucket_by_length`) so a full bucket
needs little padding.

History is carried in all three forms (O6) so the model's `history_arch`
picks one: full event rows (+ their pointer rows), turn tokens, and
summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agent import spec_v2 as S
from agent.history import N_RF, N_RI, ROW_INTS, TURN_FIELDS, summaries, turn_tokens

BLOCK_NAMES = tuple(b.name for b in S.BLOCKS)
TURN_SCALARS = tuple(f for f in TURN_FIELDS if f not in ("turn", "actor", "house", "played", "discarded", "archived",
                                                         "hand_start", "hand_end")) + ("hand_start", "hand_end")
SUMMARY_ENTITY = ("played", "reaped", "fought", "used", "since_seen", "last_public", "left_by", "revealed",
                  "drawn_turn")
SUMMARY_GLOBAL = 2 * (7 + 3 + 7 + 2 + 3)  # per side: house counts, last three, since, mulligans + declined, hands


@dataclass
class TokenBlock:
    ints: torch.Tensor  # [B, R, n_int] long
    floats: torch.Tensor  # [B, R, n_float]
    mask: torch.Tensor  # [B, R] bool

    def to(self, device) -> "TokenBlock":
        return TokenBlock(self.ints.to(device, non_blocking=True), self.floats.to(device, non_blocking=True),
                          self.mask.to(device, non_blocking=True))


@dataclass
class BatchV2:
    blocks: Dict[str, TokenBlock]
    hist_ints: torch.Tensor  # [B, H, N_RI]
    hist_floats: torch.Tensor  # [B, H, N_RF]
    hist_mask: torch.Tensor  # [B, H]
    hist_ptr: torch.Tensor  # [B, Q, 3] (row, role, entity), row -1 = padding
    turn_ints: torch.Tensor  # [B, U, 3] (turn offset, actor, house)
    turn_floats: torch.Tensor  # [B, U, len(TURN_SCALARS)]
    turn_sets: torch.Tensor  # [B, U, 3, 72] played / discarded / archived, multi-hot over entities
    turn_mask: torch.Tensor  # [B, U]
    summary_entity: torch.Tensor  # [B, 72, len(SUMMARY_ENTITY)]
    summary_global: torch.Tensor  # [B, SUMMARY_GLOBAL]
    turn_now: torch.Tensor  # [B]

    @property
    def size(self) -> int:
        return self.turn_now.shape[0]

    def to(self, device) -> "BatchV2":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "blocks":
                kw["blocks"] = {k: b.to(device) for k, b in v.items()}
            else:
                kw[f.name] = v.to(device, non_blocking=True)
        return BatchV2(**kw)


def _pad_block(encs, name: str, cols: S.Columns) -> TokenBlock:
    B = len(encs)
    R = max(1, max(e[name].n for e in encs))
    ints = np.zeros((B, R, max(cols.n_int, 1)), dtype=np.int64)
    floats = np.zeros((B, R, max(cols.n_float, 1)), dtype=np.float32)
    mask = np.zeros((B, R), dtype=bool)
    for b, e in enumerate(encs):
        blk = e[name]
        n = blk.n
        if n:
            if cols.n_int:
                ints[b, :n, :cols.n_int] = np.frombuffer(blk.ints.tobytes(), dtype=np.int64).reshape(n, cols.n_int)
            if cols.n_float:
                floats[b, :n, :cols.n_float] = np.frombuffer(blk.floats.tobytes(), dtype=np.float32).reshape(n, cols.n_float)
            mask[b, :n] = True
    return TokenBlock(torch.from_numpy(ints), torch.from_numpy(floats), torch.from_numpy(mask))


def _turn_rows(h, turn_now: int):
    toks = turn_tokens(h)
    ints = np.zeros((len(toks), 3), dtype=np.int64)
    floats = np.zeros((len(toks), len(TURN_SCALARS)), dtype=np.float32)
    sets = np.zeros((len(toks), 3, S.N_ENTITIES), dtype=np.float32)
    for k, t in enumerate(toks):
        ints[k] = (turn_now - t["turn"], t["actor"], t["house"])
        for j, f in enumerate(TURN_SCALARS):
            v = t[f]
            if isinstance(v, dict):  # hand sizes: the actor's
                v = v.get(t["actor"], 0) if t["actor"] else 0
            floats[k, j] = float(v)
        for j, f in enumerate(("played", "discarded", "archived")):
            for p in t[f]:
                if 0 <= p < S.N_ENTITIES:
                    sets[k, j, p] = 1.0
    return ints, floats, sets


def _summary_rows(h, turn_now: int):
    sm = summaries(h, turn_now)
    ent = np.zeros((S.N_ENTITIES, len(SUMMARY_ENTITY)), dtype=np.float32)
    for i, e in enumerate(sm["entities"][:S.N_ENTITIES]):
        for j, f in enumerate(SUMMARY_ENTITY):
            v = e[f]
            ent[i, j] = -1.0 if v is None else float(v)
    glob = []
    for side in (1, 2):
        g = sm["global"][side]
        glob.extend(g["house_counts"])
        glob.extend((list(g["last_three"]) + [0, 0, 0])[:3])
        glob.extend(-1.0 if x is None else float(x) for x in g["since"])
        glob.extend((g["mulligans"], g["declined_archive"]))
        glob.extend((list(g["hand_at_end_of_last_three"]) + [0, 0, 0])[:3])
    return ent, np.asarray(glob, dtype=np.float32)


def collate_v2(items: Sequence[Tuple[object, Optional[object], int]]) -> BatchV2:
    """`items`: (EncodedV2, HistoryEncoder or None, current turn number)."""
    encs = [it[0] for it in items]
    blocks = {b.name: _pad_block(encs, b.name, b) for b in S.BLOCKS}
    B = len(items)
    H = max([1] + [it[1].n for it in items if it[1] is not None])
    Q = max([1] + [len(it[1].pointers) // 3 for it in items if it[1] is not None])
    hist_ints = np.zeros((B, H, N_RI), dtype=np.int64)
    hist_floats = np.zeros((B, H, N_RF), dtype=np.float32)
    hist_mask = np.zeros((B, H), dtype=bool)
    hist_ptr = np.full((B, Q, 3), -1, dtype=np.int64)
    turn_parts, summary_ent, summary_glob = [], [], []
    for b, (enc, h, turn_now) in enumerate(items):
        if h is None:
            turn_parts.append((np.zeros((0, 3), np.int64), np.zeros((0, len(TURN_SCALARS)), np.float32),
                               np.zeros((0, 3, S.N_ENTITIES), np.float32)))
            summary_ent.append(np.zeros((S.N_ENTITIES, len(SUMMARY_ENTITY)), np.float32))
            summary_glob.append(np.zeros(SUMMARY_GLOBAL, np.float32))
            continue
        n = h.n
        if n:
            hist_ints[b, :n] = np.frombuffer(h.ints.tobytes(), dtype=np.int64).reshape(n, N_RI)
            hist_floats[b, :n] = np.frombuffer(h.floats.tobytes(), dtype=np.float32).reshape(n, N_RF)
            hist_mask[b, :n] = True
        p = np.frombuffer(h.pointers.tobytes(), dtype=np.int64).reshape(-1, 3)
        hist_ptr[b, :len(p)] = p
        turn_parts.append(_turn_rows(h, turn_now))
        e, g = _summary_rows(h, turn_now)
        summary_ent.append(e)
        summary_glob.append(g)
    U = max([1] + [len(t[0]) for t in turn_parts])
    turn_ints = np.zeros((B, U, 3), np.int64)
    turn_floats = np.zeros((B, U, len(TURN_SCALARS)), np.float32)
    turn_sets = np.zeros((B, U, 3, S.N_ENTITIES), np.float32)
    turn_mask = np.zeros((B, U), bool)
    for b, (ti, tf, ts) in enumerate(turn_parts):
        u = len(ti)
        turn_ints[b, :u], turn_floats[b, :u], turn_sets[b, :u], turn_mask[b, :u] = ti, tf, ts, True
    t = torch.from_numpy
    return BatchV2(blocks=blocks, hist_ints=t(hist_ints), hist_floats=t(hist_floats), hist_mask=t(hist_mask),
                   hist_ptr=t(hist_ptr), turn_ints=t(turn_ints), turn_floats=t(turn_floats), turn_sets=t(turn_sets),
                   turn_mask=t(turn_mask), summary_entity=t(np.stack(summary_ent)),
                   summary_global=t(np.stack(summary_glob)),
                   turn_now=torch.tensor([it[2] for it in items], dtype=torch.long))


def bucket_by_length(lengths: Sequence[int], batch_size: int) -> List[List[int]]:
    """Index batches with similar sequence lengths (sorted, then chunked)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[k:k + batch_size] for k in range(0, len(order), batch_size)]


def static_tables_tensor():
    """(attributes [V, A], signatures [V, G], text rows [V, R, W] + mask
    [V, R], embedding [V, D] or None) by card vocabulary id."""
    from agent.static_v2 import TEXT_ROW_WIDTH, tables

    attrs, texts, sigs, emb = tables()
    V = len(attrs)
    A = next(len(a) for a in attrs if a is not None)
    G = next(len(g) for g in sigs if g is not None)
    R = max(1, max(len(t) for t in texts))
    at = np.zeros((V, A), np.float32)
    sg = np.zeros((V, G), np.float32)
    tx = np.zeros((V, R, TEXT_ROW_WIDTH), np.float32)
    tm = np.zeros((V, R), bool)
    for v in range(V):
        if attrs[v] is not None:
            at[v] = np.frombuffer(attrs[v].tobytes(), np.float32)
            sg[v] = np.frombuffer(sigs[v].tobytes(), np.float32)
        for r, (trig, verb, amount, scope, cond) in enumerate(texts[v]):
            tx[v, r, :4] = (trig, verb, amount, scope)
            tx[v, r, 4:] = cond
            tm[v, r] = True
    em = None
    if emb is not None:
        from keyforge.cards.vocabulary import CARD_VOCAB

        dim = len(next(iter(emb.values())))
        em = np.zeros((V, dim), np.float32)
        for name, vec in emb.items():
            if name in CARD_VOCAB:
                em[CARD_VOCAB[name]] = vec
    t = torch.from_numpy
    return t(at), t(sg), t(tx), t(tm), (None if em is None else t(em))

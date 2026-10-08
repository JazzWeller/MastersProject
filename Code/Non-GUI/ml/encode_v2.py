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

from array import array
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agent import spec_v2 as S
from agent.history import (N_RF, N_RI, SUMMARY_ENTITY, SUMMARY_GLOBAL, TURN_SCALARS, HistoryFolded,
                           HistoryRows)

BLOCK_NAMES = tuple(b.name for b in S.BLOCKS)


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


def _where(ns: np.ndarray):
    """Rows laid end to end (item b has ns[b]) -> each row's (item, slot)."""
    total = int(ns.sum())
    item = np.repeat(np.arange(len(ns)), ns)
    starts = np.cumsum(ns) - ns
    return item, np.arange(total) - np.repeat(starts, ns)


def _scatter(chunks, ns: np.ndarray, R: int, width: int, dtype, fill=0) -> np.ndarray:
    """Each item's rows (bytes, `width` wide) into a padded [B, R, width]:
    one join and one scatter for the whole batch, not one copy per item."""
    out = np.full((len(ns), R, max(width, 1)), fill, dtype=dtype)
    if width and ns.sum():
        flat = np.frombuffer(b"".join(chunks), dtype=dtype).reshape(-1, width)
        item, slot = _where(ns)
        out[item, slot, :width] = flat
    return out


def _pad_block(encs, name: str, cols: S.Columns, min_rows: int = 1) -> TokenBlock:
    blks = [e.blocks[name] for e in encs]
    ns = np.fromiter((b.n for b in blks), dtype=np.int64, count=len(blks))
    R = max(1 if name == "global" else min_rows, int(ns.max()))  # (one global token, always)
    ints = _scatter([b.ints.tobytes() for b in blks], ns, R, cols.n_int, np.int64)
    floats = _scatter([b.floats.tobytes() for b in blks], ns, R, cols.n_float, np.float32)
    mask = np.arange(R)[None, :] < ns[:, None]
    return TokenBlock(torch.from_numpy(ints), torch.from_numpy(floats), torch.from_numpy(mask))


def collate_v2(items: Sequence[Tuple[object, Optional[object], int]], min_rows: int = 1) -> BatchV2:
    """`items`: (EncodedV2, history, current turn number). `history` is a
    `HistoryEncoder`, `HistoryRows` (bare rows: a cached prefix's suffix --
    no turn tokens or summaries), `HistoryFolded` (turn tokens and summaries
    only) or None. `min_rows`: every padded dimension has at least this
    many rows (2 for a compiled network: a dimension of 0 or 1 would compile
    a graph of its own)."""
    encs = [it[0] for it in items]
    blocks = {b.name: _pad_block(encs, b.name, b, min_rows) for b in S.BLOCKS}
    B = len(items)
    hists = [it[1] for it in items]
    rowed = [h is not None and not isinstance(h, HistoryFolded) for h in hists]
    empty = array("q")
    ns = np.fromiter((h.n if r else 0 for h, r in zip(hists, rowed)), dtype=np.int64, count=B)
    qs = np.fromiter((len(h.pointers) // 3 if r else 0 for h, r in zip(hists, rowed)), dtype=np.int64, count=B)
    H, Q = max(min_rows, int(ns.max())), max(min_rows, int(qs.max()))
    hist_ints = _scatter([h.ints.tobytes() if r else b"" for h, r in zip(hists, rowed)], ns, H, N_RI, np.int64)
    hist_floats = _scatter([h.floats.tobytes() if r else b"" for h, r in zip(hists, rowed)], ns, H, N_RF, np.float32)
    hist_mask = np.arange(H)[None, :] < ns[:, None]
    hist_ptr = _scatter([(h.pointers if r else empty).tobytes() for h, r in zip(hists, rowed)], qs, Q, 3, np.int64,
                        fill=-1)
    folded = [None if h is None or isinstance(h, HistoryRows) else
              (h if isinstance(h, HistoryFolded) else HistoryFolded.of(h, turn_now))
              for _e, h, turn_now in items]
    us = np.fromiter((f.u if f is not None else 0 for f in folded), dtype=np.int64, count=B)
    U = max(min_rows, int(us.max()))
    turn_ints = _scatter([f.turn_ints.tobytes() if f is not None else b"" for f in folded], us, U, 3, np.int64)
    turn_floats = _scatter([f.turn_floats.tobytes() if f is not None else b"" for f in folded], us, U,
                           len(TURN_SCALARS), np.float32)
    turn_mask = np.arange(U)[None, :] < us[:, None]
    turn_sets = np.zeros((B, U, 3, S.N_ENTITIES), np.float32)
    summary_ent = np.zeros((B, S.N_ENTITIES, len(SUMMARY_ENTITY)), np.float32)
    summary_glob = np.zeros((B, SUMMARY_GLOBAL), np.float32)
    for b, f in enumerate(folded):
        if f is None:
            continue
        if len(f.turn_sets):
            k, j, p = np.frombuffer(f.turn_sets.tobytes(), np.int64).reshape(-1, 3).T
            turn_sets[b, k, j, p] = 1.0
        summary_ent[b] = np.frombuffer(f.summary_entity.tobytes(), np.float32).reshape(S.N_ENTITIES, -1)
        summary_glob[b] = np.frombuffer(f.summary_global.tobytes(), np.float32)
    t = torch.from_numpy
    return BatchV2(blocks=blocks, hist_ints=t(hist_ints), hist_floats=t(hist_floats), hist_mask=t(hist_mask),
                   hist_ptr=t(hist_ptr), turn_ints=t(turn_ints), turn_floats=t(turn_floats), turn_sets=t(turn_sets),
                   turn_mask=t(turn_mask), summary_entity=t(summary_ent), summary_global=t(summary_glob),
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

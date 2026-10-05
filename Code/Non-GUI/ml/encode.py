"""Compact encodings (`agent.features.Encoded`) -> batched tensors
(Agent Training Plan, Milestone M1/M2).

`collate(encodeds, device)` stacks a list of `Encoded` into a `Batch`.
Everything is built from the raw bytes with one `frombuffer` per field and
expanded on the device -- the dense per-entity rows (`STATIC[card_id] ++
ENTITY`) never exist on the CPU side at all.

`Batch` is also what the training shards load into (`ml/dataset.py` builds
the same fields straight from numpy arrays), so the model has exactly one
input format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from agent import spec
from agent.features import Encoded, static_table

N = spec.N_ENTITIES
G = spec.GLOBAL.width
O = spec.OPTION.width
P = spec.INPLAY.width
ZONE_SLOTS = spec.ENTITY.span("zone")[1]
FLAG_SLOTS = spec.ENTITY.span("flags")[1]
RESERVED_ENTITY = spec.ENTITY.span("reserved_entity")[1]


@dataclass
class Batch:
    card_ids: torch.Tensor  # [B, 72] long
    zones: torch.Tensor  # [B, 72] long
    flags: torch.Tensor  # [B, 72] uint8 (bit-packed)
    inplay: torch.Tensor  # [B, 72, P] float
    selected: torch.Tensor  # [B, 72] float
    globals: torch.Tensor  # [B, G] float
    options: torch.Tensor  # [B, K, O] float (K = max options in the batch)
    pointers: torch.Tensor  # [B, K] long, -1 = no entity
    option_mask: torch.Tensor  # [B, K] bool, True = a real option

    @property
    def size(self) -> int:
        return self.card_ids.shape[0]

    def to(self, device) -> "Batch":
        return Batch(**{k: getattr(self, k).to(device, non_blocking=True) for k in self.__dataclass_fields__})

    def slice(self, a: int, b: int) -> "Batch":
        """Rows a..b (views; the option padding K is kept)."""
        return Batch(**{k: getattr(self, k)[a:b] for k in self.__dataclass_fields__})


def static_table_tensor() -> torch.Tensor:
    """[vocab, STATIC.width], the attribute rows looked up by card id."""
    rows = static_table()
    return torch.tensor(np.stack([np.frombuffer(r.tobytes(), dtype=np.float32) for r in rows]))


def unpack_flags(flags: torch.Tensor) -> torch.Tensor:
    """[..., 72] uint8 -> [..., 72, FLAG_SLOTS] float bits (LSB first,
    matching `keyforge.infoset.FLAG_BITS` order)."""
    shifts = torch.arange(FLAG_SLOTS, device=flags.device, dtype=torch.int32)
    return ((flags.to(torch.int32).unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def entity_block(batch: Batch) -> torch.Tensor:
    """[B, 72, ENTITY.width]: zone one-hot, flag bits, in-play block,
    prefix flag, reserved zeros -- `spec.ENTITY`'s exact field order."""
    B = batch.card_ids.shape[0]
    zone = torch.nn.functional.one_hot(batch.zones, ZONE_SLOTS).to(torch.float32)
    flags = unpack_flags(batch.flags)
    reserved = torch.zeros(B, N, RESERVED_ENTITY, device=zone.device)
    return torch.cat([zone, flags, batch.inplay, batch.selected.unsqueeze(-1), reserved], dim=-1)


def collate(
    items: Sequence[Encoded], device: Optional[torch.device] = None, *, rows: Optional[int] = None, width: Optional[int] = None,
) -> Batch:
    """One `Batch` from encoded observations, built with whole-array NumPy
    operations (no per-item tensor writes). `rows`/`width` pad it to that
    many rows and options -- the inference server's CUDA-graph shapes. A
    padded row is card 0 everywhere with no options; nothing reads it."""
    B = len(items)
    R = rows or B
    n_opts = np.fromiter((e.n_options for e in items), dtype=np.int64, count=B)
    K = width or max(int(n_opts.max()) if B else 0, 1)

    card_ids = np.zeros((R, N), np.int64)
    card_ids[:B] = np.frombuffer(b"".join(e.card_ids.tobytes() for e in items), dtype=np.int16).reshape(B, N)
    zones = np.zeros((R, N), np.int64)
    zones[:B] = np.frombuffer(b"".join(e.zones for e in items), dtype=np.uint8).reshape(B, N)
    flags = np.zeros((R, N), np.uint8)
    flags[:B] = np.frombuffer(b"".join(e.flags for e in items), dtype=np.uint8).reshape(B, N)
    globals_ = np.zeros((R, G), np.float32)
    globals_[:B] = np.frombuffer(b"".join(e.globals.tobytes() for e in items), dtype=np.float32).reshape(B, G)

    inplay = np.zeros((R * N, P), np.float32)
    counts = np.fromiter((len(e.inplay_index) for e in items), dtype=np.int64, count=B)
    if counts.sum():
        idx = np.frombuffer(b"".join(e.inplay_index for e in items), dtype=np.uint8).astype(np.int64)
        inplay[idx + np.repeat(np.arange(B, dtype=np.int64) * N, counts)] = np.frombuffer(
            b"".join(e.inplay.tobytes() for e in items), dtype=np.float32).reshape(-1, P)

    selected = np.zeros((R, N), np.float32)
    with_sel = [b for b, e in enumerate(items) if e.selected is not None]
    if with_sel:
        selected[with_sel] = np.frombuffer(b"".join(items[b].selected for b in with_sel), dtype=np.uint8).reshape(-1, N)

    options = np.zeros((R, K, O), np.float32)
    pointers = np.full((R, K), -1, np.int64)
    mask = np.zeros((R, K), bool)
    total = int(n_opts.sum())
    if total:
        offered = [e for e in items if e.n_options]
        b_of = np.repeat(np.arange(B), n_opts)
        k_of = np.arange(total) - np.repeat(np.cumsum(n_opts) - n_opts, n_opts)
        options[b_of, k_of] = np.frombuffer(b"".join(e.options.tobytes() for e in offered), dtype=np.float32).reshape(-1, O)
        pointers[b_of, k_of] = np.frombuffer(b"".join(e.pointers.tobytes() for e in offered), dtype=np.int8)
        mask[b_of, k_of] = True
    t = torch.from_numpy
    batch = Batch(t(card_ids), t(zones), t(flags), t(inplay).view(R, N, P), t(selected), t(globals_), t(options), t(pointers), t(mask))
    return batch.to(device) if device is not None else batch



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
from typing import List, Optional, Sequence

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


def collate(items: Sequence[Encoded], device: Optional[torch.device] = None) -> Batch:
    B = len(items)
    card_ids = torch.from_numpy(
        np.frombuffer(b"".join(e.card_ids.tobytes() for e in items), dtype=np.int16).reshape(B, N).astype(np.int64)
    )
    zones = torch.from_numpy(np.frombuffer(b"".join(e.zones for e in items), dtype=np.uint8).reshape(B, N).astype(np.int64))
    flags = torch.from_numpy(np.frombuffer(b"".join(e.flags for e in items), dtype=np.uint8).reshape(B, N).copy())
    globals_ = torch.from_numpy(np.frombuffer(b"".join(e.globals.tobytes() for e in items), dtype=np.float32).reshape(B, G).copy())

    inplay = torch.zeros(B * N, P)
    counts = [len(e.inplay_index) for e in items]
    if sum(counts):
        rows = torch.from_numpy(np.frombuffer(b"".join(e.inplay.tobytes() for e in items), dtype=np.float32).reshape(-1, P).copy())
        idx = np.frombuffer(b"".join(e.inplay_index for e in items), dtype=np.uint8).astype(np.int64)
        base = np.repeat(np.arange(B, dtype=np.int64) * N, counts)
        inplay.index_copy_(0, torch.from_numpy(idx + base), rows)
    inplay = inplay.view(B, N, P)

    selected = torch.zeros(B, N)
    for b, e in enumerate(items):
        if e.selected is not None:
            selected[b] = torch.from_numpy(np.frombuffer(e.selected, dtype=np.uint8).astype(np.float32))

    n_opts = [e.n_options for e in items]
    K = max(max(n_opts), 1)
    options = torch.zeros(B, K, O)
    pointers = torch.full((B, K), -1, dtype=torch.int64)
    mask = torch.zeros(B, K, dtype=torch.bool)
    for b, e in enumerate(items):
        k = n_opts[b]
        if k:
            options[b, :k] = torch.from_numpy(np.frombuffer(e.options.tobytes(), dtype=np.float32).reshape(k, O).copy())
            pointers[b, :k] = torch.from_numpy(np.frombuffer(e.pointers.tobytes(), dtype=np.int8).astype(np.int64))
            mask[b, :k] = True
    batch = Batch(card_ids, zones, flags, inplay, selected, globals_, options, pointers, mask)
    return batch.to(device) if device is not None else batch


def encoded_to_arrays(e: Encoded) -> dict:
    """One `Encoded` as numpy arrays -- the per-record form the training
    shards store (`ml/dataset.py`)."""
    k = e.n_options
    return {
        "card_ids": np.frombuffer(e.card_ids.tobytes(), dtype=np.int16),
        "zones": np.frombuffer(e.zones, dtype=np.uint8),
        "flags": np.frombuffer(e.flags, dtype=np.uint8),
        "globals": np.frombuffer(e.globals.tobytes(), dtype=np.float32),
        "inplay_index": np.frombuffer(e.inplay_index, dtype=np.uint8),
        "inplay": np.frombuffer(e.inplay.tobytes(), dtype=np.float32).reshape(-1, P),
        "options": np.frombuffer(e.options.tobytes(), dtype=np.float32).reshape(k, O),
        "pointers": np.frombuffer(e.pointers.tobytes(), dtype=np.int8),
    }


def pad_list(seqs: List[List[int]], fill: int = -1) -> torch.Tensor:
    width = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max(width, 1)), fill, dtype=torch.int64)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.int64)
    return out

"""Content-hashed, platform-portable checkpoints (Agent Training Plan,
Milestones M2 and M11).

**Format.** Not `torch.save`/pickle: a small explicit container -- a JSON
metadata header, then each tensor as `(name, dtype, shape, raw
little-endian bytes)` in sorted-name order. Its `weights_digest` (sha256 of
the tensor section) is identical on Windows and WSL for identical weights,
which a pickle's bytes aren't guaranteed to be; the file itself is named
by the digest of the whole container.

**What a checkpoint records and checks.** `agent.spec.stamp()` (feature
layout version + shape hash, vocabulary hash, `ENGINE_VERSION`), the full
card vocabulary it was trained against (name -> id), the network config,
and the run's config hash.
- A different major feature layout is refused outright (`spec.check_stamp`).
- A vocabulary that *reassigned* an id is refused; one that only *appended*
  cards is grown into (M11): new embedding rows are initialized from the
  attribute path -- a least-squares map from `STATIC` rows to the trained
  embedding rows, applied to the new cards' `STATIC` rows -- and every
  existing row is preserved exactly.
- A different `ENGINE_VERSION` is recorded and warned about, not refused by
  default: per the plan's "what invalidates what" table, a rules fix
  invalidates *data*, not checkpoints (`strict_engine=True` refuses).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from agent import spec
from keyforge.cards.vocabulary import CARD_VOCAB

MAGIC = b"KFCKPT1\n"
_DTYPES = {
    torch.float32: 1, torch.float16: 2, torch.bfloat16: 3, torch.int64: 4, torch.int32: 5, torch.uint8: 6,
    torch.bool: 7, torch.int16: 8, torch.int8: 9, torch.float64: 10,
}
_DTYPES_BACK = {v: k for k, v in _DTYPES.items()}


class CheckpointMismatch(ValueError):
    pass


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def serialize(tensors: Dict[str, torch.Tensor], meta: dict) -> Tuple[bytes, str]:
    """-> (container bytes, weights digest)."""
    body = io.BytesIO()
    wd = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        raw = _tensor_bytes(t)
        nb = name.encode("utf-8")
        header = struct.pack("<I", len(nb)) + nb + struct.pack("<BB", _DTYPES[t.dtype], t.dim())
        header += struct.pack(f"<{t.dim()}Q", *t.shape) + struct.pack("<Q", len(raw))
        body.write(header)
        body.write(raw)
        wd.update(header)
        wd.update(raw)
    weights_digest = wd.hexdigest()
    meta = dict(meta, weights_digest=weights_digest)
    mj = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<Q", len(mj)) + mj + body.getvalue(), weights_digest


def deserialize(data: bytes) -> Tuple[Dict[str, torch.Tensor], dict]:
    if not data.startswith(MAGIC):
        raise CheckpointMismatch("not a KeyForge checkpoint (bad magic)")
    at = len(MAGIC)
    (ml,) = struct.unpack_from("<Q", data, at)
    at += 8
    meta = json.loads(data[at : at + ml])
    at += ml
    tensors: Dict[str, torch.Tensor] = {}
    wd = hashlib.sha256()
    while at < len(data):
        start = at
        (nl,) = struct.unpack_from("<I", data, at)
        at += 4
        name = data[at : at + nl].decode("utf-8")
        at += nl
        code, ndim = struct.unpack_from("<BB", data, at)
        at += 2
        shape = struct.unpack_from(f"<{ndim}Q", data, at)
        at += 8 * ndim
        (nb,) = struct.unpack_from("<Q", data, at)
        at += 8
        wd.update(data[start:at])
        raw = data[at : at + nb]
        wd.update(raw)
        at += nb
        dtype = _DTYPES_BACK[code]
        if dtype == torch.bfloat16:
            arr = np.frombuffer(raw, dtype=np.int16).copy()
            tensors[name] = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
        else:
            np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
            tensors[name] = torch.from_numpy(np.frombuffer(raw, dtype=np_dtype).copy()).reshape(shape)
    if wd.hexdigest() != meta.get("weights_digest"):
        raise CheckpointMismatch("checkpoint weights don't match their recorded digest -- corrupted?")
    return tensors, meta


class CheckpointStore:
    """A directory of `<container sha256>.kfc` files (content-named -- see
    `agent.telemetry`'s artifact naming rule). A relative directory lands
    under the data root."""

    SUFFIX = ".kfc"

    def __init__(self, directory: str):
        from sim import data_root

        self.dir = data_root.resolve(directory)
        os.makedirs(self.dir, exist_ok=True)

    def save_bytes(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.dir, h + self.SUFFIX)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return h

    def path_of(self, prefix: str) -> str:
        matches = sorted(n for n in os.listdir(self.dir) if n.startswith(prefix) and n.endswith(self.SUFFIX))
        if not matches:
            raise FileNotFoundError(f"no checkpoint {prefix!r} in {self.dir}")
        if len(matches) > 1:
            raise ValueError(f"checkpoint prefix {prefix!r} is ambiguous in {self.dir}")
        return os.path.join(self.dir, matches[0])

    def list(self):
        return sorted(n[: -len(self.SUFFIX)] for n in os.listdir(self.dir) if n.endswith(self.SUFFIX))


def checkpoint_meta(model, *, config_hash: str = "", extra: Optional[dict] = None) -> dict:
    meta = {
        "stamp": spec.stamp(),
        "net_cfg": dict(model.net_cfg),
        "vocab_size": model.vocab_size,
        "vocab": dict(CARD_VOCAB),
        "config_hash": config_hash,
        "created": time.time(),
    }
    if extra:
        meta["extra"] = extra
    return meta


def save_model(model, target, *, config_hash: str = "", optimizer=None, extra: Optional[dict] = None) -> Tuple[str, str]:
    """Saves `model` (and, if given, the optimizer state, flattened into the
    same container) to `target` -- a `CheckpointStore` or a file path.
    Returns `(content hash or path, weights digest)`."""
    tensors = {f"model.{k}": v for k, v in model.state_dict().items()}
    meta = checkpoint_meta(model, config_hash=config_hash, extra=extra)
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        scalars = {}
        for pid, st in opt_state["state"].items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    tensors[f"opt.{pid}.{k}"] = v
                else:
                    scalars[f"{pid}.{k}"] = v
        meta["optimizer"] = {"param_groups": opt_state["param_groups"], "scalars": scalars}
    data, digest = serialize(tensors, meta)
    if isinstance(target, CheckpointStore):
        return target.save_bytes(data), digest
    with open(target, "wb") as f:
        f.write(data)
    return target, digest


def _check_vocab(saved: Dict[str, int]) -> None:
    for name, vid in saved.items():
        now = CARD_VOCAB.get(name)
        if now is not None and now != vid:
            raise CheckpointMismatch(f"vocabulary reassigned {name!r}: id {vid} in the checkpoint, {now} now")


def load_model(source, *, device="cpu", strict_engine: bool = False, net_overrides: Optional[dict] = None):
    """-> (model, meta, optimizer_state or None). `source` is a path, a
    `(CheckpointStore, hash prefix)` pair, or raw container bytes."""
    from keyforge.version import ENGINE_VERSION

    from .model import KeyForgeNet

    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, tuple):
        store, prefix = source
        with open(store.path_of(prefix), "rb") as f:
            data = f.read()
    else:
        with open(source, "rb") as f:
            data = f.read()
    tensors, meta = deserialize(data)
    stamp = meta["stamp"]
    spec.check_stamp(stamp, "checkpoint")
    _check_vocab(meta["vocab"])
    if stamp.get("engine_version") != ENGINE_VERSION:
        msg = f"checkpoint was trained on engine {stamp.get('engine_version')!r}, this is {ENGINE_VERSION!r}"
        if strict_engine:
            raise CheckpointMismatch(msg)
        warnings.warn(msg + " -- loading anyway (rules fixes invalidate data, not checkpoints)")

    net_cfg = dict(meta["net_cfg"])
    if net_overrides:
        net_cfg.update(net_overrides)
    saved_v = int(meta["vocab_size"])
    new_v = max(saved_v, len(CARD_VOCAB), spec.VOCAB_CAPACITY)
    model_sd = {k[len("model.") :]: v for k, v in tensors.items() if k.startswith("model.")}
    if new_v > saved_v:
        model_sd = grow_vocabulary(model_sd, saved_v, new_v)
    zero_newer_inputs(model_sd, int(stamp.get("feature_minor", 0)), int(net_cfg["card_embed"]))
    model = KeyForgeNet(net_cfg, vocab_size=new_v)
    model.load_state_dict(model_sd)
    model.to(device)
    opt = None
    if "optimizer" in meta:
        state: Dict[int, dict] = {}
        for k, v in tensors.items():
            if k.startswith("opt."):
                _, pid, name = k.split(".", 2)
                state.setdefault(int(pid), {})[name] = v
        for k, v in meta["optimizer"]["scalars"].items():
            pid, name = k.split(".", 1)
            state.setdefault(int(pid), {})[name] = v
        opt = {"state": state, "param_groups": meta["optimizer"]["param_groups"], "grown_from": saved_v if new_v > saved_v else None}
    return model, meta, opt


def zero_newer_inputs(sd: Dict[str, torch.Tensor], saved_minor: int, card_embed: int) -> None:
    """Zeroes the input-weight columns of every feature added in a minor
    version newer than the checkpoint's (`spec.MINOR_ADDITIONS`), so those
    dimensions start out exactly inert -- the reserved-slot rule's promise
    that an old checkpoint keeps working unchanged."""
    targets = {
        "option": ("option_in.weight", 0, spec.OPTION),
        "global": ("global_in.weight", 0, spec.GLOBAL),
        "entity": ("entity_in.weight", card_embed + spec.STATIC.width, spec.ENTITY),
        "static": ("entity_in.weight", card_embed, spec.STATIC),
    }
    for minor, additions in spec.MINOR_ADDITIONS.items():
        if minor <= saved_minor:
            continue
        for block_name, field in additions:
            key, base, block = targets[block_name]
            off, width = block.span(field)
            sd[key][:, base + off : base + off + width] = 0.0


def grow_vocabulary(sd: Dict[str, torch.Tensor], old_v: int, new_v: int) -> Dict[str, torch.Tensor]:
    """M11: extend every vocabulary-sized tensor from `old_v` to `new_v`
    cards. New embedding rows come from the attribute path (a ridge
    least-squares map from STATIC rows to the trained embeddings, fit on
    the old cards); existing rows are copied untouched."""
    from .encode import static_table_tensor

    sd = dict(sd)
    table = static_table_tensor()
    if table.shape[0] < new_v:
        table = torch.cat([table, torch.zeros(new_v - table.shape[0], table.shape[1])])
    emb = sd["card_embedding.weight"].float()
    X = table[:old_v].double()
    Y = emb.double()
    lam = 1e-2
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
    new_rows = (table[old_v:new_v].double() @ W).to(emb.dtype)
    sd["card_embedding.weight"] = torch.cat([sd["card_embedding.weight"], new_rows.to(sd["card_embedding.weight"].dtype)])

    if "fixed_head.weight" in sd:
        sd["fixed_head.weight"], sd["fixed_head.bias"] = _grow_fixed(sd["fixed_head.weight"], sd["fixed_head.bias"], old_v, new_v)

    ow = sd["oracle_hidden.weight"]  # [d, 2*old_v]: opponent hand counts | my next draws
    grown = torch.zeros(ow.shape[0], 2 * new_v, dtype=ow.dtype)
    grown[:, :old_v] = ow[:, :old_v]
    grown[:, new_v : new_v + old_v] = ow[:, old_v:]
    sd["oracle_hidden.weight"] = grown
    return sd


def _grow_fixed(fw: torch.Tensor, fb: torch.Tensor, old_v: int, new_v: int):
    """The fixed-vocabulary head's `verb * vocab + card` slots move when the
    vocabulary grows; carry every trained slot to its new index."""
    from .model import FIXED_PAYLOAD_BUCKETS, VERB_W

    d = fw.shape[1]
    new_fw = torch.zeros(VERB_W * new_v + VERB_W * FIXED_PAYLOAD_BUCKETS, d, dtype=fw.dtype)
    new_fb = torch.zeros(new_fw.shape[0], dtype=fb.dtype)
    for verb in range(VERB_W):
        new_fw[verb * new_v : verb * new_v + old_v] = fw[verb * old_v : (verb + 1) * old_v]
        new_fb[verb * new_v : verb * new_v + old_v] = fb[verb * old_v : (verb + 1) * old_v]
    new_fw[VERB_W * new_v :] = fw[VERB_W * old_v :]
    new_fb[VERB_W * new_v :] = fb[VERB_W * old_v :]
    return new_fw, new_fb


def load_model_v2(source, *, device="cpu"):
    """A network-v2 checkpoint (Agent Observation Plan, O9; saved with the v2
    stamp in `extra`) -> (model, meta). Refuses one whose v2 layout or
    vocabularies differ from today's: v2 vocabularies are append-only, but a
    layout change moves columns."""
    from agent import spec_v2 as S

    from .model_v2 import KeyForgeNetV2

    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, tuple):
        store, prefix = source
        with open(store.path_of(prefix), "rb") as f:
            data = f.read()
    else:
        with open(source, "rb") as f:
            data = f.read()
    tensors, meta = deserialize(data)
    extra = meta.get("extra") or {}
    if extra.get("feature_version") != 2:
        raise CheckpointMismatch("not a network-v2 checkpoint (no feature_version 2 in its stamp)")
    now = S.stamp()
    for key in ("layout_hash", "vocab_v2_hash"):
        if extra.get(key) != now.get(key):
            raise CheckpointMismatch(f"checkpoint {key} {str(extra.get(key))[:12]} != today's {str(now.get(key))[:12]}")
    _check_vocab(meta["vocab"])
    model = KeyForgeNetV2(dict(meta["net_cfg"]))
    model.load_state_dict({k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")})
    return model.to(device), meta

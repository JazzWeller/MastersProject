"""Checkpoint registry (Agent Interface Plan, Milestone H): versioned
weights identified by a content hash, so every training-data record can
name exactly which checkpoint produced it, and a self-play actor can
detect and reload a new one at a game boundary.

`weights` is deliberately untyped -- this registry only ever stores and
retrieves whatever a training pipeline hands it (a real deployment: a
state dict), via `pickle`. It has no opinion on model architecture.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from sim import data_root

_FILENAME_RE = re.compile(r"^(\d{6})_([0-9a-f]{16})\.ckpt$")


@dataclass(frozen=True)
class Checkpoint:
    version: int
    content_hash: str
    path: str


class CheckpointRegistry:
    """One directory of `{version:06d}_{content_hash[:16]}.ckpt` files.
    `version` is assigned by the registry (monotonically increasing);
    `content_hash` is a property of the weights themselves, so two saves of
    identical weights are distinguishable by version but verifiably
    identical in content."""

    def __init__(self, directory: str):
        # A relative directory lands under the data root (sim.data_root).
        self._dir = data_root.resolve(directory)
        os.makedirs(self._dir, exist_ok=True)

    def _entries(self) -> List[Checkpoint]:
        out = []
        for name in os.listdir(self._dir):
            m = _FILENAME_RE.match(name)
            if m is not None:
                out.append(Checkpoint(version=int(m.group(1)), content_hash=m.group(2), path=os.path.join(self._dir, name)))
        return sorted(out, key=lambda c: c.version)

    def save(self, weights: Any) -> Checkpoint:
        payload = pickle.dumps(weights)
        content_hash = hashlib.sha256(payload).hexdigest()
        version = self.latest_version() + 1
        path = os.path.join(self._dir, f"{version:06d}_{content_hash[:16]}.ckpt")
        with open(path, "wb") as f:
            f.write(payload)
        return Checkpoint(version=version, content_hash=content_hash, path=path)

    def load(self, checkpoint: Checkpoint) -> Any:
        with open(checkpoint.path, "rb") as f:
            payload = f.read()
        # A Checkpoint listed from disk only knows the 16-digit prefix its
        # filename carries; one returned by save() knows the full hash.
        if not hashlib.sha256(payload).hexdigest().startswith(checkpoint.content_hash):
            raise ValueError(f"checkpoint {checkpoint.path!r} content hash does not match its filename -- corrupted?")
        return pickle.loads(payload)

    def latest_version(self) -> int:
        entries = self._entries()
        return entries[-1].version if entries else 0

    def latest(self) -> Optional[Checkpoint]:
        entries = self._entries()
        return entries[-1] if entries else None

    def list(self) -> List[Checkpoint]:
        return self._entries()

    def find(self, content_hash: str) -> Optional[Checkpoint]:
        """The newest checkpoint whose content hash starts with
        `content_hash` (at least 16 hex digits, as filenames carry). Records
        name checkpoints this way rather than by absolute path, which
        differs between Windows and WSL."""
        prefix = content_hash[:16]
        matches = [c for c in self._entries() if c.content_hash.startswith(prefix)]
        return matches[-1] if matches else None

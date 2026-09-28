#!/usr/bin/env python3
"""Verifies a checkpoint container without torch (Agent Training Plan, M2):
recomputes the weights digest from the raw bytes and checks it against the
one recorded inside, then prints the stamps. The container format is
explicit little-endian (ml/checkpoints.py), so a checkpoint written in WSL
verifies to the identical digest on Windows -- which has no torch to load
it with, but can still tell it's intact and what it was trained against.

    python -m tools.verify_checkpoint \\\\wsl$\\Ubuntu\\root\\keyforge-data\\runs\\<run>\\checkpoints\\<hash>.kfc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys

MAGIC = b"KFCKPT1\n"


def verify(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(MAGIC):
        raise ValueError("not a KeyForge checkpoint (bad magic)")
    at = len(MAGIC)
    (ml,) = struct.unpack_from("<Q", data, at)
    at += 8
    meta = json.loads(data[at : at + ml])
    at += ml
    wd = hashlib.sha256()
    tensors = 0
    while at < len(data):
        start = at
        (nl,) = struct.unpack_from("<I", data, at)
        at += 4 + nl
        _code, ndim = struct.unpack_from("<BB", data, at)
        at += 2 + 8 * ndim
        (nb,) = struct.unpack_from("<Q", data, at)
        at += 8
        wd.update(data[start:at])
        wd.update(data[at : at + nb])
        at += nb
        tensors += 1
    digest = wd.hexdigest()
    return {
        "ok": digest == meta.get("weights_digest"),
        "weights_digest": digest,
        "container_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": tensors,
        "stamp": meta.get("stamp"),
        "net_cfg": meta.get("net_cfg"),
        "config_hash": meta.get("config_hash"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    args = parser.parse_args()
    out = verify(args.path)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()

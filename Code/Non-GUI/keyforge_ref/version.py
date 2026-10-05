"""Engine version stamp, written into every replay record.

A replay record made by an older engine version, or against different card
text or effect code, can still *fit* a newer engine -- same decision shapes,
same option counts -- while silently playing a different game (a card fix
changes what an option does, not how many there are). That would poison
training data and ratings without ever raising an error. So every record
carries `ENGINE_VERSION` plus `RULES_HASH`, and deserializing a record
(`keyforge.replay.config_from_dict`, `keyforge.match.match_config_from_dict`)
refuses outright on a mismatch, rather than relying on "the record doesn't
fit" to surface the problem.

Bump `ENGINE_VERSION` on any change that could alter how a saved choice
record replays (a rules fix, an effect rewrite, a change to decision shapes
or option ordering). `RULES_HASH` is computed automatically from the engine
and card-data source, so an unbumped-but-edited effect file is still caught.
"""

from __future__ import annotations

import hashlib
import os
import platform

ENGINE_VERSION = "1.1.0"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class EngineVersionMismatch(ValueError):
    """A replay record's engine version or rules hash doesn't match this
    running engine's -- see the module docstring for why this refuses
    outright instead of attempting to replay anyway."""


def _compute_rules_hash() -> str:
    """SHA-256 over the source of every `.py`/`.json` file in the `keyforge`
    package: card data plus every effect module, not just the obviously
    rules-affecting ones -- deciding which files "count" is exactly the
    kind of judgment call that goes stale."""
    paths = []
    for root, dirs, files in os.walk(_PACKAGE_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith((".py", ".json")):
                paths.append(os.path.join(root, name))
    h = hashlib.sha256()
    for path in sorted(paths):
        rel = os.path.relpath(path, _PACKAGE_DIR).replace(os.sep, "/")
        with open(path, "rb") as f:
            contents = f.read()
        h.update(rel.encode("utf-8"))
        h.update(len(contents).to_bytes(8, "big"))
        h.update(contents)
    return h.hexdigest()


RULES_HASH = _compute_rules_hash()


def interpreter_stamp() -> str:
    """Recorded for diagnosis only -- never checked on replay, since a
    portable record (Milestone A) is meant to replay identically on any
    interpreter."""
    return f"{platform.python_implementation()} {platform.python_version()}"


def version_stamp() -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "rules_hash": RULES_HASH,
        "interpreter": interpreter_stamp(),
    }


def check_version_stamp(data: dict, what: str = "record") -> None:
    """Raises `EngineVersionMismatch` unless `data` (a dict previously
    produced by `version_stamp()`, embedded in a serialized record) matches
    this running engine's version and rules hash. A record with no stamp at
    all (made before this check existed) is treated as a mismatch too."""
    engine_version = data.get("engine_version")
    rules_hash = data.get("rules_hash")
    if engine_version == ENGINE_VERSION and rules_hash == RULES_HASH:
        return
    raise EngineVersionMismatch(
        f"{what} was made by engine_version={engine_version!r} rules_hash={(rules_hash or '')[:12]!r}, "
        f"but this engine is engine_version={ENGINE_VERSION!r} rules_hash={RULES_HASH[:12]!r}. "
        "It cannot be replayed safely -- regenerate it."
    )

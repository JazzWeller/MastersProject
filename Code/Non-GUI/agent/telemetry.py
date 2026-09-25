"""Runs, journals, metrics and artifacts (Agent Training Plan, Milestones
M0 and M10). Standard library only, so actors can use it too.

A run owns `$KEYFORGE_DATA/runs/<run_id>/`:

- `config.json` -- the resolved config (`agent.config.resolve`), and its
  hash in `config.sha256`. Every record written below carries that hash.
- `journal.jsonl` -- append-only, one line per *human intervention* (a
  learning-rate change, a tier downgrade, a restart, a killed actor), each
  stamped with the game index at which it happened, so a learning curve
  can always be read against what was actually done to it.
- `metrics.jsonl` -- append-only telemetry, written by every actor and the
  learner at most once per `interval_seconds` per source (plus on demand).
- `<kind>/<content_hash>` -- artifacts, named by content, never by a
  hyperparameter (a renamed artifact is an unfindable artifact).

`tools/monitor.py` reads all of this back and prints the derived numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from typing import Any, Dict, Iterator, List, Optional

from sim import data_root

from . import config as config_mod

RUNS_DIRNAME = "runs"


def runs_root() -> str:
    return os.path.join(data_root.data_root(), RUNS_DIRNAME)


def _append_jsonl(path: str, record: dict) -> None:
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class Run:
    """One experiment's directory. `Run.create` resolves and writes the
    config; `Run.open` reattaches to an existing one (resuming refuses a
    different config -- a resumed run must be the run it claims to be,
    and a deliberate change goes through `journal` instead)."""

    def __init__(self, run_id: str, root: str, resolved: dict, cfg_hash: str):
        self.run_id = run_id
        self.root = root
        self.config = resolved
        self.config_hash = cfg_hash

    @classmethod
    def create(cls, run_id: str, source: Any = None, overrides: Optional[dict] = None, *, root: Optional[str] = None) -> "Run":
        resolved = config_mod.resolve(source, overrides)
        cfg_hash = config_mod.config_hash(resolved)
        run_root = os.path.join(root or runs_root(), run_id)
        os.makedirs(run_root, exist_ok=True)
        cfg_path = os.path.join(run_root, "config.json")
        if os.path.exists(cfg_path):
            with open(os.path.join(run_root, "config.sha256"), "r", encoding="utf-8") as f:
                existing = f.read().strip()
            if existing != cfg_hash:
                raise ValueError(
                    f"run {run_id!r} already exists with config {existing[:12]}, not {cfg_hash[:12]} -- "
                    "open it with Run.open() to resume, or pick a new run id"
                )
        else:
            with open(cfg_path, "wb") as f:
                f.write(config_mod.canonical_bytes(resolved))
            with open(os.path.join(run_root, "config.sha256"), "w", encoding="utf-8") as f:
                f.write(cfg_hash + "\n")
        run = cls(run_id, run_root, resolved, cfg_hash)
        run.journal("run_created" if not os.path.exists(run.journal_path) else "run_reopened", game_index=0)
        return run

    @classmethod
    def open(cls, run_id: str, *, root: Optional[str] = None) -> "Run":
        run_root = os.path.join(root or runs_root(), run_id)
        with open(os.path.join(run_root, "config.json"), "rb") as f:
            raw = f.read()
        resolved = json.loads(raw)
        cfg_hash = hashlib.sha256(raw).hexdigest()
        return cls(run_id, run_root, resolved, cfg_hash)

    # ------------------------------------------------------------ paths --
    @property
    def journal_path(self) -> str:
        return os.path.join(self.root, "journal.jsonl")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.root, "metrics.jsonl")

    def artifact_dir(self, kind: str) -> str:
        d = os.path.join(self.root, kind)
        os.makedirs(d, exist_ok=True)
        return d

    def artifact_path(self, kind: str, content_hash: str, suffix: str = "") -> str:
        return os.path.join(self.artifact_dir(kind), content_hash + suffix)

    def write_artifact(self, kind: str, payload: bytes, suffix: str = "") -> str:
        """Writes `payload` under its own content hash; returns the path.
        Idempotent: identical content lands at the identical path."""
        h = hashlib.sha256(payload).hexdigest()
        path = self.artifact_path(kind, h, suffix)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, path)
        return path

    # ---------------------------------------------------------- records --
    def journal(self, event: str, *, game_index: int, **details) -> dict:
        record = {"t": time.time(), "event": event, "game_index": game_index, "config_hash": self.config_hash, **details}
        _append_jsonl(self.journal_path, record)
        return record

    def journal_entries(self) -> List[dict]:
        return read_jsonl(self.journal_path)

    def metrics(self, source: str, interval_seconds: Optional[float] = None) -> "MetricsWriter":
        if interval_seconds is None:
            interval_seconds = float(self.config.get("telemetry", {}).get("interval_seconds", 60))
        return MetricsWriter(self.metrics_path, source, self.config_hash, interval_seconds)

    def metric_records(self) -> List[dict]:
        return read_jsonl(self.metrics_path)


class MetricsWriter:
    """Accumulates counters/gauges for one source (an actor, the learner)
    and appends one JSONL record per `interval_seconds` -- `tick()` is
    cheap to call every decision; only the rate-limited flush touches disk.

    `count(name, n)` sums; `gauge(name, v)` keeps the last value;
    `observe(name, v)` keeps a running mean (and max). Each flushed record
    carries per-second rates for every counter over the interval."""

    def __init__(self, path: str, source: str, config_hash: str, interval_seconds: float = 60.0):
        self.path = path
        self.source = source
        self.config_hash = config_hash
        self.interval = interval_seconds
        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._reset(time.time())

    def _reset(self, now: float) -> None:
        self._t0 = now
        self._counts: Dict[str, float] = {}
        self._gauges: Dict[str, Any] = {}
        self._sums: Dict[str, float] = {}
        self._ns: Dict[str, int] = {}
        self._maxes: Dict[str, float] = {}

    def count(self, name: str, n: float = 1) -> None:
        self._counts[name] = self._counts.get(name, 0) + n

    def gauge(self, name: str, value: Any) -> None:
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        self._sums[name] = self._sums.get(name, 0.0) + value
        self._ns[name] = self._ns.get(name, 0) + 1
        if value > self._maxes.get(name, float("-inf")):
            self._maxes[name] = value

    def tick(self, now: Optional[float] = None) -> Optional[dict]:
        now = time.time() if now is None else now
        if now - self._t0 >= self.interval:
            return self.flush(now)
        return None

    def flush(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        dt = max(now - self._t0, 1e-9)
        record = {
            "t": now,
            "source": self.source,
            "host": self._host,
            "pid": self._pid,
            "config_hash": self.config_hash,
            "interval": dt,
            "counts": dict(self._counts),
            "rates": {k: v / dt for k, v in self._counts.items()},
            "gauges": dict(self._gauges),
            "means": {k: self._sums[k] / self._ns[k] for k in self._sums},
            "maxes": dict(self._maxes),
        }
        _append_jsonl(self.path, record)
        self._reset(now)
        return record


def iter_runs(root: Optional[str] = None) -> Iterator[str]:
    base = root or runs_root()
    if not os.path.isdir(base):
        return iter(())
    return iter(sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "config.json"))))

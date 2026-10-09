"""Pausing and resuming a long training run (BC v1 and v2).

A run in progress saves its whole state -- weights, optimizer, schedule,
RNGs, the position in the epoch, the logged history -- to one state file
every `save_minutes` minutes (default 10) and whenever a pause is asked
for, then picks up from that file when started again with the same
command. A pause is asked for by

- SIGTERM or Ctrl-C (SIGINT) -- `tools/pause_training.sh` sends SIGTERM to
  every training process in WSL;
- a file named `PAUSE` next to the state file (the run directory), which is
  removed when the run resumes.

The run stops at the next step boundary, saves, and exits with code
`PAUSED` (75), so a chain script can tell a pause from a failure. The
state file is replaced atomically, so a hard kill loses at most the work
since the last save. When the run finishes, the state file is deleted.

Resumed batches are the ones the uninterrupted run would have drawn
(the epoch's shuffle is replayed from its seed); history dropout's own
draws differ after a resume.
"""

from __future__ import annotations

import os
import signal
import time
from typing import Optional

import torch

PAUSED = 75  # EX_TEMPFAIL


class Paused(SystemExit):
    """Raised by a training loop once it has saved its state for a pause."""

    def __init__(self, path: str):
        super().__init__(PAUSED)
        self.path = path


class Pauser:
    """Watches for a pause request (signal or PAUSE file) and paces the
    periodic saves. Signal handlers are installed only on the main thread."""

    def __init__(self, state_path: str, save_minutes: float = 10.0):
        self.state_path = state_path
        self.flag_path = os.path.join(os.path.dirname(os.path.abspath(state_path)), "PAUSE")
        self.save_seconds = float(save_minutes) * 60
        self._asked = False
        self._last_save = time.monotonic()
        self._last_flag_check = 0.0
        self._previous = {}
        if os.path.exists(self.flag_path):  # starting again is the resume
            os.remove(self.flag_path)
        import threading

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._previous[sig] = signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, _frame):
        if self._asked:  # a second request: stop now, without saving
            raise SystemExit(128 + signum)
        self._asked = True
        print(f"pause requested (signal {signum}): saving at the next step", flush=True)

    def close(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

    def requested(self) -> bool:
        now = time.monotonic()
        if not self._asked and now - self._last_flag_check > 5:
            self._last_flag_check = now
            if os.path.exists(self.flag_path):
                self._asked = True
                print(f"pause requested ({self.flag_path}): saving at the next step", flush=True)
        return self._asked

    def save_due(self) -> bool:
        return time.monotonic() - self._last_save > self.save_seconds

    def saved(self) -> None:
        self._last_save = time.monotonic()


def rng_state() -> dict:
    out = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_state(path: str, *, model, opt, sched, config_hash: str, **progress) -> None:
    """Atomically writes the training state (`progress`: step, epoch, batches
    into the epoch, history, ...)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "rng": rng_state(), "config_hash": config_hash, "progress": progress}
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_state(path: str, *, model, opt, sched, config_hash: str) -> Optional[dict]:
    """Restores a saved state into `model` / `opt` / `sched` and returns its
    progress dict, or None when there is nothing to resume."""
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob["config_hash"] != config_hash:
        raise SystemExit(f"{path} was saved by a run with a different config -- delete it to start over")
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    set_rng_state(blob["rng"])
    return blob["progress"]


def config_digest(cfg: dict, *extra) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps([cfg, *extra], sort_keys=True, default=str).encode()).hexdigest()[:16]


def finished(path: str) -> None:
    for p in (path, path + ".tmp"):
        if os.path.exists(p):
            os.remove(p)

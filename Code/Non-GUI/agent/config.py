"""Experiment configuration and provenance (Agent Training Plan, Milestone
M0).

Everything that can be varied lives in **one config file per experiment**
(`Code/Non-GUI/configs/*.json`; `.yaml` too when PyYAML happens to be
installed -- it's never required, since the plan's only mandatory
dependency is torch). A run resolves its config -- file values deep-merged
over `DEFAULTS`, plus the version stamps of the code it runs against --
and hashes the resolved form. That hash goes into every artifact the run
produces (checkpoints, shards, metric records, evaluation rows), so two
runs are comparable iff their config hashes differ in ways you can
enumerate (`diff_configs`).

Resolution is deterministic: the same file on the same code re-resolves to
the same bytes, and so the same hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import spec

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

DEFAULTS: Dict[str, Any] = {
    "name": "unnamed",
    "tier": 0,
    "seed": 0,
    "pool": {
        "decks": ["fignor", "igor"],
        "format": "archon",
        "deck_source": "presets",  # presets | random | alliance
        "max_turns": 200,
    },
    "network": {
        "d_model": 128,
        "layers": 4,
        "heads": 4,
        "ff": 256,
        "card_embed": 64,
        "dropout": 0.1,
        "identity": "both",  # id | attr | both
        "policy_head": "pointer",  # pointer | fixed
        "multi_select": "enumerate",  # enumerate | sequential | topk
        "enumerate_cap": 1024,
    },
    "bc": {
        "heuristic_games": 20000,
        "epsilon_games": 5000,
        "epsilon": 0.1,
        "random_games": 5000,
        "split": [0.9, 0.05, 0.05],
        "lr": 1e-3,
        "lr_min": 1e-5,
        "weight_decay": 1e-4,
        "batch": 512,
        "epochs": 4,
        "grad_clip": 1.0,
        "workers": 5,
        "precision": "bf16",  # fp32 | bf16 (autocast; 1.5x faster steps, measured 2026-10-02)
        "compile": True,  # torch.compile the trunk on CUDA (+18%; needs a C compiler)
    },
    "search": {
        "regime": "within_turn",  # within_turn | full_game
        "leaf": "student",  # student | belief_oracle | heuristic
        "rollout": "heuristic",  # heuristic | network | random
        # all | own_deck | opponent_private. Only "all" is honest: own_deck
        # leaves the opponent's true hand in every world, opponent_private my
        # true future draws (plan status, departure 4). The other two exist
        # for the hidden-information diagnostic, which sets them explicitly.
        "resample": "all",
        "simulations": 100,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.8,
        "dirichlet_eps": 0.25,
        "virtual_loss": 1,
        "leaves_in_flight": 8,
        "temperature_moves": 15,
        "temperature_high": 1.0,
        "temperature_low": 0.1,
        "belief_samples": 8,
        "branch_opponent_mid_turn": False,
        "quiet_leaves": False,  # full-game: evaluate leaves at the end of their own turn
    },
    "selfplay": {
        "games": 50000,
        # 8 actors: +28% searches/s over 5 on this machine (6 cores / 12
        # threads), leaving room for the learner and the inference server;
        # 10 adds only +4% more (tools/bench_actor_demand.py, 2026-10-02).
        "workers": 8,
        "games_per_worker": 16,
        "precision": "bf16",  # learner: fp32 | bf16
        "compile": True,  # learner: torch.compile the trunk on CUDA
        "playout_cap_full_fraction": 0.25,
        "playout_cap_small": 25,
        "playout_cap_full": 100,
        "lr": 2e-3,
        "lr_min": 1e-5,
        "batch": 1024,
        "grad_clip": 1.0,
        "buffer_positions": 1000000,
        "positions_per_step": 12,
        "gate_every_steps": 1000,
        "gate_games": 400,
        "mirror_augmentation": True,
        "training_max_turns": 60,
        "resign_threshold": None,
        "resign_consecutive": 3,
        "resign_exempt_fraction": 0.1,
        "loss_weights": {"policy": 1.0, "value": 1.0, "belief": 0.25, "oracle": 0.25, "distill": 0.0},
        "weight_decay": 1e-4,
    },
    "dmc": {
        "epsilon_start": 0.3,
        "epsilon_end": 0.01,
        "epsilon_decay_games": 20000,
    },
    "telemetry": {"interval_seconds": 60},
}


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        here = f"{path}.{k}" if path else k
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # optional
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError(f"{path}: reading YAML needs PyYAML; use a .json config instead") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def code_stamp() -> dict:
    """The version stamps of the code a config resolves against."""
    from keyforge.version import RULES_HASH

    stamp = spec.stamp()
    stamp["rules_hash"] = RULES_HASH
    return stamp


def resolve(source: Any = None, overrides: Optional[dict] = None) -> dict:
    """`source` is a path (absolute, or relative to `configs/`), a dict, or
    None (defaults only). `overrides` is merged last. Unknown top-level
    keys are an error -- a typo'd section would otherwise be silently
    ignored and the run would use defaults it didn't mean to."""
    if source is None:
        raw: dict = {}
    elif isinstance(source, dict):
        raw = source
    else:
        path = source if os.path.isabs(source) or os.path.exists(source) else os.path.join(CONFIG_DIR, source)
        raw = _read_file(path)
    merged = _deep_merge(DEFAULTS, raw)
    if overrides:
        merged = _deep_merge(merged, overrides)
    unknown = sorted(set(merged) - set(DEFAULTS) - {"code", "notes"})
    if unknown:
        raise ValueError(f"config has unknown top-level keys: {unknown}")
    merged["code"] = code_stamp()
    return merged


def canonical_bytes(resolved: dict) -> bytes:
    return json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def config_hash(resolved: dict) -> str:
    return hashlib.sha256(canonical_bytes(resolved)).hexdigest()


def diff_configs(a: dict, b: dict, path: str = "") -> List[Tuple[str, Any, Any]]:
    """Every leaf path where `a` and `b` differ: `(path, a_value, b_value)`
    -- "comparable iff the differences are enumerable", made literal."""
    out = []
    for k in sorted(set(a) | set(b)):
        here = f"{path}.{k}" if path else k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(diff_configs(va, vb, here))
        elif va != vb:
            out.append((here, va, vb))
    return out

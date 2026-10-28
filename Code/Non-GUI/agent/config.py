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
    # The network's input (Agent Observation Plan): 1 = v1 (KeyForgeNet), 2 =
    # network v2 (KeyForgeNetV2; its options are the `network` section's, as
    # configs/tier0b_bc.json's). Self-play only so far; BC v2 is ml.bc_train_v2.
    "observation": {"version": 1},
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
        # Layer implementation, not architecture: both compute the same
        # function with the same parameters and checkpoint keys
        # (ml/layers.py). sdpa calls scaled_dot_product_attention with
        # `attention_kernel` first in priority (flash | cudnn | efficient |
        # math | auto): +3-8% BC samples/s and +6-25% inference at 73 tokens,
        # +45-52% training at 400 (2026-10-05). torch is the layer the
        # Tier 0 checkpoints were trained with; they load into either.
        "attention": "sdpa",  # sdpa | torch
        "attention_kernel": "flash",
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
        # Memory: forward/backward in pieces of at most this many rows, with
        # the same gradient as the whole batch (ml/accumulate.py). None = off.
        "micro_batch": None,
        # Memory: recompute each trunk layer in backward (needs attention=sdpa).
        "activation_checkpointing": False,
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
        # How each world is sampled (keyforge/determinize.py, observation
        # plan O3): uniform | constrained | chance_exact | belief.
        "determinization": "chance_exact",
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
        "micro_batch": None,  # learner: as bc.micro_batch
        "activation_checkpointing": False,  # learner: as bc.activation_checkpointing
        "playout_cap_full_fraction": 0.25,
        "playout_cap_small": 25,
        "playout_cap_full": 100,
        "lr": 2e-3,
        "lr_min": 1e-5,
        "batch": 1024,
        "grad_clip": 1.0,
        "buffer_positions": 1000000,
        # How hard the learner may use each position: one gradient step per
        # batch / sample_reuse new positions, so a position is sampled about
        # `sample_reuse` times while it is in the buffer. Tier 2 round 1 ran
        # at 85 (positions_per_step 12, batch 1024) and its value head
        # memorized the buffer's games (train MSE 0.05 -> 0.0002).
        # `positions_per_step`, if set, overrides (the smoke configs).
        "sample_reuse": 6.0,
        "positions_per_step": None,
        # How many times one position may train each group of targets (None:
        # no limit). The policy group is the search's visit target (options
        # or multi-select); the outcome group is everything regressed onto
        # the game's result z (value, oracle, distillation). A position past
        # its limit drops that group's targets, and one past both is no
        # longer sampled -- so 1 / 1 with sample_reuse 1 trains every
        # position exactly once. Tier 2 round 2 (reuse 6, no limits): the
        # value head still memorized games (held-out MSE 1.1 vs 0.07), as
        # each game's ~210 positions share one z.
        "policy_reuse": None,
        "value_reuse": None,
        # Games held out of training (by seed), their value and policy loss
        # logged every `validation_every_steps` steps: memorization shows
        # as a train/validation gap in the monitor while the run goes.
        "validation_fraction": 0.02,
        "validation_games": 400,
        "validation_every_steps": 250,
        "gate_every_steps": 1000,
        "gate_games": 400,
        # Processes a gate plays over (ml.selfplay_gate.run_gate); 1 = the gate process alone. In Tier 2 round 3
        # one-process gates took 1-4.4 h beside 28 actors.
        "gate_workers": 1,
        # The promotion rule (ml.selfplay_gate.gate_rule): an SPRT between these
        # Elo bounds, else at `gate_games` a fixed-N test at this many standard
        # errors over 0.5.
        "gate_sprt": [0.0, 35.0],
        "gate_fallback_z": 2.0,
        "mirror_augmentation": True,
        "training_max_turns": 60,
        "resign_threshold": None,
        "resign_consecutive": 3,
        "resign_exempt_fraction": 0.1,
        "loss_weights": {"policy": 1.0, "value": 1.0, "belief": 0.25, "oracle": 0.25, "distill": 0.0},
        "weight_decay": 1e-4,
        # v2 only: processes that replay and encode arriving games for the
        # learner (ml/selfplay_v2.py), and the memory cap per training piece
        # (as bc.max_tokens; None = whole batches).
        "encode_workers": 2,
        "max_tokens": None,
    },
    "dmc": {
        "epsilon_start": 0.3,
        "epsilon_end": 0.01,
        "epsilon_decay_games": 20000,
    },
    "telemetry": {"interval_seconds": 60},
}


def positions_per_step(sp: dict) -> float:
    """New positions per learner step: `selfplay.positions_per_step`, or
    batch / `sample_reuse`."""
    if sp.get("positions_per_step"):
        return float(sp["positions_per_step"])
    return float(sp["batch"]) / float(sp.get("sample_reuse") or 6.0)


def held_out(seed: int, fraction: float) -> bool:
    """Whether a self-play game (by its seed) is a validation game."""
    if fraction <= 0:
        return False
    import hashlib as _h

    d = _h.blake2b(f"validation|{seed}".encode(), digest_size=4).digest()
    return int.from_bytes(d, "big") / 2 ** 32 < fraction


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

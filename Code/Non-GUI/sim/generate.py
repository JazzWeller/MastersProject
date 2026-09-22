"""Trajectory schema and self-play data generation (Agent Interface Plan,
Milestone J).

Storage trade-off, decided explicitly by the plan: store replay records
and regenerate observations on demand (compact, always consistent with
the current encoder) rather than encoded tensors (fast to train from, but
invalidated by any encoder change). At ~30k labelled decisions/sec
(Milestone 0's baseline), regenerating from a replay record is cheap. So a
`GameTrajectory` here is deliberately thin: per decision, the option keys,
the policy distribution over them (if the agent has one), the chosen
index, and a value estimate -- never a serialized `Observation`, which a
consumer rebuilds later from `(config, choice_record)` via `keyforge.
observation.build_observation` at whatever point they need it.

A separate, flagged `PrivilegedTrajectory` holds the true hidden state
(the final hand and deck order for both players) -- the free labels for
belief networks and oracle-guided value training, kept apart so they can
never leak into an ordinary agent's inputs by accident.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from bots.base import Controller
from keyforge.config import GameConfig
from keyforge.encoding import option_key
from keyforge.game import Game
from keyforge.replay import config_to_dict
from keyforge.version import ENGINE_VERSION, RULES_HASH


@dataclass
class DecisionRecord:
    player: int
    kind: str
    intent: Optional[str]
    option_keys: List[Any]
    policy: Optional[List[float]]  # None if the agent producing it has no policy distribution
    value_estimate: Optional[float]
    # Indices into option_keys -- length 1 for a single-choice decision,
    # 0..len(option_keys) for a multi-select one (CHOOSE_CARDS/
    # ORDER_EFFECTS submit a list, not one option).
    chosen_indices: List[int]


@dataclass
class GameTrajectory:
    engine_version: str
    rules_hash: str
    config: Dict[str, Any]
    choice_record: List[Any]
    decisions: List[DecisionRecord] = field(default_factory=list)
    outcome: Optional[Dict[str, int]] = None  # {"1": +1/0/-1, "2": ...}


@dataclass
class PrivilegedTrajectory:
    """Kept in a separate file from `GameTrajectory` -- never merge these
    into an ordinary agent's training inputs."""

    final_hand: Dict[str, List[int]]  # {"1": [instance_id, ...], "2": [...]}
    final_deck_order: Dict[str, List[int]]
    final_archive: Dict[str, List[int]]


def play_and_record(
    config: GameConfig,
    agents: Dict[int, Controller],
    *,
    policy_of: Optional[Dict[int, Any]] = None,
    value_of: Optional[Dict[int, Any]] = None,
) -> "tuple[GameTrajectory, PrivilegedTrajectory]":
    """Plays one game with `agents` (a plain `Controller` per seat -- no
    driver, no batching, matching `sim/simulate.py`'s own style) and
    returns its trajectory plus the privileged companion.

    `policy_of`/`value_of`, if given, are `{seat: callable}` -- called as
    `policy_of[pid](view, decision) -> List[float]` (aligned with
    `decision.options`) and `value_of[pid](view) -> float` respectively,
    for an agent that can produce them (e.g. a real policy/value network).
    An agent with neither recorded gets `policy=None, value_estimate=None`
    for its decisions -- still a valid, playable trajectory, just without
    training targets for those two fields.
    """
    game = Game(config)
    decisions: List[DecisionRecord] = []
    while not game.is_over:
        d = game.pending_decision
        view = game.view_for(d.player)
        choice = agents[d.player].decide(view, d)
        keys = [option_key(d, o) for o in d.options]
        chosen_items = choice if isinstance(choice, list) else [choice]
        chosen_indices = [keys.index(option_key(d, o)) for o in chosen_items]
        policy = None
        if policy_of and d.player in policy_of:
            policy = list(policy_of[d.player](view, d))
        value_estimate = None
        if value_of and d.player in value_of:
            value_estimate = float(value_of[d.player](view))
        decisions.append(
            DecisionRecord(
                player=d.player, kind=d.kind.name, intent=(d.intent.name if d.intent is not None else None),
                option_keys=keys, policy=policy, value_estimate=value_estimate, chosen_indices=chosen_indices,
            )
        )
        game.submit(choice)

    outcome = {"1": game.outcome_for(1), "2": game.outcome_for(2)}
    trajectory = GameTrajectory(
        engine_version=ENGINE_VERSION, rules_hash=RULES_HASH,
        config=config_to_dict(config), choice_record=list(game.choice_record),
        decisions=decisions, outcome=outcome,
    )
    privileged = PrivilegedTrajectory(
        final_hand={str(pid): [c.instance_id for c in p.hand.cards()] for pid, p in game.players.items()},
        final_deck_order={str(pid): [c.instance_id for c in p.deck.cards()] for pid, p in game.players.items()},
        final_archive={str(pid): [c.instance_id for c in p.archive.cards()] for pid, p in game.players.items()},
    )
    return trajectory, privileged


def write_shard(path: str, trajectories: List[GameTrajectory]) -> None:
    """Appends one JSON object per line -- sharded, append-only trajectory
    files (Milestone H): safe for one worker to keep writing to, and safe
    to concatenate shards from many workers without parsing anything."""
    with open(path, "a", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(asdict(t), separators=(",", ":")))
            f.write("\n")


def write_privileged_shard(path: str, trajectories: List[PrivilegedTrajectory]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(asdict(t), separators=(",", ":")))
            f.write("\n")


def read_shard(path: str) -> List[GameTrajectory]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["decisions"] = [DecisionRecord(**d) for d in data["decisions"]]
            out.append(GameTrajectory(**data))
    return out

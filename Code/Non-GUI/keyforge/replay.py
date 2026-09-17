"""Compact, exact game records.

The engine is deterministic for a given `GameConfig.seed`, so a whole game
is described by its config plus, for each decision, which option was
chosen. Choices are stored as indices into `decision.options` (a list of
indices for CHOOSE_CARDS / ORDER_EFFECTS), which is independent of card
instance ids and of object identity, so a record can be serialized to JSON
and replayed in a fresh process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import GameConfig
from .enums import DecisionKind

_LIST_KINDS = (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS)


def _index_of(options: List[Any], item: Any, used: set) -> int:
    for i, opt in enumerate(options):
        if i not in used and opt is item:
            return i
    for i, opt in enumerate(options):
        if i not in used and opt == item:
            return i
    raise ValueError(f"{item!r} is not one of the decision's options")


def encode_choice(decision, choice) -> Any:
    if decision.kind in _LIST_KINDS:
        used: set = set()
        out = []
        for item in choice:
            i = _index_of(decision.options, item, used)
            used.add(i)
            out.append(i)
        return out
    return _index_of(decision.options, choice, set())


def decode_choice(decision, encoded) -> Any:
    if decision.kind in _LIST_KINDS:
        return [decision.options[i] for i in encoded]
    return decision.options[encoded]


def config_to_dict(config: GameConfig) -> Dict[str, Any]:
    return {
        "decks": list(config.decks),
        "first_player": config.first_player,
        "seed": config.seed,
        "max_turns": config.max_turns,
    }


def config_from_dict(data: Dict[str, Any]) -> GameConfig:
    return GameConfig(
        decks=tuple(data["decks"]),
        first_player=data.get("first_player"),
        seed=data.get("seed"),
        max_turns=data.get("max_turns"),
    )


def replay(config: GameConfig, record: List[Any], upto: Optional[int] = None):
    """A fresh Game advanced through the first `upto` recorded choices (all
    of them if None). Raises ValueError if the record doesn't fit the game,
    e.g. it was made by a different engine version."""
    from .game import Game  # local import: game.py imports this module

    game = Game(config)
    steps = record if upto is None else record[:upto]
    for n, encoded in enumerate(steps):
        if game.is_over or game.pending_decision is None:
            raise ValueError(f"record has more choices than the game ({n})")
        game.submit(decode_choice(game.pending_decision, encoded))
    return game

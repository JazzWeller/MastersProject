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

from .cards.decks import Deck, deck_from_dict, deck_to_dict, resolve_deck
from .config import GameConfig
from .enums import DecisionKind
from .version import check_version_stamp, version_stamp

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


def _encode_deck(source) -> Dict[str, Any]:
    """Embeds the full resolved decklist (not just a preset name), so a
    replay stays reproducible even if that preset or user deck is edited or
    deleted later."""
    return deck_to_dict(resolve_deck(source))


def config_to_dict(config: GameConfig) -> Dict[str, Any]:
    data = {
        "decks": [_encode_deck(d) for d in config.decks],
        "first_player": config.first_player,
        "seed": config.seed,
        "max_turns": config.max_turns,
        "starting_chains": dict(config.starting_chains) if config.starting_chains else None,
    }
    data.update(version_stamp())
    return data


def config_from_dict(data: Dict[str, Any]) -> GameConfig:
    """Raises `keyforge.version.EngineVersionMismatch` if `data` was stamped
    by a different engine version or rules hash than this one -- see
    keyforge/version.py for why that refuses outright rather than replaying
    a record that merely happens to still fit the current decision shapes."""
    check_version_stamp(data, what="game replay record")
    starting_chains = data.get("starting_chains")
    raw_decks = data["decks"]
    # Old records (Phase 1/pre-Milestone-D) stored bare preset-name strings.
    decks = tuple(d if isinstance(d, str) else deck_from_dict(d) for d in raw_decks)
    return GameConfig(
        decks=decks,
        first_player=data.get("first_player"),
        seed=data.get("seed"),
        max_turns=data.get("max_turns"),
        starting_chains={int(k): v for k, v in starting_chains.items()} if starting_chains else None,
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

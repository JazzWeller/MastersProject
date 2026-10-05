"""Game configuration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GameConfig:
    # Each element is a preset name, a path to a deck JSON file, or a
    # `keyforge.cards.decks.Deck` object (see `cards/decks.py`).
    decks: Tuple[Any, Any] = ("fignor", "igor")
    first_player: Optional[int] = None  # 1 or 2; None = random
    seed: Optional[int] = None
    max_turns: Optional[int] = None
    starting_chains: Optional[Dict[int, int]] = None  # {1: n, 2: n}; None = no starting chains
    # A deterministic list of setup operations, applied after normal setup
    # (deal, opening hands, mulligans) and before the first decision --
    # "positions as replayable data" (Agent Interface Plan, Milestone D).
    # Each element is a plain, JSON-safe tuple/list `(op, ...)`; see
    # `Game._apply_setup_script` for the supported ops. Every card an op
    # names is taken from that player's own dealt cards (wherever they
    # currently sit), never conjured, so the 36-card pool invariant holds
    # and the position is exactly as replayable as a normal seeded game.
    setup_script: Optional[List[Any]] = None

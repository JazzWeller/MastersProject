"""Game configuration."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class GameConfig:
    # Each element is a preset name, a path to a deck JSON file, or a
    # `keyforge.cards.decks.Deck` object (see `cards/decks.py`).
    decks: Tuple[Any, Any] = ("fignor", "igor")
    first_player: Optional[int] = None  # 1 or 2; None = random
    seed: Optional[int] = None
    max_turns: Optional[int] = None
    starting_chains: Optional[Dict[int, int]] = None  # {1: n, 2: n}; None = no starting chains

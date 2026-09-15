"""Game configuration."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GameConfig:
    decks: Tuple[str, str] = ("fignor", "igor")
    first_player: Optional[int] = None  # 1 or 2; None = random
    seed: Optional[int] = None
    max_turns: Optional[int] = None

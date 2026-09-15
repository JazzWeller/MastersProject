"""Owns the `Game` plus which seats are bots, and produces the
(before, events, after) triple `Director.build` needs for every `submit()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from bots.random_bot import RandomBot
from keyforge.config import GameConfig
from keyforge.game import Game

from .snapshot import BoardSnapshot, build_snapshot

SEAT_HUMAN = "human"
SEAT_BOT = "bot"


@dataclass
class MatchSettings:
    p1_deck: str = "fignor"
    p2_deck: str = "igor"
    p1_seat: str = SEAT_HUMAN
    p2_seat: str = SEAT_BOT
    first_player: Optional[int] = None
    seed: Optional[int] = None
    max_turns: Optional[int] = None


class EngineBridge:
    def __init__(self, settings: MatchSettings):
        self.settings = settings
        config = GameConfig(
            decks=(settings.p1_deck, settings.p2_deck),
            first_player=settings.first_player,
            seed=settings.seed,
            max_turns=settings.max_turns,
        )
        self.game = Game(config)
        self.seats: Dict[int, str] = {1: settings.p1_seat, 2: settings.p2_seat}
        self.bots = {
            pid: RandomBot(seed=(settings.seed or 0) + pid)
            for pid, seat in self.seats.items()
            if seat == SEAT_BOT
        }

    @property
    def is_over(self) -> bool:
        return self.game.is_over

    @property
    def pending_decision(self):
        return self.game.pending_decision

    def seat_is_bot(self, pid: int) -> bool:
        return self.seats.get(pid) == SEAT_BOT

    def any_human(self) -> bool:
        return any(seat == SEAT_HUMAN for seat in self.seats.values())

    def snapshot(self, viewer: int) -> BoardSnapshot:
        return build_snapshot(self.game, viewer)

    def bot_choice(self, pid: int):
        d = self.game.pending_decision
        return self.bots[pid].decide(self.game.view_for(pid), d)

    def submit(self, choice, viewer: int) -> Tuple[BoardSnapshot, list, BoardSnapshot]:
        before = self.snapshot(viewer)
        start = len(self.game.log.events)
        self.game.submit(choice)
        events = self.game.log.events[start:]
        after = self.snapshot(viewer)
        return before, events, after

"""Owns the `Game` plus which seats are bots, and produces the
(before, events, after) triple `Director.build` needs for every `submit()`.

`EngineBridge` is a live game (optionally recorded to the game history);
`ReplayBridge` rebuilds a recorded game and feeds it its recorded choices,
one step at a time, forwards or backwards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from bots.heuristic_bot import HeuristicBot
from keyforge.config import GameConfig
from keyforge.enums import DecisionKind
from keyforge.game import Game
from keyforge.replay import decode_choice, replay

from .snapshot import BoardSnapshot, build_snapshot

SEAT_HUMAN = "human"
SEAT_BOT = "bot"
SEAT_REPLAY = "replay"


@dataclass
class MatchSettings:
    p1_deck: str = "fignor"
    p2_deck: str = "igor"
    p1_seat: str = SEAT_HUMAN
    p2_seat: str = SEAT_BOT
    first_player: Optional[int] = None
    seed: Optional[int] = None
    max_turns: Optional[int] = None

    def with_concrete_seed(self) -> "MatchSettings":
        """A copy whose seed is always set: games must be reproducible to be
        replayable, so "random" means "a randomly chosen, recorded seed"."""
        if self.seed is not None:
            return self
        return replace(self, seed=random.SystemRandom().randrange(1, 2**31))

    def config(self) -> GameConfig:
        return GameConfig(
            decks=(self.p1_deck, self.p2_deck),
            first_player=self.first_player,
            seed=self.seed,
            max_turns=self.max_turns,
        )


class EngineBridge:
    def __init__(self, settings: MatchSettings, history=None):
        self.settings = settings.with_concrete_seed()
        self.config = self.settings.config()
        self.game = Game(self.config)
        self.seats: Dict[int, str] = {1: self.settings.p1_seat, 2: self.settings.p2_seat}
        self.bots = {
            pid: HeuristicBot(seed=(self.settings.seed or 0) + pid)
            for pid, seat in self.seats.items()
            if seat == SEAT_BOT
        }
        self.history = history
        self.history_id: Optional[int] = None
        if history is not None:
            self.history_id = history.start(self.config, self.seats[1], self.seats[2], self.game)
        self.last_history_id = self.history_id  # survives close(), for "Watch Replay"

    @property
    def is_over(self) -> bool:
        return self.game.is_over

    @property
    def pending_decision(self):
        return self.game.pending_decision

    def seat_is_bot(self, pid: int) -> bool:
        return self.seats.get(pid) in (SEAT_BOT, SEAT_REPLAY)

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
        self._save_if_boundary()
        return before, events, after

    def _save_if_boundary(self) -> None:
        """Persist at turn boundaries (and at the end) rather than on every
        decision: cheap enough to be invisible, frequent enough that a crash
        loses at most the turn in progress."""
        if self.history is None or self.history_id is None:
            return
        d = self.game.pending_decision
        if self.game.is_over or (d is not None and d.kind == DecisionKind.CHOOSE_HOUSE):
            self.history.update(self.history_id, self.config, self.game)

    def close(self, abandoned: bool = True) -> None:
        """Final save when the game scene is left (quit, back to menu, rematch)."""
        if self.history is not None and self.history_id is not None:
            self.history.update(self.history_id, self.config, self.game, abandoned=abandoned and not self.game.is_over)
            self.history_id = None


class ReplayBridge(EngineBridge):
    """A recorded game, stepped through its recorded choices. Every seat is a
    "replay" seat, so the game scene treats it like a bot-vs-bot game and
    never asks anyone for input."""

    def __init__(self, config: GameConfig, record: List, settings: Optional[MatchSettings] = None):
        self.config = config
        self.record = list(record)
        self.settings = settings or MatchSettings(
            p1_deck=config.decks[0], p2_deck=config.decks[1],
            p1_seat=SEAT_REPLAY, p2_seat=SEAT_REPLAY,
            first_player=config.first_player, seed=config.seed, max_turns=config.max_turns,
        )
        self.seats = {1: SEAT_REPLAY, 2: SEAT_REPLAY}
        self.bots = {}
        self.history = None
        self.history_id = None
        self.last_history_id = None
        self.game = Game(config)

    @property
    def position(self) -> int:
        return len(self.game.choice_record)

    @property
    def length(self) -> int:
        return len(self.record)

    @property
    def at_end(self) -> bool:
        return self.game.is_over or self.position >= len(self.record)

    def bot_choice(self, pid: int):
        return decode_choice(self.game.pending_decision, self.record[self.position])

    def rebuild_to(self, position: int) -> None:
        position = max(0, min(position, len(self.record)))
        self.game = replay(self.config, self.record, upto=position)

    def turn_start_positions(self) -> List[int]:
        """Record positions of the first decision of each turn (for next /
        previous-turn stepping), found by replaying once silently. Keyed on
        the engine's turn counter, so turns whose house was forced (no house
        decision) still count."""
        if not hasattr(self, "_turn_starts"):
            starts = [0]
            game = Game(self.config)
            last_turn = game.turn_number
            for i, encoded in enumerate(self.record):
                d = game.pending_decision
                if d is None:
                    break
                if game.turn_number != last_turn:
                    starts.append(i)
                    last_turn = game.turn_number
                game.submit(decode_choice(d, encoded))
            starts.append(len(self.record))
            self._turn_starts = sorted(set(starts))
        return self._turn_starts

    def _save_if_boundary(self) -> None:
        pass

    def close(self, abandoned: bool = True) -> None:
        pass

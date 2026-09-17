"""Game history: every game played is stored in a local SQLite database so it
can be looked up and replayed step by step later (Past Games in the menu).

A game is stored as its settings plus the engine's `choice_record` (the
index of the option chosen at every decision -- see keyforge/replay.py),
serialized as JSON and zlib-compressed. With the recorded seed that is
enough to reproduce the game exactly. Summary columns (decks, seats,
winner, turns, ...) are kept alongside so the list view never has to
decompress anything.

The database lives at `Code/GUI/data/history.sqlite3` by default; set the
`KEYFORGE_HISTORY_DB` environment variable to use another file (the tests
point it at a temporary file).
"""

from __future__ import annotations

import json
import os
import sqlite3
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from . import settings as S

from keyforge.config import GameConfig
from keyforge.replay import config_from_dict, config_to_dict

RECORD_FORMAT = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    status        TEXT NOT NULL,            -- 'in progress' | 'finished' | 'abandoned'
    p1_deck       TEXT NOT NULL,
    p2_deck       TEXT NOT NULL,
    p1_seat       TEXT NOT NULL,
    p2_seat       TEXT NOT NULL,
    first_player  INTEGER,
    seed          INTEGER NOT NULL,
    max_turns     INTEGER,
    winner        INTEGER,
    reason        TEXT,
    turns         INTEGER NOT NULL DEFAULT 0,
    decisions     INTEGER NOT NULL DEFAULT 0,
    p1_keys       INTEGER NOT NULL DEFAULT 0,
    p2_keys       INTEGER NOT NULL DEFAULT 0,
    record        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS games_started ON games(started_at DESC);
"""


def default_db_path() -> str:
    return os.environ.get("KEYFORGE_HISTORY_DB") or os.path.join(S.GUI_DIR, "data", "history.sqlite3")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def encode_record(config: GameConfig, choices: List[Any]) -> bytes:
    payload = {"format": RECORD_FORMAT, "config": config_to_dict(config), "choices": choices}
    return zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), 9)


def decode_record(blob: bytes):
    payload = json.loads(zlib.decompress(blob).decode("utf-8"))
    return config_from_dict(payload["config"]), payload["choices"]


@dataclass
class GameSummary:
    id: int
    started_at: str
    updated_at: str
    status: str
    p1_deck: str
    p2_deck: str
    p1_seat: str
    p2_seat: str
    first_player: Optional[int]
    seed: int
    max_turns: Optional[int]
    winner: Optional[int]
    reason: Optional[str]
    turns: int
    decisions: int
    p1_keys: int
    p2_keys: int

    @property
    def result_text(self) -> str:
        if self.status == "finished":
            if self.winner:
                return f"Player {self.winner} won ({self.reason})"
            return f"Draw ({self.reason})"
        return self.status.capitalize()


_SUMMARY_COLUMNS = (
    "id, started_at, updated_at, status, p1_deck, p2_deck, p1_seat, p2_seat, first_player, seed, "
    "max_turns, winner, reason, turns, decisions, p1_keys, p2_keys"
)


class GameHistory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        # Anything still marked in progress belongs to a session that ended
        # without closing its game (crash, killed process).
        self._conn.execute("UPDATE games SET status='abandoned' WHERE status='in progress'")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ writing ----

    def start(self, config: GameConfig, p1_seat: str, p2_seat: str, game) -> int:
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO games (started_at, updated_at, status, p1_deck, p2_deck, p1_seat, p2_seat, "
            "first_player, seed, max_turns, record) VALUES (?, ?, 'in progress', ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now, now, config.decks[0], config.decks[1], p1_seat, p2_seat,
                game.first_player, config.seed, config.max_turns, encode_record(config, []),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update(self, game_id: int, config: GameConfig, game, abandoned: bool = False) -> None:
        result = game.result or {}
        if game.is_over:
            status = "finished"
        elif abandoned:
            status = "abandoned"
        else:
            status = "in progress"
        self._conn.execute(
            "UPDATE games SET updated_at=?, status=?, winner=?, reason=?, turns=?, decisions=?, "
            "p1_keys=?, p2_keys=?, record=? WHERE id=?",
            (
                _now(), status, result.get("winner"), result.get("reason"), game.turn_number,
                len(game.choice_record), game.players[1].keys, game.players[2].keys,
                encode_record(config, list(game.choice_record)), game_id,
            ),
        )
        self._conn.commit()

    def delete(self, game_id: int) -> None:
        self._conn.execute("DELETE FROM games WHERE id=?", (game_id,))
        self._conn.commit()

    # ------------------------------------------------------------ reading ----

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def list(self, limit: int = 20, offset: int = 0) -> List[GameSummary]:
        rows = self._conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM games ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [GameSummary(*row) for row in rows]

    def get(self, game_id: int) -> Optional[GameSummary]:
        row = self._conn.execute(f"SELECT {_SUMMARY_COLUMNS} FROM games WHERE id=?", (game_id,)).fetchone()
        return GameSummary(*row) if row else None

    def load(self, game_id: int):
        """(GameConfig, choice record) for a stored game."""
        row = self._conn.execute("SELECT record FROM games WHERE id=?", (game_id,)).fetchone()
        if row is None:
            raise KeyError(game_id)
        return decode_record(row[0])

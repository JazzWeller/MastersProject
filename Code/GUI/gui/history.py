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

from keyforge.cards.decks import deck_label
from keyforge.config import GameConfig
from keyforge.match import Match, MatchConfig, match_config_from_dict, match_config_to_dict
from keyforge.replay import config_from_dict, config_to_dict

RECORD_FORMAT = 1

# Table creation and index creation are split so `_migrate` can add columns
# a pre-existing `games` table is missing *before* an index referencing
# them is created (sqlite errors creating an index on a nonexistent column).
_SCHEMA_TABLES = """
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
    record        BLOB NOT NULL,
    match_id      INTEGER,
    match_game_index INTEGER
);

CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    status        TEXT NOT NULL,            -- 'in progress' | 'finished' | 'abandoned'
    format        TEXT NOT NULL,
    p1_deck       TEXT NOT NULL,
    p2_deck       TEXT NOT NULL,
    p1_seat       TEXT NOT NULL,
    p2_seat       TEXT NOT NULL,
    first_player  INTEGER,
    seed          INTEGER NOT NULL,
    max_turns     INTEGER,
    winner        INTEGER,
    reason        TEXT,
    games_played  INTEGER NOT NULL DEFAULT 0,
    bid_owner     INTEGER,
    bid_winner    INTEGER,
    bid_amount    INTEGER,
    bid_deck      TEXT,
    record        BLOB NOT NULL
);
"""

_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS games_started ON games(started_at DESC);
CREATE INDEX IF NOT EXISTS games_match ON games(match_id);
CREATE INDEX IF NOT EXISTS matches_started ON matches(started_at DESC);
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


def encode_match_record(config: MatchConfig, choices: List[Any]) -> bytes:
    payload = {"format": RECORD_FORMAT, "config": match_config_to_dict(config), "choices": choices}
    return zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), 9)


def decode_match_record(blob: bytes):
    payload = json.loads(zlib.decompress(blob).decode("utf-8"))
    return match_config_from_dict(payload["config"]), payload["choices"]


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
    match_id: Optional[int] = None
    match_game_index: Optional[int] = None

    @property
    def result_text(self) -> str:
        if self.status == "finished":
            if self.winner:
                return f"Player {self.winner} won ({self.reason})"
            return f"Draw ({self.reason})"
        return self.status.capitalize()


@dataclass
class MatchSummary:
    id: int
    started_at: str
    updated_at: str
    status: str
    format: str
    p1_deck: str
    p2_deck: str
    p1_seat: str
    p2_seat: str
    first_player: Optional[int]
    seed: int
    max_turns: Optional[int]
    winner: Optional[int]
    reason: Optional[str]
    games_played: int
    bid_owner: Optional[int]
    bid_winner: Optional[int]
    bid_amount: Optional[int]
    bid_deck: Optional[str]

    @property
    def result_text(self) -> str:
        if self.status == "finished":
            if self.winner:
                return f"Player {self.winner} won ({self.reason})"
            return f"Undecided ({self.reason})"
        return self.status.capitalize()


_SUMMARY_COLUMNS = (
    "id, started_at, updated_at, status, p1_deck, p2_deck, p1_seat, p2_seat, first_player, seed, "
    "max_turns, winner, reason, turns, decisions, p1_keys, p2_keys, match_id, match_game_index"
)
_MATCH_SUMMARY_COLUMNS = (
    "id, started_at, updated_at, status, format, p1_deck, p2_deck, p1_seat, p2_seat, first_player, seed, "
    "max_turns, winner, reason, games_played, bid_owner, bid_winner, bid_amount, bid_deck"
)


class GameHistory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA_TABLES)
        self._migrate()
        self._conn.executescript(_SCHEMA_INDEXES)
        # Anything still marked in progress belongs to a session that ended
        # without closing its game (crash, killed process).
        self._conn.execute("UPDATE games SET status='abandoned' WHERE status='in progress'")
        self._conn.execute("UPDATE matches SET status='abandoned' WHERE status='in progress'")
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database file's first creation.
        `CREATE TABLE IF NOT EXISTS` in `_SCHEMA` doesn't touch existing
        tables, so a pre-Phase-2 `games` table needs these added by hand."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(games)")}
        if "match_id" not in cols:
            self._conn.execute("ALTER TABLE games ADD COLUMN match_id INTEGER")
        if "match_game_index" not in cols:
            self._conn.execute("ALTER TABLE games ADD COLUMN match_game_index INTEGER")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ writing ----

    def start(self, config: GameConfig, p1_seat: str, p2_seat: str, game, match_id: Optional[int] = None,
              match_game_index: Optional[int] = None) -> int:
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO games (started_at, updated_at, status, p1_deck, p2_deck, p1_seat, p2_seat, "
            "first_player, seed, max_turns, record, match_id, match_game_index) "
            "VALUES (?, ?, 'in progress', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now, now, deck_label(config.decks[0]), deck_label(config.decks[1]), p1_seat, p2_seat,
                game.first_player, config.seed, config.max_turns, encode_record(config, []),
                match_id, match_game_index,
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

    def start_match(self, config: MatchConfig, p1_seat: str, p2_seat: str, match: Match) -> int:
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO matches (started_at, updated_at, status, format, p1_deck, p2_deck, p1_seat, p2_seat, "
            "first_player, seed, max_turns, record) VALUES (?, ?, 'in progress', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now, now, config.format, deck_label(config.decks[0]), deck_label(config.decks[1]), p1_seat, p2_seat,
                config.first_player, config.seed, config.max_turns, encode_match_record(config, []),
            ),
        )
        self._conn.commit()
        match_id = cur.lastrowid
        self._sync_match_games(match_id, match, p1_seat, p2_seat)
        return match_id

    def update_match(self, match_id: int, config: MatchConfig, match: Match, abandoned: bool = False) -> None:
        result = match.result or {}
        if match.is_over:
            status = "finished"
        elif abandoned:
            status = "abandoned"
        else:
            status = "in progress"
        bid = match.bid
        self._conn.execute(
            "UPDATE matches SET updated_at=?, status=?, winner=?, reason=?, games_played=?, "
            "bid_owner=?, bid_winner=?, bid_amount=?, bid_deck=?, record=? WHERE id=?",
            (
                _now(), status, result.get("winner"), result.get("reason"), len(match.games),
                bid.owner if bid else None, bid.winner if bid else None, bid.amount if bid else None,
                bid.deck if bid else None, encode_match_record(config, list(match.choice_record)), match_id,
            ),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT p1_seat, p2_seat FROM matches WHERE id=?", (match_id,)).fetchone()
        if row is not None:
            self._sync_match_games(match_id, match, row[0], row[1])

    def _sync_match_games(self, match_id: int, match: Match, p1_seat: str, p2_seat: str) -> None:
        """Save every game the match has finished so far as an ordinary row
        in `games` (reusing the existing single-game replay machinery
        unchanged), tagged with `match_id` so it groups under its match."""
        existing = {
            row[0]
            for row in self._conn.execute(
                "SELECT match_game_index FROM games WHERE match_id=?", (match_id,)
            ).fetchall()
        }
        for i, (record, game) in enumerate(zip(match.games, match.finished_games)):
            if i in existing:
                continue
            game_id = self.start(record.config, p1_seat, p2_seat, game, match_id=match_id, match_game_index=i)
            self.update(game_id, record.config, game)

    def delete_match(self, match_id: int) -> None:
        self._conn.execute("DELETE FROM games WHERE match_id=?", (match_id,))
        self._conn.execute("DELETE FROM matches WHERE id=?", (match_id,))
        self._conn.commit()

    # ------------------------------------------------------------ reading ----

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def list(self, limit: int = 20, offset: int = 0, match_id: Optional[int] = None) -> List[GameSummary]:
        if match_id is not None:
            rows = self._conn.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM games WHERE match_id=? ORDER BY match_game_index ASC LIMIT ? OFFSET ?",
                (match_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM games ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [GameSummary(*row) for row in rows]

    def get(self, game_id: int) -> Optional[GameSummary]:
        row = self._conn.execute(f"SELECT {_SUMMARY_COLUMNS} FROM games WHERE id=?", (game_id,)).fetchone()
        return GameSummary(*row) if row else None

    def count_matches(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    def list_matches(self, limit: int = 20, offset: int = 0) -> List[MatchSummary]:
        rows = self._conn.execute(
            f"SELECT {_MATCH_SUMMARY_COLUMNS} FROM matches ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [MatchSummary(*row) for row in rows]

    def get_match(self, match_id: int) -> Optional[MatchSummary]:
        row = self._conn.execute(f"SELECT {_MATCH_SUMMARY_COLUMNS} FROM matches WHERE id=?", (match_id,)).fetchone()
        return MatchSummary(*row) if row else None

    def load_match(self, match_id: int):
        """(MatchConfig, choice record) for a stored match -- feed to
        `keyforge.match.match_replay` to reconstruct it exactly."""
        row = self._conn.execute("SELECT record FROM matches WHERE id=?", (match_id,)).fetchone()
        if row is None:
            raise KeyError(match_id)
        return decode_match_record(row[0])

    def load(self, game_id: int):
        """(GameConfig, choice record) for a stored game."""
        row = self._conn.execute("SELECT record FROM games WHERE id=?", (game_id,)).fetchone()
        if row is None:
            raise KeyError(game_id)
        return decode_record(row[0])

"""Match play: Reversal (v1.2) and Adaptive (v1.3), spanning multiple games.

`Match` drives one or more `Game`s through the same `pending_decision` /
`submit()` / `view_for()` coroutine API as `Game` itself, so any Controller,
the text UI, and the GUI can drive a match exactly as they drive a single
game -- the only new thing they need to handle is two new `DecisionKind`s
(`BID_CHAINS`, `CHOOSE_FIRST_PLAYER`) that `Match` asks directly, between
games.

Format rules (see Code/PHASE_2_PLAN.md, "v1.2 / v1.3: Match formats"):
- "archon": a single game, own deck each. (Wrapped in `Match` too, so the
  text UI and sim have one uniform entry point across all formats.)
- "reversal": a single game where each player uses the *other* player's
  deck. Ownership follows whoever is assigned the deck for that game, so no
  special owner/controller handling is needed.
- "adaptive": game 1 with each player's own deck, game 2 with decks
  swapped. If the same player won both, they win the match 2-0. Otherwise
  the swap structure guarantees the *same deck* won both games (once under
  each player) -- game 3 bids chains for the right to pilot that deck,
  starting at 0 chains offered by its owner, bids alternating and capped at
  24. The bid winner plays the proven deck with that many starting chains;
  the loser plays the other deck at 0. Whoever lost the previous game
  chooses to go first or second in the next one; game 1 is random.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import random

from .cards.decks import deck_from_dict, deck_label, deck_to_dict, resolve_deck
from .config import GameConfig
from .decision import Decision
from .enums import DecisionKind
from .game import Game
from .replay import encode_choice
from .version import check_version_stamp, version_stamp

MAX_BID = 24
FORMATS = ("archon", "reversal", "adaptive")


@dataclass
class MatchConfig:
    format: str
    decks: Tuple[Any, Any] = ("fignor", "igor")  # each: preset name, file path, or `Deck`
    first_player: Optional[int] = None  # applies to the format's first game only; None = random
    seed: Optional[int] = None
    max_turns: Optional[int] = None  # forwarded to every Game (sim/fuzz use only)

    def __post_init__(self):
        if self.format not in FORMATS:
            raise ValueError(f"Unknown match format: {self.format!r} (expected one of {FORMATS})")


@dataclass
class GameRecord:
    """One game's config and result, kept for history/replay and for the
    "how decisively did it win" bidding heuristic."""

    config: GameConfig
    winner: Optional[int]
    turns: int
    reason: str
    seat_decks: Dict[int, str]
    final_chains: Dict[int, int]
    final_keys: Dict[int, int]


@dataclass
class BidRecord:
    owner: int
    winner: int
    amount: int
    deck: str


@dataclass
class MatchView:
    """What a Controller sees for a match-level decision (BID_CHAINS,
    CHOOSE_FIRST_PLAYER): no board, just the match so far."""

    viewer: int
    format: str
    games: List[GameRecord] = field(default_factory=list)
    score: Dict[int, int] = field(default_factory=dict)


class Match:
    def __init__(self, config: MatchConfig):
        self.config = config
        self.format = config.format
        self.decks = config.decks
        self.rng = random.Random(config.seed)
        self.games: List[GameRecord] = []
        # The actual finished Game objects, parallel to `self.games` -- kept
        # around (not just their GameRecord summaries) so a finished game's
        # own `choice_record` is still available, e.g. for the GUI to persist
        # each sub-game through the existing single-game replay machinery.
        self.finished_games: List[Game] = []
        self.bid: Optional[BidRecord] = None
        self.current_game: Optional[Game] = None
        self.score: Dict[int, int] = {1: 0, 2: 0}
        self.is_over = False
        self.result: Optional[dict] = None
        self.bidding_deck: Optional[str] = None  # set while a BID_CHAINS decision is pending
        self.choice_log: list = []
        # Same shape as `Game.choice_record` (option indices), but spanning
        # every match-level and per-game decision in order: with the
        # match's seed, this reproduces the whole match. See `match_replay`.
        self.choice_record: list = []
        self._driver = self._run()
        self.pending_decision: Optional[Decision] = None
        self._prime()

    # ---------------------------------------------------------- driver ----

    def _prime(self):
        try:
            self.pending_decision = next(self._driver)
        except StopIteration:
            self.is_over = True
            self.pending_decision = None

    def submit(self, choice) -> None:
        if self.is_over or self.pending_decision is None:
            raise RuntimeError("No pending decision to submit a choice for")
        if not self.pending_decision.validate(choice):
            raise ValueError(f"Invalid choice {choice!r} for decision {self.pending_decision}")
        self.choice_log.append(choice)
        self.choice_record.append(encode_choice(self.pending_decision, choice))
        try:
            self.pending_decision = self._driver.send(choice)
        except StopIteration:
            self.is_over = True
            self.pending_decision = None

    def view_for(self, pid: int):
        if self.current_game is not None and not self.current_game.is_over:
            return self.current_game.view_for(pid)
        return MatchView(viewer=pid, format=self.format, games=list(self.games), score=dict(self.score))

    def _next_game_seed(self) -> int:
        return self.rng.randrange(2**31)

    def _score_game(self, record: GameRecord) -> None:
        if record.winner is not None:
            self.score[record.winner] = self.score.get(record.winner, 0) + 1

    # -------------------------------------------------------------- run ----

    def _run(self):
        if self.format in ("archon", "reversal"):
            seat_decks = self.decks if self.format == "archon" else (self.decks[1], self.decks[0])
            yield from self._play_game(decks=seat_decks, first_player=self.config.first_player)
            g = self.games[-1]
            self.result = {
                "winner": g.winner, "games": [g.winner], "format": self.format, "bid": None,
                "reason": None if g.winner is not None else "game did not conclude",
            }
            return
        yield from self._run_adaptive()

    def _run_adaptive(self):
        yield from self._play_game(decks=self.decks, first_player=self.config.first_player)
        g1 = self.games[-1]
        if g1.winner is None:
            self.result = {"winner": None, "reason": "game 1 did not conclude", "games": [None], "format": self.format}
            return

        next_first = yield from self._choose_first_player(3 - g1.winner)
        yield from self._play_game(decks=(self.decks[1], self.decks[0]), first_player=next_first)
        g2 = self.games[-1]
        if g2.winner is None:
            self.result = {
                "winner": None, "reason": "game 2 did not conclude",
                "games": [g1.winner, None], "format": self.format,
            }
            return

        if g1.winner == g2.winner:
            self.result = {
                "winner": g1.winner, "games": [g1.winner, g2.winner], "format": self.format, "bid": None,
                "reason": None,
            }
            return

        # A split necessarily means the same deck won both games (once
        # under each player), since decks swap seats between games 1 and 2.
        bid_deck = g1.seat_decks[g1.winner]
        bid_owner = 1 if deck_label(bid_deck) == deck_label(self.decks[0]) else 2
        other_deck = self.decks[1] if bid_owner == 1 else self.decks[0]

        self.bidding_deck = bid_deck
        winner_pid, amount = yield from self._run_bid(bid_owner)
        self.bidding_deck = None
        self.bid = BidRecord(owner=bid_owner, winner=winner_pid, amount=amount, deck=deck_label(bid_deck))
        loser_pid = 3 - winner_pid
        seat_decks = {winner_pid: bid_deck, loser_pid: other_deck}
        starting_chains = {winner_pid: amount, loser_pid: 0}

        next_first = yield from self._choose_first_player(3 - g2.winner)
        yield from self._play_game(
            decks=(seat_decks[1], seat_decks[2]), first_player=next_first, starting_chains=starting_chains,
        )
        g3 = self.games[-1]
        self.result = {
            "winner": g3.winner,
            "games": [g1.winner, g2.winner, g3.winner],
            "format": self.format,
            "bid": self.bid,
            "reason": None if g3.winner is not None else "game 3 did not conclude",
        }

    def _play_game(self, decks, first_player, starting_chains=None):
        game_config = GameConfig(
            decks=decks,
            first_player=first_player,
            seed=self._next_game_seed(),
            max_turns=self.config.max_turns,
            starting_chains=starting_chains,
        )
        game = Game(game_config)
        self.current_game = game
        while not game.is_over:
            choice = yield game.pending_decision
            game.submit(choice)
        result = game.result or {}
        record = GameRecord(
            config=game_config,
            winner=result.get("winner"),
            turns=game.turn_number,
            reason=result.get("reason", ""),
            seat_decks={1: decks[0], 2: decks[1]},
            final_chains={pid: p.chains for pid, p in game.players.items()},
            final_keys={pid: p.keys for pid, p in game.players.items()},
        )
        self.games.append(record)
        self.finished_games.append(game)
        self._score_game(record)

    def _choose_first_player(self, pid: int):
        choice = yield Decision(
            pid, DecisionKind.CHOOSE_FIRST_PLAYER,
            "You lost the last game: choose whether to go first or second next game",
            ["first", "second"], 1, 1,
        )
        return pid if choice == "first" else 3 - pid

    def _run_bid(self, owner_pid: int):
        """Ascending chain auction for the deck that won both games 1 and 2.
        The owner's implicit opening offer is 0 chains; bids alternate from
        there, and whoever passes first loses the bid to whoever bid last
        (0, if nobody ever bid). A bid of 24 (the rulebook cap) ends it."""
        opponent_pid = 3 - owner_pid
        current_bid = 0
        current_bidder: Optional[int] = None
        turn = opponent_pid
        while True:
            options = ["pass"] + list(range(current_bid + 1, MAX_BID + 1))
            if current_bidder is None:
                standing = "unclaimed at 0"
            elif current_bidder == turn:
                standing = f"{current_bid} chains, by you"
            else:
                standing = f"{current_bid} chains, by your opponent"
            prompt = f"Bid chains to pilot {deck_label(self.bidding_deck)} (won both games). Current bid: {standing}."
            choice = yield Decision(turn, DecisionKind.BID_CHAINS, prompt, options, 1, 1)
            if choice == "pass":
                winner_pid = current_bidder if current_bidder is not None else owner_pid
                return winner_pid, current_bid
            current_bid = choice
            current_bidder = turn
            if current_bid >= MAX_BID:
                return current_bidder, current_bid
            turn = 3 - turn


def match_config_to_dict(config: MatchConfig) -> Dict[str, Any]:
    data = {
        "format": config.format,
        # Full decklists, not bare names, so a replay survives later deck edits.
        "decks": [deck_to_dict(resolve_deck(d)) for d in config.decks],
        "first_player": config.first_player,
        "seed": config.seed,
        "max_turns": config.max_turns,
    }
    data.update(version_stamp())
    return data


def match_config_from_dict(data: Dict[str, Any]) -> MatchConfig:
    """Raises `keyforge.version.EngineVersionMismatch` on a stale stamp --
    see `keyforge.replay.config_from_dict`, which guards the same way."""
    check_version_stamp(data, what="match replay record")
    raw_decks = data["decks"]
    # Old records (before this embedding) stored bare preset-name strings.
    decks = tuple(d if isinstance(d, str) else deck_from_dict(d) for d in raw_decks)
    return MatchConfig(
        format=data["format"],
        decks=decks,
        first_player=data.get("first_player"),
        seed=data.get("seed"),
        max_turns=data.get("max_turns"),
    )


def match_decode_choice(decision: Decision, encoded) -> Any:
    from .replay import decode_choice

    return decode_choice(decision, encoded)


def match_replay(config: MatchConfig, record: List[Any], upto: Optional[int] = None) -> Match:
    """A fresh Match advanced through the first `upto` recorded choices (all
    of them if None). Raises ValueError if the record doesn't fit -- e.g. it
    was made by a different engine version, or a different match format."""
    match = Match(config)
    steps = record if upto is None else record[:upto]
    for n, encoded in enumerate(steps):
        if match.is_over or match.pending_decision is None:
            raise ValueError(f"record has more choices than the match ({n})")
        match.submit(match_decode_choice(match.pending_decision, encoded))
    return match

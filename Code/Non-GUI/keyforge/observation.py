"""Immutable, serializable per-player observations (Agent Interface Plan,
Milestone C).

`PlayerView` (keyforge/view.py) stays exactly as it is -- the GUI and
`RandomBot`/`HeuristicBot` read it unchanged. `Observation` is a separate,
richer artifact for anything that has to cross a process boundary, get
written to disk, or reason about the game's full public history:

- **No live `Card` references.** Cards appear by `instance_id` (stable per
  game, per Milestone A) plus their public state, so a consumer can't
  mutate engine state through its own observation, or read a field it
  isn't entitled to.
- **The full public history**, not a 20-event tail: the viewer's
  projection (keyforge/projection.py, Agent Observation Plan O1) -- every
  zone move, log event and decision as the viewer saw it, in game order,
  each with its visibility code (`vis`).
- **Match context in every observation**: format, which game this is
  within the match, the match score, and starting chains -- constants in a
  single Archon game, but what let one agent play every format without a
  per-format special case.
- **Generic decision handling**: every decision reaches the agent as a
  kind, an intent, and a list of `(option_key, option_features)` pairs
  (keyforge/encoding.py) -- a new decision kind is a new feature value, not
  an agent rewrite.
- **JSON-serializable by construction**: every field is already a plain
  int/float/str/bool/None/dict/tuple; `observation_to_dict` is a thin
  `dataclasses.asdict` wrapper.

`full_state_observation` is the privileged, everything-visible variant for
oracle baselines and the hidden-information diagnostic (Milestone J) --
reachable only through the privileged capability once Milestone G's
capability enforcement lands.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .cards.card import Card
from .encoding import option_features, option_key
from .state_hash import canonical_card_state, canonicalize
from .version import ENGINE_VERSION


def _card_identity(card: Card) -> dict:
    """Static identity only -- never current zone/state -- for a decklist
    entry. Both players' full 36-card decklists are public from the start
    (keyforge/view.py's existing `decklist` field), but a decklist card
    currently sitting in a hidden zone (deck, either archive, an enemy
    hand) must not otherwise leak its position or state through here."""
    return {"instance_id": card.instance_id, "name": card.name, "house": card.house.value, "type": card.type.value}


@dataclass(frozen=True)
class ObservedDecision:
    player: int
    kind: str
    intent: Optional[str]
    source_card: Optional[int]
    affects: Optional[str]
    min_n: int
    max_n: int
    optional: bool
    # (option_key, option_features) pairs, in `Decision.options` order.
    # Empty for a decision pending for a DIFFERENT player than this
    # observation's viewer -- the kind/intent of "someone is being asked
    # something" is public, but the options themselves might be drawn from
    # that other player's own hidden hand.
    options: Tuple[Tuple[Any, dict], ...] = ()


@dataclass(frozen=True)
class HistoryEntry:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    # "log" (`kind` is the event's), "zone" (a journal entry; `kind` is
    # "zone", or "revealed_card" for one card of a revealed hand) or
    # "decision"
    stream: str = "log"


@dataclass(frozen=True)
class ObservedPlayer:
    id: int
    aember: int
    keys: int
    chains: int
    hand_count: int
    hand: Optional[Tuple[dict, ...]]  # None if hidden
    archive_count: int
    archive: Optional[Tuple[dict, ...]]  # None if hidden
    deck_count: int
    discard: Tuple[dict, ...]
    purged: Tuple[dict, ...]
    creatures: Tuple[dict, ...]
    artifacts: Tuple[dict, ...]
    decklist: Tuple[dict, ...]
    selected_house: Optional[str]


@dataclass(frozen=True)
class Observation:
    viewer: int
    engine_version: str
    format: str
    game_number: int  # 1-based, within the match (or 1 for a standalone game)
    match_score: Dict[int, int]
    starting_chains: Dict[int, int]
    turn_number: int
    active_player: Optional[int]
    is_over: bool
    outcome: Optional[int]  # +1/0/-1 from `viewer`'s perspective, once known
    players: Dict[int, ObservedPlayer]
    history: Tuple[HistoryEntry, ...]
    pending_decision: Optional[ObservedDecision]
    privileged: bool = False


def _in_play_state(card: Card, viewer: int, privileged: bool) -> dict:
    """`canonical_card_state`, minus the identity of any card facedown
    beneath this one (Masterplan) that `viewer` didn't put there: the count
    is public, the identity isn't."""
    state = canonical_card_state(card)
    if not privileged and card.under_cards:
        state["under_cards"] = [
            {"__card__": u.instance_id} if u.owner == viewer else {"__card__": None} for u in card.under_cards
        ]
    return state


def _observed_player(player, viewer: int, privileged: bool) -> ObservedPlayer:
    is_mine = privileged or player.id == viewer
    visible_hand = is_mine or viewer in player.hand_revealed_to
    return ObservedPlayer(
        id=player.id,
        aember=player.aember,
        keys=player.keys,
        chains=player.chains,
        hand_count=len(player.hand),
        hand=tuple(canonical_card_state(c) for c in player.hand.cards()) if visible_hand else None,
        archive_count=len(player.archive),
        archive=tuple(canonical_card_state(c) for c in player.archive.cards()) if is_mine else None,
        deck_count=len(player.deck),
        discard=tuple(canonical_card_state(c) for c in player.discard.cards()),
        purged=tuple(canonical_card_state(c) for c in player.purged.cards()),
        creatures=tuple(_in_play_state(c, viewer, privileged) for c in player.play_area.creatures),
        artifacts=tuple(_in_play_state(c, viewer, privileged) for c in player.play_area.artifacts),
        decklist=tuple(_card_identity(c) for c in player.all_cards),
        selected_house=player.selected_house.value if player.selected_house else None,
    )


def _observed_decision(decision, viewer: int, privileged: bool) -> Optional[ObservedDecision]:
    if decision is None:
        return None
    reveal_options = privileged or decision.player == viewer
    options = tuple((option_key(decision, o), option_features(decision, o)) for o in decision.options) if reveal_options else ()
    return ObservedDecision(
        player=decision.player,
        kind=decision.kind.name,
        intent=decision.intent.name if decision.intent is not None else None,
        source_card=decision.source_card.instance_id if isinstance(decision.source_card, Card) else None,
        affects=decision.affects.value if decision.affects is not None else None,
        min_n=decision.min_n,
        max_n=decision.max_n,
        optional=decision.optional,
        options=options,
    )


_ZONE_FIELDS = ("seq", "turn", "iid", "owner", "controller", "from", "to", "op", "epoch", "position", "cause", "vis")
_DECISION_FIELDS = ("index", "player", "kind", "intent", "source", "affects", "optional", "chosen", "offered", "peek",
                    "vis")


def _history_entry(item) -> HistoryEntry:
    tag = item[0]
    if tag == "log":
        _, index, kind, data, code = item
        return HistoryEntry(kind=kind, data={**data, "vis": code}, stream="log")
    if tag == "zone":
        return HistoryEntry(kind="zone", data=canonicalize(dict(zip(_ZONE_FIELDS, item[1:]))), stream="zone")
    if tag == "reveal":
        _, seq, iid, zone, code = item
        return HistoryEntry(kind="revealed_card", data=canonicalize({"seq": seq, "iid": iid, "zone": zone, "vis": code}),
                            stream="zone")
    return HistoryEntry(kind="decision", data=canonicalize(dict(zip(_DECISION_FIELDS, item[1:]))), stream="decision")


def _history(game, viewer: int, privileged: bool) -> Tuple[HistoryEntry, ...]:
    return tuple(_history_entry(item) for item in game.projected(None if privileged else viewer))


def build_observation(game, viewer: int, *, match=None, privileged: bool = False) -> Observation:
    """`match`, if given, supplies format/game-number/score context; a
    standalone `Game` (no match) is reported as a single Archon game."""
    fmt = match.format if match is not None else "archon"
    game_number = len(match.games) + 1 if match is not None else 1
    match_score = dict(match.score) if match is not None else {1: 0, 2: 0}
    starting_chains = dict(game.config.starting_chains) if game.config.starting_chains else {1: 0, 2: 0}
    outcome = game.outcome_for(viewer) if game.is_over else None
    return Observation(
        viewer=viewer,
        engine_version=ENGINE_VERSION,
        format=fmt,
        game_number=game_number,
        match_score=match_score,
        starting_chains=starting_chains,
        turn_number=game.turn_number,
        active_player=None if game.is_over else game.active_player_id,
        is_over=game.is_over,
        outcome=outcome,
        players={pid: _observed_player(p, viewer, privileged) for pid, p in sorted(game.players.items())},
        history=_history(game, viewer, privileged),
        pending_decision=_observed_decision(game.pending_decision, viewer, privileged),
        privileged=privileged,
    )


def build_match_observation(match, viewer: int) -> Observation:
    """For `BID_CHAINS` / `CHOOSE_FIRST_PLAYER`, which arrive between games
    -- there's no live `Game` at that point (`Match.pending_decision` came
    from `Match` itself). At least one game has always finished by the time
    either of these can fire (see `Match._run_adaptive`), so `match.games`/
    `match.finished_games` are never empty here."""
    last_record = match.games[-1]
    last_game = match.finished_games[-1]
    players = {
        pid: ObservedPlayer(
            id=pid,
            aember=0,
            keys=last_record.final_keys.get(pid, 0),
            chains=last_record.final_chains.get(pid, 0),
            hand_count=0,
            hand=None,
            archive_count=0,
            archive=None,
            deck_count=0,
            discard=(),
            purged=(),
            creatures=(),
            artifacts=(),
            decklist=tuple(_card_identity(c) for c in last_game.players[pid].all_cards),
            selected_house=None,
        )
        for pid in (1, 2)
    }
    outcome = None
    if match.is_over and match.result is not None:
        winner = match.result.get("winner")
        outcome = None if winner is None else (1 if winner == viewer else -1)
    history = tuple(
        HistoryEntry(kind="game_result", data=canonicalize({"winner": g.winner, "turns": g.turns, "reason": g.reason}))
        for g in match.games
    )
    return Observation(
        viewer=viewer,
        engine_version=ENGINE_VERSION,
        format=match.format,
        game_number=len(match.games) + 1,
        match_score=dict(match.score),
        starting_chains={1: 0, 2: 0},
        turn_number=0,
        active_player=None,
        is_over=match.is_over,
        outcome=outcome,
        players=players,
        history=history,
        pending_decision=_observed_decision(match.pending_decision, viewer, False),
        privileged=False,
    )


def full_state_observation(game, viewer: int = 1, *, match=None) -> Observation:
    """The privileged, everything-visible observation: both players' hands,
    archives and decklists are fully shown regardless of `viewer`. Gated by
    the privileged capability once Milestone G's driver enforces it --
    calling this directly is itself the privilege check today."""
    return build_observation(game, viewer, match=match, privileged=True)


def observation_to_dict(obs: Observation) -> dict:
    return dataclasses.asdict(obs)


def observation_key(obs: Observation) -> str:
    """A canonical hash of `obs` -- the information-set key for ISMCTS
    nodes and transposition tables. Id-stable per Milestone A: two viewers
    (or two independent replays) with the same knowledge hash equal."""
    payload = json.dumps(observation_to_dict(obs), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

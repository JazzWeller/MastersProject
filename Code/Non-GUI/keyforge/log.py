"""Structured event log."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

_BOTH_PLAYERS = frozenset({1, 2})
_NO_PRIVATE: Dict[str, FrozenSet[int]] = {}

# Every event kind's redaction schema (Agent Observation Plan, O1): its
# private fields, each with the data field naming the one player who may see
# it. A field not listed is seen by everyone the event is logged for.
# `tests/test_agent_observation_o1.py` fails on an event kind logged
# anywhere in the engine that is missing here.
EVENT_SCHEMAS: Dict[str, Dict[str, str]] = {kind: {} for kind in (
    "aember_stored_lost", "archive", "archive_decision", "arise", "capture", "capture_released", "choose_house",
    "damage", "damage_prevented", "damage_redirected", "destroyed", "destroyed_in_fight", "discard",
    "discard_from_hand", "discard_random", "draw", "duration_effect", "exhaust", "fight", "forge_key",
    "forge_skipped", "gain", "gain_chains", "heal", "help_from_future_self", "house_forced", "lose",
    "mimicry_copy", "move_aember", "mulligan", "mulligan_decision", "pay", "place_aember", "play_card",
    "power_counter", "purge", "put_into_play", "put_on_bottom", "put_on_top", "ready", "reap", "reshuffle",
    "return_to_hand", "reveal", "reveal_hand", "reveal_top", "shed_chain", "shortfall", "shuffle_into_deck",
    "spend_stored_aember", "steal", "stun", "stun_consumed", "swap", "take_archive", "take_control",
    "timetraveler_shuffle", "turn_start", "under_card", "unforge", "use_action", "use_omni",
)}
EVENT_SCHEMAS["draw"] = {"iids": "player"}  # the count is public, the cards the drawer's
# Kinds whose call sites may log the whole event for one player only: each
# names a card that came from a hidden zone when it did (archived from a
# hand or a deck, returned to a hand from an archive, put facedown under a
# card). The move itself is public through the journal.
SITE_RESTRICTED = frozenset({"archive", "return_to_hand", "under_card"})
_PRIVATE_FIELDS = {kind: fields for kind, fields in EVENT_SCHEMAS.items() if fields}


@dataclass(slots=True)
class LogEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    # Which players are entitled to this event's content -- both by default.
    # A handful of zone-transition events name a card that was sitting in a
    # hidden zone (hand, deck, or a player's own archive) immediately before
    # the event; those are logged with `visible_to` narrowed to whoever
    # already legitimately knows that card (almost always just its owner),
    # so a player's own view (`view.build_view`'s `log_tail`) can filter
    # entries down to what they're actually entitled to see. See
    # Code/AGENT_INTERFACE_PLAN.md, Milestone C ("first, the leak").
    visible_to: FrozenSet[int] = field(default_factory=lambda: _BOTH_PLAYERS)
    # Fields only some of those players may see, by name: an event can be
    # public while one of its fields is not -- a `draw` is public, its count
    # too, but which cards were drawn is the drawer's alone (Agent
    # Observation Plan, Milestone O0). `redacted_for` drops a field from the
    # copy a player is shown; the event itself keeps every field, for the
    # rules and the GUI.
    private: Dict[str, FrozenSet[int]] = field(default_factory=lambda: _NO_PRIVATE)

    def __repr__(self):
        return f"{self.kind}({self.data})"

    def redacted_for(self, pid: int) -> "LogEvent":
        """This event as `pid` may see it: itself, or a copy without the
        fields `pid` isn't entitled to."""
        hidden = [k for k, who in self.private.items() if pid not in who]
        if not hidden:
            return self
        data = {k: v for k, v in self.data.items() if k not in hidden}
        return LogEvent(self.kind, data, visible_to=self.visible_to)


class GameLog:
    def __init__(self):
        self.events: List[LogEvent] = []
        # Same events, grouped by kind -- a few rules queries (Key Hammer,
        # The Warchest, Lifeweb: game.py's forged_key_on_turn and friends)
        # only ever care about one kind, and scanning just that kind instead
        # of the whole log matters as games lengthen (Milestone L). Derived
        # entirely from `events`/`add()`, so it's correct for every caller,
        # including a test that pokes `log.add(...)` directly rather than
        # going through the engine's own call sites.
        self.by_kind: Dict[str, List[LogEvent]] = defaultdict(list)

    def add(self, kind: str, visible_to: Optional[Any] = None, private: Optional[Dict[str, Any]] = None, **data) -> None:
        entitled = _BOTH_PLAYERS if visible_to is None else frozenset(visible_to)
        if private is None and kind in _PRIVATE_FIELDS:
            private = {f: (data[holder],) for f, holder in _PRIVATE_FIELDS[kind].items() if f in data}
        fields = _NO_PRIVATE if not private else {k: frozenset(who) for k, who in private.items()}
        event = LogEvent(kind, data, visible_to=entitled, private=fields)
        self.events.append(event)
        self.by_kind[kind].append(event)

    def tail(self, n: int) -> List[LogEvent]:
        return self.events[-n:]

    def visible_to(self, pid: int) -> List[LogEvent]:
        """Every event `pid` is entitled to see, engine-log order preserved,
        each with any field `pid` isn't entitled to removed."""
        return [e.redacted_for(pid) if e.private else e for e in self.events if pid in e.visible_to]

    def visible_tail(self, pid: int, n: int) -> List[LogEvent]:
        """`visible_to(pid)[-n:]`, reading only the end of the log."""
        out = []
        events = self.events
        i = len(events) - 1
        while i >= 0 and len(out) < n:
            e = events[i]
            if pid in e.visible_to:
                out.append(e.redacted_for(pid) if e.private else e)
            i -= 1
        out.reverse()
        return out

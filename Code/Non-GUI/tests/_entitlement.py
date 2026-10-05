"""A privileged recorder for the content-level leak test (Agent Observation
Plan, Milestone O0).

`recorded_game(config)` builds a `Game` whose log is wrapped, so that for
every event it also records which card identities each player was entitled
to at that moment. A card is hidden from a viewer only while it sits in a
zone that viewer can't see: either deck, the opponent's hand (unless it is
revealed to the viewer), the opponent's archive, or under one of the
opponent's cards. Everything else is visible, including a card between
zones (an action being played, a card leaving play).

An event is logged after the moves it describes, sometimes after other
events too, so its entitlement is every card that was visible to the viewer
at any event since the last submitted decision, up to and including this
one.

`check_history(game, recorder, viewer, shown)` compares the identities in
what a viewer was shown against that entitlement and returns every leak.
Identities are the `*iid`/`*iids` fields and any string field that names a
card in either decklist. `source` fields are exempt: they name the card
behind an effect, which was public when the effect began, and say nothing
about where that card is now.

Events whose purpose is to show a card to the viewer (the reveal kinds
below) are themselves the entitlement for the cards they name.
"""

from __future__ import annotations

import sys
from typing import Dict, FrozenSet, List, Set

from keyforge import game as game_module
from keyforge.log import GameLog

# Events that exist to reveal the cards they name, to everyone they are
# logged for.
REVEAL_KINDS = frozenset({"reveal", "reveal_top", "help_from_future_self", "timetraveler_shuffle", "mimicry_copy"})

EXEMPT_FIELDS = frozenset({"source"})


def _hidden_iids(game, viewer: int) -> FrozenSet[int]:
    hidden: Set[int] = set()
    for pid, p in game.players.items():
        hidden.update(c.instance_id for c in p.deck.cards())
        if pid != viewer:
            if viewer not in p.hand_revealed_to:
                hidden.update(c.instance_id for c in p.hand.cards())
            hidden.update(c.instance_id for c in p.archive.cards())
        for c in p.play_area.creatures + p.play_area.artifacts:
            for u in c.under_cards:
                if u.owner != viewer:
                    hidden.add(u.instance_id)
    return frozenset(hidden)


class EntitlementRecorder:
    def __init__(self, game, log):
        self.game = game
        self.all_iids: FrozenSet[int] = frozenset()
        # Per event: the cards visible to each viewer at that event, and
        # which submit's window it falls in. Per window: what was visible
        # when its decision was submitted.
        self.visible: List[Dict[int, FrozenSet[int]]] = []
        self.window_of: List[int] = []
        self.window_starts: List[int] = [0]
        self.window_visible: List[Dict[int, FrozenSet[int]]] = [{1: frozenset(), 2: frozenset()}]
        original = log.add

        def add(kind, visible_to=None, **data):
            if not self.all_iids:
                self.all_iids = frozenset(c.instance_id for p in game.players.values() for c in p.all_cards)
            self.visible.append(self._now())
            self.window_of.append(len(self.window_starts) - 1)
            original(kind, visible_to=visible_to, **data)

        log.add = add

        for name in ("submit", "submit_index"):
            method = getattr(game, name)

            def wrapped(choice, _method=method):
                self.window_starts.append(len(self.visible))
                self.window_visible.append(self._now())
                return _method(choice)

            setattr(game, name, wrapped)

    def _now(self) -> Dict[int, FrozenSet[int]]:
        return {v: self.all_iids - _hidden_iids(self.game, v) for v in (1, 2)}

    def entitled(self, index: int, viewer: int) -> FrozenSet[int]:
        window = self.window_of[index]
        out: Set[int] = set(self.window_visible[window][viewer])
        for i in range(self.window_starts[window], index + 1):
            out |= self.visible[i][viewer]
        return frozenset(out)


def recorded_game(config):
    """`(game, recorder)`: a new `Game` whose log is recorded from its first
    event. `Game.__init__` builds its log after its players and before it
    deals, so the log factory below finds the game under construction."""
    recorders = []

    def make_log():
        game = sys._getframe(1).f_locals["self"]
        log = GameLog()
        recorders.append(EntitlementRecorder(game, log))
        return log

    saved = game_module.GameLog
    game_module.GameLog = make_log
    try:
        game = game_module.Game(config)
    finally:
        game_module.GameLog = saved
    return game, recorders[0]


def _iid_values(key: str, value):
    if key == "iid" or key.endswith("_iid"):
        if isinstance(value, int) and not isinstance(value, bool):
            yield value
    elif key == "iids" or key.endswith("_iids"):
        if isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, int) and not isinstance(v, bool):
                    yield v


def _name_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                yield v


def check_history(game, recorder: EntitlementRecorder, viewer: int, shown) -> List[str]:
    """`shown` is a list of `(log_index, kind, data)` triples: what `viewer`
    was given, each tagged with the index of the full-log event it came
    from."""
    names = {c.name for p in game.players.values() for c in p.all_cards}
    name_of = {c.instance_id: c.name for p in game.players.values() for c in p.all_cards}
    leaks = []
    for index, kind, data in shown:
        if kind in REVEAL_KINDS:
            continue
        entitled = recorder.entitled(index, viewer)
        entitled_names = {name_of[i] for i in entitled if i in name_of}
        for key, value in data.items():
            if key in EXEMPT_FIELDS:
                continue
            for iid in _iid_values(key, value):
                if iid not in entitled:
                    leaks.append(f"event {index} {kind}.{key}: card {iid} ({name_of.get(iid)}) shown to player {viewer}")
            for name in _name_values(value):
                if name in names and name not in entitled_names:
                    leaks.append(f"event {index} {kind}.{key}: name {name!r} shown to player {viewer}")
    return leaks

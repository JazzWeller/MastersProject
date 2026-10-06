"""The complete current state, as one viewer may know it (Agent
Observation Plan, Milestone O4): the v2 information-set extract.

v1 (`keyforge/infoset.py`) stays exactly as it is, for v1 checkpoints. This
extract adds what v1 drops (the plan's Inventory D), and is checked by the
coverage registry (`agent/state_registry.py`, I4) to read every attribute
of the engine's state objects or say why not:

- **entities**: the 72 cards, in public decklist order (the viewer's, then
  the opponent's), each with the viewer's knowledge of it
  (`knowledge.card_knowledge`, O2/O3) and, where the viewer can see it, its
  whole state -- every `Card` and type-object attribute, cards it refers to
  as entity pointers -- and its position (side, zone, index, flank; the host
  entity for an upgrade or a card under another);
- **players**: every counter and flag of both players, and their zone sizes;
- **effects**: one token per lasting effect of any kind -- kind, name
  (variable, event or kind), op, value and condition evaluated now,
  remaining duration (None: infinite), affected and controlling side,
  source pointer;
- **cleanups** (pending end-of-turn operations), **temporary control**
  (original controller per card), **redirected hits**;
- **resolution**: the explicit stack (`resolution.resolution_view`, R7),
  one token per frame, redacted for the viewer;
- **turn**: number, active side, first player, `max_turns` and turns left;
- **match**: one token per previous game of the match.

Everything is plain data (ints, strings, tuples, dicts), sides are
relative to the viewer (0 = me, 1 = them), and nothing reads a hidden
card's state: the invariance test (I1) holds the extract byte-identical
across every determinized world.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .cards.card import Card
from .effects.effect_object import INFINITE
from .knowledge import tracker_for
from .state_hash import canonicalize

FEATURE_STATE_VERSION = 1

_ATOMS = (int, bool, str, float, type(None))

# Card attributes the extract leaves out, each with why (the registry
# repeats these; a new attribute must be classified there).
_CARD_SKIPPED = {
    "card_def": "static: the card's definition, by its name",
    "type_object": "read separately (the type object's own attributes)",
    "instance_id": "the entity's index is its identity",
    "extra_triggers": "read as the names of the hooks it holds",
    "under_cards": "each under-card points to its host",
    "_ember_imp_effect": "bookkeeping: the effect itself is an effect token",
}
_TYPE_SKIPPED = {"upgrades": "each upgrade points to its host", "host": "the upgrade's own host pointer"}


def _side(pid: Optional[int], viewer: int) -> Optional[int]:
    return None if pid is None else (0 if pid == viewer else 1)


class _Pointers:
    """Instance id -> entity index, and the viewer's visibility of each."""

    def __init__(self, game, viewer: int):
        me, them = game.players[viewer], game.players[3 - viewer]
        self.order = [c.instance_id for c in me.all_cards] + [c.instance_id for c in them.all_cards]
        self.index = {iid: i for i, iid in enumerate(self.order)}
        self.tracker = tracker_for(game, viewer)
        self.viewer = viewer

    def visible(self, iid: int) -> bool:
        m = self.tracker.mask.get(iid)
        if m is None or len(m) != 1:
            return False
        (zone,) = m
        return self.tracker.visible(zone, self.tracker.owner.get(iid))

    def ptr(self, iid: int):
        """An entity pointer: the index, if the viewer sees the card; else
        ("hidden", side of its owner)."""
        if self.visible(iid):
            return self.index[iid]
        return ("hidden", _side(self.tracker.owner.get(iid), self.viewer))

    def value(self, x):
        """`canonicalize`d, with every card as an entity pointer."""
        t = type(x)
        if t in _ATOMS:
            return x
        if t is Card:
            return {"entity": self.ptr(x.instance_id)}
        if isinstance(x, Enum):
            return x.value
        c = canonicalize(x)
        return self._swap(c)

    def _swap(self, c):
        if isinstance(c, dict):
            if set(c) == {"__card__"}:
                return {"entity": self.ptr(c["__card__"])} if c["__card__"] is not None else None
            if "__fn__" in c:
                return {"fn": c["__fn__"]}  # what it is, never what it closes over
            return {k: self._swap(v) for k, v in c.items()}
        if isinstance(c, list):
            return [self._swap(v) for v in c]
        return c


def _zone_code(zone, viewer: int):
    kind, key = zone
    if kind in ("attached", "under"):
        return (kind, None)
    return (kind, _side(key, viewer))


def _position(game, card: Card, viewer: int, ptrs: _Pointers):
    for pid, p in game.players.items():
        if card in p.play_area.creatures:
            i = p.play_area.creatures.index(card)
            n = len(p.play_area.creatures)
            flank = "left" if i == 0 else ("right" if i == n - 1 else "center")
            return {"zone": ("battleline", _side(pid, viewer)), "index": i, "of": n,
                    "flank": flank if n > 1 else "both"}
        if card in p.play_area.artifacts:
            return {"zone": ("artifacts", _side(pid, viewer)), "index": p.play_area.artifacts.index(card)}
    host = getattr(card.type_object, "host", None)
    if host is not None:
        return {"zone": ("attached", None), "host": ptrs.ptr(host.instance_id)}
    for c in game._cards_by_id.values():
        if card in c.under_cards:
            return {"zone": ("under", None), "host": ptrs.ptr(c.instance_id), "index": c.under_cards.index(card)}
    return None


def _card_state(card: Card, ptrs: _Pointers) -> dict:
    out = {}
    for k in Card.__slots__:
        if k in _CARD_SKIPPED:
            continue
        out[k] = ptrs.value(getattr(card, k))
    out["extra_triggers"] = sorted((ev, len(fns)) for ev, fns in (card.extra_triggers or {}).items() if fns)
    to = card.type_object
    for k in getattr(type(to), "__slots__", ()):
        if k in _TYPE_SKIPPED:
            continue
        out["type." + k] = ptrs.value(getattr(to, k))
    out["under"] = len(card.under_cards)
    return out


def _entities(game, viewer: int, ptrs: _Pointers) -> list:
    from .determinize import chance_filter
    from .knowledge import _card_knowledge, second_order_tracker

    rows = []
    turn = game.turn_number
    t, s, f = ptrs.tracker, second_order_tracker(game, viewer), chance_filter(game, viewer)
    for iid in ptrs.order:
        card = game._cards_by_id[iid]
        k = _card_knowledge(t, s, f, viewer, iid, turn)
        k["zones"] = tuple(_zone_code(z, viewer) for z in k["zones"])
        if k.get("last_seen") is not None:
            k["last_seen"] = _zone_code(k["last_seen"], viewer)
        if k.get("opponent_knows") is not None:
            ok = dict(k["opponent_knows"])
            ok["zones"] = tuple(_zone_code(z, viewer) for z in ok["zones"])
            k["opponent_knows"] = ok
        row = {"card": card.name, "owner": _side(card.owner, viewer), "knowledge": k}
        if ptrs.visible(iid):
            row["state"] = _card_state(card, ptrs)
            row["position"] = _position(game, card, viewer, ptrs)
            row["controller"] = _side(card.controller, viewer)
        rows.append(row)
    return rows


_PLAYER_ZONES = ("deck", "hand", "discard", "archive", "purged", "play_area", "all_cards")


def _players(game, viewer: int, ptrs: _Pointers) -> list:
    out = []
    for pid in (viewer, 3 - viewer):
        p = game.players[pid]
        d = {k: ptrs.value(v) for k, v in sorted(vars(p).items()) if k not in _PLAYER_ZONES and k != "id"}
        d["hand_revealed_to"] = sorted(_side(x, viewer) for x in p.hand_revealed_to)
        d["sizes"] = {z: len(getattr(p, z)) for z in ("deck", "hand", "discard", "archive", "purged")}
        d["sizes"]["creatures"] = len(p.play_area.creatures)
        d["sizes"]["artifacts"] = len(p.play_area.artifacts)
        out.append(d)
    return out


def _effect_tokens(game, viewer: int, ptrs: _Pointers) -> list:
    ae = game.active_effects
    tokens = []

    def src(e):
        return ptrs.ptr(e.source_card.instance_id) if e.source_card is not None else None

    def remaining(e):
        r = getattr(e, "remaining_duration", INFINITE)
        return None if r == INFINITE else r

    for e in ae.duration_effects:
        value = e.value(game) if callable(e.value) else e.value
        tokens.append({"kind": "duration", "name": e.variable, "op": e.op, "value": ptrs.value(value),
                       "remaining": remaining(e), "active": bool(e.is_active(game)),
                       "conditional": e.conditional is not None, "affected": _side(e.player_affected, viewer),
                       "controller": _side(e.controller, viewer), "source": src(e)})
    for e in ae.trigger_effects:
        tokens.append({"kind": "trigger", "name": e.event, "remaining": remaining(e),
                       "controller": _side(e.controller, viewer), "source": src(e),
                       "handler": getattr(e.handler, "__qualname__", None)})
    for e in ae.instead_effects:
        tokens.append({"kind": "instead", "name": e.kind, "controller": _side(e.controller, viewer), "source": src(e),
                       "handler": getattr(e.handler, "__qualname__", None)})
    for e in ae.modifier_effects:
        tokens.append({"kind": "modifier", "name": e.kind, "controller": _side(e.controller, viewer), "source": src(e),
                       "handler": getattr(e.handler, "__qualname__", None)})
    return tokens


def _resolution(game, viewer: int, ptrs: _Pointers) -> list:
    if game.is_over or game.execution != "compiled":
        return []
    from .resolution import resolution_view

    frames = resolution_view(game, viewer)
    out = []
    for f in frames:
        g = dict(f)
        out.append(_res_swap(g, ptrs, viewer))
    return out


def _res_swap(x, ptrs: _Pointers, viewer: int):
    """The resolution view's ("card", iid) values as entity pointers."""
    if isinstance(x, tuple) and len(x) == 2 and x[0] == "card" and isinstance(x[1], int):
        return ("entity", ptrs.ptr(x[1]))
    if isinstance(x, dict):
        return {k: _res_swap(v, ptrs, viewer) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_res_swap(v, ptrs, viewer) for v in x)
    if isinstance(x, Card):
        return ("entity", ptrs.ptr(x.instance_id))
    return x


def _match(match, viewer: int) -> list:
    if match is None:
        return []
    out = []
    seats0 = match.games[0].seat_decks if match.games else None
    for g in match.games:
        out.append({"winner": _side(g.winner, viewer), "turns": g.turns, "reason": g.reason,
                    "keys": sorted((_side(p, viewer), k) for p, k in g.final_keys.items()),
                    "chains": sorted((_side(p, viewer), c) for p, c in g.final_chains.items()),
                    "first": _side(g.config.first_player, viewer),
                    "swapped": g.seat_decks != seats0})
    return out


def build_infoset_v2(game, viewer: int, *, match=None) -> dict:
    ptrs = _Pointers(game, viewer)
    max_turns = game.config.max_turns
    return {
        "version": FEATURE_STATE_VERSION,
        "turn": {"number": game.turn_number, "active": _side(game.active_player_id, viewer),
                 "first": _side(game._first_player, viewer), "max_turns": max_turns,
                 "remaining": None if max_turns is None else max(0, max_turns - game.turn_number),
                 "elusive_suppressed": game._elusive_suppressed, "is_over": game.is_over},
        "entities": _entities(game, viewer, ptrs),
        "players": _players(game, viewer, ptrs),
        "effects": _effect_tokens(game, viewer, ptrs),
        "cleanups": [(op, ptrs.ptr(iid) if iid in ptrs.index else iid) for op, iid in game._end_of_turn_cleanups],
        "temp_control": sorted(
            [(ptrs.ptr(c.instance_id), _side(orig, viewer), ptrs.ptr(src) if src in ptrs.index else src)
             for src, entries in game._temp_control.items() for c, orig in entries], key=repr),
        "redirected_hits": ptrs.value(game._redirected_hits),
        "resolution": _resolution(game, viewer, ptrs),
        "match": _match(match, viewer),
    }


def infoset_v2_bytes(info: dict) -> bytes:
    """Canonical bytes of an extract (the invariance tests compare these)."""
    return json.dumps(info, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")

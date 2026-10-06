"""The per-viewer projection (Agent Observation Plan, Milestone O1).

`Game.projected(viewer)` is an append-only stream of what `viewer` saw
happen, built incrementally (behind a cursor) from the privileged sources:
the zone journal (keyforge/journal.py), the log (redacted by each event
kind's schema, keyforge/log.py) and the journal's decision records. Their
order is the game's: each journal entry and decision record notes how many
log events came before it.

Items are plain tuples, tagged by their first element, each ending with a
visibility code (`PUBLIC`, `PRIVATE`, `REVEALED`):

    ("zone", seq, turn, iid, owner, controller, from_zone, to_zone, op, deck_epoch, position, cause, code)
    ("reveal", seq, iid, zone, code)          # one card of a revealed hand
    ("log", index, kind, data, code)
    ("decision", index, player, kind, intent, source, affects, optional, chosen, offered, peek, code)

**The visibility rule** (`visible_zone`): the fact of every entry is
public -- op, zones, owner, count. A card's identity (its `iid`; decklists
are public, so an instance id is an identity) is shown to a viewer who can
see into its source zone or its destination, or when the entry reveals it:
- hand: its owner, and the player it is revealed to (`reveal_hand`, until
  `unreveal_hand`); archive: its owner; deck: nobody; under a card: the
  card's owner; discard, purged, the board, attached, limbo: both.
Otherwise the item carries `iid` None.

**Codes**, from the viewer's side: `PUBLIC` -- both players get this
content; `PRIVATE` -- it shows the viewer a card the other player doesn't
get; `REVEALED` -- it shows both players a card of the viewer's that was
hidden from the other one (played from hand, revealed).

**Decisions.** Each chosen option is (verb, iid, zone) for a card or an
action on one -- `iid` None when the other player couldn't see that zone
at that moment, for them -- else (verb, payload). `offered` is the whole
option set when every option was public, else None (so the size of a
choice from a hidden hand is not shown). `peek` -- to the chooser only --
is each offered card from a zone hidden from them (a search of their own
deck).

Viewer None is the privileged projection: every identity, code `PUBLIC`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .journal import CARD_VERBS, LIMBO, PUBLIC_KINDS
from .state_hash import canonicalize

PUBLIC = "public"
PRIVATE = "private"
REVEALED = "revealed"

_NOTES_SHOWN_TO_BOTH = frozenset({"reveal", "search_reveal"})


class Projector:
    """One viewer's projection of one game, advanced by `update()`."""

    def __init__(self, viewer: Optional[int], cut=None):
        self.viewer = viewer
        self.other = None if viewer is None else 3 - viewer
        # Inside a determinized world (keyforge/determinize.py, O3 "drop"):
        # (the world's viewer, entries, log events, decisions) at its start.
        # Before that, this viewer is shown only what the world's viewer saw
        # too: their own private past belongs to the true game.
        self.cut = cut
        self.items: List[tuple] = []
        self.i = 0  # next journal entry
        self.k = 0  # next log event
        self.d = 0  # next decision record
        self.where: Dict[int, tuple] = {}
        self.hands: Dict[int, Set[int]] = {1: set(), 2: set()}
        self.revealed: Dict[int, Set[int]] = {1: set(), 2: set()}

    def copy(self) -> "Projector":
        new = Projector.__new__(Projector)
        new.viewer, new.other, new.cut = self.viewer, self.other, self.cut
        new.items = list(self.items)
        new.i, new.k, new.d = self.i, self.k, self.d
        new.where = dict(self.where)
        new.hands = {p: set(s) for p, s in self.hands.items()}
        new.revealed = {p: set(s) for p, s in self.revealed.items()}
        return new

    # ------------------------------------------------------ visibility ----

    def visible_zone(self, zone, pid, owner) -> bool:
        kind = zone[0]
        if kind in PUBLIC_KINDS:
            return True
        if kind == "hand":
            return zone[1] == pid or pid in self.revealed[zone[1]]
        if kind == "archive":
            return zone[1] == pid
        if kind == "under":
            return owner == pid
        return False  # deck

    def shown(self, e, pid) -> bool:
        if pid is None:
            return True
        op = e[7]
        if op in _NOTES_SHOWN_TO_BOTH:
            return True
        return self.visible_zone(e[5], pid, e[3]) or self.visible_zone(e[6], pid, e[3])

    def code(self, shown_me: bool, shown_them: bool, hidden_from_them_before: bool) -> str:
        if not shown_me or self.viewer is None:
            return PUBLIC
        if not shown_them:
            return PRIVATE
        return REVEALED if hidden_from_them_before else PUBLIC

    # ---------------------------------------------------------- update ----

    def update(self, journal, log_events) -> List[tuple]:
        records = getattr(journal, "records", None)
        entries, marks = journal.entries, journal.marks
        decisions = records() if records is not None else journal.decisions
        ne, nl, nd = len(entries), len(log_events), len(decisions)
        while True:
            if self.d < nd:
                rec = decisions[self.d]
                if rec[0] <= self.i and rec[1] <= self.k:
                    self._decision(rec)
                    self.d += 1
                    continue
            if self.k < nl and (self.i >= ne or self.k < marks[self.i]):
                self._log(log_events[self.k])
                self.k += 1
                continue
            if self.i < ne:
                self._entry(entries[self.i])
                self.i += 1
                continue
            break
        return self.items

    def _entry(self, e) -> None:
        v, o = self.viewer, self.other
        iid, owner, frm, to, op = e[2], e[3], e[5], e[6], e[7]
        if op == "reveal_hand" or op == "unreveal_hand":
            pid, to_whom = frm[1], e[9]
            if op == "reveal_hand":
                self.revealed[pid].add(to_whom)
            else:
                self.revealed[pid].discard(to_whom)
            self.items.append(("zone",) + e[:2] + (None,) + e[3:] + (PUBLIC,))
            if op == "reveal_hand" and (v is None or v == to_whom or v == pid):
                code = PUBLIC if v is None or v == to_whom else REVEALED
                for c in sorted(self.hands[pid]):
                    self.items.append(("reveal", e[0], c, frm, code))
            return
        if iid is not None and frm != to:
            if frm[0] == "hand":
                self.hands[frm[1]].discard(iid)
            if to[0] == "hand":
                self.hands[to[1]].add(iid)
            self.where[iid] = to
        if iid is None:
            self.items.append(("zone",) + e + (PUBLIC,))
            return
        me = self.shown(e, v)
        cut = self.cut
        if cut is not None and e[0] < cut[1]:
            me = me and self.shown(e, cut[0])
        them = True if v is None else self.shown(e, o)
        before = v is not None and owner == v and not self.visible_zone(frm, o, owner)
        code = self.code(me, them, before)
        shown_iid = iid if me else None
        cause = e[10]
        self.items.append(("zone", e[0], e[1], shown_iid) + e[3:10] + (cause, code))

    def _log(self, event) -> None:
        v = self.viewer
        idx = self.k
        if v is None:
            self.items.append(("log", idx, event.kind, canonicalize(event.data), PUBLIC))
            return
        if v not in event.visible_to:
            return
        cut = self.cut
        if cut is not None and idx < cut[2]:
            if cut[0] not in event.visible_to:
                return
            if event.private:
                data = {k: x for k, x in event.data.items()
                        if k not in event.private or (v in event.private[k] and cut[0] in event.private[k])}
                self.items.append(("log", idx, event.kind, canonicalize(data), PUBLIC))
                return
        shown = event.redacted_for(v) if event.private else event
        o = self.other
        if o not in event.visible_to:
            code = PRIVATE
        elif event.private and any(v in who and o not in who for who in event.private.values()):
            code = PRIVATE
        else:
            code = PUBLIC
        self.items.append(("log", idx, event.kind, canonicalize(shown.data), code))

    def _decision(self, rec) -> None:
        v, o = self.viewer, self.other
        _, _, player, kind, intent, source, affects, optional, chosen, offered = rec
        idx = self.d
        chosen = tuple(self._placed(f) for f in chosen)
        offered = tuple(self._placed(f) for f in offered)
        public = all(self._public(f) for f in offered)
        offered_shown = offered if public else None
        peek = () if public else tuple(
            (f[1], f[2]) for f in offered if len(f) == 3 and not self.visible_zone(f[2], player, player))
        if v is None:
            self.items.append(("decision", idx, player, kind, intent, source, affects, optional,
                               chosen, offered_shown, peek, PUBLIC))
            return
        cut = self.cut
        if cut is not None and idx < cut[3] and v == player and v != cut[0]:
            # before the world: their choice as the world's viewer saw it
            self.items.append(("decision", idx, player, kind, intent, source, affects, optional,
                               tuple(self._option(f, cut[0]) for f in chosen), offered_shown, (), PUBLIC))
            return
        if v == player:  # the chooser saw every option they were offered
            theirs = tuple(self._option(f, o) for f in chosen)
            code = PRIVATE if (chosen != theirs or peek) else PUBLIC
            self.items.append(("decision", idx, player, kind, intent, source, affects, optional,
                               chosen, offered_shown, peek, code))
        else:
            self.items.append(("decision", idx, player, kind, intent, source, affects, optional,
                               tuple(self._option(f, v) for f in chosen), offered_shown, (), PUBLIC))

    def _placed(self, f):
        """An option form with its card's zone at this moment: (verb, iid,
        zone)."""
        if f[0] in CARD_VERBS:
            return (f[0], f[1], self.where.get(f[1], LIMBO))
        if f[0] == "seq":
            return ("seq", tuple(self._placed(x) for x in f[1]))
        return f

    def _public(self, f) -> bool:
        if len(f) == 3:
            z = f[2]
            return z[0] in PUBLIC_KINDS or (self.visible_zone(z, 1, None) and self.visible_zone(z, 2, None))
        if f[0] == "seq":
            return all(self._public(x) for x in f[1])
        return True

    def _option(self, f, pid):
        if len(f) == 3:
            verb, iid, zone = f
            if self.visible_zone(zone, pid, pid):  # (under a card: only its owner chooses it)
                return f
            return (verb, None, zone)
        if f[0] == "seq":
            return ("seq", tuple(self._option(x, pid) for x in f[1]))
        return f


def projected(game, viewer: Optional[int]) -> List[tuple]:
    """`viewer`'s projection so far (append-only; read it, don't change it).
    The projector lives on the game, so each call only reads what is new."""
    cache = game.__dict__.get("_projectors")
    if cache is None:
        cache = game.__dict__["_projectors"] = {}
    p = cache.get(viewer)
    if p is None:
        cut = game.__dict__.get("_world_cut")
        p = cache[viewer] = Projector(viewer, cut if cut is not None and cut[0] != viewer else None)
    return p.update(game.journal, game.log.events)

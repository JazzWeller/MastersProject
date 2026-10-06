"""The knowledge tracker (Agent Observation Plan, Milestone O2).

One `Tracker` per viewer, fed only by that viewer's projection
(keyforge/projection.py) and the public decklists, at O(cards in a zone)
per event. For every card it keeps the set of zones it may be in (its
*mask*), so its knowledge is one of:

- **exact** -- a single zone (a card in play, in a discard or purged; the
  viewer's own hand and archive; a card known to be in the opponent's hand;
  a card known to be in some deck);
- **constrained** -- several hidden zones of one side, for example
  {hand, archive} after "a card left their hand face down" while this one
  was known to be in that hand.

Also kept:
- each card's **provenance group**: the public event at which it entered its
  side's hidden pool (the deal, a reshuffled discard, a card shuffled in or
  put on a deck), the entry's `seq` (0 for the deal);
- **deck positions**: per deck, the known cards from the top and from the
  bottom, valid until the deck is shuffled; draws from the top consume them;
- the **public count** of every zone;
- **memory of reveals**: the turn each card was last seen and where, and how
  it last left the viewer's sight (the op);
- **second-order knowledge** (`second_order`): the same tracker run over the
  part of the viewer's projection the opponent sees too, which is what the
  opponent knows about the viewer's own cards.

Soundness: every card's true zone is always in its mask, and every known
deck position is true (tools/o2_acceptance.py checks it at every decision).
Precision comes from three rules: a move whose card is shown makes it exact;
a draw from a deck whose top is known moves that card; and the pigeonhole
rule -- a zone holding as many cards as there are cards that may be in it
holds exactly those (and an empty zone holds none).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .journal import LIMBO, PUBLIC_KINDS

Zone = Tuple[str, Optional[int]]

EXACT = "exact"
CONSTRAINED = "constrained"

_NOTE_REVEALS = frozenset({"reveal", "search_reveal"})


class Tracker:
    __slots__ = ("viewer", "mask", "poss", "counts", "top", "bottom", "revealed", "group", "seen_turn",
                 "seen_zone", "exit_op", "_pending_group", "cursor", "_dealt", "owner")

    def __init__(self, viewer: int, decklists: Dict[int, Iterable[int]]):
        self.viewer = viewer
        self.mask: Dict[int, FrozenSet[Zone]] = {}
        self.poss: Dict[Zone, Set[int]] = {}
        self.counts: Dict[Zone, int] = {}
        self.top: Dict[int, List[Optional[int]]] = {1: [], 2: []}
        self.bottom: Dict[int, List[Optional[int]]] = {1: [], 2: []}
        self.revealed: Dict[int, Set[int]] = {1: set(), 2: set()}
        self.group: Dict[int, int] = {}
        self.seen_turn: Dict[int, int] = {}
        self.seen_zone: Dict[int, Zone] = {}
        self.exit_op: Dict[int, str] = {}
        self._pending_group: Dict[int, Set[int]] = {1: set(), 2: set()}
        self._dealt: Set[int] = set()
        self.owner: Dict[int, int] = {}
        self.cursor = 0
        for pid, iids in decklists.items():
            setup = ("setup", pid)
            self.counts[setup] = 0
            for iid in iids:
                self.owner[iid] = pid
                self._place(iid, frozenset((setup,)))
                self.counts[setup] += 1
                self.group[iid] = 0

    # ----------------------------------------------------------- queries ----

    def kind(self, iid: int) -> str:
        return EXACT if len(self.mask[iid]) == 1 else CONSTRAINED

    def zone(self, iid: int) -> Optional[Zone]:
        m = self.mask[iid]
        return next(iter(m)) if len(m) == 1 else None

    def deck_position(self, iid: int) -> Optional[int]:
        """0, 1, ... from the top, -1, -2, ... from the bottom, if known."""
        for pid in (1, 2):
            t = self.top[pid]
            if iid in t:
                return t.index(iid)
            b = self.bottom[pid]
            if iid in b:
                return b.index(iid) - len(b)
        return None

    def export(self, iid: int, turn: int) -> dict:
        m = self.mask[iid]
        seen = self.seen_turn.get(iid)
        return {
            "kind": EXACT if len(m) == 1 else CONSTRAINED,
            "zones": tuple(sorted(m, key=repr)),
            "deck_position": self.deck_position(iid),
            "group": self.group.get(iid),
            "turns_since_seen": None if seen is None else turn - seen,
            "last_seen": self.seen_zone.get(iid),
            "exit_op": self.exit_op.get(iid),
        }

    # ------------------------------------------------------------ update ----

    def update(self, items) -> None:
        """Feeds the items of the viewer's projection not yet seen."""
        for i in range(self.cursor, len(items)):
            self.feed(items[i])
        self.cursor = len(items)

    def feed(self, item) -> None:
        tag = item[0]
        if tag == "zone":
            self._zone(item)
        elif tag == "reveal":
            _, seq, iid, zone, code = item
            self._known(iid, zone)
            self._seen(iid, zone, None)
        elif tag == "decision":
            for f in item[8]:
                if len(f) == 3 and f[1] is not None:
                    self._known(f[1], f[2])
            if item[9] is not None:
                for f in item[9]:
                    if len(f) == 3 and f[1] is not None:
                        self._known(f[1], f[2])
            for iid, zone in item[10]:
                self._known(iid, zone)

    # ------------------------------------------------------------ helpers ----

    def visible(self, zone: Zone, owner: Optional[int] = None) -> bool:
        kind = zone[0]
        if kind in PUBLIC_KINDS:
            return True
        v = self.viewer
        if kind == "hand":
            return zone[1] == v or v in self.revealed[zone[1]]
        if kind == "archive":
            return zone[1] == v
        if kind == "under":
            return owner == v
        return False

    def _place(self, iid: int, mask: FrozenSet[Zone]) -> None:
        old = self.mask.get(iid)
        if old is not None:
            for z in old:
                if z not in mask:
                    self.poss[z].discard(iid)
        for z in mask:
            s = self.poss.get(z)
            if s is None:
                s = self.poss[z] = set()
            s.add(iid)
        self.mask[iid] = mask

    def _known(self, iid: int, zone: Zone) -> None:
        if self.mask.get(iid) != frozenset((zone,)):
            touched = self.mask.get(iid, frozenset())
            self._place(iid, frozenset((zone,)))
            for z in touched:
                if z != zone:
                    self._pigeonhole(z)
            self._pigeonhole(zone)

    def _seen(self, iid: int, zone: Zone, turn: Optional[int]) -> None:
        if turn is not None:
            self.seen_turn[iid] = turn
        self.seen_zone[iid] = zone

    def _count(self, zone: Zone, d: int) -> None:
        self.counts[zone] = self.counts.get(zone, 0) + d

    def _pigeonhole(self, zone: Zone) -> None:
        """A zone holding as many cards as may be in it holds exactly those;
        an empty zone holds none. Repeats while it settles anything."""
        work = [zone]
        while work:
            z = work.pop()
            cands = self.poss.get(z)
            if not cands:
                continue
            n = self.counts.get(z, 0)
            if n == 0:
                for iid in list(cands):
                    m = self.mask[iid] - {z}
                    if m:
                        self._place(iid, m)
                        work.extend(m)
            elif n == len(cands):
                for iid in list(cands):
                    m = self.mask[iid]
                    if len(m) > 1:
                        self._place(iid, frozenset((z,)))
                        work.extend(x for x in m if x != z)

    def _deck_clear(self, pid: int) -> None:
        self.top[pid] = []
        self.bottom[pid] = []

    def _zone(self, item) -> None:
        _, seq, turn, iid, owner, controller, frm, to, op, epoch, position, cause, code = item
        if op in ("reveal_hand", "unreveal_hand"):
            if op == "reveal_hand":
                self.revealed[frm[1]].add(position)
            else:
                self.revealed[frm[1]].discard(position)
            return
        if op == "shuffle":
            pid = frm[1]
            self._deck_clear(pid)
            for x in self._pending_group[pid]:
                self.group[x] = seq
            self._pending_group[pid].clear()
            return
        if frm == to:  # a note
            if iid is not None:
                if frm == LIMBO:
                    # about a card in mid-move: its move's entry (completed in
                    # place) already says where it went
                    self._seen(iid, frm, turn)
                    return
                self._known(iid, frm)
                self._seen(iid, frm, turn)
                if op in _NOTE_REVEALS and frm[0] == "deck" and position == "top":
                    t = self.top[frm[1]]
                    if t:
                        t[0] = iid
                    else:
                        t.append(iid)
            return
        if frm[0] == "setup":
            self._deal(iid, frm, to)
            return
        # a move
        self._count(frm, -1)
        self._count(to, 1)
        moved = iid
        if frm[0] == "deck":
            pid = frm[1]
            t, b = self.top[pid], self.bottom[pid]
            if position == "top":
                if t:
                    x = t.pop(0)
                    if moved is None:
                        moved = x
                elif b and self.counts[frm] + 1 == len(b):
                    x = b.pop(0)
                    if moved is None:
                        moved = x
            elif position == "bottom":
                if b:
                    x = b.pop()
                    if moved is None:
                        moved = x
                elif t and self.counts[frm] + 1 == len(t):
                    x = t.pop()
                    if moved is None:
                        moved = x
            else:
                if moved is None:
                    self._deck_clear(pid)
                else:
                    if moved in t:
                        t.remove(moved)
                    if moved in b:
                        b.remove(moved)
        if moved is not None:
            old = self.mask.get(moved, frozenset())
            self._place(moved, frozenset((to,)))
            for z in old:
                if z != to:
                    self._pigeonhole(z)
            self._pigeonhole(frm)
            self._pigeonhole(to)
            if iid is not None:
                vis_from, vis_to = self.visible(frm, owner), self.visible(to, owner)
                if vis_from or vis_to:
                    self._seen(moved, to if vis_to else frm, turn)
                if vis_from and not vis_to:
                    self.exit_op[moved] = op
        else:
            # some card that may have been in `frm` went to `to` -- not one
            # known to sit at the other end of a deck the move took from
            # (the deck still holds at least as many cards as that end has)
            exclude = ()
            if frm[0] == "deck" and position in ("top", "bottom"):
                pid = frm[1]
                other = self.bottom[pid] if position == "top" else self.top[pid]
                if other and self.counts[frm] >= len(other):
                    exclude = set(x for x in other if x is not None)
            for x in list(self.poss.get(frm, ())):
                m = self.mask[x]
                if to not in m and x not in exclude:
                    self._place(x, m | {to})
            self._pigeonhole(frm)
            self._pigeonhole(to)
        self._enter_deck(moved, to, seq, position)

    def _enter_deck(self, moved: Optional[int], to: Zone, seq: int, position) -> None:
        if to[0] != "deck":
            if moved is not None and to[0] in ("hand", "archive"):
                self.group[moved] = seq
            return
        pid = to[1]
        if position == "top":
            self.top[pid].insert(0, moved)
        elif position == "bottom":
            self.bottom[pid].append(moved)
        else:
            # somewhere unknown (then, as a rule, shuffled): what is known of
            # the ends no longer is
            self._deck_clear(pid)
        if moved is not None:
            if position in ("top", "bottom"):
                self.group[moved] = seq
            else:
                self._pending_group[pid].add(moved)

    def _deal(self, iid: Optional[int], frm: Zone, to: Zone) -> None:
        """One card dealt into its owner's deck. The first such entry puts
        the whole decklist there (the deal is one public event; nothing is
        asked of anyone before it ends)."""
        pid = frm[1]
        self._count(frm, -1)
        self._count(to, 1)
        if pid not in self._dealt:
            self._dealt.add(pid)
            for x, o in self.owner.items():
                if o == pid:
                    self._place(x, frozenset((to,)))

    def copy(self) -> "Tracker":
        new = Tracker.__new__(Tracker)
        new.viewer = self.viewer
        new.mask = dict(self.mask)
        new.poss = {z: set(s) for z, s in self.poss.items()}
        new.counts = dict(self.counts)
        new.top = {p: list(t) for p, t in self.top.items()}
        new.bottom = {p: list(b) for p, b in self.bottom.items()}
        new.revealed = {p: set(s) for p, s in self.revealed.items()}
        new.group = dict(self.group)
        new.seen_turn = dict(self.seen_turn)
        new.seen_zone = dict(self.seen_zone)
        new.exit_op = dict(self.exit_op)
        new._pending_group = {p: set(s) for p, s in self._pending_group.items()}
        new._dealt = set(self._dealt)
        new.owner = self.owner
        new.cursor = self.cursor
        return new


# ------------------------------------------------------- per game ----


def _decklists(game) -> Dict[int, List[int]]:
    return {pid: [c.instance_id for c in p.all_cards] for pid, p in game.players.items()}


def tracker_for(game, viewer: int) -> Tracker:
    """`viewer`'s tracker, brought up to date with their projection. Kept on
    the game (and copied with it), so each call reads only what is new."""
    cache = game.__dict__.get("_trackers")
    if cache is None:
        cache = game.__dict__["_trackers"] = {}
    t = cache.get(viewer)
    if t is None:
        t = cache[viewer] = Tracker(viewer, _decklists(game))
    t.update(game.projected(viewer))
    return t


class SecondOrder:
    """What the opponent knows about `viewer`'s cards: a tracker for the
    opponent, fed the part of `viewer`'s projection the opponent sees too --
    every event the opponent sees about `viewer`'s cards is one `viewer`
    sees, so this needs nothing `viewer` doesn't have."""

    __slots__ = ("viewer", "tracker", "revealed", "cursor")

    def __init__(self, viewer: int, decklists):
        self.viewer = viewer
        self.tracker = Tracker(3 - viewer, decklists)
        self.revealed: Dict[int, Set[int]] = {1: set(), 2: set()}
        self.cursor = 0

    def _sees(self, zone: Zone, owner) -> bool:
        o = 3 - self.viewer
        kind = zone[0]
        if kind in PUBLIC_KINDS:
            return True
        if kind == "hand":
            return zone[1] == o or o in self.revealed[zone[1]]
        if kind == "archive":
            return zone[1] == o
        if kind == "under":
            return owner == o
        return False

    def _as_theirs(self, item):
        tag = item[0]
        if tag == "zone":
            iid, frm, to, op = item[3], item[6], item[7], item[8]
            if op == "reveal_hand":
                self.revealed[frm[1]].add(item[10])
            elif op == "unreveal_hand":
                self.revealed[frm[1]].discard(item[10])
            if iid is None or op in _NOTE_REVEALS or self._sees(frm, item[4]) or self._sees(to, item[4]):
                return item
            return item[:3] + (None,) + item[4:]
        if tag == "decision" and item[2] == self.viewer:
            chosen = tuple((f[0], f[1] if self._sees(f[2], None) else None, f[2]) if len(f) == 3 else f
                           for f in item[8])
            return item[:8] + (chosen, item[9], ()) + item[11:]
        return item

    def update(self, items) -> Tracker:
        t = self.tracker
        for i in range(self.cursor, len(items)):
            t.feed(self._as_theirs(items[i]))
        self.cursor = len(items)
        return t

    def copy(self) -> "SecondOrder":
        new = SecondOrder.__new__(SecondOrder)
        new.viewer = self.viewer
        new.tracker = self.tracker.copy()
        new.revealed = {p: set(s) for p, s in self.revealed.items()}
        new.cursor = self.cursor
        return new


def second_order_tracker(game, viewer: int) -> Tracker:
    """What `viewer`'s opponent knows -- read it for `viewer`'s own cards."""
    cache = game.__dict__.get("_trackers2")
    if cache is None:
        cache = game.__dict__["_trackers2"] = {}
    s = cache.get(viewer)
    if s is None:
        s = cache[viewer] = SecondOrder(viewer, _decklists(game))
    return s.update(game.projected(viewer))


def card_knowledge(game, viewer: int, iid: int) -> dict:
    """Everything `viewer` knows about one card, as the plan's O2 exports
    it: the tracker's kind, zones, deck position, provenance group, turns
    since last seen and how it last left sight; for `viewer`'s own cards,
    what the opponent knows (`opponent_knows`: their kind and zones); and
    for the opponent's hidden cards, chance_exact's priors (O3) P(hand),
    P(archive), P(deck), P(next draw)."""
    from .determinize import chance_filter

    return _card_knowledge(tracker_for(game, viewer), second_order_tracker(game, viewer), chance_filter(game, viewer),
                           viewer, iid, game.turn_number)


def _card_knowledge(t, s, f, viewer: int, iid: int, turn: int) -> dict:
    """`card_knowledge` given the viewer's tracker, second-order tracker
    and chance filter, already up to date."""
    out = t.export(iid, turn)
    if t.owner[iid] == viewer:
        m = s.mask[iid]
        out["opponent_knows"] = {"kind": EXACT if len(m) == 1 else CONSTRAINED, "zones": tuple(sorted(m, key=repr))}
        out["priors"] = None
    else:
        pz = f.p_zone(iid)
        out["priors"] = None if pz is None else {
            "hand": pz[0], "archive": pz[1], "deck": pz[2], "next_draw": f.p_next_draw(iid)}
        out["opponent_knows"] = None
    return out

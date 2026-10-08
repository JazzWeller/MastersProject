"""History encoding (Agent Observation Plan, Milestone O6): a viewer's
projected stream (`Game.projected`, O1) as compact integer rows, in three
representations, all built (O10 picks the default):

1. **Full events** (`HistoryEncoder.rows`): one row per projected item --
   a zone move, a revealed card, a log event, a decision -- uncapped, with
   its pointers as (row, role, entity) rows beside it. A redacted identity
   is `HIDDEN_MINE` / `HIDDEN_THEIRS` with its zone.
2. **Turn tokens** (`turn_tokens`): one per half-turn -- the actor, the
   house chosen, the cards played, discarded and archived (pointer sets),
   reaps, fights and uses, Æmber gained / stolen / captured / lost, keys
   forged, cards drawn, hand size at its start and end, the archive taken
   or declined, chains shed.
3. **Summaries** (`summaries`): per entity, times played / reaped / fought
   / used, turns since last seen in public, last public zone, how it left
   view, ever in a revealed hand, the turn it was drawn (own cards); and
   globally each side's house choices (counts, the last three, turns since
   each), hand sizes at the end of their last three turns, mulligans and
   declined archive pickups.

**Incremental and shared.** `HistoryEncoder(game, viewer)` consumes the
projection behind a cursor. `freeze()` marks a prefix (a real decision):
every simulation of a search shares it, and a world (`copy()`) appends only
its own suffix. The prefix carries a rolling blake2b digest; cache keys use
the digest, never the bytes. `to_bytes` / `from_bytes` round-trip the
rows (the shard form, O9).
"""

from __future__ import annotations

import hashlib
import struct
from array import array
from typing import Dict, List, Optional, Tuple

from keyforge.enums import DecisionIntent, DecisionKind, House
from keyforge.infoset import HOUSE_INDEX

from . import spec_v2 as S
from .vocab import load_all

_V = load_all()
STREAMS = ("zone", "reveal", "log", "decision")
VIS = {"public": 1, "private": 2, "revealed": 3}
ROLES = ("card", "cause", "source", "chosen", "offered", "peek", "iid", "iids", "attacker_iid", "target_iid",
         "under_iid", "to_iid", "other_iid")
_ROLE = {r: i + 1 for i, r in enumerate(ROLES)}
NUMERIC = ("amount", "n", "absorbed", "cost", "keys", "total", "fewer", "aember", "value")
BOOLS = ("from_deck_top", "took", "permanent", "reverted", "found", "as_if_yours", "value")
POSITIONS = {"top": 1, "bottom": 2, "left": 3, "right": 4}
_KINDS = {k.name: i + 1 for i, k in enumerate(DecisionKind)}
_INTENTS = {k.name: i + 1 for i, k in enumerate(DecisionIntent)}

ROW_INTS = ("stream", "kind", "actor", "from_zone", "to_zone", "house", "flank", "vis", "turn", "step", "seq",
            "position", "flags", "intent", "affects", "optional")
ROW_FLOATS = NUMERIC
_RI = {c: i for i, c in enumerate(ROW_INTS)}
_RF = {c: i for i, c in enumerate(ROW_FLOATS)}
N_RI, N_RF = len(ROW_INTS), len(ROW_FLOATS)
_AFFECTS = {}


class HistoryRows:
    """Bare event rows: a prefix or a suffix of a `HistoryEncoder`, as the
    inference protocol carries them (O8). `first` is the row number of the
    first row; pointer triples keep their absolute row numbers."""

    __slots__ = ("n", "ints", "floats", "pointers", "first")

    def __init__(self, n: int, ints: array, floats: array, pointers: array, first: int = 0):
        self.n, self.ints, self.floats, self.pointers, self.first = n, ints, floats, pointers, first

    def to_bytes(self) -> bytes:
        return (struct.pack("<qqq", self.n, len(self.pointers), self.first) + self.ints.tobytes()
                + self.floats.tobytes() + self.pointers.tobytes())

    @staticmethod
    def from_bytes(data: bytes) -> "HistoryRows":
        n, n_ptr, first = struct.unpack_from("<qqq", data)
        at = 24
        ints, floats, pointers = array("q"), array("f"), array("q")
        ints.frombytes(data[at:at + 8 * n * N_RI])
        at += 8 * n * N_RI
        floats.frombytes(data[at:at + 4 * n * N_RF])
        at += 4 * n * N_RF
        pointers.frombytes(data[at:at + 8 * n_ptr])
        return HistoryRows(n, ints, floats, pointers, first)


class HistoryFolded:
    """Turn tokens and summaries as flat arrays, built on the engine side
    (the `turn_tokens` / `summary` architectures' history; in a request the
    workers build it, not the one server thread, O8):
    - `u` turn tokens: `turn_ints` (u x 3: turn offset, actor, house),
      `turn_floats` (u x `TURN_SCALARS`), `turn_sets` ((token, set, entity)
      triples: played / discarded / archived);
    - `summary_entity` (72 x `SUMMARY_ENTITY`, -1 = never), `summary_global`
      (`SUMMARY_GLOBAL`)."""

    __slots__ = ("u", "turn_ints", "turn_floats", "turn_sets", "summary_entity", "summary_global")

    @staticmethod
    def of(h: "HistoryEncoder", turn_now: int) -> "HistoryFolded":
        new = HistoryFolded()
        toks = turn_tokens(h)
        new.u = len(toks)
        ti, tf, ts = array("q"), array("f"), array("q")
        for k, t in enumerate(toks):
            actor = t["actor"]
            ti.extend((turn_now - t["turn"], actor, t["house"]))
            for f in TURN_SCALARS:
                v = t[f]
                if isinstance(v, dict):  # hand sizes: the actor's
                    v = v.get(actor, 0) if actor else 0
                tf.append(float(v))
            for j, f in enumerate(("played", "discarded", "archived")):
                for p in t[f]:
                    if 0 <= p < S.N_ENTITIES:
                        ts.extend((k, j, p))
        new.turn_ints, new.turn_floats, new.turn_sets = ti, tf, ts
        sm = summaries(h, turn_now)
        ent = array("f", [0.0] * (S.N_ENTITIES * len(SUMMARY_ENTITY)))
        for i, e in enumerate(sm["entities"][:S.N_ENTITIES]):
            for j, f in enumerate(SUMMARY_ENTITY):
                v = e[f]
                ent[i * len(SUMMARY_ENTITY) + j] = -1.0 if v is None else float(v)
        glob = array("f")
        for side in (1, 2):
            g = sm["global"][side]
            glob.extend(g["house_counts"])
            glob.extend((list(g["last_three"]) + [0, 0, 0])[:3])
            glob.extend(-1.0 if x is None else float(x) for x in g["since"])
            glob.extend((g["mulligans"], g["declined_archive"]))
            glob.extend((list(g["hand_at_end_of_last_three"]) + [0, 0, 0])[:3])
        new.summary_entity, new.summary_global = ent, glob
        return new

    def __getstate__(self):
        return tuple(getattr(self, k) for k in self.__slots__)

    def __setstate__(self, state):
        for k, v in zip(self.__slots__, state):
            setattr(self, k, v)


class HistoryEncoder:
    __slots__ = ("viewer", "order_index", "owner_of", "ints", "floats", "pointers", "n", "cursor", "_digest",
                 "_prefix", "_turn", "_step", "zone_code_cache", "folds")

    def __init__(self, game, viewer: int):
        me, them = game.players[viewer], game.players[3 - viewer]
        order = [c.instance_id for c in me.all_cards] + [c.instance_id for c in them.all_cards]
        self.viewer = viewer
        self.order_index = {iid: i for i, iid in enumerate(order)}
        self.owner_of = {c.instance_id: c.owner for c in me.all_cards + them.all_cards}
        self.ints = array("q")
        self.floats = array("f")
        self.pointers = array("q")  # (row, role, entity) triples
        self.n = 0
        self.cursor = 0
        # The digest starts from both public decklists: equal rows in games
        # with different cards are different prefixes (a cached prefix's
        # keys and values depend on the cards its rows point at).
        self._digest = hashlib.blake2b(digest_size=16)
        self._digest.update("\x1f".join(c.name for c in me.all_cards + them.all_cards).encode())
        self._prefix = (0, self._digest.copy())
        self._turn = None
        self._step = 0
        self.zone_code_cache: Dict[tuple, int] = {}
        self.folds = None  # turn tokens and summaries, folded as rows come

    # --------------------------------------------------------- pieces ----

    def side(self, pid) -> int:
        return 0 if pid is None else (1 if pid == self.viewer else 2)

    def zone(self, z) -> int:
        if z is None:
            return 0
        code = self.zone_code_cache.get(z)
        if code is None:
            kind, key = z
            if kind in ("attached", "under", "limbo"):
                code = S.zone_code(kind, None)
            else:
                code = S.zone_code(kind, None if key is None else (0 if key == self.viewer else 1))
            self.zone_code_cache[z] = code
        return code

    def ptr(self, iid, owner=None) -> int:
        if iid is None:
            return S.HIDDEN_MINE if owner == self.viewer else S.HIDDEN_THEIRS
        return self.order_index.get(iid, S.NO_POINTER)

    def _emit(self, ints: list, floats: list, ptrs: List[Tuple[int, int]]) -> None:
        row = self.n
        self.ints.extend(ints)
        self.floats.extend(floats)
        for role, p in ptrs:
            self.pointers.extend((row, role, p))
        self._digest.update(struct.pack(f"<{N_RI}q{N_RF}f", *ints, *floats))
        for role, p in ptrs:
            self._digest.update(struct.pack("<3q", row, role, p))
        self.n += 1

    def _base(self, stream: str, turn: int) -> list:
        if turn != self._turn:
            self._turn, self._step = turn, 0
        self._step += 1
        r = [0] * N_RI
        r[_RI["stream"]] = STREAMS.index(stream) + 1
        r[_RI["turn"]] = turn
        r[_RI["step"]] = self._step
        r[_RI["seq"]] = self.n + 1
        return r

    # --------------------------------------------------------- update ----

    def update(self, items, turn_now: int = 0) -> None:
        last_turn = self._turn if self._turn is not None else 0
        for i in range(self.cursor, len(items)):
            it = items[i]
            tag = it[0]
            if tag == "zone":
                _, seq, turn, iid, owner, controller, frm, to, op, epoch, position, cause, code = it
                last_turn = turn
                r = self._base("zone", turn)
                r[_RI["kind"]] = _V["journal_ops"].id(op)
                r[_RI["actor"]] = self.side(controller if controller is not None else owner)
                r[_RI["from_zone"]], r[_RI["to_zone"]] = self.zone(frm), self.zone(to)
                r[_RI["vis"]] = VIS[code]
                if isinstance(position, str):
                    r[_RI["position"]] = POSITIONS.get(position, 0)
                ptrs = []
                if op not in ("shuffle", "reveal_hand", "unreveal_hand"):
                    ptrs.append((_ROLE["card"], self.ptr(iid, owner)))
                if cause is not None:
                    ptrs.append((_ROLE["cause"], self.ptr(cause)))
                self._emit(r, [0.0] * N_RF, ptrs)
            elif tag == "reveal":
                _, seq, iid, zone, code = it
                r = self._base("reveal", last_turn)
                r[_RI["to_zone"]] = self.zone(zone)
                r[_RI["vis"]] = VIS[code]
                self._emit(r, [0.0] * N_RF, [(_ROLE["card"], self.ptr(iid))])
            elif tag == "log":
                _, idx, kind, data, code = it
                turn = data["turn"] if isinstance(data.get("turn"), int) else last_turn
                last_turn = turn
                r = self._base("log", turn)
                r[_RI["kind"]] = _V["log_kinds"].id(kind)
                pid = data.get("player", data.get("controller", data.get("owner")))
                r[_RI["actor"]] = self.side(pid if isinstance(pid, int) else None)
                r[_RI["vis"]] = VIS[code]
                h = data.get("house")
                if isinstance(h, str):
                    r[_RI["house"]] = _house_by_value(h)
                fl = data.get("flank")
                if isinstance(fl, str):
                    r[_RI["flank"]] = POSITIONS.get(fl, 0)
                dest = data.get("destination")
                if isinstance(dest, str) and dest in S.ZONE_KINDS:
                    r[_RI["to_zone"]] = S.zone_code(dest, None)
                flags = 0
                for b, name in enumerate(BOOLS):
                    if data.get(name) is True:
                        flags |= 1 << b
                r[_RI["flags"]] = flags
                f = [0.0] * N_RF
                for name in NUMERIC:
                    v = data.get(name)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        f[_RF[name]] = float(v)
                ptrs = []
                for name in ("iid", "attacker_iid", "target_iid", "under_iid", "to_iid", "other_iid"):
                    v = data.get(name)
                    if isinstance(v, int):
                        ptrs.append((_ROLE[name], self.ptr(v)))
                vs = data.get("iids")
                if isinstance(vs, list):
                    for v in vs:
                        if isinstance(v, int):
                            ptrs.append((_ROLE["iids"], self.ptr(v)))
                self._emit(r, f, ptrs)
            else:  # decision
                _, idx, player, kind, intent, source, affects, optional, chosen, offered, peek, code = it
                r = self._base("decision", last_turn)
                r[_RI["kind"]] = _KINDS[kind]
                r[_RI["intent"]] = _INTENTS.get(intent, 0)
                r[_RI["affects"]] = _affects_id(affects)
                r[_RI["optional"]] = int(bool(optional))
                r[_RI["actor"]] = self.side(player)
                r[_RI["vis"]] = VIS[code]
                f = [0.0] * N_RF
                ptrs = []
                if source is not None:
                    ptrs.append((_ROLE["source"], self.ptr(source)))
                for role, forms in (("chosen", chosen), ("offered", offered or ())):
                    for form in forms:
                        if len(form) == 3:
                            verb, iid, zone = form
                            if iid is not None:
                                ptrs.append((_ROLE[role], self.ptr(iid)))
                            else:  # hidden from this viewer: whose zone it was in
                                ptrs.append((_ROLE[role], S.HIDDEN_MINE if zone[1] == self.viewer else S.HIDDEN_THEIRS))
                        elif role == "chosen":
                            verb, payload = form[0], form[1]
                            if verb == "house":
                                r[_RI["house"]] = _house_by_value(payload)
                            elif verb in ("value", "bool") and isinstance(payload, (int, float)):
                                f[_RF["value"]] = float(payload)
                for iid, zone in peek:
                    ptrs.append((_ROLE["peek"], self.ptr(iid)))
                self._emit(r, f, ptrs)
        self.cursor = len(items)

    # --------------------------------------------------------- prefix ----

    def freeze(self) -> Tuple[int, str]:
        """Marks what has been read so far as the prefix; returns (rows,
        digest)."""
        self._prefix = (self.n, self._digest.copy())
        return self.n, self._digest.hexdigest()

    def prefix_digest(self) -> str:
        return self._prefix[1].hexdigest()

    def digest(self) -> str:
        return self._digest.hexdigest()

    def suffix(self) -> Tuple[array, array, array]:
        """The rows after the prefix (what a world adds). Pointer triples are
        in row order, so the suffix's are a tail of them."""
        k = self._prefix[0]
        p = self.pointers
        start = len(p)
        while start >= 3 and p[start - 3] >= k:
            start -= 3
        return self.ints[k * N_RI:], self.floats[k * N_RF:], p[start:]

    def prefix_rows(self) -> HistoryRows:
        """The frozen prefix's rows."""
        k = self._prefix[0]
        p = self.pointers
        end = next((t for t in range(0, len(p), 3) if p[t] >= k), len(p))
        return HistoryRows(k, self.ints[:k * N_RI], self.floats[:k * N_RF], p[:end], 0)

    def suffix_rows(self) -> HistoryRows:
        """The rows after the frozen prefix (what a world added)."""
        ints, floats, ptrs = self.suffix()
        k = self._prefix[0]
        return HistoryRows(self.n - k, ints, floats, ptrs, k)

    def prefix_n(self) -> int:
        return self._prefix[0]

    def to_rows(self) -> HistoryRows:
        """Every row (the naive request path)."""
        return HistoryRows(self.n, self.ints, self.floats, self.pointers, 0)

    def copy(self) -> "HistoryEncoder":
        new = HistoryEncoder.__new__(HistoryEncoder)
        new.viewer, new.order_index, new.owner_of = self.viewer, self.order_index, self.owner_of
        new.ints, new.floats, new.pointers = array("q", self.ints), array("f", self.floats), array("q", self.pointers)
        new.n, new.cursor = self.n, self.cursor
        new._digest = self._digest.copy()
        new._prefix = (self._prefix[0], self._prefix[1].copy())
        new._turn, new._step = self._turn, self._step
        new.zone_code_cache = self.zone_code_cache
        new.folds = None if self.folds is None else self.folds.copy()
        return new

    # --------------------------------------------------------- shards ----

    def to_bytes(self) -> bytes:
        return (struct.pack("<qq", self.n, len(self.pointers)) + self.ints.tobytes() + self.floats.tobytes()
                + self.pointers.tobytes())

    @staticmethod
    def rows_from_bytes(data: bytes) -> Tuple[array, array, array]:
        n, n_ptr = struct.unpack_from("<qq", data)
        at = 16
        ints = array("q")
        ints.frombytes(data[at:at + 8 * n * N_RI])
        at += 8 * n * N_RI
        floats = array("f")
        floats.frombytes(data[at:at + 4 * n * N_RF])
        at += 4 * n * N_RF
        pointers = array("q")
        pointers.frombytes(data[at:at + 8 * n_ptr])
        return ints, floats, pointers


def _house_by_value(v) -> int:
    if isinstance(v, House):
        return HOUSE_INDEX[v] + 1
    for h in HOUSE_INDEX:
        if h.value == v:
            return HOUSE_INDEX[h] + 1
    return 0


def _affects_id(a) -> int:
    from keyforge.enums import Affects

    if a is None:
        return 0
    for i, x in enumerate(Affects):
        if x.value == a:
            return i + 1
    return 0


def history_for(game, viewer: int) -> HistoryEncoder:
    """`viewer`'s history encoder, up to date (kept on the game and copied
    with it, like the projection)."""
    cache = game.__dict__.get("_history")
    if cache is None:
        cache = game.__dict__["_history"] = {}
    h = cache.get(viewer)
    if h is None:
        h = cache[viewer] = HistoryEncoder(game, viewer)
    h.update(game.projected(viewer))
    return h


# ----------------------------------------------------- turn tokens ----

TURN_FIELDS = ("turn", "actor", "house", "played", "discarded", "archived", "reaps", "fights", "uses", "gained",
               "stolen", "captured", "lost", "keys", "drawn", "hand_start", "hand_end", "archive_taken",
               "archive_declined", "chains_shed")
TURN_SCALARS = tuple(f for f in TURN_FIELDS if f not in ("turn", "actor", "house", "played", "discarded", "archived",
                                                         "hand_start", "hand_end")) + ("hand_start", "hand_end")
SUMMARY_ENTITY = ("played", "reaped", "fought", "used", "since_seen", "last_public", "left_by", "revealed",
                  "drawn_turn")
SUMMARY_GLOBAL = 2 * (7 + 3 + 7 + 2 + 3)  # per side: house counts, last three, since, mulligans + declined, hands
_LOG_COUNT = {"reap": "reaps", "fight": "fights", "use_action": "uses", "use_omni": "uses", "forge_key": "keys",
              "shed_chain": "chains_shed"}
_LOG_AMOUNT = {"gain": "gained", "steal": "stolen", "capture": "captured", "lose": "lost"}
_LOG_SET = {"play_card": "played", "discard": "discarded", "discard_from_hand": "discarded",
            "discard_random": "discarded", "archive": "archived"}
_COUNT_OF = {"play_card": "played", "reap": "reaped", "fight": "fought", "use_action": "used", "use_omni": "used"}
_HAND_CODES = {S.zone_code("hand", 0): 1, S.zone_code("hand", 1): 2}
_PUBLIC = ({S.zone_code(k, s) for k in ("discard", "purged", "battleline", "artifacts") for s in (0, 1)}
           | {S.zone_code(k, None) for k in ("attached", "limbo")})
_MY_HAND = S.zone_code("hand", 0)
_TOOK = 1 << BOOLS.index("took")


class _Folds:
    """Turn tokens and summaries, folded row by row as the encoder grows
    (so reading them costs only the new rows)."""

    __slots__ = ("row", "ptr", "tokens", "hand", "per", "houses", "mulligans", "declined")

    def __init__(self, n_entities: int):
        self.row = 0
        self.ptr = 0
        self.tokens: Dict[int, dict] = {}
        self.hand = {1: 0, 2: 0}
        self.per = [{"played": 0, "reaped": 0, "fought": 0, "used": 0, "seen_turn": None, "last_public": 0,
                     "left_by": 0, "revealed": 0, "drawn_turn": None} for _ in range(n_entities)]
        self.houses = {1: [], 2: []}
        self.mulligans = {1: 0, 2: 0}
        self.declined = {1: 0, 2: 0}

    def copy(self) -> "_Folds":
        new = _Folds.__new__(_Folds)
        new.row, new.ptr = self.row, self.ptr
        new.tokens = {t: {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                          for k, v in tok.items()} for t, tok in self.tokens.items()}
        new.hand = dict(self.hand)
        new.per = [dict(p) for p in self.per]
        new.houses = {s: list(v) for s, v in self.houses.items()}
        new.mulligans, new.declined = dict(self.mulligans), dict(self.declined)
        return new

    def advance(self, h: "HistoryEncoder") -> None:
        kinds = _V["log_kinds"].entries
        pointers = h.pointers
        n_ptr = len(pointers)
        while self.row < h.n:
            row = self.row
            ints = h.ints[row * N_RI:(row + 1) * N_RI]
            floats = h.floats[row * N_RF:(row + 1) * N_RF]
            ptrs = []
            while self.ptr < n_ptr and pointers[self.ptr] == row:
                ptrs.append((pointers[self.ptr + 1], pointers[self.ptr + 2]))
                self.ptr += 3
            turn = ints[_RI["turn"]]
            tok = self.tokens.get(turn)
            if tok is None:
                tok = self.tokens[turn] = {f: 0 for f in TURN_FIELDS}
                tok.update(turn=turn, played=[], discarded=[], archived=[], hand_start=dict(self.hand))
            stream = STREAMS[ints[_RI["stream"]] - 1]
            if stream == "zone":
                frm, to = ints[_RI["from_zone"]], ints[_RI["to_zone"]]
                if frm in _HAND_CODES:
                    self.hand[_HAND_CODES[frm]] -= 1
                if to in _HAND_CODES:
                    self.hand[_HAND_CODES[to]] += 1
                for role, p in ptrs:
                    if role != _ROLE["card"] or p < 0:
                        continue
                    e = self.per[p]
                    if to in _PUBLIC or frm in _PUBLIC:
                        e["seen_turn"] = turn
                        e["last_public"] = to if to in _PUBLIC else frm
                    if frm in _PUBLIC and to not in _PUBLIC:
                        e["left_by"] = ints[_RI["kind"]]
                    if to == _MY_HAND and e["drawn_turn"] is None:
                        e["drawn_turn"] = turn
            elif stream == "reveal":
                for role, p in ptrs:
                    if p >= 0:
                        self.per[p]["revealed"] = 1
            elif stream == "log":
                kind = kinds[ints[_RI["kind"]] - 1]
                actor = ints[_RI["actor"]]
                if kind == "turn_start":
                    tok["actor"] = actor
                elif kind == "choose_house":
                    tok["house"] = ints[_RI["house"]]
                    if actor:
                        self.houses[actor].append((turn, ints[_RI["house"]]))
                elif kind in _LOG_COUNT:
                    tok[_LOG_COUNT[kind]] += 1
                elif kind in _LOG_AMOUNT:
                    tok[_LOG_AMOUNT[kind]] += floats[_RF["amount"]]
                elif kind == "draw":
                    tok["drawn"] += floats[_RF["n"]]
                elif kind == "archive_decision":
                    if ints[_RI["flags"]] & _TOOK:
                        tok["archive_taken"] = 1
                    else:
                        tok["archive_declined"] = 1
                        if actor:
                            self.declined[actor] += 1
                elif kind == "mulligan" and actor:
                    self.mulligans[actor] += 1
                if kind in _LOG_SET:
                    tok[_LOG_SET[kind]].extend(p for role, p in ptrs if role == _ROLE["iid"])
                if kind in _COUNT_OF:
                    for role, p in ptrs:
                        if role == _ROLE["iid"] and p >= 0:
                            self.per[p][_COUNT_OF[kind]] += 1
            tok["hand_end"] = dict(self.hand)
            self.row += 1


def _folds(h: "HistoryEncoder") -> _Folds:
    if h.folds is None:
        h.folds = _Folds(len(h.order_index))
    h.folds.advance(h)
    return h.folds


def turn_tokens(h: "HistoryEncoder") -> List[dict]:
    """One token per half-turn (turn number), from the full event rows."""
    f = _folds(h)
    return [f.tokens[t] for t in sorted(f.tokens)]


# ------------------------------------------------------- summaries ----


def summaries(h: "HistoryEncoder", turn_now: int) -> dict:
    """Per entity and global summaries of the full event rows."""
    f = _folds(h)
    per = []
    for p in f.per:
        q = dict(p)
        seen = q.pop("seen_turn")
        q["since_seen"] = None if seen is None else turn_now - seen
        per.append(q)
    ends = {1: [], 2: []}
    for t in sorted(f.tokens):
        tok = f.tokens[t]
        if tok["actor"]:
            ends[tok["actor"]].append(tok["hand_end"][tok["actor"]])
    glob = {}
    for side in (1, 2):
        chosen = f.houses[side]
        counts = [0] * len(HOUSE_INDEX)
        since = [None] * len(HOUSE_INDEX)
        for turn, hid in chosen:
            if hid:
                counts[hid - 1] += 1
                since[hid - 1] = turn_now - turn
        glob[side] = {"house_counts": counts, "last_three": [hid for _, hid in chosen[-3:]], "since": since,
                      "mulligans": f.mulligans[side], "declined_archive": f.declined[side],
                      "hand_at_end_of_last_three": ends[side][-3:]}
    return {"entities": per, "global": glob}

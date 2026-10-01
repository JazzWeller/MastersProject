"""InfoSet -> flat feature arrays (Agent Training Plan, Milestone M1).

Pure standard library (`array`, no numpy), so it runs unchanged in engine
workers under CPython or PyPy; `ml/encode.py` turns batches of the result
into tensors on the torch side.

    encode(infoset, prefix=None) -> Encoded

`Encoded` is the **compact** form -- what crosses a process boundary or is
written to a training shard: per-entity vocabulary ids, zone codes and flag
bytes, the in-play block only for entities actually in play, the global
vector, and one row plus one pointer per option. `Encoded.dense()` expands
it into the plan's four arrays (`entities`, `globals`, `options`,
`pointers`), with each entity row being `STATIC[card_id] ++ ENTITY`; the
network does the same expansion on the GPU instead, where it's free.

Everything is from the deciding player's perspective ("mine"/"theirs",
never "player 1"/"player 2"), so one network serves both seats.
"""

from __future__ import annotations

import re
import zlib
from array import array
from typing import List, Optional, Sequence

from keyforge.cards.card_data import CARD_DEFS
from keyforge.cards.vocabulary import CARD_VOCAB
from keyforge.infoset import (
    FLAG_BITS,
    HOUSE_INDEX,
    IP_ARMOR,
    IP_ARMOR_USED,
    IP_ASSAULT,
    IP_BL_INDEX,
    IP_BL_SIZE,
    IP_CAPTURED,
    IP_DAMAGE,
    IP_DAMAGE_PREVENTED,
    IP_EFF_HOUSE,
    IP_EXHAUSTED,
    IP_FLANK,
    IP_FOUGHT,
    IP_HAZARDOUS,
    IP_IS_CREATURE,
    IP_KEYWORDS,
    IP_N_UNDER,
    IP_N_UPGRADES,
    IP_POWER,
    IP_POWER_COUNTERS,
    IP_STORED,
    IP_STUNNED,
    IP_ARMOR_NEGATED,
    VERB,
    ZONE,
    InfoSet,
)

from . import spec
from .spec import ENTITY, GLOBAL, INPLAY, OPTION, STATIC

# ------------------------------------------------------------- lookups --
_KIND_INDEX = {k: i for i, k in enumerate(spec.DECISION_KINDS)}
_INTENT_INDEX = {x: i + 1 for i, x in enumerate(spec.INTENTS)}  # 0 = none
_AFFECTS_INDEX = {x: i + 1 for i, x in enumerate(spec.AFFECTS)}  # 0 = not given
_FORMAT_INDEX = {f: i for i, f in enumerate(spec.FORMATS)}
_EFFECT_INDEX = {v: i for i, v in enumerate(spec.EFFECT_VARIABLES)}
_EFFECT_OTHER = spec.EFFECT_SLOTS - 1
_KEYWORD_INDEX = {k: i for i, k in enumerate(spec.KEYWORDS)}
_TYPE_INDEX = {t: i for i, t in enumerate(spec.CARD_TYPES)}

_G = {name: GLOBAL.offset(name) for name, _w in GLOBAL.fields}
_O = {name: OPTION.offset(name) for name, _w in OPTION.fields}
_I = {name: INPLAY.offset(name) for name, _w in INPLAY.fields}
_E_ZONE = ENTITY.offset("zone")
_E_FLAGS = ENTITY.offset("flags")
_E_INPLAY = ENTITY.offset("inplay")
_E_PREFIX = ENTITY.offset("selected_in_prefix")

_PL_SCALES = (
    spec.SCALE["aember"], spec.SCALE["keys"], spec.SCALE["chains"], spec.SCALE["hand"], spec.SCALE["deck"],
    spec.SCALE["pile"], spec.SCALE["small_pile"], spec.SCALE["small_pile"], spec.SCALE["board"],
    spec.SCALE["board"], spec.SCALE["key_cost"], 1.0, spec.SCALE["n"], spec.SCALE["tokens"],
    spec.SCALE["tokens"], 1.0, 1.0,
)


def stable_bucket(text: str, buckets: int) -> int:
    """A process- and platform-independent hash bucket (crc32, never
    Python's randomized `hash`)."""
    return zlib.crc32(text.lower().encode("utf-8")) % buckets


# --------------------------------------------------------- static table --
_TEXT_VERB_RES = [re.compile(r"\b" + re.escape(v), re.IGNORECASE) for v in spec.TEXT_VERBS]


def static_row(cdef) -> array:
    """The `STATIC` block for one card definition -- a pure function of
    printed card data, so a card never seen in training still gets a
    meaningful row."""
    row = array("f", bytes(4 * STATIC.width))
    row[STATIC.offset("house") + HOUSE_INDEX[cdef.house]] = 1.0
    row[STATIC.offset("type") + _TYPE_INDEX[cdef.type]] = 1.0
    row[STATIC.offset("power")] = cdef.power / spec.SCALE["power"]
    row[STATIC.offset("armor")] = cdef.armor / spec.SCALE["armor"]
    row[STATIC.offset("aember")] = cdef.aember_on_play / 4.0
    kw = STATIC.offset("keywords")
    for k in cdef.keywords:
        idx = _KEYWORD_INDEX.get(k)
        if idx is None:
            raise ValueError(f"static_row: keyword {k!r} on {cdef.name!r} has no slot -- add it to spec.KEYWORDS")
        row[kw + idx] = 1.0
    if cdef.assault:
        row[kw + _KEYWORD_INDEX["assault"]] = 1.0
    if cdef.hazardous:
        row[kw + _KEYWORD_INDEX["hazardous"]] = 1.0
    row[STATIC.offset("assault")] = cdef.assault / spec.SCALE["tokens"]
    row[STATIC.offset("hazardous")] = cdef.hazardous / spec.SCALE["tokens"]
    for t in cdef.tags:
        row[STATIC.offset("traits") + stable_bucket(t, spec.TRAIT_BUCKETS)] = 1.0
    ab = STATIC.offset("abilities")
    present = (
        cdef.on_play, cdef.on_reap, cdef.on_fight, cdef.on_action, cdef.on_omni, cdef.on_destroyed,
        cdef.register_passive, cdef.on_before_fight,
    )
    for i, hook in enumerate(present):
        if hook is not None:
            row[ab + i] = 1.0
    row[STATIC.offset("upgrade_power_bonus")] = cdef.power_bonus / spec.SCALE["tokens"]
    row[STATIC.offset("upgrade_armor_bonus")] = cdef.armor_bonus / spec.SCALE["tokens"]
    gk = STATIC.offset("grants_keywords")
    for k in cdef.grants_keywords:
        row[gk + _KEYWORD_INDEX[k]] = 1.0
    tv = STATIC.offset("text_verbs")
    text = cdef.text or ""
    for i, rx in enumerate(_TEXT_VERB_RES):
        if rx.search(text):
            row[tv + i] = 1.0
    return row


_STATIC_TABLE: Optional[List[array]] = None
_VOCAB_IDS = CARD_VOCAB  # name -> id (append-only; keyforge/cards/vocabulary.py)


def static_table() -> List[array]:
    """One `STATIC` row per vocabulary id, for the whole registry (a card
    in the vocabulary but not in `CARD_DEFS` -- impossible today -- would
    get an all-zero row)."""
    global _STATIC_TABLE
    if _STATIC_TABLE is None:
        size = max(_VOCAB_IDS.values()) + 1 if _VOCAB_IDS else 0
        table = [array("f", bytes(4 * STATIC.width)) for _ in range(size)]
        for name, vid in _VOCAB_IDS.items():
            cdef = CARD_DEFS.get(name)
            if cdef is not None:
                table[vid] = static_row(cdef)
        _STATIC_TABLE = table
    return _STATIC_TABLE


# -------------------------------------------------------------- encoded --
class Encoded:
    """The compact encoding of one decision point. See the module docstring."""

    __slots__ = ("card_ids", "zones", "flags", "inplay_index", "inplay", "globals", "options", "pointers", "selected")

    def __init__(self, card_ids, zones, flags, inplay_index, inplay, globals_, options, pointers, selected=None):
        self.card_ids: array = card_ids  # array('h')[72]
        self.zones: bytes = zones  # [72]
        self.flags: bytes = flags  # [72]
        self.inplay_index: bytes = inplay_index  # entity index per in-play row
        self.inplay: array = inplay  # array('f')[len(inplay_index) * INPLAY.width]
        self.globals: array = globals_  # array('f')[GLOBAL.width]
        self.options: array = options  # array('f')[n_options * OPTION.width]
        self.pointers: array = pointers  # array('b')[n_options], -1 = no entity
        self.selected: Optional[bytes] = selected  # [72] 0/1, sequential prefix; None = empty

    @property
    def n_options(self) -> int:
        return len(self.pointers)

    def to_bytes(self) -> bytes:
        """A deterministic byte serialization (the determinism acceptance
        test compares these across processes and platforms)."""
        parts = [
            self.card_ids.tobytes(), self.zones, self.flags, len(self.inplay_index).to_bytes(2, "little"),
            self.inplay_index, self.inplay.tobytes(), self.globals.tobytes(), len(self.pointers).to_bytes(2, "little"),
            self.options.tobytes(),
            self.pointers.tobytes(), self.selected or b"",
        ]
        return b"".join(parts)

    def entity_rows(self) -> array:
        """The per-decision `ENTITY` block, dense: 72 rows."""
        w = ENTITY.width
        out = array("f", bytes(4 * spec.N_ENTITIES * w))
        n_flags = len(FLAG_BITS)
        for i in range(spec.N_ENTITIES):
            base = i * w
            out[base + _E_ZONE + self.zones[i]] = 1.0
            f = self.flags[i]
            for b in range(n_flags):
                if f & (1 << b):
                    out[base + _E_FLAGS + b] = 1.0
            if self.selected is not None and self.selected[i]:
                out[base + _E_PREFIX] = 1.0
        pw = INPLAY.width
        for r, i in enumerate(self.inplay_index):
            out[i * w + _E_INPLAY : i * w + _E_INPLAY + pw] = self.inplay[r * pw : (r + 1) * pw]
        return out

    def dense(self):
        """The plan's `(entities, globals, options, pointers)`: `entities` is
        `72 * (STATIC.width + ENTITY.width)` floats, each row the card's
        static attributes followed by its per-decision block."""
        table = static_table()
        ent = self.entity_rows()
        ew = ENTITY.width
        out = array("f")
        for i in range(spec.N_ENTITIES):
            out.extend(table[self.card_ids[i]])
            out.extend(ent[i * ew : (i + 1) * ew])
        return out, self.globals, self.options, self.pointers


# --------------------------------------------------------------- encode --
def _encode_inplay(t: tuple, out: array, base: int) -> None:
    out[base + _I["damage"]] = t[IP_DAMAGE] / 10.0
    out[base + _I["power"]] = t[IP_POWER] / 10.0
    out[base + _I["armor"]] = t[IP_ARMOR] / 5.0
    out[base + _I["armor_used"]] = t[IP_ARMOR_USED] / 5.0
    out[base + _I["captured"]] = t[IP_CAPTURED] / 5.0
    out[base + _I["stored"]] = t[IP_STORED] / 5.0
    out[base + _I["power_counters"]] = t[IP_POWER_COUNTERS] / 5.0
    if t[IP_STUNNED]:
        out[base + _I["stunned"]] = 1.0
    if not t[IP_EXHAUSTED]:
        out[base + _I["ready"]] = 1.0
    out[base + _I["flank"] + t[IP_FLANK]] = 1.0
    size = t[IP_BL_SIZE]
    idx = t[IP_BL_INDEX]
    if idx >= 0 and size > 0:
        out[base + _I["battleline_index"]] = idx / (size - 1) if size > 1 else 0.5
    out[base + _I["n_upgrades"]] = t[IP_N_UPGRADES] / 5.0
    out[base + _I["n_under"]] = t[IP_N_UNDER] / 5.0
    if t[IP_DAMAGE_PREVENTED]:
        out[base + _I["damage_prevented"]] = 1.0
    if t[IP_ARMOR_NEGATED]:
        out[base + _I["armor_negated"]] = 1.0
    if t[IP_FOUGHT]:
        out[base + _I["fought_this_turn"]] = 1.0
    house = t[IP_EFF_HOUSE]
    if house is not None:
        out[base + _I["effective_house"] + HOUSE_INDEX[house]] = 1.0
    kw_base = base + _I["keywords"]
    for k in t[IP_KEYWORDS]:
        idx = _KEYWORD_INDEX.get(k)
        if idx is None:
            raise ValueError(f"encode: keyword {k!r} has no slot -- add it to spec.KEYWORDS")
        out[kw_base + idx] = 1.0
    if t[IP_HAZARDOUS]:
        out[kw_base + _KEYWORD_INDEX["hazardous"]] = 1.0
        out[base + _I["hazardous"]] = t[IP_HAZARDOUS] / 5.0
    if t[IP_ASSAULT]:
        out[kw_base + _KEYWORD_INDEX["assault"]] = 1.0
        out[base + _I["assault"]] = t[IP_ASSAULT] / 5.0
    if t[IP_IS_CREATURE]:
        out[base + _I["is_creature"]] = 1.0


def _encode_player(p: tuple, out: array, base: int) -> None:
    for j, scale in enumerate(_PL_SCALES):
        v = p[j]
        if v:
            out[base + j] = float(v) / scale


_HOUSE_OF_NAME = {name: HOUSE_INDEX[cdef.house] for name, cdef in CARD_DEFS.items()}
_Z_MY_HAND = ZONE["my_hand"]
_Z_MY_CREATURE = ZONE["my_creature"]
_Z_MY_ARTIFACT = ZONE["my_artifact"]


def _house_context(info: InfoSet) -> list:
    """Per house: [my hand cards, my creatures, my ready creatures, my
    artifacts] of that house (effective house for cards in play)."""
    ctx = [[0, 0, 0, 0] for _ in HOUSE_INDEX]
    zones = info.zones
    for i, name in enumerate(info.entity_names):
        z = zones[i]
        if z == _Z_MY_HAND:
            ctx[_HOUSE_OF_NAME[name]][0] += 1
        elif z == _Z_MY_CREATURE or z == _Z_MY_ARTIFACT:
            t = info.inplay.get(i)
            h = HOUSE_INDEX[t[IP_EFF_HOUSE]] if t is not None and t[IP_EFF_HOUSE] is not None else _HOUSE_OF_NAME[name]
            if z == _Z_MY_CREATURE:
                ctx[h][1] += 1
                if t is not None and not t[IP_EXHAUSTED]:
                    ctx[h][2] += 1
            else:
                ctx[h][3] += 1
    return ctx


def _encode_option(entry: tuple, out: array, base: int, house_ctx=None) -> None:
    verb, ptr, payload = entry
    out[base + _O["verb"] + verb] = 1.0
    if ptr >= 0:
        out[base + _O["has_pointer"]] = 1.0
    if payload is None:
        return
    if verb == VERB["house"]:
        h = HOUSE_INDEX[payload]
        out[base + _O["house"] + h] = 1.0
        if house_ctx is not None:
            hand, creatures, ready, artifacts = house_ctx[h]
            hc = base + _O["house_context"]
            out[hc] = hand / 6.0
            out[hc + 1] = creatures / 6.0
            out[hc + 2] = ready / 6.0
            out[hc + 3] = artifacts / 4.0
    elif verb == VERB["bool"]:
        out[base + _O["bool"]] = 1.0 if payload else 0.0
    elif verb == VERB["flank"]:
        out[base + _O["flank"] + (0 if payload == "left" else 1)] = 1.0
    elif verb == VERB["first_player"]:
        out[base + _O["first_player"] + (0 if payload == "first" else 1)] = 1.0
    elif verb == VERB["number"]:
        out[base + _O["number"]] = payload / spec.SCALE["number"]
    elif verb == VERB["bid"]:
        out[base + _O["number"]] = payload / spec.SCALE["bid"]
    elif verb in (VERB["mode"], VERB["trigger"]):
        out[base + _O["string"] + stable_bucket(str(payload), spec.STRING_BUCKETS)] = 1.0
    else:
        raise ValueError(f"encode: unexpected payload {payload!r} for verb {verb}")


def encode(info: InfoSet, prefix: Optional[Sequence[int]] = None, *, with_stop: bool = False) -> Encoded:
    """`prefix`, for the sequential multi-select treatment, is the option
    indices already picked: their entities get `selected_in_prefix` and the
    prefix length goes into the decision scalars. `with_stop` appends the
    "stop here" pseudo-option (verb `spec.STOP_VERB`, no pointer)."""
    n = len(info.entity_iids)
    if n != spec.N_ENTITIES:
        raise ValueError(f"encode: {n} entities, the layout needs exactly {spec.N_ENTITIES}")
    card_ids = array("h", [_VOCAB_IDS[name] for name in info.entity_names])

    # In-play rows, sorted by entity index so the encoding is canonical.
    rows = sorted(info.inplay)
    pw = INPLAY.width
    inplay = array("f", bytes(4 * pw * len(rows)))
    for r, i in enumerate(rows):
        _encode_inplay(info.inplay[i], inplay, r * pw)

    g = array("f", bytes(4 * GLOBAL.width))
    _encode_player(info.players[0], g, _G["me"])
    _encode_player(info.players[1], g, _G["them"])
    g[_G["turn"]] = info.turn_number / spec.SCALE["turn"]
    if info.is_my_turn:
        g[_G["is_my_turn"]] = 1.0
    if info.i_went_first:
        g[_G["i_went_first"]] = 1.0
    ah = info.active_house
    g[_G["active_house"] + (HOUSE_INDEX[ah] if ah is not None else len(HOUSE_INDEX))] = 1.0
    for h in info.my_houses:
        g[_G["my_houses"] + HOUSE_INDEX[h]] = 1.0
    for h in info.their_houses:
        g[_G["their_houses"] + HOUSE_INDEX[h]] = 1.0
    for variable, affects_me in info.effects:
        slot = _EFFECT_INDEX.get(variable, _EFFECT_OTHER)
        g[_G["effects_me" if affects_me else "effects_them"] + slot] += 0.5
    fmt, game_number, my_score, their_score, my_chains, their_chains, swapped = info.match
    g[_G["format"] + _FORMAT_INDEX.get(fmt, len(spec.FORMATS))] = 1.0
    m = _G["match"]
    g[m] = game_number / spec.SCALE["match_game"]
    g[m + 1] = my_score / spec.SCALE["score"]
    g[m + 2] = their_score / spec.SCALE["score"]
    g[m + 3] = my_chains / spec.SCALE["chains"]
    g[m + 4] = their_chains / spec.SCALE["chains"]
    g[m + 5] = 1.0 if swapped else 0.0

    n_opts = len(info.options) + (1 if with_stop else 0)
    if info.decision is not None:
        kind, intent, affects, optional, min_n, max_n = info.decision
        g[_G["kind"] + _KIND_INDEX[kind]] = 1.0
        g[_G["intent"] + (_INTENT_INDEX[intent] if intent is not None else 0)] = 1.0
        g[_G["affects"] + (_AFFECTS_INDEX[affects] if affects is not None else 0)] = 1.0
        ds = _G["decision_scalars"]
        g[ds] = 1.0 if optional else 0.0
        g[ds + 1] = min(min_n, 40) / spec.SCALE["n"]
        g[ds + 2] = min(max_n, 40) / spec.SCALE["n"]
        g[ds + 3] = n_opts / spec.SCALE["n_options"]
        g[ds + 4] = (len(prefix) if prefix else 0) / spec.SCALE["n"]

    ow = OPTION.width
    options = array("f", bytes(4 * ow * n_opts))
    pointers = array("b", [-1]) * n_opts
    house_ctx = _house_context(info) if any(entry[0] == VERB["house"] for entry in info.options) else None
    for k, entry in enumerate(info.options):
        _encode_option(entry, options, k * ow, house_ctx)
        pointers[k] = entry[1]
    if with_stop:
        options[(n_opts - 1) * ow + _O["verb"] + spec.STOP_VERB] = 1.0

    selected = None
    if prefix:
        sel = bytearray(n)
        for k in prefix:
            ptr = info.options[k][1]
            if ptr >= 0:
                sel[ptr] = 1
        selected = bytes(sel)

    return Encoded(
        card_ids, bytes(info.zones), bytes(info.flags), bytes(rows), inplay, g, options, pointers, selected,
    )

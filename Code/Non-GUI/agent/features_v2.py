"""The v2 encoding (Agent Observation Plan, Milestone O5): game -> the
token sets of `agent/spec_v2.py`, for one viewer.

    encode_v2(game, viewer, match=None, prefix=None, with_stop=False) -> EncodedV2

Pure standard library, like v1. It reads the live game only through what
the viewer is entitled to: v1's redacted extract (`keyforge/infoset.py`)
for the decision, the options and in-play values; the viewer's knowledge
tracker, second-order tracker and chance filter (O2, O3) for every card;
the full state of the cards the viewer can see; the effect list; the
resolution view (R7, redacted); cleanups, temporary control and the match.
The invariance test (I1) holds the bytes identical in every world
consistent with what the viewer knows.

Nothing is capped or hashed: counts are raw (scaled, never clipped), every
categorical is a vocabulary id (`agent/vocab/`), and an unknown one raises.
"""

from __future__ import annotations

import math
from array import array
from typing import Dict, List, Optional, Sequence

from keyforge.cards.card import Card, CreatureType
from keyforge.cards.vocabulary import CARD_VOCAB
from keyforge.effects.effect_object import INFINITE
from keyforge.enums import DecisionKind, House
from keyforge.infoset import (
    HOUSE_INDEX,
    IP_ARMOR,
    IP_ARMOR_USED,
    IP_ASSAULT,
    IP_CAPTURED,
    IP_DAMAGE,
    IP_HAZARDOUS,
    IP_N_UNDER,
    IP_N_UPGRADES,
    IP_POWER,
    IP_POWER_COUNTERS,
    IP_STORED,
    VERB_NAMES,
    build_infoset,
)

from . import spec_v2 as S
from .features_v1 import _house_context
from .vocab import load_all

_V = load_all()
_HOUSES = list(HOUSE_INDEX)
_KIND_IDS = {k: i + 1 for i, k in enumerate(DecisionKind)}
_FLAG_BIT = {f: 1 << i for i, f in enumerate(S.ENTITY_FLAGS)}
_EFFECT_KIND = {k: i + 1 for i, k in enumerate(S.EFFECT_KINDS)}
_VALUE_KIND = {k: i for i, k in enumerate(S.VALUE_KINDS)}
_FLANK = {f: i + 1 for i, f in enumerate(S.FLANKS)}


def _house_id(h) -> int:
    return 0 if h is None else HOUSE_INDEX[h] + 1


def _house_bits(houses) -> int:
    out = 0
    for h in houses:
        out |= 1 << HOUSE_INDEX[h]
    return out


def _log1p(x) -> float:
    return math.log1p(max(0, x))


class Block:
    """One token block's rows: an int and a float array, row-major."""

    __slots__ = ("cols", "ints", "floats", "n")

    def __init__(self, cols: S.Columns):
        self.cols = cols
        self.ints = array("q")
        self.floats = array("f")
        self.n = 0

    def row(self, **values) -> None:
        cols = self.cols
        ints = [0] * cols.n_int
        floats = [0.0] * cols.n_float
        for k, v in values.items():
            j = cols.i.get(k)
            if j is not None:
                ints[j] = int(v)
            else:
                floats[cols.f[k]] = float(v)
        self.ints.extend(ints)
        self.floats.extend(floats)
        self.n += 1

    def to_bytes(self) -> bytes:
        return self.n.to_bytes(4, "little") + self.ints.tobytes() + self.floats.tobytes()


class EncodedV2:
    __slots__ = ("blocks",)

    def __init__(self, blocks: Dict[str, Block]):
        self.blocks = blocks

    def __getitem__(self, name: str) -> Block:
        return self.blocks[name]

    def to_bytes(self) -> bytes:
        return b"".join(self.blocks[b.name].to_bytes() for b in S.BLOCKS)


class _Ctx:
    def __init__(self, game, viewer: int):
        from keyforge.determinize import chance_filter
        from keyforge.knowledge import second_order_tracker, tracker_for

        self.game = game
        self.viewer = viewer
        self.me = game.players[viewer]
        self.them = game.players[3 - viewer]
        self.order = [c.instance_id for c in self.me.all_cards] + [c.instance_id for c in self.them.all_cards]
        self.index = {iid: i for i, iid in enumerate(self.order)}
        self.t = tracker_for(game, viewer)
        self.s = second_order_tracker(game, viewer)
        self.f = chance_filter(game, viewer)
        self._zones = {}
        self._masks = {}

    def side(self, pid) -> Optional[int]:
        return None if pid is None else (0 if pid == self.viewer else 1)

    def zone(self, z) -> int:
        code = self._zones.get(z)
        if code is None:
            kind, key = z
            code = S.zone_code(kind, None if kind in ("attached", "under", "limbo") else self.side(key))
            self._zones[z] = code
        return code

    def mask_bits(self, mask) -> int:
        bits = self._masks.get(mask)
        if bits is None:
            bits = 0
            for z in mask:
                bits |= 1 << self.zone(z)
            self._masks[mask] = bits
        return bits

    def visible(self, iid: int) -> bool:
        m = self.t.mask.get(iid)
        if m is None or len(m) != 1:
            return False
        (z,) = m
        return self.t.visible(z, self.t.owner.get(iid))

    def ptr(self, iid) -> int:
        if iid is None:
            return S.NO_POINTER
        if isinstance(iid, Card):
            iid = iid.instance_id
        if self.visible(iid):
            return self.index[iid]
        return S.HIDDEN_MINE if self.t.owner.get(iid) == self.viewer else S.HIDDEN_THEIRS


def _groups(ctx: _Ctx) -> Dict[int, tuple]:
    """Each card's provenance group as (rank among its side's groups, its
    size): a group's id is a journal seq, which says when, not which."""
    by_side: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
    for iid in ctx.order:
        g = ctx.t.group.get(iid)
        if g is not None:
            side = ctx.side(ctx.t.owner[iid])
            by_side[side][g] = by_side[side].get(g, 0) + 1
    out = {}
    for iid in ctx.order:
        g = ctx.t.group.get(iid)
        if g is None:
            continue
        side = ctx.side(ctx.t.owner[iid])
        ranks = sorted(by_side[side])
        out[iid] = (ranks.index(g) + 1, by_side[side][g])
    return out


def _entities(ctx: _Ctx, info, block: Block, prefix_set) -> None:
    game, t, s, f = ctx.game, ctx.t, ctx.s, ctx.f
    turn = game.turn_number
    groups = _groups(ctx)
    temp = {}
    for src, entries in game._temp_control.items():
        for c, orig in entries:
            temp[c.instance_id] = orig
    positions = {}
    for pid, p in game.players.items():
        n = len(p.play_area.creatures)
        for i, c in enumerate(p.play_area.creatures):
            flank = "both" if n == 1 else ("left" if i == 0 else ("right" if i == n - 1 else "center"))
            positions[c.instance_id] = (("battleline", pid), i, n, flank, None, 0)
        for i, c in enumerate(p.play_area.artifacts):
            positions[c.instance_id] = (("artifacts", pid), i, len(p.play_area.artifacts), None, None, 0)
    for host in game._cards_by_id.values():
        for u in getattr(host.type_object, "upgrades", None) or ():
            positions[u.instance_id] = (("attached", host.instance_id), 0, 0, None, host.instance_id, 0)
        for k, u in enumerate(host.under_cards):
            positions[u.instance_id] = (("under", host.instance_id), 0, 0, None, host.instance_id, k + 1)
    events = _V["events"]
    keywords = _V["keywords"]
    journal_ops = _V["journal_ops"]
    destined = _V["destined_zones"]
    marginals = f.marginals()
    next_draw = f.next_draw_fn()
    cols = block.cols
    NI, NF = cols.n_int, cols.n_float
    I, F = cols.i, cols.f
    i_card = I["card"]
    i_controller = I["controller"]
    i_deck_pos = I["deck_pos"]
    i_destined_zone = I["destined_zone"]
    i_exact = I["exact"]
    i_exit_op = I["exit_op"]
    i_extra_triggers = I["extra_triggers"]
    i_flags = I["flags"]
    i_flank = I["flank"]
    i_group = I["group"]
    i_group_size = I["group_size"]
    i_host = I["host"]
    i_house = I["house"]
    i_house_override = I["house_override"]
    i_index = I["index"]
    i_keywords = I["keywords"]
    i_last_seen = I["last_seen"]
    i_mask = I["mask"]
    i_of = I["of"]
    i_opp_exact = I["opp_exact"]
    i_opp_mask = I["opp_mask"]
    i_owner = I["owner"]
    i_played_this_turn = I["played_this_turn"]
    i_position = I["position"]
    i_purged_by = I["purged_by"]
    i_redirect_to = I["redirect_to"]
    i_selected = I["selected"]
    i_since_seen = I["since_seen"]
    i_temp_original = I["temp_original"]
    i_under_index = I["under_index"]
    i_uses_this_turn = I["uses_this_turn"]
    i_visible = I["visible"]
    i_zone = I["zone"]
    f_base_armor = F["base_armor"]
    f_base_power = F["base_power"]
    f_captured = F["captured"]
    f_damage = F["damage"]
    f_n_under = F["n_under"]
    f_p_archive = F["p_archive"]
    f_p_deck = F["p_deck"]
    f_p_elsewhere = F["p_elsewhere"]
    f_p_hand = F["p_hand"]
    f_p_next_draw = F["p_next_draw"]
    f_power_counters = F["power_counters"]
    f_stored = F["stored"]
    ints = array("q", bytes(8 * NI * len(ctx.order)))
    floats = array("f", bytes(4 * NF * len(ctx.order)))
    zone = ctx.zone
    mask_bits = ctx.mask_bits
    viewer = ctx.viewer
    side = ctx.side
    for e, iid in enumerate(ctx.order):
        bi, bf = e * NI, e * NF
        card = game._cards_by_id[iid]
        mask = t.mask[iid]
        exact = len(mask) == 1
        ints[bi + i_card] = CARD_VOCAB[card.name]
        ints[bi + i_owner] = side(card.owner)
        ints[bi + i_exact] = exact
        ints[bi + i_mask] = mask_bits(mask)
        if exact:
            ints[bi + i_zone] = zone(next(iter(mask)))
        dp = t.deck_position(iid)
        if dp is not None:
            ints[bi + i_deck_pos] = dp + 1 if dp >= 0 else dp
        g = groups.get(iid)
        if g is not None:
            ints[bi + i_group], ints[bi + i_group_size] = g
        seen = t.seen_turn.get(iid)
        if seen is not None:
            ints[bi + i_since_seen] = turn - seen + 1
        z = t.seen_zone.get(iid)
        if z is not None:
            ints[bi + i_last_seen] = zone(z)
        op = t.exit_op.get(iid)
        if op is not None:
            ints[bi + i_exit_op] = journal_ops.id(op)
        if t.owner[iid] == viewer:
            om = s.mask[iid]
            ints[bi + i_opp_exact] = len(om) == 1
            ints[bi + i_opp_mask] = mask_bits(om)
        else:
            pz = marginals.get(iid)
            if pz is not None:
                floats[bf + f_p_hand], floats[bf + f_p_archive], floats[bf + f_p_deck], \
                    floats[bf + f_p_elsewhere] = pz
                floats[bf + f_p_next_draw] = next_draw(iid)
        if e in prefix_set:
            ints[bi + i_selected] = 1
        if not ctx.visible(iid):
            continue
        ints[bi + i_visible] = 1
        ints[bi + i_controller] = side(card.controller) + 1
        pos = positions.get(iid)
        ints[bi + i_position] = zone(pos[0] if pos is not None else next(iter(mask)))
        if pos is not None:
            _, i, n, flank, host, under_index = pos
            ints[bi + i_index], ints[bi + i_of] = i + 1, n
            if flank is not None:
                ints[bi + i_flank] = _FLANK[flank]
            if host is not None:
                ints[bi + i_host] = ctx.ptr(host)
            ints[bi + i_under_index] = under_index
        ints[bi + i_house] = _house_id(card.house)
        ints[bi + i_house_override] = _house_id(card.house_override)
        ints[bi + i_destined_zone] = destined.id(card.destined_zone)
        if iid in temp:
            ints[bi + i_temp_original] = side(temp[iid]) + 1
        ints[bi + i_redirect_to] = ctx.ptr(card.redirect_fight_damage_to)
        ints[bi + i_purged_by] = ctx.ptr(card.purged_by)
        to = card.type_object
        is_creature = isinstance(to, CreatureType)
        flags = 0
        for name, on in (("exhausted", card.Exhausted), ("stunned", card.stunned), ("can_be_used", card.CanBeUsed),
                         ("destroyed", card.destroyed), ("fought_this_turn", card.fought_this_turn),
                         ("forced_flank", card.forced_flank), ("granted_action", card.granted_action is not None),
                         ("archive_return_to_owner", card.archive_return_to_owner),
                         ("damage_prevented", card.damage_prevented), ("armor_negated", card.armor_negated),
                         ("is_creature", is_creature), ("ignore_elusive", card.IgnoreElusive)):
            if on:
                flags |= _FLAG_BIT[name]
        ints[bi + i_flags] = flags
        kw = 0
        for k in (game.get_keywords(card) if pos is not None and pos[0][0] == "battleline" else card.keywords):
            kw |= 1 << keywords.id(k)
        ints[bi + i_keywords] = kw
        et = 0
        for ev, fns in (card.extra_triggers or {}).items():
            if fns:
                et |= 1 << events.id(ev)
        ints[bi + i_extra_triggers] = et
        owner = game.players[card.controller]
        ints[bi + i_uses_this_turn] = owner.used_this_turn.get(card.name, 0)
        ints[bi + i_played_this_turn] = owner.CardsPlayed.get(card.name, 0)
        ip = info.inplay.get(e)
        if ip is not None:
            for col, k in (("damage", IP_DAMAGE), ("power", IP_POWER), ("armor", IP_ARMOR),
                           ("armor_used", IP_ARMOR_USED), ("captured", IP_CAPTURED), ("stored", IP_STORED),
                           ("power_counters", IP_POWER_COUNTERS), ("hazardous", IP_HAZARDOUS),
                           ("assault", IP_ASSAULT), ("n_upgrades", IP_N_UPGRADES), ("n_under", IP_N_UNDER)):
                floats[bf + F[col]] = ip[k]
        else:
            floats[bf + f_captured] = card.aember_captured
            floats[bf + f_stored] = card.aember_stored
            floats[bf + f_power_counters] = card.power_counters
            floats[bf + f_n_under] = len(card.under_cards)
            if is_creature:
                floats[bf + f_damage] = to.damage
        if is_creature:
            floats[bf + f_base_power] = to.base_power
            floats[bf + f_base_armor] = to.base_armor
    block.ints.extend(ints)
    block.floats.extend(floats)
    block.n += len(ctx.order)


def _player(ctx: _Ctx, p, other, prefix: str, row: dict) -> None:
    game = ctx.game
    limit = p.get_card_played_limit(game)
    counters = {
        "aember": p.aember, "keys": p.keys, "chains": p.chains, "CardDrawModifier": p.CardDrawModifier,
        "DrawUpToLimit": p.DrawUpToLimit, "KeyForgeCost": p.KeyForgeCost, "NonLogosCardsPlayable": p.NonLogosCardsPlayable,
        "creatures_played_this_turn": p.creatures_played_this_turn,
        "cards_played_or_discarded_this_turn": p.cards_played_or_discarded_this_turn,
        "hand_plays_this_turn": p.hand_plays_this_turn, "key_cost_now": p.get_key_forge_cost(game),
        "n_hand": len(p.hand), "n_deck": len(p.deck), "n_discard": len(p.discard), "n_archive": len(p.archive),
        "n_purged": len(p.purged), "n_creatures": len(p.play_area.creatures), "n_artifacts": len(p.play_area.artifacts),
        "card_played_limit": limit or 0, "uses_this_turn": sum(p.used_this_turn.values()),
        "cards_played_kinds": len(p.CardsPlayed),
    }
    for k in S.PLAYER_COUNTERS:
        row[f"{prefix}_{k}"] = counters[k] / 10.0
    flags = {
        "CanKeyForge": p.CanKeyForge, "CanPlayActions": p.CanPlayActions, "CanPlayCreatures": p.CanPlayCreatures,
        "ReapGainBecomesSteal": p.ReapGainBecomesSteal, "CanFight": p.CanFight, "CannotUseCards": p.CannotUseCards,
        "CanPlayCards": p.CanPlayCards, "FirstCreatureEntersReady": p.FirstCreatureEntersReady,
        "CannotBeDealtDamage": p.CannotBeDealtDamage, "CanOnlyFight": p.CanOnlyFight,
        "CannotBeStolenFrom": p.CannotBeStolenFrom, "next_entry_ready": p.next_entry_ready,
        "next_mars_creature_ready": p.next_mars_creature_ready,
        "discarded_untamed_this_turn": p.discarded_untamed_this_turn,
        "hand_revealed_to_other": other.id in p.hand_revealed_to, "has_card_played_limit": limit is not None,
    }
    for k in S.PLAYER_FLAGS:
        row[f"{prefix}_{k}"] = int(bool(flags[k]))


def _global(ctx: _Ctx, info, block: Block, n_opts: int, prefix) -> None:
    game = ctx.game
    me, them = ctx.me, ctx.them
    max_turns = game.config.max_turns
    row = {
        "turn": game.turn_number / 50.0, "is_my_turn": int(info.is_my_turn), "i_went_first": int(info.i_went_first),
        "has_max_turns": int(max_turns is not None), "elusive_suppressed": int(bool(game._elusive_suppressed)),
        "is_over": int(game.is_over), "active_house": _house_id(info.active_house),
        "my_houses": _house_bits(info.my_houses), "their_houses": _house_bits(info.their_houses),
        "my_selected_house": _house_id(me.selected_house), "their_selected_house": _house_id(them.selected_house),
        "my_house_selection": _house_id(me.HouseSelection if isinstance(me.HouseSelection, House) else None),
        "their_house_selection": _house_id(them.HouseSelection if isinstance(them.HouseSelection, House) else None),
        "my_extra_house_playable": _house_bits(h for h, n in me.ExtraHousePlayable.items() if n),
        "their_extra_house_playable": _house_bits(h for h, n in them.ExtraHousePlayable.items() if n),
        "n_options": _log1p(n_opts), "prefix": _log1p(len(prefix) if prefix else 0),
    }
    if max_turns is not None:
        row["max_turns"] = max_turns / 50.0
        row["turns_left"] = max(0, max_turns - game.turn_number) / 50.0
    fmt, game_number, my_score, their_score, my_chains, their_chains, swapped = info.match
    row.update(format=("archon", "reversal", "adaptive").index(fmt) + 1, game_number=game_number,
               my_score=my_score, their_score=their_score, my_start_chains=my_chains / 6.0,
               their_start_chains=their_chains / 6.0, swapped=int(bool(swapped)))
    d = game.pending_decision
    if d is not None:
        row.update(kind=_KIND_IDS[d.kind], intent=0 if d.intent is None else list(type(d.intent)).index(d.intent) + 1,
                   affects=0 if d.affects is None else list(type(d.affects)).index(d.affects) + 1,
                   optional=int(bool(d.optional)), min_n=_log1p(d.min_n), max_n=_log1p(d.max_n),
                   decider=ctx.side(d.player) + 1,
                   decision_source=ctx.ptr(d.source_card) if isinstance(d.source_card, Card) else S.NO_POINTER)
    _player(ctx, me, them, "my", row)
    _player(ctx, them, me, "their", row)
    block.row(**row)


def _value(v):
    if v is None:
        return 0.0, "none"
    if isinstance(v, bool):
        return float(v), "bool"
    if isinstance(v, (int, float)):
        return float(v), "number"
    if isinstance(v, House):
        return float(HOUSE_INDEX[v] + 1), "house"
    return 0.0, "other"


def _effects(ctx: _Ctx, block: Block) -> None:
    game = ctx.game
    ae = game.active_effects

    def common(e):
        return {"controller": ctx.side(e.controller) + 1, "source": ctx.ptr(e.source_card)}

    def remaining(e):
        r = getattr(e, "remaining_duration", INFINITE)
        return {"infinite": 1} if r == INFINITE else {"remaining": r / 5.0}

    for e in ae.duration_effects:
        value, vk = _value(e.value(game) if callable(e.value) else e.value)
        block.row(kind=_EFFECT_KIND["duration"], variable=_V["effect_variables"].id(e.variable),
                  op=_V["effect_ops"].id(e.op), value=value, value_kind=_VALUE_KIND[vk],
                  active=int(bool(e.is_active(game))), conditional=int(e.conditional is not None),
                  affected=ctx.side(e.player_affected) + 1, **remaining(e), **common(e))
    for e in ae.trigger_effects:
        block.row(kind=_EFFECT_KIND["trigger"], event=_V["events"].id(e.event), active=1, **remaining(e), **common(e))
    for kind, effects in (("instead", ae.instead_effects), ("modifier", ae.modifier_effects)):
        for e in effects:
            block.row(kind=_EFFECT_KIND[kind], effect_kind=_V["effect_kinds"].id(e.kind), active=1, infinite=1,
                      **common(e))


def _cards_in(value, out: list) -> None:
    """The card references inside a resolution-view value: ("card", iid),
    and ("hidden_counts", ((zone, side, n), ...))."""
    if isinstance(value, tuple) and value:
        tag = value[0]
        if tag == "card" and len(value) == 2:
            out.append(("card", value[1]))
            return
        if tag == "hidden" and len(value) == 3:
            out.append(("hidden", value[1], value[2], 1))
            return
        if tag == "hidden_counts":
            for zone, side, n in value[1]:
                out.append(("hidden", zone, side, n))
            return
        for v in value:
            _cards_in(v, out)


def _resolution(ctx: _Ctx, frames_b: Block, ptr_b: Block, op_b: Block) -> None:
    game = ctx.game
    if game.is_over or game.execution != "compiled":
        return
    from keyforge.resolution import resolution_view

    from .vocab import op_name

    roles, routines, sites, kinds, events, ops = (_V[n] for n in ("roles", "routines", "sites", "ability_kinds",
                                                                  "events", "op_names"))
    for k, fr in enumerate(resolution_view(game, ctx.viewer)):
        qual = fr["routine"].split(":", 1)[1] if ":" in fr["routine"] else fr["routine"]
        src = fr["source"]
        source = S.NO_POINTER
        if isinstance(src, tuple) and src and src[0] == "card":
            source = ctx.ptr(src[1])
        elif isinstance(src, tuple) and src and src[0] == "hidden":
            source = S.HIDDEN_MINE if src[2] == "mine" else S.HIDDEN_THEIRS
        frames_b.row(routine=routines.id(qual), site=sites.id(f"{qual}:{fr['pc']}"), depth=fr["depth"],
                     kind=kinds.id(fr["kind"]), event=events.id(fr["event"]), source=source, pc=fr["pc"],
                     n_locals=len(fr["locals"]))
        for role, value in fr["locals"]:
            found: list = []
            _cards_in(value, found)
            for pos, ref in enumerate(found):
                if ref[0] == "card":
                    ptr_b.row(frame=k, role=roles.id(role), entity=ctx.ptr(ref[1]), position=pos + 1, count=1)
                else:
                    _, zone, side, n = ref
                    hidden = S.HIDDEN_MINE if side == "mine" else S.HIDDEN_THEIRS
                    ptr_b.row(frame=k, role=roles.id(role), entity=hidden, count=n,
                              zone=S.zone_code(zone, 0 if side == "mine" else 1) if zone in S.ZONE_KINDS else 0)
        for j, op in enumerate(fr["ops"]):
            op_b.row(frame=k, op=ops.id(op_name(op)), order=j + 1)


def _options(ctx: _Ctx, info, block: Block, with_stop: bool) -> None:
    verbs, modes, events = _V["verbs"], _V["modes"], _V["events"]
    house_ctx = _house_context(info) if any(entry[0] == VERB_NAMES.index("house") for entry in info.options) else None
    for verb, ptr, payload in info.options:
        name = VERB_NAMES[verb]
        row = {"verb": verbs.id(name), "pointer": ptr if ptr >= 0 else S.NO_POINTER}
        if payload is not None:
            if name == "house":
                h = HOUSE_INDEX[payload]
                row["house"] = h + 1
                if house_ctx is not None:
                    hand, creatures, ready, artifacts = house_ctx[h]
                    row.update(hand_of_house=hand, creatures_of_house=creatures, ready_of_house=ready,
                               artifacts_of_house=artifacts)
            elif name == "bool":
                row["bool"] = 2 if payload else 1
            elif name == "flank":
                row["flank"] = 1 if payload == "left" else 2
            elif name == "first_player":
                row["first_player"] = 1 if payload == "first" else 2
            elif name in ("number", "bid"):
                row["number"] = payload / 10.0
            elif name == "mode":
                row["mode"] = modes.id(payload)
            elif name == "trigger":
                if payload in events._ids:
                    row["event"] = events.id(payload)
                else:
                    row["tag"] = _V["option_tags"].id(payload)
        block.row(**row)
    if with_stop:
        block.row(verb=verbs.id("stop"))


def encode_v2(game, viewer: int, match=None, prefix: Optional[Sequence[int]] = None, *, with_stop: bool = False) -> EncodedV2:
    info = build_infoset(game, viewer, match=match)
    ctx = _Ctx(game, viewer)
    blocks = {b.name: Block(b) for b in S.BLOCKS}
    prefix_set = set()
    if prefix:
        for k in prefix:
            ptr = info.options[k][1]
            if ptr >= 0:
                prefix_set.add(ptr)
    n_opts = len(info.options) + (1 if with_stop else 0)
    _global(ctx, info, blocks["global"], n_opts, prefix)
    _entities(ctx, info, blocks["entity"], prefix_set)
    _effects(ctx, blocks["effect"])
    _resolution(ctx, blocks["resolution"], blocks["res_pointer"], blocks["res_op"])
    for op, iid in game._end_of_turn_cleanups:
        blocks["cleanup"].row(op=_V["cleanup_ops"].id(op), target=ctx.ptr(iid) if iid in ctx.index else S.NO_POINTER)
    if match is not None:
        _match_rows(ctx.side, match, blocks["match"])
    _options(ctx, info, blocks["option"], with_stop)
    return EncodedV2(blocks)


def _match_rows(ctx_side, match, block: Block) -> None:
    seats0 = match.games[0].seat_decks if match.games else None
    for g in match.games:
        keys = {ctx_side(p): k for p, k in g.final_keys.items()}
        chains = {ctx_side(p): c for p, c in g.final_chains.items()}
        block.row(winner=0 if g.winner is None else ctx_side(g.winner) + 1, my_keys=keys.get(0, 0),
                  their_keys=keys.get(1, 0), turns=g.turns / 50.0, my_chains=chains.get(0, 0),
                  their_chains=chains.get(1, 0),
                  first=0 if g.config.first_player is None else ctx_side(g.config.first_player) + 1,
                  swapped=int(g.seat_decks != seats0))


def encode_v2_match(match, viewer: int) -> EncodedV2:
    """A match-level decision (BID_CHAINS, CHOOSE_FIRST_PLAYER), between
    games: the global row (match context, the decision), the two decks of
    the game just finished as entities (the deck being bid on flagged
    `subject`), the previous games, the options."""
    from keyforge.infoset import SUBJECT, build_match_infoset

    info = build_match_infoset(match, viewer)
    blocks = {b.name: Block(b) for b in S.BLOCKS}

    def side(pid):
        return None if pid is None else (0 if pid == viewer else 1)

    d = match.pending_decision
    fmt, game_number, my_score, their_score, _, _, _ = info.match
    row = {"format": ("archon", "reversal", "adaptive").index(fmt) + 1, "game_number": game_number,
           "my_score": my_score, "their_score": their_score, "my_houses": _house_bits(info.my_houses),
           "their_houses": _house_bits(info.their_houses), "n_options": _log1p(len(info.options))}
    if d is not None:
        row.update(kind=_KIND_IDS[d.kind], intent=0 if d.intent is None else list(type(d.intent)).index(d.intent) + 1,
                   affects=0 if d.affects is None else list(type(d.affects)).index(d.affects) + 1,
                   optional=int(bool(d.optional)), min_n=_log1p(d.min_n), max_n=_log1p(d.max_n),
                   decider=side(d.player) + 1)
    blocks["global"].row(**row)
    last = match.finished_games[-1]
    for i, name in enumerate(info.entity_names):
        card = last._cards_by_id[info.entity_iids[i]]
        blocks["entity"].row(card=CARD_VOCAB[name], owner=side(card.owner),
                             flags=_FLAG_BIT["subject"] if info.flags[i] & SUBJECT else 0)
    _match_rows(side, match, blocks["match"])
    verbs = _V["verbs"]
    for verb, ptr, payload in info.options:
        name = VERB_NAMES[verb]
        r = {"verb": verbs.id(name), "pointer": S.NO_POINTER}
        if name == "first_player":
            r["first_player"] = 1 if payload == "first" else 2
        elif name == "bid":
            r["number"] = payload / 10.0
        blocks["option"].row(**r)
    return EncodedV2(blocks)


def encode_v2_for(obj, viewer: int, prefix=None, *, with_stop: bool = False) -> EncodedV2:
    """`encode_v2` / `encode_v2_match` for a `Game` or a `Match`, whichever
    it is and whichever point it's at."""
    from keyforge.match import Match

    if isinstance(obj, Match):
        live = obj.current_game
        if live is not None and not live.is_over:
            return encode_v2(live, viewer, match=obj, prefix=prefix, with_stop=with_stop)
        return encode_v2_match(obj, viewer)
    return encode_v2(obj, viewer, prefix=prefix, with_stop=with_stop)

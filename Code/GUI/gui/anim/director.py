"""Turns one `submit()`'s worth of raw log events into an animation
sequence: `Director.build(board, before, after, events) -> Playable`.

Each handler moves/pulses/particles the *specific* card(s) the event names
(found by instance id, so Mother x2 / Wild Wormhole x3 animate correctly);
`Board.settle_beat` always runs last as a catch-all so the board can never
end up visually out of sync with the engine, even for an event kind this
module doesn't special-case.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .. import settings as S
from ..sprites.overlays import Banner, FloatingText, Toast
from .easing import ease_in_out_sine, ease_out_back, ease_out_cubic
from .tween import Call, Delay, Parallel, Playable, Sequence, Tween

Handler = Callable[["Ctx"], Optional[Playable]]


class Ctx:
    __slots__ = ("board", "before", "after", "event")

    def __init__(self, board, before, after, event):
        self.board = board
        self.before = before
        self.after = after
        self.event = event

    @property
    def data(self):
        return self.event.data


_HANDLERS: Dict[str, Handler] = {}


def handler(kind: str):
    def deco(fn: Handler) -> Handler:
        _HANDLERS[kind] = fn
        return fn

    return deco


def _move_to(board, snap, iid, duration, ease=ease_out_cubic) -> Optional[Playable]:
    cs = snap.cards.get(iid)
    if cs is None:
        return None
    sprite = board.sprite_for(iid)
    x, y, rot, w, h = board.slot(cs, snap)
    sprite.set_size(w, h)
    return Parallel(
        Tween(sprite, "x", x, duration, ease),
        Tween(sprite, "y", y, duration, ease),
        Tween(sprite, "rot", rot, duration, ease),
    )


def _hud_pos(board, pid):
    r = board.layout.hud_rect(pid)
    return (r.left + 40, r.centery)


def _pop(sprite, up=1.15, down=1.0, dur=140):
    return Sequence(Parallel(Tween(sprite, "scale", up, dur)), Parallel(Tween(sprite, "scale", down, dur)))


# ------------------------------------------------------------------ draw ----


@handler("draw")
def _h_draw(ctx: Ctx):
    pid = ctx.data["player"]
    iids = ctx.data.get("iids", [])
    parts = []
    for i, iid in enumerate(iids):
        before_cs = ctx.before.cards.get(iid)
        sprite = ctx.board.sprite_for(iid)
        if before_cs is not None:
            bx, by, brot, bw, bh = ctx.board.slot(before_cs, ctx.before)
            sprite.snap_to(bx, by, brot)
        beat = _move_to(ctx.board, ctx.after, iid, S.T_DEAL, ease_out_back)
        if beat is not None:
            parts.append(Sequence(Delay(i * 70), beat))
    return Parallel(*parts) if parts else None


@handler("reshuffle")
def _h_reshuffle(ctx: Ctx):
    pid = ctx.data["player"]
    rect = ctx.board.layout.pile_rect(pid, "deck")

    def fx():
        ctx.board.particles.emit_swirl(rect.centerx, rect.centery, color=S.TEXT_DIM, n=10)

    return Sequence(Call(fx), Delay(S.T_RESHUFFLE * 0.5))


@handler("mulligan")
def _h_mulligan(ctx: Ctx):
    pid = ctx.data["player"]
    hand = ctx.before.zone_cards(pid, "hand")
    deck_rect = ctx.board.layout.pile_rect(pid, "deck")
    parts = []
    for cs in hand:
        sprite = ctx.board.sprite_for(cs.iid)
        parts.append(Parallel(Tween(sprite, "x", float(deck_rect.centerx), 260), Tween(sprite, "y", float(deck_rect.centery), 260)))
    return Sequence(Parallel(*parts) if parts else Delay(0), Delay(200))


# --------------------------------------------------------------- turn flow ----


@handler("choose_house")
def _h_choose_house(ctx: Ctx):
    pid = ctx.data["player"]
    house = ctx.data["house"]
    label = "You" if pid == ctx.board.viewer else "Your opponent"
    ctx.board.banners.append(Banner(f"{label} chooses {house}", color=S.HOUSE_COLORS.get(house, S.AEMBER), life_ms=S.T_BANNER))
    return Delay(220)


@handler("take_archive")
def _h_take_archive(ctx: Ctx):
    pid = ctx.data["player"]
    parts = []
    for cs in ctx.after.zone_cards(pid, "hand"):
        before_cs = ctx.before.cards.get(cs.iid)
        if before_cs is None or before_cs.zone != "archive":
            continue
        sprite = ctx.board.sprite_for(cs.iid)
        bx, by, brot, bw, bh = ctx.board.slot(before_cs, ctx.before)
        sprite.snap_to(bx, by, brot)
        beat = _move_to(ctx.board, ctx.after, cs.iid, S.T_ARCHIVE_TAKE)
        if beat is not None:
            parts.append(beat)
    return Parallel(*parts) if parts else None


@handler("forge_key")
def _h_forge_key(ctx: Ctx):
    pid = ctx.data["player"]
    hud = ctx.board.layout.hud_rect(pid)
    x, y = hud.left + 100 + (ctx.data["keys"] - 1) * 20, hud.centery

    def fx():
        ctx.board.particles.emit_burst_ring(x, y, color=S.KEY_GOLD, n=26)
        who = "You" if pid == ctx.board.viewer else "Your opponent"
        ctx.board.banners.append(Banner("Key Forged!", f"{who} now ha{'ve' if pid == ctx.board.viewer else 's'} {ctx.data['keys']} key(s)", color=S.KEY_GOLD, life_ms=S.T_KEY_FORGE))

    return Sequence(Call(fx), Delay(S.T_KEY_FORGE * 0.6))


# ------------------------------------------------------------- hand/play ----


@handler("play_card")
def _h_play_card(ctx: Ctx):
    iid = ctx.data["iid"]
    ctype = ctx.data.get("type")
    sprite = ctx.board.sprite_for(iid)
    if ctype == "Action":
        stage_x = S.PLAY_X + S.PLAY_W / 2
        stage_y = (S.BAND_PROMPT[0] + S.BAND_YOUR_CREATURES[0]) / 2
        return Sequence(
            Parallel(
                Tween(sprite, "x", stage_x, S.T_PLAY_ACTION_FLY, ease_out_back),
                Tween(sprite, "y", stage_y, S.T_PLAY_ACTION_FLY, ease_out_back),
                Tween(sprite, "scale", 1.3, S.T_PLAY_ACTION_FLY),
                Tween(sprite, "rot", 0.0, S.T_PLAY_ACTION_FLY),
            ),
            Delay(S.T_PLAY_ACTION_HOLD),
        )
    beat = _move_to(ctx.board, ctx.after, iid, S.T_PLAY_MOVE, ease_out_back)
    return beat


@handler("discard_from_hand")
@handler("discard")
@handler("discard_random")
def _h_discard(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    return _move_to(ctx.board, ctx.after, iid, S.T_DISCARD)


@handler("archive")
def _h_archive(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    return _move_to(ctx.board, ctx.after, iid, S.T_MOVE_ZONE)


@handler("return_to_hand")
def _h_return(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    return _move_to(ctx.board, ctx.after, iid, S.T_MOVE_ZONE)


@handler("shuffle_into_deck")
@handler("put_on_top")
@handler("put_on_bottom")
@handler("timetraveler_shuffle")
def _h_zone_move(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    return _move_to(ctx.board, ctx.after, iid, S.T_MOVE_ZONE)


@handler("purge")
def _h_purge(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    sprite = ctx.board.sprite_for(iid)
    ox, oy = sprite.x, sprite.y

    def fx():
        ctx.board.particles.emit_swirl(ox, oy, color=S.PURGE, n=14)

    move = _move_to(ctx.board, ctx.after, iid, S.T_PURGE)
    return Sequence(Call(fx), move) if move else Call(fx)


@handler("arise")
def _h_arise(ctx: Ctx):
    pid = ctx.data["player"]
    parts = []
    for iid in ctx.data.get("iids", []):
        beat = _move_to(ctx.board, ctx.after, iid, S.T_MOVE_ZONE)
        if beat is not None:
            parts.append(beat)
    return Parallel(*parts) if parts else None


@handler("help_from_future_self")
def _h_hffs(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    return _move_to(ctx.board, ctx.after, iid, S.T_MOVE_ZONE)


# ------------------------------------------------------------ reap/fight ----


@handler("reap")
def _h_reap(ctx: Ctx):
    iid = ctx.data["iid"]
    sprite = ctx.board.sprite_for(iid)

    def fx():
        ctx.board.particles.emit_sparks(sprite.x, sprite.y, n=8)
        ctx.board.floaters.append(FloatingText(sprite.x, sprite.y - 24, "+1 Æmber", S.AEMBER))

    return Sequence(Call(fx), _pop(sprite))


@handler("fight")
def _h_fight(ctx: Ctx):
    a_iid, t_iid = ctx.data["attacker_iid"], ctx.data["target_iid"]
    a_sprite = ctx.board.sprite_for(a_iid)
    t_sprite = ctx.board.sprite_for(t_iid)
    ax0, ay0 = a_sprite.x, a_sprite.y
    tx, ty = t_sprite.x, t_sprite.y

    def lunge_to():
        return ax0 + (tx - ax0) * 0.55

    def lunge_to_y():
        return ay0 + (ty - ay0) * 0.55

    lunge = Parallel(Tween(a_sprite, "x", lunge_to, 260, ease_in_out_sine), Tween(a_sprite, "y", lunge_to_y, 260, ease_in_out_sine))
    shake = Parallel(
        Tween(t_sprite, "x", lambda: tx + 5, 60), Tween(t_sprite, "x", lambda: tx - 5, 60),
    )
    back = Parallel(Tween(a_sprite, "x", lambda: ax0, 220, ease_in_out_sine), Tween(a_sprite, "y", lambda: ay0, 220, ease_in_out_sine))
    return Sequence(lunge, shake, back)


@handler("damage")
def _h_damage(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    sprite = ctx.board.sprite_for(iid)
    amount = ctx.data["amount"]
    ox = sprite.x

    def fx():
        ctx.board.floaters.append(FloatingText(sprite.x, sprite.y - 20, f"-{amount}", S.DANGER, big=True))

    shake = Sequence(
        Parallel(Tween(sprite, "x", lambda: ox + 6, 55)),
        Parallel(Tween(sprite, "x", lambda: ox - 6, 55)),
        Parallel(Tween(sprite, "x", lambda: ox + 3, 55)),
        Parallel(Tween(sprite, "x", lambda: ox, 55)),
    )
    return Sequence(Call(fx), shake)


@handler("heal")
def _h_heal(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    sprite = ctx.board.sprite_for(iid)
    amount = ctx.data["amount"]

    def fx():
        ctx.board.floaters.append(FloatingText(sprite.x, sprite.y - 20, f"+{amount}", S.HEAL))

    return Call(fx)


@handler("destroyed")
def _h_destroyed(ctx: Ctx):
    iid = ctx.data["iid"]
    sprite = ctx.board.sprite_for(iid)
    ox, oy = sprite.x, sprite.y

    def fx():
        ctx.board.particles.emit_embers(ox, oy, n=16)

    shrink = Parallel(Tween(sprite, "scale", 0.15, S.T_DESTROY), Tween(sprite, "alpha", 40.0, S.T_DESTROY))
    move = _move_to(ctx.board, ctx.after, iid, S.T_DESTROY * 0.6)
    tail = Parallel(shrink, move) if move else shrink
    return Sequence(Call(fx), tail)


# ------------------------------------------------------------------ aember ----


@handler("gain")
def _h_gain(ctx: Ctx):
    pid = ctx.data["player"]
    amount = ctx.data["amount"]
    x, y = _hud_pos(ctx.board, pid)

    def fx():
        ctx.board.particles.emit_sparks(x, y, n=6)
        ctx.board.floaters.append(FloatingText(x, y - 14, f"+{amount} Æ", S.AEMBER))

    return Call(fx)


@handler("steal")
def _h_steal(ctx: Ctx):
    frm, to = ctx.data["frm"], ctx.data["to"]
    amount = ctx.data["amount"]
    fx0, fy0 = _hud_pos(ctx.board, frm)
    fx1, fy1 = _hud_pos(ctx.board, to)

    def fx():
        ctx.board.particles.emit_sparks(fx0, fy0, color=S.DANGER, n=6)
        ctx.board.particles.emit_sparks(fx1, fy1, color=S.AEMBER, n=6)
        ctx.board.floaters.append(FloatingText(fx0, fy0 - 14, f"-{amount} Æ", S.DANGER))
        ctx.board.floaters.append(FloatingText(fx1, fy1 - 14, f"+{amount} Æ", S.AEMBER))

    return Call(fx)


@handler("capture")
def _h_capture(ctx: Ctx):
    iid = ctx.data.get("iid")
    amount = ctx.data["amount"]
    if iid is None:
        return None
    sprite = ctx.board.sprite_for(iid)

    def fx():
        ctx.board.particles.emit_sparks(sprite.x, sprite.y, n=6)
        ctx.board.floaters.append(FloatingText(sprite.x, sprite.y - 24, f"+{amount} captured", S.AEMBER))

    return Call(fx)


@handler("duration_effect")
def _h_duration_effect(ctx: Ctx):
    iid = ctx.data.get("iid")
    if iid is None:
        return None
    sprite = ctx.board.sprite_for(iid)

    def fx():
        ctx.board.particles.emit_sparks(sprite.x, sprite.y, color=S.PURGE, n=8)

    return Call(fx)


class Director:
    @staticmethod
    def build(board, before, after, events: List) -> Playable:
        board.apply_visual(after)
        beats: List[Playable] = []
        for ev in events:
            fn = _HANDLERS.get(ev.kind)
            if fn is None:
                beat = _move_to(board, after, ev.data.get("iid"), S.T_MOVE_ZONE) if ev.data.get("iid") is not None else None
            else:
                beat = fn(Ctx(board, before, after, ev))
            if beat is not None:
                beats.append(beat)
        beats.append(board.settle_beat(after, S.T_SETTLE))
        return Sequence(*beats)

    @staticmethod
    def game_over_beat(board, after) -> Playable:
        cx = S.PLAY_X + S.PLAY_W / 2

        def fx():
            board.particles.emit_confetti(cx, 40, n=60)

        return Sequence(Call(fx), Delay(S.T_GAME_OVER))

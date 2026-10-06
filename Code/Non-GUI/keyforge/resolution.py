"""The resolution stack as information (Agent Observation Plan, Part R, R7).

`resolution_view(game, viewer)` describes what is still left to resolve, one
entry per suspended frame, top (innermost) first:

- `routine` (a stable id, for a vocabulary), `pc`, `depth`, the source `line`;
- `site`: what the frame waits on -- a callee's source text, or "decision";
- `kind`: the ability kind, from the call site that started the frame
  (play, reap, fight, action, omni, destroyed, before_fight,
  destroyed_fighting, after_reap, after_fight, trigger), "turn" for the
  root, else "kernel" (a rules routine) or "effect" (a card-effect helper);
- `event`: the trigger event, for a trigger;
- `source`: the card whose ability it is, where there is one;
- `locals`: `(name, value)` pairs -- loop state, chosen cards, pending
  trigger batches and all;
- `ops`: the operations still reachable from `pc` (the compiler's static
  table: each choice with its intent and `affects`, each primitive step with
  its literal arguments, loops marked).

Values are plain data, redacted for `viewer`: a card the viewer may see is
`("card", instance id)`; one in a zone hidden from them is `("hidden",
zone, side)` -- "a hidden card from zone Z of side S". Sides are relative
to the viewer ("mine" / "theirs"). Abilities resolve face up, so the frames
themselves are public; only identities are redacted.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from . import vm
from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card
from .decision import Decision
from .effects.effect_object import EffectObject

_ACTIONS = (DiscardCard, Fight, PlayCard, Reap, UseAction, UseOmni)

# Zones whose contents both players see.
PUBLIC_ZONES = frozenset({"discard", "purged", "battleline", "artifacts", "attached", "limbo"})


def card_zones(game) -> Dict[int, Tuple[str, int]]:
    """instance id -> (zone kind, the player whose zone it is). A card in
    none of them is between zones ("limbo": being played, leaving play)."""
    out: Dict[int, Tuple[str, int]] = {}
    for pid, p in game.players.items():
        for kind, zone in (("hand", p.hand), ("deck", p.deck), ("discard", p.discard), ("archive", p.archive), ("purged", p.purged)):
            for c in zone.cards():
                out[c.instance_id] = (kind, pid)
        for c in p.play_area.creatures:
            out[c.instance_id] = ("battleline", pid)
            for u in getattr(c.type_object, "upgrades", ()):
                out[u.instance_id] = ("attached", pid)
            for u in c.under_cards:
                out[u.instance_id] = ("under", u.owner)
        for c in p.play_area.artifacts:
            out[c.instance_id] = ("artifacts", pid)
            for u in c.under_cards:
                out[u.instance_id] = ("under", u.owner)
    return out


def visible_to(game, zone: Tuple[str, int], viewer: int) -> bool:
    kind, pid = zone
    if kind in PUBLIC_ZONES:
        return True
    if kind in ("hand", "archive", "under"):
        return pid == viewer or (kind == "hand" and viewer in game.players[pid].hand_revealed_to)
    return False  # decks: nobody


class _Encoder:
    def __init__(self, game, viewer: int):
        self.game = game
        self.viewer = viewer
        self.zones = card_zones(game)

    def side(self, pid) -> str:
        return "mine" if pid == self.viewer else "theirs"

    def card(self, c: Card):
        zone = self.zones.get(c.instance_id, ("limbo", c.owner))
        if visible_to(self.game, zone, self.viewer):
            return ("card", c.instance_id)
        return ("hidden", zone[0], self.side(zone[1]))

    def value(self, x, depth: int = 0) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, Enum):
            return ("enum", f"{type(x).__name__}.{x.name}")
        if isinstance(x, Card):
            return self.card(x)
        t = type(x).__name__
        if t == "Player":
            return ("player", self.side(x.id))
        if t == "Game":
            return ("game",)
        if isinstance(x, (list, tuple)):
            return ("list", tuple(self.value(v, depth + 1) for v in x))
        if isinstance(x, (set, frozenset)):
            return ("set", tuple(sorted((self.value(v, depth + 1) for v in x), key=repr)))
        if isinstance(x, dict):
            return ("dict", tuple((self.value(k, depth + 1), self.value(v, depth + 1)) for k, v in x.items()))
        if isinstance(x, EffectObject):
            what = getattr(x, "event", None) or getattr(x, "kind", None) or getattr(x, "variable", None)
            src = self.card(x.source_card) if x.source_card is not None else None
            return ("effect", type(x).__name__, what, src)
        if isinstance(x, _ACTIONS):
            return ("action", type(x).__name__, self.card(x.card))
        if isinstance(x, EndTurn):
            return ("action", "EndTurn")
        if isinstance(x, Decision):
            return ("decision", x.kind.name)
        if callable(x):
            return ("fn", getattr(x, "__qualname__", type(x).__name__))
        return ("object", t)


# The call site text in the frame below -> the ability kind it starts.
_HOOK_KINDS = (
    (".on_play", "play"), (".on_reap", "reap"), (".on_fight", "fight"), (".on_action", "action"),
    (".on_omni", "omni"), (".on_destroyed_fighting", "destroyed_fighting"), (".on_destroyed", "destroyed"),
    (".on_before_fight", "before_fight"), ("trig.handler", "trigger"),
)
_LOCAL_SITE_KINDS = {  # (caller routine, callee text) for calls through a local
    ("Game._use_action", "effect"): "action",
    ("Game.use_artifact_ability", "ability"): "action",
    ("Game._reap", "extra"): "after_reap",
    ("Game._fight", "extra"): "after_fight",
    ("Game.destroy_cards", "extra"): "destroyed",
}
_KERNEL_MODULES = frozenset({"game", "match", "effects.steps"})

# `Game._caused` runs every ability (it notes the cause of the zone moves it
# makes, O1); the view leaves that wrapper out and reads the ability's kind
# off it: the hook it runs, else the routine that called it.
_CAUSED = "Game._caused"
_HOOK_ATTRS = (
    ("on_play", "play"), ("on_reap", "reap"), ("on_fight", "fight"), ("on_action", "action"),
    ("on_omni", "omni"), ("on_destroyed_fighting", "destroyed_fighting"), ("on_destroyed", "destroyed"),
    ("on_before_fight", "before_fight"),
)
_CAUSED_CALLER_KINDS = {
    "Game._fire_event": "trigger", "Game._run_play_trigger_check": "trigger",
    "Game._use_action": "action", "Game.use_artifact_ability": "action",
    "Game._reap": "after_reap", "Game._fight": "after_fight", "Game.destroy_cards": "destroyed",
}


def _caused_kind(caused, caller) -> str:
    src, fn = _local(caused, "source"), _local(caused, "fn")
    cdef = getattr(src, "card_def", None)
    for attr, kind in _HOOK_ATTRS:
        if cdef is not None and getattr(cdef, attr, None) is fn:
            return kind
    return _CAUSED_CALLER_KINDS.get(caller.routine.qualname if caller is not None else None, "effect")


def _ability_kind(frame, below) -> str:
    if below is None:
        return "turn"
    site = below.routine.site_of_pc.get(below.pc, "")
    for text, kind in _HOOK_KINDS:
        if text in site:
            return kind
    kind = _LOCAL_SITE_KINDS.get((below.routine.qualname, site))
    if kind is not None:
        return kind
    return "kernel" if frame.routine.module in _KERNEL_MODULES else "effect"


def _local(frame, name: str):
    r = frame.routine
    for slot in (name, f"_kfc_{name}"):
        if slot in r.local_names:
            v = frame.L[r.local_names.index(slot)]
            if slot.startswith("_kfc_"):
                try:
                    v = v.cell_contents
                except ValueError:
                    return None
            return None if v is vm.UNBOUND else v
    return None


def _role(slot: str) -> Optional[str]:
    """A frame slot's name as the view shows it, or None to leave it out."""
    if slot.startswith("_kfc_"):
        return slot[len("_kfc_"):]
    if slot.startswith("_kfs"):
        return "_loop_sequence"
    if slot.startswith("_kfi"):
        return "_loop_index"
    if slot.startswith("_kft"):
        return "_test"
    if slot.startswith("_kf"):
        return None
    return slot


def _index(frames, frame) -> int:
    for j, f in enumerate(frames):
        if f is frame:
            return j
    raise ValueError("frame not on the stack")


def resolution_view(game, viewer: int) -> List[Dict[str, Any]]:
    if game.is_over:
        return []
    if game.execution != "compiled":
        raise ValueError("the resolution view needs compiled execution (Part R): a native game's stack is suspended generators")
    full = game._machine.stack
    stack = [f for f in full if f.routine.qualname != _CAUSED]
    enc = _Encoder(game, viewer)
    out = []
    n = len(stack)
    for depth in range(n):
        i = n - 1 - depth  # top first
        frame = stack[i]
        below = stack[i - 1] if i > 0 else None
        r = frame.routine
        j = _index(full, frame)
        caused = full[j - 1] if j > 0 and full[j - 1].routine.qualname == _CAUSED else None
        event = None
        source = None
        if caused is not None:
            kind = _caused_kind(caused, below)
            v = _local(caused, "source")
            if isinstance(v, Card):
                source = v
        else:
            kind = _ability_kind(frame, below)
        if kind == "trigger" and below is not None:
            trig = _local(below, "trig")
            event = _local(below, "event_name") if below.routine.qualname == "Game._fire_event" else getattr(trig, "event", None)
            if trig is not None and trig.source_card is not None:
                source = trig.source_card
        if source is None:
            for name in ("card", "source_card", "attacker", "source"):
                v = _local(frame, name)
                if isinstance(v, Card):
                    source = v
                    break
        locals_ = []
        for slot, v in zip(r.local_names, frame.L):
            role = _role(slot)
            if role is None:
                continue
            if slot.startswith("_kfc_"):
                try:
                    v = v.cell_contents
                except ValueError:
                    continue
            if v is vm.UNBOUND or type(v).__name__ == "Game":
                continue
            locals_.append((role, enc.value(v)))
        out.append({
            "routine": r.rid, "pc": frame.pc, "depth": depth, "line": r.line_of_pc.get(frame.pc),
            "site": r.site_of_pc.get(frame.pc), "kind": kind, "event": event,
            "source": enc.card(source) if source is not None else None,
            "locals": tuple(locals_), "ops": r.ops_of_pc.get(frame.pc, ()),
        })
    return out

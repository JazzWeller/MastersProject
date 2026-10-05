"""Lasting effect objects: DurationEffect, InsteadEffect, TriggerEffect."""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

INFINITE = -1

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "=": lambda base, value: value,
}

# End-of-turn cleanups as data, not closures (Agent Interface Plan, Milestone
# E2: "cleanups as data" -- a closure captures a specific Card object by
# reference, which `Game.copy()`'s snapshot-copy backend can't rebind the
# way replaying naturally would; a `(operation, instance_id)` pair is plain,
# copyable data instead). Each `effects/named/*.py` file that needs a
# cleanup registers its own resolver here, at import time, keeping `game.py`
# and this module ignorant of any specific card's behavior.
_CLEANUP_OPERATIONS: Dict[str, Callable[[object, Optional[int]], None]] = {}


def register_cleanup_operation(name: str, resolver: Callable[[object, Optional[int]], None]) -> None:
    """`resolver(game, instance_id)` performs the cleanup -- `instance_id`
    is `None` for a game-level (not card-specific) cleanup."""
    if name in _CLEANUP_OPERATIONS:
        raise ValueError(f"cleanup operation {name!r} already registered")
    _CLEANUP_OPERATIONS[name] = resolver


def resolve_cleanup(game, operation: str, instance_id: Optional[int]) -> None:
    _CLEANUP_OPERATIONS[operation](game, instance_id)


def _remove_effects_from_source_op(game, instance_id: Optional[int]) -> None:
    game.active_effects.remove_from_source(game.card_by_id(instance_id))


register_cleanup_operation("remove_effects_from_source", _remove_effects_from_source_op)


class EffectObject:
    def __init__(self, source_card, controller: int):
        self.source_card = source_card
        self.controller = controller


class DurationEffect(EffectObject):
    """Modifies a player variable for a limited (or infinite) duration."""

    def __init__(
        self,
        source_card,
        controller: int,
        remaining_duration,
        player_affected: int,
        variable: str,
        op: str,
        value,
        conditional: Optional[Callable] = None,
    ):
        super().__init__(source_card, controller)
        self.remaining_duration = remaining_duration
        self.player_affected = player_affected
        self.variable = variable
        self.op = op
        self.value = value
        self.conditional = conditional

    def is_active(self, game) -> bool:
        if self.conditional is not None:
            return self.conditional(game)
        return True

    def apply(self, base, game):
        if not self.is_active(game):
            return base
        value = self.value(game) if callable(self.value) else self.value
        return _OPS[self.op](base, value)

    def tick(self) -> bool:
        """Advance one turn. Returns True if the effect should be removed."""
        if self.remaining_duration == INFINITE:
            return False
        self.remaining_duration -= 1
        return self.remaining_duration <= 0


class ModifierEffect(EffectObject):
    """A passive from one card that continuously affects OTHER cards'
    computed stats -- King of the Crag's -2 power to enemy Brobnar
    creatures, a flank-conditional fight-damage bonus (Valdr), a keyword
    grant to other creatures (Halacor). `handler(game, card, *extra)`
    returns the contribution for that card (0 / an empty frozenset if the
    modifier doesn't apply to it) -- the modifier itself decides who it
    applies to, so `Game.get_power`/`get_keywords`/etc. just sum or union
    every registered handler's result unconditionally."""

    def __init__(self, source_card, controller: int, kind: str, handler: Callable):
        super().__init__(source_card, controller)
        self.kind = kind
        self.handler = handler


class InsteadEffect(EffectObject):
    """Replaces a destroy/move/forge step. `handler(game, **kwargs) -> bool` returns True if it fired."""

    def __init__(self, source_card, controller: int, kind: str, handler: Callable):
        super().__init__(source_card, controller)
        self.kind = kind
        self.handler = handler


class TriggerEffect(EffectObject):
    """Fires when a matching game event occurs. `handler` is a generator function(game, event)."""

    def __init__(self, source_card, controller: int, event: str, handler: Callable, remaining_duration=INFINITE):
        super().__init__(source_card, controller)
        self.event = event
        self.handler = handler
        self.remaining_duration = remaining_duration

    def tick(self) -> bool:
        if self.remaining_duration == INFINITE:
            return False
        self.remaining_duration -= 1
        return self.remaining_duration <= 0


class ActiveEffectList:
    """`duration_effects_for(variable, player)` is by far the hottest query
    here (460k calls over 150 profiled games, Milestone L) -- indexed by
    `(variable, player_affected)` in `_duration_by_key`, kept up to date in
    `add`/`remove_from_source`/`end_of_turn_tick` (the only three places
    `duration_effects` ever changes -- see those methods' own comments) so
    a read is a dict lookup instead of a scan of every active effect.
    `variable`/`player_affected` are set once at construction and never
    mutated (DurationEffect.__init__), so an effect's index key never goes
    stale during its own lifetime."""

    def __init__(self):
        self.duration_effects: List[DurationEffect] = []
        self.trigger_effects = []
        self.instead_effects = []
        self.modifier_effects = []
        self._duration_by_key: Dict[Tuple[str, int], List[DurationEffect]] = defaultdict(list)

    def add(self, effect):
        if isinstance(effect, DurationEffect):
            self.duration_effects.append(effect)
            self._duration_by_key[(effect.variable, effect.player_affected)].append(effect)
        elif isinstance(effect, TriggerEffect):
            self.trigger_effects.append(effect)
        elif isinstance(effect, InsteadEffect):
            self.instead_effects.append(effect)
        elif isinstance(effect, ModifierEffect):
            self.modifier_effects.append(effect)

    def remove_from_source(self, card):
        removed = [e for e in self.duration_effects if e.source_card is card]
        self.duration_effects = [e for e in self.duration_effects if e.source_card is not card]
        for e in removed:
            self._drop_from_index(e)
        self.trigger_effects = [e for e in self.trigger_effects if e.source_card is not card]
        self.instead_effects = [e for e in self.instead_effects if e.source_card is not card]
        self.modifier_effects = [e for e in self.modifier_effects if e.source_card is not card]

    def _drop_from_index(self, effect: DurationEffect) -> None:
        key = (effect.variable, effect.player_affected)
        bucket = self._duration_by_key.get(key)
        if bucket:
            self._duration_by_key[key] = [e for e in bucket if e is not effect]

    def duration_effects_for(self, variable, player):
        return list(self._duration_by_key.get((variable, player), ()))

    def triggers_for(self, event):
        return [e for e in self.trigger_effects if e.event == event]

    def insteads_for(self, kind):
        return [e for e in self.instead_effects if e.kind == kind]

    def modifiers_for(self, kind):
        return [e for e in self.modifier_effects if e.kind == kind]

    def end_of_turn_tick(self):
        survivors = []
        for e in self.duration_effects:
            # `tick()` mutates `remaining_duration` -- call it exactly once
            # per effect, here, never again (e.g. in a second filter pass).
            if e.tick():
                self._drop_from_index(e)
            else:
                survivors.append(e)
        self.duration_effects = survivors
        self.trigger_effects = [e for e in self.trigger_effects if not e.tick()]

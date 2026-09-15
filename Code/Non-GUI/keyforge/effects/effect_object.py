"""Lasting effect objects: DurationEffect, InsteadEffect, TriggerEffect."""

from __future__ import annotations

import operator
from typing import Callable, Optional

INFINITE = -1

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "=": lambda base, value: value,
}


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
        return _OPS[self.op](base, self.value)

    def tick(self) -> bool:
        """Advance one turn. Returns True if the effect should be removed."""
        if self.remaining_duration == INFINITE:
            return False
        self.remaining_duration -= 1
        return self.remaining_duration <= 0


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
    def __init__(self):
        self.duration_effects = []
        self.trigger_effects = []
        self.instead_effects = []

    def add(self, effect):
        if isinstance(effect, DurationEffect):
            self.duration_effects.append(effect)
        elif isinstance(effect, TriggerEffect):
            self.trigger_effects.append(effect)
        elif isinstance(effect, InsteadEffect):
            self.instead_effects.append(effect)

    def remove_from_source(self, card):
        self.duration_effects = [e for e in self.duration_effects if e.source_card is not card]
        self.trigger_effects = [e for e in self.trigger_effects if e.source_card is not card]
        self.instead_effects = [e for e in self.instead_effects if e.source_card is not card]

    def duration_effects_for(self, variable, player):
        return [
            e
            for e in self.duration_effects
            if e.variable == variable and e.player_affected == player
        ]

    def triggers_for(self, event):
        return [e for e in self.trigger_effects if e.event == event]

    def insteads_for(self, kind):
        return [e for e in self.instead_effects if e.kind == kind]

    def end_of_turn_tick(self):
        self.duration_effects = [e for e in self.duration_effects if not e.tick()]
        self.trigger_effects = [e for e in self.trigger_effects if not e.tick()]

"""A bot that picks uniformly at random among the legal options."""

from __future__ import annotations

import random

from keyforge.enums import DecisionKind

from .base import Controller


class RandomBot(Controller):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def decide(self, view, decision):
        if decision.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            options = list(decision.options)
            self.rng.shuffle(options)
            n = self.rng.randint(decision.min_n, decision.max_n) if decision.kind == DecisionKind.CHOOSE_CARDS else len(options)
            return options[:n]
        return self.rng.choice(decision.options)

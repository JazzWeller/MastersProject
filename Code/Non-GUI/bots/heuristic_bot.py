"""A simple, rules-aware opponent that plays like a reasonable human.

RandomBot picks End Turn as often as anything else and mulligans at random,
which makes for an opponent that doesn't feel like KeyForge at all. This
bot keeps things deliberately simple and readable:

- choose the house that lets it do the most this turn;
- play every card it sensibly can, use artifacts/actions, reap with ready
  creatures, fight only when the fight is favorable;
- end the turn only when nothing useful is left.

It only reads the redacted `PlayerView`, like any other Controller, and is
seeded so games stay reproducible.
"""

from __future__ import annotations

import random

from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.cards.card import Card
from keyforge.enums import CardType, DecisionKind, House

from .base import Controller

# Cards whose effect is usually bad to fire blindly.
_CAREFUL_PLAYS = {"Gateway to Dis", "Three Fates", "Pawn Sacrifice", "One Last Job"}
_NEVER_USE = {"Timetraveler", "Lifeward"}  # shuffles itself away / sacrifices itself


def _power(card: Card) -> int:
    return getattr(card.type_object, "base_power", 0)


def _remaining(card: Card) -> int:
    to = card.type_object
    return getattr(to, "base_power", 0) - getattr(to, "damage", 0)


class HeuristicBot(Controller):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ helpers ----

    def _house_score(self, view, house) -> float:
        me = view.me()
        score = 0.0
        for c in me.hand or []:
            if c.house == house:
                score += 1.0
        for c in me.creatures:
            if c.house == house:
                score += 1.5
        for c in me.artifacts:
            if c.house == house and (c.card_def.on_action is not None):
                score += 0.75
        return score + self.rng.random() * 0.01

    def _is_good_play(self, view, card: Card) -> bool:
        me, opp = view.me(), view.opponent()
        name = card.name
        if name == "Gateway to Dis":
            return len(opp.creatures) >= len(me.creatures) + 2
        if name == "Three Fates":
            ranked = sorted(me.creatures + opp.creatures, key=_power, reverse=True)[:3]
            return sum(1 for c in ranked if c in opp.creatures) >= 2
        if name == "Pawn Sacrifice":
            return bool(me.creatures) and len(opp.creatures) >= 1
        if name == "One Last Job":
            shadows = [c for c in me.creatures if c.house == House.SHADOWS]
            return len(shadows) >= 2 and opp.aember >= len(shadows)
        if name == "Too Much To Protect":
            return opp.aember > 6
        if name in ("The Terror",):
            return True
        return True

    def _fight_value(self, attacker: Card, target: Card) -> float:
        """Positive when the fight is worth it."""
        kills = _power(attacker) >= _remaining(target)
        survives = attacker.Skirmish or _power(target) < _remaining(attacker)
        elusive_blocks = target.Elusive and not target.fought_this_turn
        if elusive_blocks:
            return -1.0
        if kills and survives:
            return 3.0 + _power(target) * 0.1 + target.aember_captured
        if kills and not survives:
            return 1.0 if _power(target) > _power(attacker) else -0.5
        return -1.0

    # ------------------------------------------------------------ decide ----

    def decide(self, view, decision):
        kind = decision.kind
        opts = list(decision.options)
        me, opp = view.me(), view.opponent()

        if kind == DecisionKind.MULLIGAN:
            hand = me.hand or []
            best = max((sum(1 for c in hand if c.house == h) for h in {c.house for c in hand}), default=0)
            return True if best < 2 and True in opts else False

        if kind == DecisionKind.CHOOSE_HOUSE:
            return max(opts, key=lambda h: self._house_score(view, h))

        if kind == DecisionKind.TAKE_ARCHIVE:
            return True

        if kind == DecisionKind.CHOOSE_FLANK:
            return "right" if "right" in opts else opts[0]

        if kind == DecisionKind.YES_NO:
            return True

        if kind == DecisionKind.ORDER_EFFECTS:
            return opts

        if kind == DecisionKind.CHOOSE_HOUSE_FOR_EFFECT:
            prompt = decision.prompt.lower()
            if "arise" in prompt:
                return max(opts, key=lambda h: sum(1 for c in me.discard if c.house == h and c.type == CardType.CREATURE))
            # Control the Weak: force the house they have the fewest creatures in.
            return min(opts, key=lambda h: sum(1 for c in opp.creatures if c.house == h))

        if kind == DecisionKind.CHOOSE_ACTION:
            return self._choose_action(view, opts)

        if kind == DecisionKind.CHOOSE_CARDS:
            return self._choose_cards(view, decision, opts)

        return self.rng.choice(opts)

    def _choose_action(self, view, opts):
        me, opp = view.me(), view.opponent()
        # 1. Play cards: creatures and artifacts first (board presence), then actions.
        plays = [o for o in opts if isinstance(o, PlayCard) and self._is_good_play(view, o.card)]
        type_rank = {CardType.CREATURE: 0, CardType.ARTIFACT: 1, CardType.UPGRADE: 2, CardType.ACTION: 3}
        if plays:
            plays.sort(key=lambda o: (type_rank.get(o.card.type, 4), -o.card.aember_on_play))
            return plays[0]
        # 2. Use card abilities.
        uses = [
            o for o in opts
            if isinstance(o, (UseAction, UseOmni)) and o.card.name not in _NEVER_USE
        ]
        if uses:
            return uses[0]
        # 3. Favorable fights, then reaps.
        fights = [o for o in opts if isinstance(o, Fight)]
        best_fight, best_value = None, 0.0
        for f in fights:
            value = max((self._fight_value(f.card, t) for t in opp.creatures), default=-1.0)
            if value > best_value:
                best_fight, best_value = f, value
        if best_fight is not None:
            return best_fight
        reaps = [o for o in opts if isinstance(o, Reap)]
        if reaps:
            return reaps[0]
        # 4. Discard cards that couldn't be played anyway (thins the hand for the draw step).
        plays_possible = {id(o.card) for o in opts if isinstance(o, PlayCard)}
        discards = [o for o in opts if isinstance(o, DiscardCard) and id(o.card) not in plays_possible]
        if discards:
            return discards[0]
        return next(o for o in opts if isinstance(o, EndTurn))

    def _choose_cards(self, view, decision, opts):
        me, opp = view.me(), view.opponent()
        prompt = decision.prompt.lower()
        n = decision.max_n if decision.max_n >= decision.min_n else decision.min_n

        def pick(sorted_opts, count):
            return list(sorted_opts[:count])

        cards = [o for o in opts if isinstance(o, Card)]
        if len(cards) != len(opts):
            # Non-card options: discard pile ids, "reap"/"fight"/"action", etc.
            if all(isinstance(o, int) and not isinstance(o, bool) for o in opts):
                return [opp.id] if opp.id in opts else [opts[0]]
            for preferred in ("fight", "reap", "action"):
                if preferred in opts:
                    if preferred == "fight" and not opp.creatures:
                        continue
                    return [preferred]
            return opts[: max(decision.min_n, 1)]

        mine = set(id(c) for c in me.creatures + me.artifacts + (me.hand or []) + me.discard + (me.archive or []))

        if "sacrifice" in prompt:
            return pick(sorted(cards, key=_power), decision.min_n or 1)
        if "to fight" in prompt:
            attacker = next((c for c in me.creatures if f"for {c.name.lower()} to fight" in prompt), None)
            if attacker is not None:
                return [max(cards, key=lambda t: self._fight_value(attacker, t))]
            return [min(cards, key=_remaining)]
        if "archive" in prompt or "discard a card" in prompt:
            house = me.selected_house
            ranked = sorted(cards, key=lambda c: (c.house == house, c.aember_on_play))
            return pick(ranked, max(decision.min_n, 1))
        if "heal" in prompt or "attach" in prompt:
            friendly = [c for c in cards if id(c) in mine]
            return [friendly[0] if friendly else cards[0]]
        # Default: hurt the opponent -- prefer their cards, strongest first.
        ranked = sorted(cards, key=lambda c: (id(c) in mine, -_power(c)))
        count = max(decision.min_n, min(n, len([c for c in ranked if id(c) not in mine]) or decision.min_n))
        return pick(ranked, count)

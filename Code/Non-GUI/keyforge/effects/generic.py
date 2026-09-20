"""Reusable generic effect factories, used to build the per-card effects in
`cards/card_data.py`. Each factory returns a generator function with the
signature `(game, card) -> generator` suitable for use as a CardDef hook
(on_play, on_reap, on_fight, on_action, on_omni)."""

from __future__ import annotations

from ..effects.effect_object import DurationEffect, INFINITE, TriggerEffect
from . import steps


def opponent_of(game, card):
    return game.players[3 - card.controller]


def controller_of(game, card):
    return game.players[card.controller]


def gain_n(n: int):
    def effect(game, card):
        steps.gain(game, controller_of(game, card), n)
        return
        yield

    return effect


def lose_n(n: int):
    def effect(game, card):
        steps.lose(game, opponent_of(game, card), n)
        return
        yield

    return effect


def steal_n(n: int):
    def effect(game, card):
        steps.steal(game, opponent_of(game, card), controller_of(game, card), n, source=card)
        return
        yield

    return effect


def archive_n(n: int):
    def effect(game, card):
        player = controller_of(game, card)
        for i in range(n):
            if not player.hand.cards():
                what = "archives nothing" if i == 0 else f"archives only {i} of {n} cards"
                steps.shortfall(game, card, f"{what}: {{pos:{player.id}}} hand is empty", "Hand is empty")
                break
            options = player.hand.cards()
            choice = yield from game.choose_cards(
                player.id, f"Archive a card ({card.name})", options, 1, 1
            )
            steps.archive_card(game, player, choice[0])

    return effect


def draw_n(n: int):
    def effect(game, card):
        steps.draw(game, controller_of(game, card), n, source=card)
        return
        yield

    return effect


def heal_self_n(n: int):
    def effect(game, card):
        steps.heal(game, card, n)
        return
        yield

    return effect


def capture_n(n: int):
    def effect(game, card):
        steps.capture(game, card, n)
        return
        yield

    return effect


def deal_damage_to_chosen(n: int, targets="any"):
    def effect(game, card):
        player = controller_of(game, card)
        options = game.all_creatures(targets, card)
        if not options:
            kind = {"friendly": "friendly creatures", "enemy": "enemy creatures"}.get(targets, "creatures")
            steps.shortfall(game, card, f"deals no damage: there are no {kind} in play", "No creature to damage")
            return
        choice = yield from game.choose_cards(
            player.id, f"Deal {n} damage ({card.name})", options, 1, 1
        )
        steps.deal_damage(game, choice[0], n)
        yield from game.check_destroyed(choice)

    return effect


def move_aember_to_card(n: int = 1):
    """Pocket Universe, Safe Place: 'Action: Move N Æ from your pool to
    this card' -- paired with `spendable_for_keys=True` on the CardDef."""

    def effect(game, card):
        player = controller_of(game, card)
        amount = min(n, player.aember)
        if amount <= 0:
            steps.shortfall(game, card, f"moves nothing: {{pos:{player.id}}} Æmber pool is empty", "Pool is empty")
            return
        player.aember -= amount
        card.aember_stored += amount
        game.log.add("move_aember", player=player.id, card=card.name, iid=card.instance_id, amount=amount, to="card")
        return
        yield

    return effect


def choose_most_powerful(game, pid, creatures, prompt):
    """Yields the single most powerful creature among `creatures`, letting
    `pid` break a tie for the max. Returns None if `creatures` is empty."""
    if not creatures:
        return None
    max_power = max(game.get_power(c) for c in creatures)
    tied = [c for c in creatures if game.get_power(c) == max_power]
    if len(tied) == 1:
        return tied[0]
    choice = yield from game.choose_cards(pid, prompt, tied, 1, 1)
    return choice[0]


def deal_damage_to_chosen_with_splash(main: int, splash: int, targets="any"):
    """'Deal `main` damage to a creature with `splash` damage splash': the
    chosen creature's neighbors (at the moment of the hit, before anything
    dies) also take `splash` damage (Lava Ball)."""

    def effect(game, card):
        player = controller_of(game, card)
        options = game.all_creatures(targets, card)
        if not options:
            kind = {"friendly": "friendly creatures", "enemy": "enemy creatures"}.get(targets, "creatures")
            steps.shortfall(game, card, f"deals no damage: there are no {kind} in play", "No creature to damage")
            return
        choice = yield from game.choose_cards(
            player.id, f"Deal {main} damage with {splash} splash ({card.name})", options, 1, 1
        )
        target = choice[0]
        area = game.find_play_area(target)
        neighbors = area.neighbors(target) if area is not None else []
        steps.deal_damage(game, target, main)
        for n in neighbors:
            steps.deal_damage(game, n, splash)
        yield from game.check_destroyed([target] + neighbors)

    return effect


def duration_effect(variable: str, op: str, value, duration, scope: str):
    """scope: 'self', 'enemy', or 'both' (relative to the source card's controller)."""

    def effect(game, card):
        controller = card.controller
        targets = []
        if scope in ("self", "both"):
            targets.append(controller)
        if scope in ("enemy", "both"):
            targets.append(3 - controller)
        for player_affected in targets:
            game.active_effects.add(
                DurationEffect(card, controller, duration, player_affected, variable, op, value)
            )
        game.log.add(
            "duration_effect", card=card.name, iid=card.instance_id, variable=variable, op=op, value=value,
            player=controller, affected=targets,
        )
        return
        yield

    return effect

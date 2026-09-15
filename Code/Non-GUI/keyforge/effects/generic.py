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


def steal_n(n: int):
    def effect(game, card):
        steps.steal(game, opponent_of(game, card), controller_of(game, card), n)
        return
        yield

    return effect


def archive_n(n: int):
    def effect(game, card):
        player = controller_of(game, card)
        for _ in range(n):
            if not player.hand.cards():
                break
            options = player.hand.cards()
            choice = yield from game.choose_cards(
                player.id, f"Archive a card ({card.name})", options, 1, 1
            )
            steps.archive_card(game, player, choice[0])

    return effect


def draw_n(n: int):
    def effect(game, card):
        steps.draw(game, controller_of(game, card), n)
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
            return
        choice = yield from game.choose_cards(
            player.id, f"Deal {n} damage ({card.name})", options, 1, 1
        )
        steps.deal_damage(game, choice[0], n)
        yield from game.check_destroyed(choice)

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
            "duration_effect", card=card.name, iid=card.instance_id, variable=variable, op=op, value=value
        )
        return
        yield

    return effect

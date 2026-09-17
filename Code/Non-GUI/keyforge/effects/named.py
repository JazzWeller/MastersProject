"""Bespoke per-card effects that don't fit the generic factories."""

from __future__ import annotations

from ..enums import CardType, House
from ..effects.effect_object import DurationEffect, TriggerEffect, INFINITE
from . import steps
from .generic import controller_of, opponent_of


# ---------------------------------------------------------------- Dis ----

def arise(game, card):
    player = controller_of(game, card)
    houses = game.player_houses(player.id)
    chosen = yield from game.choose_house(player.id, "Arise: choose one of your houses", houses)
    creatures = [c for c in player.discard.cards() if c.house == chosen and c.type == CardType.CREATURE]
    for c in creatures:
        player.discard.remove(c)
        player.hand.add(c)
    player.chains = min(24, player.chains + 1)
    game.log.add("arise", player=player.id, house=chosen.value, n=len(creatures), iids=[c.instance_id for c in creatures])


def control_the_weak(game, card):
    opponent = opponent_of(game, card)
    houses = game.player_houses(opponent.id)
    chosen = yield from game.choose_house(card.controller, "Control the Weak: choose the enemy's house", houses)
    game.active_effects.add(
        DurationEffect(card, card.controller, 2, opponent.id, "HouseSelection", "=", chosen)
    )


def creeping_oblivion(game, card):
    player_id = card.controller
    piles = []
    if game.players[1].discard.cards():
        piles.append(1)
    if game.players[2].discard.cards():
        piles.append(2)
    if not piles:
        return
    pile_choice = yield from game.choose_cards(player_id, "Choose a discard pile", piles, 1, 1)
    target_player = game.players[pile_choice[0]]
    options = target_player.discard.cards()
    n_max = min(2, len(options))
    choice = yield from game.choose_cards(
        player_id, "Purge 0-2 cards from that discard pile", options, 0, n_max
    )
    for c in choice:
        steps.purge(game, c)


def dominator_bauble(game, card):
    player = controller_of(game, card)
    # Only creatures that can actually be used: picking an exhausted one (or
    # one past the rule of six) would silently waste the Bauble.
    options = [
        c for c in player.play_area.creatures
        if not c.Exhausted and game._rule_of_six_ok(player, c.name)
    ]
    if not options:
        return
    choice = yield from game.choose_cards(player.id, "Choose a friendly creature to use", options, 1, 1)
    yield from steps.use_creature(game, choice[0])


def dust_imp_destroyed(game, card):
    steps.gain(game, game.players[card.owner], 2)
    return
    yield


def ember_imp_register(game, card):
    # Deliberately follows the spec, not the printed card: CardPlayedLimit
    # counts only cards played from hand, so effect plays (Wild Wormhole)
    # don't count toward Ember Imp's limit. Confirmed as the intended
    # behavior by the project owner; see tests/test_rules_fixes.py.
    effect = DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "CardPlayedLimit", "=", 2)
    card._ember_imp_effect = effect
    game.active_effects.add(effect)


def ember_imp_unregister(game, card):
    game.active_effects.remove_from_source(card)


def gateway_to_dis(game, card):
    player = controller_of(game, card)
    targets = game.all_creatures("any", card)
    yield from game.destroy_cards(targets)
    player.chains = min(24, player.chains + 3)


def guardian_demon(game, card):
    player = controller_of(game, card)
    creatures = game.all_creatures("any", card)
    # Healing an undamaged creature heals 0 and deals 0, so only offer ones
    # that can actually be healed.
    options = [c for c in creatures if c.type_object.damage > 0]
    if not options:
        return
    heal_choice = yield from game.choose_cards(player.id, "Guardian Demon: heal up to 2 from a creature", options, 1, 1)
    healed = steps.heal(game, heal_choice[0], 2)
    if healed <= 0:
        return
    # Printed text: deal the damage to *another* creature. None -> no damage.
    damage_targets = [c for c in creatures if c is not heal_choice[0]]
    if not damage_targets:
        return
    dmg_choice = yield from game.choose_cards(player.id, f"Deal {healed} damage to another creature", damage_targets, 1, 1)
    steps.deal_damage(game, dmg_choice[0], healed)
    yield from game.check_destroyed(dmg_choice)


def lifeward_omni(game, card):
    yield from steps.sacrifice(game, card)
    game.active_effects.add(
        DurationEffect(card, card.controller, 2, 3 - card.controller, "CanPlayCreatures", "=", False)
    )


def shooler_play(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 4:
        steps.steal(game, opponent, controller_of(game, card), 1)
    return
    yield


def snudge(game, card):
    player = controller_of(game, card)
    artifacts = game.players[1].play_area.artifacts + game.players[2].play_area.artifacts
    flank_creatures = [
        c
        for pid in (1, 2)
        for c in game.players[pid].play_area.creatures
        if game.players[pid].play_area.is_flank(c)
    ]
    options = artifacts + flank_creatures
    if not options:
        return
    choice = yield from game.choose_cards(player.id, "Snudge: return an artifact or a flank creature", options, 1, 1)
    steps.return_to_hand(game, choice[0])


def succubus_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "DrawUpToLimit", "-", 1)
    game.active_effects.add(effect)


def the_terror_play(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember == 0:
        steps.gain(game, controller_of(game, card), 2)
    return
    yield


def three_fates(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        return
    remaining = list(targets)
    chosen = []
    for _ in range(min(3, len(remaining))):
        max_power = max(game.get_power(c) for c in remaining)
        tied = [c for c in remaining if game.get_power(c) == max_power]
        if len(tied) > 1:
            pick = yield from game.choose_cards(
                card.controller, "Three Fates: choose among tied creatures", tied, 1, 1
            )
            picked = pick[0]
        else:
            picked = tied[0]
        chosen.append(picked)
        remaining.remove(picked)
    yield from game.destroy_cards(chosen)


# -------------------------------------------------------------- Logos ----

def help_from_future_self(game, card):
    player = controller_of(game, card)
    found = None
    for c in player.deck.cards():
        if c.name == "Timetraveler":
            found = c
            break
    if found is not None:
        player.deck.remove(found)
    else:
        for c in player.discard.cards():
            if c.name == "Timetraveler":
                found = c
                break
        if found is not None:
            player.discard.remove(found)
    if found is not None:
        player.hand.add(found)
        game.log.add("help_from_future_self", player=player.id, found=True, iid=found.instance_id)
    discard_cards = player.discard.take_all()
    if discard_cards:
        player.deck.shuffle_in(discard_cards, game.rng)
    return
    yield


def library_access_play(game, card):
    # Library Access is already removed from hand (it's the card currently
    # being played) and hasn't been placed anywhere yet, so purge it directly
    # rather than via the generic purge() step, which expects to find and
    # remove the card from a zone first.
    game.players[card.owner].purged.add(card)
    game.log.add("purge", card=card.name, iid=card.instance_id, owner=card.owner)
    effect = TriggerEffect(card, card.controller, "card_played", library_access_trigger, remaining_duration=1)
    game.active_effects.add(effect)
    return
    yield


def library_access_trigger(game, event):
    player = game.players[event["player"]]
    steps.draw(game, player, 1)
    return
    yield


def phase_shift(game, card):
    controller_of(game, card).NonLogosCardsPlayable += 1
    return
    yield


def mother_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, card.controller, "DrawUpToLimit", "+", 1)
    game.active_effects.add(effect)


def quixo_register(game, card):
    card.Skirmish = True


def quixo_unregister(game, card):
    card.Skirmish = False


def quixo_after_fight(game, card):
    steps.draw(game, controller_of(game, card), 1)
    return
    yield


def the_howling_pit_register(game, card):
    # Printed text: "During their 'draw cards' step, each player refills their
    # hand to 1 additional card." That is a DrawUpToLimit increase (like
    # Mother), not the spec's CardDrawModifier, which would draw nothing when
    # a hand is already full. Follows the card over the spec, as approved.
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 1, "DrawUpToLimit", "+", 1))
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 2, "DrawUpToLimit", "+", 1))


def timetraveler_action(game, card):
    player = controller_of(game, card)
    if player.play_area.remove(card):
        game.leave_play(card)
        player.deck.shuffle_in([card], game.rng)
        game.log.add("timetraveler_shuffle", player=player.id, iid=card.instance_id)
    return
    yield


def titan_mechanic_conditional(card):
    def cond(game):
        return (
            not card.destroyed
            and card in game.players[card.controller].play_area.creatures
            and game.players[card.controller].play_area.is_flank(card)
        )

    return cond


def titan_mechanic_register(game, card):
    cond = titan_mechanic_conditional(card)
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 1, "KeyForgeCost", "-", 1, conditional=cond))
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 2, "KeyForgeCost", "-", 1, conditional=cond))


def wild_wormhole(game, card):
    player = controller_of(game, card)
    top = player.deck.peek_top()
    if top is None:
        return
    yield from game.play_card_from_deck_top(player, top, ignore_house=True)


# ------------------------------------------------------------ Shadows ----

def bait_and_switch(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    while True:
        ok = steps.steal(game, opponent, player, 1)
        if not ok:
            break
        if not (player.aember < opponent.aember):
            break
    return
    yield


def booby_trap(game, card):
    non_flank = [
        c
        for pid in (1, 2)
        for c in game.players[pid].play_area.creatures
        if not game.players[pid].play_area.is_flank(c)
    ]
    if not non_flank:
        return
    choice = yield from game.choose_cards(card.controller, "Booby Trap: choose a non-flank creature", non_flank, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 4)
    owner_area = game.players[target.controller].play_area
    for neighbor in owner_area.neighbors(target):
        steps.deal_damage(game, neighbor, 2)
    yield from game.check_destroyed([target] + owner_area.neighbors(target))


def duskrunner_register(game, card):
    host = card.type_object.host
    if host is not None:
        host.extra_triggers["after_reap"].append(duskrunner_effect)


def duskrunner_unregister(game, card):
    host = card.type_object.host
    if host is not None and duskrunner_effect in host.extra_triggers["after_reap"]:
        host.extra_triggers["after_reap"].remove(duskrunner_effect)


def duskrunner_effect(game, host_card):
    controller = game.players[host_card.controller]
    steps.steal(game, game.players[3 - host_card.controller], controller, 1)
    return
    yield


def ghostly_hand(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember == 1:
        steps.steal(game, opponent, controller_of(game, card), 1)
    return
    yield


def lights_out(game, card):
    opponent = opponent_of(game, card)
    options = list(opponent.play_area.creatures)
    if not options:
        return
    # Printed text: "Return 2 enemy creatures" -- exactly 2, or all if fewer.
    n = min(2, len(options))
    choice = yield from game.choose_cards(card.controller, f"Lights Out: return {n} enemy creature{'s' if n != 1 else ''}", options, n, n)
    for c in choice:
        steps.return_to_hand(game, c)


def nerve_blast(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    ok = steps.steal(game, opponent, player, 1)
    if not ok:
        return
    targets = game.all_creatures("any", card)
    if not targets:
        return
    choice = yield from game.choose_cards(player.id, "Nerve Blast: deal 2 damage", targets, 1, 1)
    steps.deal_damage(game, choice[0], 2)
    yield from game.check_destroyed(choice)


def noddy_action(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 1)
    return
    yield


def old_bruno_play(game, card):
    steps.capture(game, card, 3)
    return
    yield


def one_last_job(game, card):
    player = controller_of(game, card)
    targets = [c for c in player.play_area.creatures if c.house == House.SHADOWS]
    n = 0
    for c in list(targets):
        if steps.purge(game, c):
            n += 1
    if n:
        steps.steal(game, opponent_of(game, card), player, n)
    return
    yield


def oubliette(game, card):
    targets = [c for c in game.all_creatures("any", card) if game.get_power(c) <= 3]
    if not targets:
        return
    choice = yield from game.choose_cards(card.controller, "Oubliette: purge a creature (power <= 3)", targets, 1, 1)
    steps.purge(game, choice[0])


def pawn_sacrifice(game, card):
    player = controller_of(game, card)
    options = list(player.play_area.creatures)
    if not options:
        return
    choice = yield from game.choose_cards(player.id, "Pawn Sacrifice: sacrifice a friendly creature", options, 1, 1)
    victim = choice[0]
    ok = yield from steps.sacrifice(game, victim)
    if not ok:
        return
    remaining_targets = game.all_creatures("any", card)
    if not remaining_targets:
        return
    if len(remaining_targets) == 1:
        targets = remaining_targets
    else:
        targets = yield from game.choose_cards(
            player.id, "Pawn Sacrifice: choose 2 different creatures", remaining_targets, 2, 2
        )
    for t in targets:
        steps.deal_damage(game, t, 3)
    yield from game.check_destroyed(targets)


def relentless_whispers(game, card):
    player = controller_of(game, card)
    targets = game.all_creatures("any", card)
    if not targets:
        return
    choice = yield from game.choose_cards(player.id, "Relentless Whispers: deal 2 damage", targets, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 2)
    destroyed = yield from game.check_destroyed([target])
    if target in destroyed:
        steps.steal(game, opponent_of(game, card), player, 1)


def silvertooth_play(game, card):
    steps.ready(game, card)
    return
    yield


def subtle_maul(game, card):
    steps.discard_random(game, opponent_of(game, card))
    return
    yield


def too_much_to_protect(game, card):
    opponent = opponent_of(game, card)
    amount = max(opponent.aember - 6, 0)
    steps.steal(game, opponent, controller_of(game, card), amount)
    return
    yield


def urchin_play(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 1)
    return
    yield


def bad_penny_destroyed(game, card):
    card.destined_zone = "hand"
    return
    yield


def elusive_register(game, card):
    card.Elusive = True


def elusive_unregister(game, card):
    card.Elusive = False


def sloppy_labwork(game, card):
    player = controller_of(game, card)
    if player.hand.cards():
        options = player.hand.cards()
        choice = yield from game.choose_cards(player.id, "Sloppy Labwork: archive a card", options, 1, 1)
        steps.archive_card(game, player, choice[0])
    if player.hand.cards():
        options = player.hand.cards()
        choice = yield from game.choose_cards(player.id, "Sloppy Labwork: discard a card", options, 1, 1)
        steps.discard_from_hand(game, player, choice[0])

"""Bespoke per-card effects for Shadows cards that don't fit the generic factories."""

from __future__ import annotations

from ...enums import CardType, House
from ..effect_object import DurationEffect, InsteadEffect, TriggerEffect, INFINITE
from .. import steps
from ..generic import controller_of, opponent_of


def bait_and_switch(game, card):
    # MRB 18.3 errata over the spec, as approved by the project owner: "If
    # your opponent has more Æ than you, steal 1Æ. Repeat the preceding
    # effect if your opponent still has more Æ than you." The FAQ is
    # explicit that this repeats AT MOST ONCE (two steals total), not
    # until the gap closes -- the check before the first steal, and the
    # one check after it, are the only two chances to steal.
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    if not opponent.aember > player.aember:
        steps.shortfall(
            game, card,
            f"steals nothing: {{pos:{opponent.id}}} Æmber ({opponent.aember}) isn't more than {{mine:{player.id}}} ({player.aember})",
            "Enemy doesn't have more Æmber",
        )
        return
    steps.steal(game, opponent, player, 1)
    if opponent.aember > player.aember:
        steps.steal(game, opponent, player, 1)
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
        steps.shortfall(game, card, "deals no damage: every creature in play is on a flank", "No non-flank creature")
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
    steps.steal(game, game.players[3 - host_card.controller], controller, 1, source="Duskrunner")
    return
    yield


def ghostly_hand(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember == 1:
        steps.steal(game, opponent, controller_of(game, card), 1)
    else:
        steps.shortfall(game, card, f"steals nothing: {{pos:{opponent.id}}} Æmber is {opponent.aember}, and it steals only when that is exactly 1", "Needs exactly 1 enemy Æmber")
    return
    yield


def lights_out(game, card):
    opponent = opponent_of(game, card)
    options = list(opponent.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "returns nothing: there are no enemy creatures in play", "No enemy creatures")
        return
    # Printed text: "Return 2 enemy creatures" -- exactly 2, or all if fewer.
    n = min(2, len(options))
    if n < 2:
        steps.shortfall(game, card, "returns only 1 creature: it was the only enemy creature in play", "Only 1 enemy creature")
    choice = yield from game.choose_cards(card.controller, f"Lights Out: return {n} enemy creature{'s' if n != 1 else ''}", options, n, n)
    for c in choice:
        steps.return_to_hand(game, c)


def nerve_blast(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    ok = steps.steal(game, opponent, player, 1)
    if not ok:
        steps.shortfall(game, card, f"steals nothing, so deals no damage: {{pos:{opponent.id}}} Æmber pool is empty", "Nothing to steal")
        return
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "steals 1 but deals no damage: there are no creatures in play", "No creature to damage")
        return
    choice = yield from game.choose_cards(player.id, "Nerve Blast: deal 2 damage", targets, 1, 1)
    steps.deal_damage(game, choice[0], 2)
    yield from game.check_destroyed(choice)


def noddy_action(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 1, source=card)
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
        steps.steal(game, opponent_of(game, card), player, n, source=card)
    else:
        steps.shortfall(game, card, "purges nothing, so steals nothing: there are no friendly Shadows creatures in play", "No Shadows creatures")
    return
    yield


def oubliette(game, card):
    targets = [c for c in game.all_creatures("any", card) if game.get_power(c) <= 3]
    if not targets:
        steps.shortfall(game, card, "purges nothing: no creature in play has power 3 or less", "No creature with power 3 or less")
        return
    choice = yield from game.choose_cards(card.controller, "Oubliette: purge a creature (power <= 3)", targets, 1, 1)
    steps.purge(game, choice[0])


def pawn_sacrifice(game, card):
    player = controller_of(game, card)
    options = list(player.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "does nothing: there is no friendly creature to sacrifice", "Nothing to sacrifice")
        return
    choice = yield from game.choose_cards(player.id, "Pawn Sacrifice: sacrifice a friendly creature", options, 1, 1)
    victim = choice[0]
    ok = yield from steps.sacrifice(game, victim)
    if not ok:
        return
    remaining_targets = game.all_creatures("any", card)
    if not remaining_targets:
        steps.shortfall(game, card, "deals no damage: no creatures are left in play after the sacrifice", "No creatures left")
        return
    if len(remaining_targets) == 1:
        steps.shortfall(game, card, "damages only 1 creature: it was the only one left in play", "Only 1 creature left")
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
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creature to damage")
        return
    choice = yield from game.choose_cards(player.id, "Relentless Whispers: deal 2 damage", targets, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 2)
    destroyed = yield from game.check_destroyed([target])
    if target in destroyed:
        steps.steal(game, opponent_of(game, card), player, 1, source=card)


def silvertooth_play(game, card):
    steps.ready(game, card)
    return
    yield


def subtle_maul(game, card):
    yield from steps.discard_random(game, opponent_of(game, card), source=card)


def too_much_to_protect(game, card):
    opponent = opponent_of(game, card)
    amount = max(opponent.aember - 6, 0)
    if amount == 0:
        steps.shortfall(game, card, f"steals nothing: {{pos:{opponent.id}}} Æmber is {opponent.aember}, and it takes only what is above 6", "6 or less to take")
    steps.steal(game, opponent, controller_of(game, card), amount)
    return
    yield


def urchin_play(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 1, source=card)
    return
    yield


def bad_penny_destroyed(game, card):
    card.destined_zone = "hand"
    return
    yield


# ---------------------------------------------------- v2.0 Milestone C ----

def finishing_blow(game, card):
    player = controller_of(game, card)
    targets = [c for c in game.all_creatures("any", card) if c.type_object.damage > 0]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature in play is damaged", "No damaged creature")
        return
    choice = yield from game.choose_cards(player.id, "Finishing Blow: destroy a damaged creature", targets, 1, 1)
    destroyed = yield from game.destroy_cards(choice)
    if choice[0] in destroyed:
        steps.steal(game, opponent_of(game, card), player, 1, source=card)


def imperial_traitor(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    game.reveal_hand(opponent.id, player.id, source=card)
    # No Sanctum card can exist in this CotA-only pool, so this compares by
    # value rather than `House.SANCTUM` (which the enum doesn't define) --
    # the comparison is always False here, correctly leaving the ability
    # dead text rather than crashing.
    targets = [c for c in opponent.hand.cards() if c.house.value == "Sanctum"]
    if not targets:
        steps.shortfall(game, card, f"purges nothing: {{pos:{opponent.id}}} hand has no Sanctum card", "No Sanctum card in hand")
        return
    choice = yield from game.choose_cards(player.id, "Imperial Traitor: you may purge a Sanctum card from the opponent's hand", targets, 0, 1)
    if choice:
        steps.purge(game, choice[0])


def key_of_darkness(game, card):
    opponent = opponent_of(game, card)
    modifier = 2 if opponent.aember == 0 else 6
    yield from game.forge_key(card.controller, cost_modifier=modifier, source=card)


def poison_wave(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creatures")
        return
    for c in targets:
        steps.deal_damage(game, c, 2)
    yield from game.check_destroyed(targets)


def routine_job(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    copies_in_discard = sum(1 for c in player.discard.cards() if c.name == "Routine Job")
    steps.steal(game, opponent, player, 1 + copies_in_discard, source=card)
    return
    yield


def treasure_map(game, card):
    player = controller_of(game, card)
    if player.hand_plays_this_turn == 1:
        steps.gain(game, player, 3)
    else:
        steps.shortfall(game, card, "gains nothing: you already played another card this turn", "Already played a card")
    game.active_effects.add(DurationEffect(card, card.controller, 1, card.controller, "CanPlayCards", "=", False))
    game.log.add(
        "duration_effect", card=card.name, iid=card.instance_id, variable="CanPlayCards", op="=",
        value=False, player=card.controller, affected=[card.controller],
    )
    return
    yield


def customs_office_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "ArtifactPlayToll", "=", (1, card.controller))
    game.active_effects.add(effect)


def evasion_sigil_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "before_fight", _evasion_sigil_handler))


def _evasion_sigil_handler(game, event):
    attacker = event["attacker"]
    controller = game.players[attacker.controller]
    top = controller.deck.draw_top()
    if top is not None:
        controller.discard.push(top)
        game.log.add("discard", player=controller.id, card=top.name, iid=top.instance_id)
        if top.house == controller.selected_house:
            event["cancelled"] = True
            steps.shortfall(game, attacker, "is exhausted with no effect: Evasion Sigil discarded a card of the active house", "Evasion Sigil: no effect")
    return
    yield


def longfused_mines(game, card):
    opponent = opponent_of(game, card)
    yield from steps.sacrifice(game, card)
    targets = [c for c in opponent.play_area.creatures if not opponent.play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "deals no damage: every enemy creature is on a flank", "No non-flank enemy creature")
        return
    for c in targets:
        steps.deal_damage(game, c, 3)
    yield from game.check_destroyed(targets)


def masterplan_play(game, card):
    player = controller_of(game, card)
    options = player.hand.cards()
    if not options:
        steps.shortfall(game, card, f"has no card to place beneath it: {{pos:{player.id}}} hand is empty", "Hand is empty")
        return
    choice = yield from game.choose_cards(player.id, "Masterplan: put a card from your hand facedown beneath it", options, 1, 1)
    c = choice[0]
    player.hand.remove(c)
    card.under_cards.append(c)
    game.log.add("under_card", player=player.id, card=card.name, iid=card.instance_id, under=c.name, under_iid=c.instance_id)


def masterplan_omni(game, card):
    player = controller_of(game, card)
    if not card.under_cards:
        steps.shortfall(game, card, "has no card to play: nothing is beneath it", "Nothing beneath it")
    else:
        to_play = card.under_cards.pop(0)
        ok = yield from game._play_card(player.id, to_play, from_deck_top=True)
        if not ok:
            card.under_cards.insert(0, to_play)
            steps.shortfall(game, card, f"can't play {to_play.name} right now; it stays beneath Masterplan", f"Can't play {to_play.name}")
    yield from steps.sacrifice(game, card)


def seeker_needle(game, card):
    player = controller_of(game, card)
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creature to damage")
        return
    choice = yield from game.choose_cards(player.id, f"{card.name}: deal 1 damage", targets, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 1)
    destroyed = yield from game.check_destroyed([target])
    if target in destroyed:
        steps.gain(game, player, 1)


mack_the_knife = seeker_needle


def skeleton_key(game, card):
    player = controller_of(game, card)
    targets = list(player.play_area.creatures)
    if not targets:
        steps.shortfall(game, card, "captures nothing: there is no friendly creature in play", "No friendly creature")
        return
    choice = yield from game.choose_cards(player.id, "Skeleton Key: choose a friendly creature to capture 1Æ", targets, 1, 1)
    steps.capture(game, choice[0], 1)


def special_delivery(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    targets = [c for pid in (1, 2) for c in game.players[pid].play_area.creatures if game.players[pid].play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no flank creatures in play", "No flank creature")
        return
    choice = yield from game.choose_cards(player.id, "Special Delivery: deal 3 damage to a flank creature", targets, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 3)
    destroyed = yield from game.check_destroyed([target])
    if target in destroyed:
        steps.purge(game, target)


def speed_sigil_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, card.controller, "FirstCreatureEntersReady", "=", True))


def the_sting_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, card.controller, "CanKeyForge", "=", False))
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "RedirectsForgePayment", "=", card.controller))


def the_sting_action(game, card):
    yield from steps.sacrifice(game, card)


def bulleteye(game, card):
    player = controller_of(game, card)
    targets = [c for pid in (1, 2) for c in game.players[pid].play_area.creatures if game.players[pid].play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there are no flank creatures in play", "No flank creature")
        return
    choice = yield from game.choose_cards(player.id, "Bulleteye: destroy a flank creature", targets, 1, 1)
    yield from game.destroy_cards(choice)


def carlo_phantom_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", _carlo_phantom_handler(card)))


def _carlo_phantom_handler(carlo_card):
    def handler(game, event):
        if event["card"].type == CardType.ARTIFACT:
            steps.steal(game, game.players[3 - carlo_card.controller], game.players[carlo_card.controller], 1, source=carlo_card)
        return
        yield
    return handler


def deipno_spymaster(game, card):
    player = controller_of(game, card)
    options = list(player.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "chooses nothing: there are no friendly creatures in play", "No friendly creature")
        return
    choice = yield from game.choose_cards(player.id, "Deipno Spymaster: choose a friendly creature", options, 1, 1)
    target = choice[0]
    if target.Exhausted:
        steps.shortfall(game, card, f"can't use {target.name}: it is already exhausted", f"{target.name} is exhausted")
        return
    may_use = yield from game.yes_no(player.id, f"Deipno Spymaster: use {target.name} this turn?")
    if may_use:
        yield from steps.use_creature(game, target)


def faygin(game, card):
    # Canonical text: "Return an Urchin from play [any Urchin in play,
    # friendly or enemy] or from your discard pile [your own only] to
    # your hand." An enemy Urchin returns to ITS OWNER's hand, not
    # Faygin's controller's -- `steps.return_to_hand` already routes by
    # `card.owner`, so the in-play branch needs no owner logic of its own.
    player = controller_of(game, card)
    in_play = [c for pid in (1, 2) for c in game.players[pid].play_area.creatures if c.name == "Urchin"]
    options = in_play + [c for c in player.discard.cards() if c.name == "Urchin"]
    if not options:
        steps.shortfall(game, card, "returns nothing: there is no Urchin in play or in your discard pile", "No Urchin")
        return
    choice = yield from game.choose_cards(player.id, "Faygin: return an Urchin to its owner's hand", options, 1, 1)
    target = choice[0]
    if target in in_play:
        steps.return_to_hand(game, target)
    else:
        player.discard.remove(target)
        player.hand.add(target)
        game.log.add("return_to_hand", card=target.name, iid=target.instance_id, owner=player.id)


def magda_the_rat_play(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 2, source=card)
    return
    yield


def magda_the_rat_unregister(game, card):
    steps.steal(game, game.players[card.controller], game.players[3 - card.controller], 2, source=card)


def nexus(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    targets = [
        a for a in opponent.play_area.artifacts
        if (a.card_def.on_action is not None or a.card_def.on_omni is not None) and not a.Exhausted
    ]
    if not targets:
        steps.shortfall(game, card, "uses nothing: the opponent has no usable artifact in play", "No usable enemy artifact")
        return
    choice = yield from game.choose_cards(player.id, "Nexus: use an opponent's artifact as if it were yours", targets, 1, 1)
    yield from game.use_artifact_ability(choice[0], player.id)


def selwyn_the_fence(game, card):
    player = controller_of(game, card)
    sources = [c for c in player.play_area.creatures if c.aember_captured > 0] + [c for c in player.play_area.artifacts if c.aember_stored > 0]
    if not sources:
        steps.shortfall(game, card, "moves nothing: none of your cards have Æmber on them", "No Æmber on your cards")
        return
    choice = yield from game.choose_cards(player.id, "Selwyn the Fence: move 1Æ from one of your cards to your pool", sources, 1, 1)
    source_card = choice[0]
    if source_card.aember_captured > 0:
        source_card.aember_captured -= 1
    else:
        source_card.aember_stored -= 1
    player.aember += 1
    game.log.add("move_aember", player=player.id, card=source_card.name, iid=source_card.instance_id, amount=1, to="pool")


def shadow_self_register(game, card):
    game.active_effects.add(InsteadEffect(card, card.controller, "damage_target", _shadow_self_redirect(card)))


def _shadow_self_redirect(shadow_self_card):
    def handler(game, creature):
        if creature is shadow_self_card:
            return None
        area = game.players[shadow_self_card.controller].play_area
        if shadow_self_card not in area.creatures or creature not in area.neighbors(shadow_self_card):
            return None
        if "Specter" in creature.tags:
            return None
        return shadow_self_card
    return handler


def smiling_ruth(game, card):
    player = controller_of(game, card)
    if not game.forged_key_on_turn(player.id, game.turn_number):
        steps.shortfall(game, card, "takes control of nothing: you have not forged a key this turn", "No key forged this turn")
        return
    opponent = opponent_of(game, card)
    targets = [c for c in opponent.play_area.creatures if opponent.play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "takes control of nothing: the opponent has no flank creature", "No enemy flank creature")
        return
    choice = yield from game.choose_cards(player.id, "Smiling Ruth: take control of an enemy flank creature", targets, 1, 1)
    yield from game.take_control(choice[0], player.id)


def sneklifter(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    targets = list(opponent.play_area.artifacts)
    if not targets:
        steps.shortfall(game, card, "takes control of nothing: the opponent has no artifact in play", "No enemy artifact")
        return
    choice = yield from game.choose_cards(player.id, "Sneklifter: take control of an enemy artifact", targets, 1, 1)
    target = choice[0]
    yield from game.take_control(target, player.id)
    if target.house not in game.player_houses(player.id):
        target.house_override = House.SHADOWS


def silent_dagger_register(game, card):
    host = card.type_object.host
    if host is not None:
        host.extra_triggers["after_reap"].append(_silent_dagger_effect)


def silent_dagger_unregister(game, card):
    host = card.type_object.host
    if host is not None and _silent_dagger_effect in host.extra_triggers["after_reap"]:
        host.extra_triggers["after_reap"].remove(_silent_dagger_effect)


def _silent_dagger_effect(game, host_card):
    targets = [c for pid in (1, 2) for c in game.players[pid].play_area.creatures if game.players[pid].play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, host_card, "deals no damage: there are no flank creatures in play", "No flank creature")
        return
    choice = yield from game.choose_cards(host_card.controller, "Silent Dagger: deal 4 damage to a flank creature", targets, 1, 1)
    steps.deal_damage(game, choice[0], 4)
    yield from game.check_destroyed(choice)

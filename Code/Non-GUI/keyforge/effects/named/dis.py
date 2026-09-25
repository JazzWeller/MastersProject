"""Bespoke per-card effects for Dis cards that don't fit the generic factories."""

from __future__ import annotations

from ...enums import Affects, CardType, DecisionIntent, House
from ..effect_object import DurationEffect, InsteadEffect, TriggerEffect, INFINITE, register_cleanup_operation
from .. import steps
from ..generic import controller_of, opponent_of


def arise(game, card):
    player = controller_of(game, card)
    houses = game.player_houses(player.id)
    chosen = yield from game.choose_house(player.id, "Arise: choose one of your houses", houses)
    creatures = [c for c in player.discard.cards() if c.house == chosen and c.type == CardType.CREATURE]
    for c in creatures:
        player.discard.remove(c)
        player.hand.add(c)
    if not creatures:
        steps.shortfall(game, card, f"returns no creatures: {{pos:{player.id}}} discard pile has no {chosen.value} creatures (the chain is still gained)", "No creatures to return")
    before = player.chains
    player.chains = min(24, player.chains + 1)
    game.log.add("gain_chains", player=player.id, n=player.chains - before, total=player.chains, card=card.name, iid=card.instance_id)
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
        steps.shortfall(game, card, "purges nothing: both discard piles are empty", "Nothing to purge")
        return
    pile_choice = yield from game.choose_cards(
        player_id, "Choose a discard pile", piles, 1, 1,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.NONE,
    )
    target_player = game.players[pile_choice[0]]
    options = target_player.discard.cards()
    n_max = min(2, len(options))
    choice = yield from game.choose_cards(
        player_id, "Purge 0-2 cards from that discard pile", options, 0, n_max,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.ANY, optional=True,
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
        if player.play_area.creatures:
            steps.shortfall(game, card, "does nothing: every friendly creature is exhausted (or already used 6 times this turn)", "No ready creature")
        else:
            steps.shortfall(game, card, "does nothing: there are no friendly creatures in play", "No creature to use")
        return
    choice = yield from game.choose_cards(
        player.id, "Choose a friendly creature to use", options, 1, 1,
        source_card=card, intent=DecisionIntent.USE_TARGET, affects=Affects.FRIENDLY,
    )
    yield from steps.use_creature(game, choice[0])


def dust_imp_destroyed(game, card):
    # Unqualified "Destroyed:" abilities benefit the controller at the
    # moment of destruction, not the owner (they can differ if a
    # control-change effect took the creature first) -- user decision.
    steps.gain(game, game.players[card.controller], 2)
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
    before = player.chains
    player.chains = min(24, player.chains + 3)
    game.log.add("gain_chains", player=player.id, n=player.chains - before, total=player.chains, card=card.name, iid=card.instance_id)


def guardian_demon(game, card):
    player = controller_of(game, card)
    creatures = game.all_creatures("any", card)
    # Healing an undamaged creature heals 0 and deals 0, so only offer ones
    # that can actually be healed.
    options = [c for c in creatures if c.type_object.damage > 0]
    if not options:
        steps.shortfall(game, card, "heals nothing and deals no damage: no creature is damaged", "No damage to heal")
        return
    heal_choice = yield from game.choose_cards(
        player.id, "Guardian Demon: heal up to 2 from a creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.HEAL, affects=Affects.ANY,
    )
    healed = steps.heal(game, heal_choice[0], 2)
    if healed <= 0:
        return
    # Printed text: deal the damage to *another* creature. None -> no damage.
    damage_targets = [c for c in creatures if c is not heal_choice[0]]
    if not damage_targets:
        steps.shortfall(game, card, f"heals {healed} but deals no damage: there is no other creature in play", "No creature to damage")
        return
    dmg_choice = yield from game.choose_cards(
        player.id, f"Deal {healed} damage to another creature", damage_targets, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
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
        steps.steal(game, opponent, controller_of(game, card), 1, source=card)
    else:
        steps.shortfall(game, card, f"steals nothing: {{pos:{opponent.id}}} Æmber is {opponent.aember}, and it needs 4 or more", "Needs 4+ enemy Æmber")
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
        steps.shortfall(game, card, "returns nothing: there are no artifacts or flank creatures in play", "Nothing to return")
        return
    choice = yield from game.choose_cards(
        player.id, "Snudge: return an artifact or a flank creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.ANY,
    )
    steps.return_to_hand(game, choice[0])


def succubus_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "DrawUpToLimit", "-", 1)
    game.active_effects.add(effect)


def the_terror_play(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember == 0:
        steps.gain(game, controller_of(game, card), 2)
    else:
        steps.shortfall(game, card, f"gains nothing: {{pos:{opponent.id}}} Æmber is {opponent.aember}, and it pays out only at 0", "Enemy has Æmber")
    return
    yield


def three_fates(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there are no creatures in play", "No creatures")
        return
    if len(targets) < 3:
        steps.shortfall(game, card, f"destroys only {len(targets)}: that is every creature in play", f"Only {len(targets)} in play")
    remaining = list(targets)
    chosen = []
    while remaining and len(chosen) < 3:
        slots = 3 - len(chosen)
        max_power = max(game.get_power(c) for c in remaining)
        tied = [c for c in remaining if game.get_power(c) == max_power]
        if len(tied) > slots:
            # Only a tie that crosses the cut needs a choice; a tie that all
            # fits is destroyed whole.
            picked = yield from game.choose_cards(
                card.controller, f"Three Fates: choose {slots} of the tied creatures to destroy", tied, slots, slots,
                source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ANY,
            )
        else:
            picked = tied
        for c in picked:
            chosen.append(c)
            remaining.remove(c)
    yield from game.destroy_cards(chosen)


# ---------------------------------------------------- v2.0 Milestone C ----

def a_fair_game(game, card):
    _fair_game_step(game, card, controller_of(game, card), opponent_of(game, card))
    _fair_game_step(game, card, opponent_of(game, card), controller_of(game, card))
    return
    yield


def _fair_game_step(game, card, gainer, target):
    top = target.deck.draw_top()
    if top is None:
        steps.shortfall(game, card, f"discards nothing from {{pos:{target.id}}} deck: it is empty", "Deck is empty")
        return
    target.discard.push(top)
    game.log.add("discard", player=target.id, card=top.name, iid=top.instance_id)
    game.reveal_hand(target.id, gainer.id, source=card)
    n = sum(1 for c in target.hand.cards() if c.house == top.house)
    steps.gain(game, gainer, n)


def dance_of_doom(game, card):
    creatures = game.all_creatures("any", card)
    if not creatures:
        steps.shortfall(game, card, "destroys nothing: there are no creatures in play", "No creatures")
        return
    powers = sorted({game.get_power(c) for c in creatures})
    chosen = yield from game.choose_number(card.controller, "Dance of Doom: choose a number", powers, source_card=card)
    targets = [c for c in creatures if game.get_power(c) == chosen]
    yield from game.destroy_cards(targets)


def fear(game, card):
    targets = game.all_creatures("enemy", card)
    if not targets:
        steps.shortfall(game, card, "returns nothing: there are no enemy creatures in play", "No enemy creatures")
        return
    choice = yield from game.choose_cards(
        card.controller, "Fear: return an enemy creature to hand", targets, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.ENEMY,
    )
    steps.return_to_hand(game, choice[0])


def gongoozle(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creature to damage")
        return
    choice = yield from game.choose_cards(
        card.controller, "Gongoozle: deal 3 damage", targets, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    target = choice[0]
    steps.deal_damage(game, target, 3)
    destroyed = yield from game.check_destroyed([target])
    if target not in destroyed:
        yield from steps.discard_random(game, game.players[target.owner], source=card)


def guilty_hearts(game, card):
    targets = [c for c in game.all_creatures("any", card) if c.aember_captured > 0]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature has any Æmber on it", "No creature with Æmber")
        return
    yield from game.destroy_cards(targets)


def hand_of_dis(game, card):
    targets = [c for c in game.all_creatures("any", card) if not game.players[c.controller].play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: every creature in play is on a flank", "No non-flank creature")
        return
    choice = yield from game.choose_cards(
        card.controller, "Hand of Dis: destroy a creature not on a flank", targets, 1, 1,
        source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ANY,
    )
    yield from game.destroy_cards(choice)


def hecatomb(game, card):
    targets = [c for c in game.all_creatures("any", card) if c.house == House.DIS]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there are no Dis creatures in play", "No Dis creatures")
        return
    counts = {1: 0, 2: 0}
    for c in targets:
        counts[c.controller] += 1
    yield from game.destroy_cards(targets)
    for pid, n in counts.items():
        steps.gain(game, game.players[pid], n)


def tendrils_of_pain(game, card):
    opponent = opponent_of(game, card)
    amount = 4 if game.forged_key_on_turn(opponent.id, game.turn_number - 1) else 1
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creatures")
        return
    for c in targets:
        steps.deal_damage(game, c, amount)
    yield from game.check_destroyed(targets)


def hysteria(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "returns nothing: there are no creatures in play", "No creatures")
        return
    for c in list(targets):
        steps.return_to_hand(game, c)
    return
    yield


def key_hammer(game, card):
    opponent = opponent_of(game, card)
    if game.forged_key_on_turn(opponent.id, game.turn_number - 1):
        opponent.keys = max(0, opponent.keys - 1)
        game.log.add("unforge", player=opponent.id, keys=opponent.keys, card=card.name, iid=card.instance_id)
    else:
        steps.shortfall(game, card, f"unforges nothing: {{pos:{opponent.id}}} did not forge a key on their previous turn", "No key forged last turn")
    steps.gain(game, opponent, 6)
    return
    yield


def mind_barb(game, card):
    yield from steps.discard_random(game, opponent_of(game, card), source=card)


def pandemonium(game, card):
    targets = [c for c in game.all_creatures("any", card) if c.type_object.damage == 0]
    if not targets:
        steps.shortfall(game, card, "captures nothing: every creature in play is damaged", "No undamaged creature")
        return
    for c in targets:
        steps.capture(game, c, 1)
    return
    yield


def poltergeist(game, card):
    targets = [
        a for a in list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
        if (a.card_def.on_action is not None or a.card_def.on_omni is not None) and not a.Exhausted
    ]
    if not targets:
        steps.shortfall(game, card, "uses nothing: there is no usable artifact in play", "No usable artifact")
        return
    choice = yield from game.choose_cards(
        card.controller, "Poltergeist: use an artifact as if it were yours", targets, 1, 1,
        source_card=card, intent=DecisionIntent.USE_TARGET, affects=Affects.ANY,
    )
    target = choice[0]
    yield from game.use_artifact_ability(target, card.controller)
    yield from game.destroy_cards([target])


def _revert_armor_negated_op(game, iid):
    game.card_by_id(iid).armor_negated = False


register_cleanup_operation("dis.clear_armor_negated", _revert_armor_negated_op)


def red_hot_armor(game, card):
    opponent = opponent_of(game, card)
    targets = [c for c in opponent.play_area.creatures if game.get_armor(c) > 0]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no enemy creature has any armor", "No armored enemy creature")
        return
    for c in targets:
        lost = game.get_armor(c)
        c.armor_negated = True
        game._end_of_turn_cleanups.append(("dis.clear_armor_negated", c.instance_id))
        steps.deal_damage(game, c, lost)
    yield from game.check_destroyed(targets)


def annihilation_ritual_register(game, card):
    game.active_effects.add(InsteadEffect(card, card.controller, "discard_destination", lambda g, c: True))


def key_to_dis(game, card):
    yield from steps.sacrifice(game, card)
    targets = game.all_creatures("any", card)
    yield from game.destroy_cards(targets)


def sacrificial_altar(game, card):
    player = controller_of(game, card)
    options = [c for c in player.play_area.creatures if "Human" in c.tags]
    if not options:
        steps.shortfall(game, card, "purges nothing: there is no friendly Human creature in play", "No Human creature")
        return
    choice = yield from game.choose_cards(
        player.id, "Sacrificial Altar: purge a friendly Human creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.FRIENDLY,
    )
    victim = choice[0]
    if not steps.purge(game, victim):
        return
    discard_creatures = [c for c in player.discard.cards() if c.type == CardType.CREATURE]
    if not discard_creatures:
        steps.shortfall(game, card, f"purges {victim.name} but plays nothing: your discard pile has no creature", "No creature in discard")
        return
    choice2 = yield from game.choose_cards(
        player.id, "Sacrificial Altar: play a creature from your discard pile", discard_creatures, 1, 1,
        source_card=card, intent=DecisionIntent.PLAY, affects=Affects.FRIENDLY,
    )
    to_play = choice2[0]
    player.discard.remove(to_play)
    ok = yield from game._play_card(player.id, to_play, from_deck_top=True)
    if not ok:
        player.discard.push(to_play)
        steps.shortfall(game, card, f"can't play {to_play.name} from the discard pile right now; it stays there", f"Can't play {to_play.name}")


def screaming_cave(game, card):
    player = controller_of(game, card)
    cards = player.hand.take_all() + player.discard.take_all()
    if not cards:
        steps.shortfall(game, card, "shuffles nothing: your hand and discard pile are both empty", "Nothing to shuffle")
        return
    player.deck.shuffle_in(cards, game.event_rng("reshuffle", player.id))
    game.log.add("reshuffle", player=player.id)
    return
    yield


def soul_snatcher_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "creature_destroyed", soul_snatcher_trigger))


def soul_snatcher_trigger(game, event):
    destroyed_card = event["card"]
    steps.gain(game, game.players[destroyed_card.owner], 1)
    return
    yield


def drumble_play(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 7:
        steps.capture(game, card, opponent.aember)
    else:
        steps.shortfall(game, card, f"captures nothing: {{pos:{opponent.id}}} Æmber is {opponent.aember}, and it needs 7 or more", "Needs 7+ enemy Æmber")
    return
    yield


def eater_of_the_dead(game, card):
    player = controller_of(game, card)
    options = [c for c in list(game.players[1].discard.cards()) + list(game.players[2].discard.cards()) if c.type == CardType.CREATURE]
    if not options:
        steps.shortfall(game, card, "purges nothing: neither discard pile has a creature", "No creature in either discard pile")
        return
    choice = yield from game.choose_cards(
        player.id, "Eater of the Dead: purge a creature from a discard pile", options, 1, 1,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.ANY,
    )
    if steps.purge(game, choice[0]):
        card.power_counters += 1
        game.log.add("power_counter", card=card.name, iid=card.instance_id, amount=1, total=card.power_counters)


def gabos_longarms_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "before_fight", _gabos_before_fight(card)))


def gabos_longarms_unregister(game, card):
    game.active_effects.remove_from_source(card)


def _gabos_before_fight(gabos_card):
    def handler(game, event):
        if event["attacker"] is not gabos_card:
            return
        targets = game.all_creatures("any", gabos_card)
        if not targets:
            return
        choice = yield from game.choose_cards(
            gabos_card.controller, "Gabos Longarms: choose a creature to deal its fight damage to instead", targets, 1, 1,
            source_card=gabos_card, intent=DecisionIntent.REDIRECT, affects=Affects.ANY,
        )
        gabos_card.redirect_fight_damage_to = choice[0]
    return handler


def overlord_greking_on_destroyed_fighting(game, survivor, victim):
    owner = game.players[victim.owner]
    if not owner.discard.remove(victim) or victim.type != CardType.CREATURE:
        return
        yield
    new_controller = survivor.controller
    victim.reset_on_leave_play()
    game.players[new_controller].play_area.add_creature(victim, "left")
    victim.controller = new_controller
    victim.Exhausted = True
    if victim.card_def.register_passive:
        victim.card_def.register_passive(game, victim)
    game.log.add("put_into_play", card=victim.name, iid=victim.instance_id, player=new_controller, source=survivor.name)
    return
    yield


def stealer_of_souls_on_destroyed_fighting(game, survivor, victim):
    # Confirmed ruling (Bad Penny vs. Stealer of Souls): the creature's own
    # "Destroyed:" ability resolves first and can redirect it out of the
    # discard pile (e.g. Bad Penny returns to hand) before this trigger
    # runs, in which case there's nothing left here to purge -- but the
    # Æmber gain still happens either way, since the creature was still
    # destroyed. Same reasoning already applied to Overlord Greking above
    # (`owner.discard.remove(victim)` there).
    owner = game.players[victim.owner]
    if victim in owner.discard.cards():
        steps.purge(game, victim)
    steps.gain(game, game.players[survivor.controller], 1)
    return
    yield


def master_of_n(n):
    def effect(game, card):
        targets = [c for c in game.all_creatures("any", card) if game.get_power(c) == n]
        if not targets:
            steps.shortfall(game, card, f"destroys nothing: no creature in play has {n} power", f"No {n}-power creature")
            return
        do_it = yield from game.yes_no(
            card.controller, f"{card.name}: destroy a creature with {n} power?",
            source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
        )
        if not do_it:
            return
        choice = yield from game.choose_cards(
            card.controller, f"{card.name}: choose a creature with {n} power to destroy", targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ANY,
        )
        yield from game.destroy_cards(choice)

    return effect


def pitlord_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, card.controller, "HouseSelection", "=", House.DIS))


def restringuntus(game, card):
    opponent = opponent_of(game, card)
    houses = game.player_houses(opponent.id)
    chosen = yield from game.choose_house(card.controller, "Restringuntus: choose a house to deny your opponent", houses)
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, opponent.id, "CannotChooseHouse", "add", chosen))


def shaffles_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "end_of_turn", _shaffles_handler(card)))


def _shaffles_handler(shaffles_card):
    def handler(game, event):
        if event["player"] == shaffles_card.controller:
            steps.lose(game, game.players[3 - shaffles_card.controller], 1)
        return
        yield
    return handler


def tocsin(game, card):
    yield from steps.discard_random(game, opponent_of(game, card), source=card)


def tolas_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "creature_destroyed", tolas_trigger))


def tolas_trigger(game, event):
    destroyed_card = event["card"]
    steps.gain(game, game.players[3 - destroyed_card.owner], 1)
    return
    yield


def truebaru_destroyed(game, card):
    # See dust_imp_destroyed: controller at destruction, not owner.
    steps.gain(game, game.players[card.controller], 5)
    return
    yield


def collar_of_subordination_register(game, card):
    host = card.type_object.host
    if host is None or host.controller == card.controller:
        return
    old_pid = host.controller
    game._temp_control.setdefault(card.instance_id, []).append((host, old_pid))
    game.players[old_pid].play_area.remove(host)
    game.players[card.controller].play_area.add_creature(host, "left")
    host.controller = card.controller
    game.log.add(
        "take_control", card=host.name, iid=host.instance_id,
        from_player=old_pid, to_player=card.controller, permanent=False, reverted=False,
    )


def collar_of_subordination_unregister(game, card):
    game._revert_temp_control(card)


def tentacus_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "ArtifactUseToll", "=", (1, card.controller))
    game.active_effects.add(effect)

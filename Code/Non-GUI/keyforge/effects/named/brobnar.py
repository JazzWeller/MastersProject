"""Named (card-specific) effects for the Brobnar house (Phase 3, Milestone D.1).
See Code/PHASE_3_CARD_POOL.md for canonical text and Code/PHASE_3_CARD_RULINGS.md
for the FAQ rulings behind each implementation."""

from __future__ import annotations

from ...enums import CardType, House
from ..effect_object import DurationEffect, INFINITE, ModifierEffect, TriggerEffect
from .. import steps
from ..generic import choose_most_powerful, controller_of, opponent_of


# ------------------------------------------------------------- actions --

def anger(game, card):
    player = controller_of(game, card)
    creatures = player.play_area.creatures
    if not creatures:
        steps.shortfall(game, card, "does nothing: there is no friendly creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly creature to ready and fight with", creatures, 1, 1
    )
    yield from game.ready_and_fight(choice[0])


def barehanded(game, card):
    artifacts = list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
    if not artifacts:
        steps.shortfall(game, card, "does nothing: there are no artifacts in play", "No artifacts")
        return
    order = artifacts
    if len(artifacts) > 1:
        order = yield from game.order_effects(
            card.controller, artifacts, f"{card.name}: choose the order these go to their decks' tops"
        )
    for a in order:
        owner = game.players[a.owner]
        area = game.find_play_area(a)
        if area is not None:
            area.remove(a)
            game.leave_play(a)
        owner.deck.put_on_top(a)
        game.log.add("put_on_top", player=owner.id, card=a.name, iid=a.instance_id)


def blood_money(game, card):
    opponent = opponent_of(game, card)
    options = list(opponent.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "places nothing: there is no enemy creature in play", "No enemy creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose an enemy creature", options, 1, 1)
    steps.place_aember(game, choice[0], 2)


def brothers_in_battle(game, card):
    player = controller_of(game, card)
    chosen = yield from game.choose_house(player.id, f"{card.name}: choose a house", list(House))
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "FightPermittedHouse", "add", chosen))
    game.log.add(
        "duration_effect", card=card.name, iid=card.instance_id, variable="FightPermittedHouse",
        op="add", value=chosen.value, player=player.id, affected=[player.id],
    )


def burn_the_stockpile(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 7:
        steps.lose(game, opponent, 4)
    else:
        steps.shortfall(game, card, f"does nothing: {{pos:{opponent.id}}} Æmber is below 7", "Opponent below 7")
    return
    yield


def champions_challenge(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)

    enemy_creatures = list(opponent.play_area.creatures)
    survivor_enemy = yield from choose_most_powerful(
        game, player.id, enemy_creatures, f"{card.name}: choose the enemy creature to survive (most powerful)"
    )
    yield from game.destroy_cards([c for c in enemy_creatures if c is not survivor_enemy])

    friendly_creatures = list(player.play_area.creatures)
    survivor_friendly = yield from choose_most_powerful(
        game, player.id, friendly_creatures, f"{card.name}: choose the friendly creature to survive (most powerful)"
    )
    yield from game.destroy_cards([c for c in friendly_creatures if c is not survivor_friendly])

    if survivor_friendly is not None and not survivor_friendly.destroyed:
        yield from game.ready_and_fight(survivor_friendly)


def cowards_end(game, card):
    player = controller_of(game, card)
    targets = [c for c in game.all_creatures("any", card) if c.type_object.damage == 0]
    yield from game.destroy_cards(targets)
    if not targets:
        steps.shortfall(game, card, "destroys nothing: every creature in play already has damage on it", "No undamaged creature")
    steps.gain_chains(game, player, card, 3)


def follow_the_leader(game, card):
    player = controller_of(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "FightPermittedHouse", "add", True))
    game.log.add(
        "duration_effect", card=card.name, iid=card.instance_id, variable="FightPermittedHouse",
        op="add", value="any house", player=player.id, affected=[player.id],
    )
    return
    yield


def _grant_gain_on_enemy_destroyed_this_turn(game, card):
    def handler(g, event):
        if event["card"].controller != card.controller:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "creature_destroyed", handler, remaining_duration=1))


def loot_the_bodies(game, card):
    _grant_gain_on_enemy_destroyed_this_turn(game, card)
    return
    yield


def take_that_smartypants(game, card):
    opponent = opponent_of(game, card)
    logos_count = 0
    for c in opponent.play_area.creatures:
        if c.house == House.LOGOS:
            logos_count += 1
        logos_count += sum(1 for u in c.type_object.upgrades if u.house == House.LOGOS)
    for a in opponent.play_area.artifacts:
        if a.house == House.LOGOS:
            logos_count += 1
    if logos_count >= 3:
        steps.steal(game, opponent, controller_of(game, card), 2, source=card)
    else:
        steps.shortfall(game, card, f"steals nothing: {{pos:{opponent.id}}} has fewer than 3 Logos cards in play", "Not enough Logos cards")
    return
    yield


def relentless_assault(game, card):
    player = controller_of(game, card)
    used = []
    for i in range(3):
        available = [c for c in player.play_area.creatures if c not in used]
        if not available:
            break
        if i > 0:
            cont = yield from game.yes_no(player.id, f"{card.name}: ready and fight with another creature?")
            if not cont:
                break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose a friendly creature to ready and fight with", available, 1, 1
        )
        target = choice[0]
        used.append(target)
        yield from game.ready_and_fight(target)


def smith(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    if len(player.play_area.creatures) > len(opponent.play_area.creatures):
        steps.gain(game, player, 2)
    else:
        steps.shortfall(game, card, "gains nothing: you do not control more creatures than your opponent", "Not enough creatures")
    return
    yield


def sound_the_horns(game, card):
    player = controller_of(game, card)
    discarded = []
    found = None
    while not player.deck.is_empty():
        top = player.deck.draw_top()
        player.discard.push(top)
        discarded.append(top)
        game.log.add("discard", player=player.id, card=top.name, iid=top.instance_id)
        if top.type == CardType.CREATURE and top.house == House.BROBNAR:
            found = top
            break
    if found is not None:
        player.discard.remove(found)
        player.hand.add(found)
        game.log.add("return_to_hand", card=found.name, iid=found.instance_id, owner=player.id)
    elif not discarded:
        steps.shortfall(game, card, f"discards nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
    else:
        steps.shortfall(game, card, f"discards {len(discarded)} card(s) but finds no Brobnar creature", "No Brobnar creature found")
    return
    yield


def tremor(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "stuns nothing: there are no creatures in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature to stun (with its neighbors)", options, 1, 1
    )
    target = choice[0]
    area = game.find_play_area(target)
    neighbors = area.neighbors(target) if area is not None else []
    steps.stun(game, target)
    for n in neighbors:
        steps.stun(game, n)


def unguarded_camp(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    excess = len(player.play_area.creatures) - len(opponent.play_area.creatures)
    if excess <= 0:
        steps.shortfall(game, card, "captures nothing: you do not have more creatures than your opponent", "No excess creatures")
        return
    used = []
    for i in range(excess):
        available = [c for c in player.play_area.creatures if c not in used]
        if not available:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose a friendly creature to capture 1Æ ({i + 1}/{excess})", available, 1, 1
        )
        target = choice[0]
        used.append(target)
        steps.capture(game, target, 1)


def warsong(game, card):
    def handler(g, event):
        if event["attacker"].controller == card.controller:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "fight_resolved", handler, remaining_duration=1))
    return
    yield


# ------------------------------------------------------------ artifacts --

def autocannon_register(game, card):
    def handler(g, event):
        entered = event["card"]
        steps.deal_damage(g, entered, 1)
        yield from g.check_destroyed([entered])

    game.active_effects.add(TriggerEffect(card, card.controller, "creature_entered_play", handler))


def banner_of_battle_register(game, card):
    def mod(g, target):
        return 1 if target.controller == card.controller else 0

    game.active_effects.add(ModifierEffect(card, card.controller, "power", mod))


def gauntlet_of_command(game, card):
    player = controller_of(game, card)
    creatures = player.play_area.creatures
    if not creatures:
        steps.shortfall(game, card, "has no friendly creature to ready and fight with", "No creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly creature to ready and fight with", creatures, 1, 1
    )
    yield from game.ready_and_fight(choice[0])


def iron_obelisk_register(game, card):
    def cost_bonus(g):
        return sum(
            1 for c in g.players[card.controller].play_area.creatures
            if c.house == House.BROBNAR and c.type_object.damage > 0
        )

    game.active_effects.add(
        DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "KeyForgeCost", "+", cost_bonus)
    )


def mighty_javelin(game, card):
    yield from steps.sacrifice(game, card)
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: deal 4 damage to a creature", options, 1, 1)
    steps.deal_damage(game, choice[0], 4)
    yield from game.check_destroyed(choice)


def pile_of_skulls_register(game, card):
    def handler(g, event):
        destroyed_card = event["card"]
        if destroyed_card.controller == card.controller:
            return
        if g.active_player_id != card.controller:
            return
        friendly = g.players[card.controller].play_area.creatures
        if not friendly:
            return
        choice = yield from g.choose_cards(
            card.controller, f"{card.name}: choose a friendly creature to capture 1Æ", friendly, 1, 1
        )
        steps.capture(g, choice[0], 1)

    game.active_effects.add(TriggerEffect(card, card.controller, "creature_destroyed", handler))


def screechbomb(game, card):
    yield from steps.sacrifice(game, card)
    steps.lose(game, opponent_of(game, card), 2)


def the_warchest(game, card):
    opponent = opponent_of(game, card)
    n = game.creatures_destroyed_in_fight_this_turn(opponent.id)
    if not steps.gain(game, controller_of(game, card), n):
        steps.shortfall(game, card, "gains nothing: no enemy creature was destroyed in a fight this turn", "None destroyed in a fight")
    return
    yield


# ------------------------------------------------------------ creatures --

def bilgum_avalanche_register(game, card):
    def handler(g, event):
        if event.get("player") != card.controller:
            return
        targets = g.all_creatures("enemy", card)
        for t in targets:
            steps.deal_damage(g, t, 2)
        yield from g.check_destroyed(targets)

    game.active_effects.add(TriggerEffect(card, card.controller, "key_forged", handler))


def valdr_register(game, card):
    def mod(g, attacker, target):
        if attacker is not card:
            return 0
        area = g.find_play_area(target)
        if area is None or not area.is_flank(target):
            return 0
        return 2

    game.active_effects.add(ModifierEffect(card, card.controller, "fight_damage", mod))


def earthshaker(game, card):
    targets = [c for c in game.all_creatures("any", card) if game.get_power(c) <= 3]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature has power 3 or lower", "No low-power creature")
        return
    yield from game.destroy_cards(targets)


def firespitter_before_fight(game, card):
    targets = game.all_creatures("enemy", card)
    for t in targets:
        steps.deal_damage(game, t, 1)
    return
    yield


def ganger_chieftain_play(game, card):
    player = controller_of(game, card)
    area = player.play_area
    neighbors = area.neighbors(card)
    if not neighbors:
        return
    do_it = yield from game.yes_no(player.id, f"{card.name}: ready and fight with a neighboring creature?")
    if not do_it:
        return
    choice = yield from game.choose_cards(player.id, f"{card.name}: choose a neighboring creature", neighbors, 1, 1)
    yield from game.ready_and_fight(choice[0])


def hebe_the_huge(game, card):
    targets = [c for c in game.all_creatures("any", card) if c is not card and c.type_object.damage == 0]
    if not targets:
        steps.shortfall(game, card, "deals no damage: there is no other undamaged creature", "No undamaged creature")
        return
    for t in targets:
        steps.deal_damage(game, t, 2)
    yield from game.check_destroyed(targets)


def kelifi_dragon_after(game, card):
    steps.gain(game, controller_of(game, card), 1)
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: deal 5 damage to a creature", options, 1, 1)
    steps.deal_damage(game, choice[0], 5)
    yield from game.check_destroyed(choice)


def king_of_the_crag_register(game, card):
    def mod(g, target):
        if target.controller == card.controller or target.house != House.BROBNAR:
            return 0
        return -2

    game.active_effects.add(ModifierEffect(card, card.controller, "power", mod))


def krump_on_destroyed_fighting(game, survivor, victim):
    steps.lose(game, game.players[victim.controller], 1)
    return
    yield


def lomir_flamefist(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 7:
        steps.lose(game, opponent, 2)
    else:
        steps.shortfall(game, card, f"does nothing: {{pos:{opponent.id}}} Æmber is below 7", "Opponent below 7")
    return
    yield


def mugwump_on_destroyed_fighting(game, survivor, victim):
    steps.fully_heal(game, survivor)
    survivor.power_counters += 1
    return
    yield


def pingle_who_annoys_register(game, card):
    def handler(g, event):
        entered = event["card"]
        if entered.controller != card.controller:
            steps.deal_damage(g, entered, 1)
            yield from g.check_destroyed([entered])

    game.active_effects.add(TriggerEffect(card, card.controller, "creature_entered_play", handler))


def rock_hurling_giant_register(game, card):
    def handler(g, event):
        if event["player"] != card.controller or g.active_player_id != card.controller:
            return
        discarded = event["card"]
        if discarded.house != House.BROBNAR:
            return
        options = g.all_creatures("any", card)
        if not options:
            return
        do_it = yield from g.yes_no(card.controller, f"{card.name}: deal 4 damage to a creature?")
        if not do_it:
            return
        choice = yield from g.choose_cards(card.controller, f"{card.name}: choose a creature", options, 1, 1)
        steps.deal_damage(g, choice[0], 4)
        yield from g.check_destroyed(choice)

    game.active_effects.add(TriggerEffect(card, card.controller, "card_discarded_from_hand", handler))


def rogue_ogre_register(game, card):
    def handler(g, event):
        if event["player"] == card.controller:
            player = g.players[card.controller]
            if sum(player.CardsPlayed.values()) == 1:
                steps.heal(g, card, 2)
                steps.capture(g, card, 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "end_of_turn", handler))


def smaaash(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "stuns nothing: there are no creatures in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to stun", options, 1, 1)
    steps.stun(game, choice[0])


def tireless_crocag_play(game, card):
    opponent = opponent_of(game, card)
    if not opponent.play_area.creatures:
        yield from game.destroy_cards([card])


def tireless_crocag_register(game, card):
    def handler(g, event):
        destroyed_card = event["card"]
        opponent_id = 3 - card.controller
        if destroyed_card.controller != opponent_id:
            return
        if g.find_play_area(card) is None:
            return
        if not g.players[opponent_id].play_area.creatures:
            yield from g.destroy_cards([card])

    game.active_effects.add(TriggerEffect(card, card.controller, "creature_destroyed", handler))


def wardrummer(game, card):
    player = controller_of(game, card)
    targets = [c for c in player.play_area.creatures if c is not card and c.house == House.BROBNAR]
    if not targets:
        steps.shortfall(game, card, "returns nothing: there is no other friendly Brobnar creature in play", "No other Brobnar creature")
    else:
        for t in targets:
            steps.return_to_hand(game, t)
    return
    yield


# -------------------------------------------------------------- upgrades --

def phoenix_heart_register(game, card):
    host = card.type_object.host
    if host is not None:
        host.extra_triggers["destroyed"].append(_phoenix_heart_effect)


def phoenix_heart_unregister(game, card):
    host = card.type_object.host
    if host is not None and _phoenix_heart_effect in host.extra_triggers["destroyed"]:
        host.extra_triggers["destroyed"].remove(_phoenix_heart_effect)


def _phoenix_heart_effect(game, host_card):
    host_card.destined_zone = "hand"
    targets = [c for c in game.all_creatures("any", host_card) if c is not host_card]
    for t in targets:
        steps.deal_damage(game, t, 3)
    yield from game.check_destroyed(targets)


def yo_mama_mastery_play(game, card):
    host = card.type_object.host
    if host is not None:
        steps.fully_heal(game, host)
    return
    yield

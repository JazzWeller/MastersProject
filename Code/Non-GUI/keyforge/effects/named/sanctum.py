"""Named (card-specific) effects for the Sanctum house (Phase 3, Milestone D.2).
See Code/PHASE_3_CARD_POOL.md for canonical text and Code/PHASE_3_CARD_RULINGS.md
for the FAQ rulings behind each implementation."""

from __future__ import annotations

from ...enums import CardType, House
from ..effect_object import DurationEffect, INFINITE, InsteadEffect, ModifierEffect, TriggerEffect
from .. import steps
from ..generic import choose_least_powerful, choose_most_powerful, controller_of, opponent_of


# ------------------------------------------------------------- actions --

def begone(game, card):
    player = controller_of(game, card)
    mode = yield from game.choose_mode(player.id, f"{card.name}: choose one", ["Destroy each Dis creature", "Gain 1Æ"])
    if mode == "Gain 1Æ":
        steps.gain(game, player, 1)
        return
    targets = [c for c in game.all_creatures("any", card) if c.house == House.DIS]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there is no Dis creature in play", "No Dis creature")
        return
    yield from game.destroy_cards(targets)


def blinding_light(game, card):
    player = controller_of(game, card)
    chosen = yield from game.choose_house(player.id, f"{card.name}: choose a house", list(House))
    targets = [c for c in game.all_creatures("any", card) if c.house == chosen]
    if not targets:
        steps.shortfall(game, card, f"stuns nothing: there is no {chosen.value} creature in play", "No matching creature")
        return
    for t in targets:
        steps.stun(game, t)


def charge(game, card):
    player = controller_of(game, card)

    def handler(g, event):
        if event["card"].type != CardType.CREATURE:
            return
        options = g.all_creatures("enemy", card)
        if not options:
            return
        choice = yield from g.choose_cards(player.id, f"{card.name}: deal 2 damage to an enemy creature", options, 1, 1)
        steps.deal_damage(g, choice[0], 2)
        yield from g.check_destroyed(choice)

    game.active_effects.add(TriggerEffect(card, player.id, "card_played", handler, remaining_duration=1))
    return
    yield


def cleansing_wave(game, card):
    player = controller_of(game, card)
    targets = game.all_creatures("any", card)
    healed_count = 0
    for t in targets:
        if steps.heal(game, t, 1):
            healed_count += 1
    if healed_count:
        steps.gain(game, player, healed_count)
    else:
        steps.shortfall(game, card, "heals nothing: no creature in play has any damage", "No damaged creature")
    return
    yield


def clear_mind(game, card):
    player = controller_of(game, card)
    for c in player.play_area.creatures:
        c.stunned = False
    return
    yield


def doorstep_to_heaven(game, card):
    for p in game.players.values():
        if p.aember >= 6:
            lost = p.aember - 5
            p.aember = 5
            game.log.add("lose", player=p.id, amount=lost)
    return
    yield


def glorious_few(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    excess = len(opponent.play_area.creatures) - len(player.play_area.creatures)
    if excess > 0:
        steps.gain(game, player, excess)
    else:
        steps.shortfall(game, card, "gains nothing: your opponent does not control more creatures than you", "No excess")
    return
    yield


def honorable_claim(game, card):
    player = controller_of(game, card)
    targets = [c for c in player.play_area.creatures if "Knight" in c.tags]
    if not targets:
        steps.shortfall(game, card, "captures nothing: there is no friendly Knight creature in play", "No Knight creature")
        return
    for t in targets:
        steps.capture(game, t, 1)
    return
    yield


def inspiration(game, card):
    player = controller_of(game, card)
    creatures = player.play_area.creatures
    if not creatures:
        steps.shortfall(game, card, "has no friendly creature to ready and use", "No creature")
        return
    choice = yield from game.choose_cards(player.id, f"{card.name}: choose a friendly creature to ready and use", creatures, 1, 1)
    target = choice[0]
    steps.ready(game, target)
    yield from game.use_creature_ability(target)


def mighty_lance(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature", options, 1, 1)
    target = choice[0]
    steps.deal_damage(game, target, 3)
    area = game.find_play_area(target)
    neighbors = area.neighbors(target) if area is not None else []
    hit_neighbor = None
    if len(neighbors) == 1:
        hit_neighbor = neighbors[0]
    elif len(neighbors) > 1:
        nchoice = yield from game.choose_cards(
            card.controller, f"{card.name}: choose a neighbor to also deal 3 damage to", neighbors, 1, 1
        )
        hit_neighbor = nchoice[0]
    if hit_neighbor is not None:
        steps.deal_damage(game, hit_neighbor, 3)
    yield from game.check_destroyed([target] + ([hit_neighbor] if hit_neighbor else []))


def oath_of_poverty(game, card):
    player = controller_of(game, card)
    targets = list(player.play_area.artifacts)
    if not targets:
        steps.shortfall(game, card, "destroys nothing: you have no artifacts in play", "No artifacts")
        return
    destroyed = []
    for a in targets:
        area = game.find_play_area(a)
        if area is not None:
            area.remove(a)
            game.leave_play(a)
            player.discard.push(a)
            game.log.add("destroyed", card=a.name, iid=a.instance_id, destination="discard")
            destroyed.append(a)
    steps.gain(game, player, len(destroyed) * 2)
    return
    yield


def one_stood_against_many(game, card):
    player = controller_of(game, card)
    creatures = player.play_area.creatures
    if not creatures:
        steps.shortfall(game, card, "has no friendly creature to ready and fight with", "No creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly creature to ready and fight 3 times", creatures, 1, 1
    )
    fighter = choice[0]
    fought_ids = set()
    for _ in range(3):
        if not game.legal_fight_targets(fighter, exclude=frozenset(fought_ids)):
            break
        target = yield from game.ready_and_fight(fighter, exclude=frozenset(fought_ids))
        if target is None:
            break
        fought_ids.add(target.instance_id)


def radiant_truth(game, card):
    opponent = opponent_of(game, card)
    targets = [c for c in opponent.play_area.creatures if not opponent.play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "stuns nothing: every enemy creature is on a flank", "No non-flank enemy creature")
        return
    for t in targets:
        steps.stun(game, t)
    return
    yield


def take_hostages(game, card):
    def handler(g, event):
        if event["attacker"].controller == card.controller:
            steps.capture(g, event["attacker"], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "fight_resolved", handler, remaining_duration=1))
    return
    yield


def terms_of_redress(game, card):
    player = controller_of(game, card)
    creatures = player.play_area.creatures
    if not creatures:
        steps.shortfall(game, card, "captures nothing: there is no friendly creature in play", "No creature")
        return
    choice = yield from game.choose_cards(player.id, f"{card.name}: choose a friendly creature to capture 2Æ", creatures, 1, 1)
    steps.capture(game, choice[0], 2)


def the_harder_they_come(game, card):
    options = [c for c in game.all_creatures("any", card) if game.get_power(c) >= 5]
    if not options:
        steps.shortfall(game, card, "purges nothing: no creature has power 5 or higher", "No high-power creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to purge", options, 1, 1)
    steps.purge(game, choice[0])


def the_spirits_way(game, card):
    targets = [c for c in game.all_creatures("any", card) if game.get_power(c) >= 3]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature has power 3 or higher", "No high-power creature")
        return
    yield from game.destroy_cards(targets)


# ------------------------------------------------------------ artifacts --

def epic_quest_play(game, card):
    player = controller_of(game, card)
    targets = [c for c in player.play_area.creatures if "Knight" in c.tags]
    if not targets:
        steps.shortfall(game, card, "archives nothing: there is no friendly Knight creature in play", "No Knight creature")
        return
    for t in targets:
        player.play_area.remove(t)
        game.leave_play(t)
        player.archive.add(t)
        game.log.add("archive", player=player.id, card=t.name, iid=t.instance_id)
    return
    yield


def epic_quest_omni(game, card):
    player = controller_of(game, card)
    name_to_house = {c.name: c.house for c in player.all_cards}
    sanctum_played = sum(n for name, n in player.CardsPlayed.items() if name_to_house.get(name) == House.SANCTUM)
    if sanctum_played < 7:
        steps.shortfall(
            game, card,
            f"cannot forge a key: only {sanctum_played} Sanctum cards played this turn (needs 7)",
            "Not enough Sanctum cards",
        )
        return
    yield from steps.sacrifice(game, card)
    yield from game.forge_key(player.id, cost_modifier=-999, source=card)


def gorm_of_omm(game, card):
    yield from steps.sacrifice(game, card)
    options = list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
    if not options:
        steps.shortfall(game, card, "destroys nothing: there is no artifact in play", "No artifact")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose an artifact to destroy", options, 1, 1)
    target = choice[0]
    owner = game.players[target.owner]
    area = game.find_play_area(target)
    if area is not None:
        area.remove(target)
        game.leave_play(target)
        owner.discard.push(target)
        game.log.add("destroyed", card=target.name, iid=target.instance_id, destination="discard")


def hallowed_blaster(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "heals nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to heal 3 damage from", options, 1, 1)
    steps.heal(game, choice[0], 3)


def potion_of_invulnerability(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "CannotBeDealtDamage", "=", True))


def round_table_register(game, card):
    def power_mod(g, target):
        return 1 if target.controller == card.controller and "Knight" in target.tags else 0

    def keyword_mod(g, target):
        if target.controller == card.controller and "Knight" in target.tags:
            return frozenset({"taunt"})
        return frozenset()

    game.active_effects.add(ModifierEffect(card, card.controller, "power", power_mod))
    game.active_effects.add(ModifierEffect(card, card.controller, "keywords", keyword_mod))


def sigil_of_brotherhood(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "UsePermittedHouse", "add", House.SANCTUM))
    for c in player.play_area.creatures:
        if c.house == House.SANCTUM:
            c.CanBeUsed = True


def whispering_reliquary(game, card):
    options = list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
    if not options:
        steps.shortfall(game, card, "returns nothing: there is no artifact in play", "No artifact")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose an artifact to return to its owner's hand", options, 1, 1)
    steps.return_to_hand(game, choice[0])


# ------------------------------------------------------------ creatures --

def bulwark_register(game, card):
    def mod(g, target):
        area = g.find_play_area(card)
        if area is None or target not in area.neighbors(card):
            return 0
        return 2

    game.active_effects.add(ModifierEffect(card, card.controller, "armor", mod))


def commander_remiel(game, card):
    player = controller_of(game, card)
    options = [c for c in player.play_area.creatures if c.house != House.SANCTUM]
    if not options:
        steps.shortfall(game, card, "does nothing: there is no friendly non-Sanctum creature in play", "No non-Sanctum creature")
        return
    choice = yield from game.choose_cards(player.id, f"{card.name}: choose a friendly non-Sanctum creature to use", options, 1, 1)
    yield from game.use_creature_ability(choice[0])


def duma_the_martyr_destroyed(game, card):
    player = controller_of(game, card)
    for c in player.play_area.creatures:
        if c is not card:
            steps.fully_heal(game, c)
    steps.draw(game, player, 2, source=card)
    return
    yield


def francus_on_destroyed_fighting(game, survivor, victim):
    steps.capture(game, survivor, 1)
    return
    yield


def grey_monk_register(game, card):
    def mod(g, target):
        return 1 if target.controller == card.controller else 0

    game.active_effects.add(ModifierEffect(card, card.controller, "armor", mod))


def grey_monk_after_reap(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "heals nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to heal 2 damage from", options, 1, 1)
    steps.heal(game, choice[0], 2)


def hayyel_the_merchant_register(game, card):
    def handler(g, event):
        if event["card"].type == CardType.ARTIFACT:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", handler))


def horseman_of_death(game, card):
    player = controller_of(game, card)
    targets = [c for c in player.discard.cards() if "Horseman" in c.tags]
    if not targets:
        steps.shortfall(game, card, "returns nothing: there is no Horseman creature in your discard pile", "No Horseman in discard")
        return
    for t in targets:
        player.discard.remove(t)
        player.hand.add(t)
        game.log.add("return_to_hand", card=t.name, iid=t.instance_id, owner=player.id)
    return
    yield


def horseman_of_famine(game, card):
    targets = game.all_creatures("any", card)
    least = yield from choose_least_powerful(game, card.controller, targets, f"{card.name}: choose the least powerful creature")
    if least is None:
        steps.shortfall(game, card, "destroys nothing: there is no creature in play", "No creature")
        return
    yield from game.destroy_cards([least])


def horseman_of_pestilence(game, card):
    targets = [c for c in game.all_creatures("any", card) if "Horseman" not in c.tags]
    if not targets:
        steps.shortfall(game, card, "deals no damage: every creature in play is a Horseman", "No non-Horseman creature")
        return
    for t in targets:
        steps.deal_damage(game, t, 1)
    yield from game.check_destroyed(targets)


def horseman_of_war(game, card):
    player = controller_of(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "UsePermittedHouse", "add", True))
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "CanOnlyFight", "=", True))
    for c in player.play_area.creatures:
        c.CanBeUsed = True
    return
    yield


def jehu_the_bureaucrat_register(game, card):
    def handler(g, event):
        if event["player"] == card.controller and event["house"] == House.SANCTUM:
            steps.gain(g, g.players[card.controller], 2)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "house_chosen", handler))


def lady_maxena_play(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "stuns nothing: there are no creatures in play", "No creature")
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to stun", options, 1, 1)
    steps.stun(game, choice[0])


def lady_maxena_action(game, card):
    steps.return_to_hand(game, card)
    return
    yield


def lord_golgotha_before_fight(game, card, target):
    area = game.find_play_area(target)
    neighbors = area.neighbors(target) if area is not None else []
    for n in neighbors:
        steps.deal_damage(game, n, 3)
    return
    yield


def numquid_the_fair(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    while True:
        options = list(opponent.play_area.creatures)
        if not options:
            steps.shortfall(game, card, "destroys nothing: there is no enemy creature in play", "No enemy creature")
            return
        choice = yield from game.choose_cards(player.id, f"{card.name}: choose an enemy creature to destroy", options, 1, 1)
        yield from game.destroy_cards(choice)
        if len(opponent.play_area.creatures) <= len(player.play_area.creatures):
            return


def _revert_damage_prevented(card):
    card.damage_prevented = False


def protectrix_after_reap(game, card):
    options = game.all_creatures("any", card)
    if not options:
        return
    do_it = yield from game.yes_no(card.controller, f"{card.name}: fully heal a creature?")
    if not do_it:
        return
    choice = yield from game.choose_cards(card.controller, f"{card.name}: choose a creature to fully heal", options, 1, 1)
    target = choice[0]
    steps.fully_heal(game, target)
    target.damage_prevented = True
    game._end_of_turn_cleanups.append(lambda t=target: _revert_damage_prevented(t))


def sanctum_guardian_after(game, card):
    player = controller_of(game, card)
    others = [c for c in player.play_area.creatures if c is not card]
    if not others:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly creature to swap battleline positions with", others, 1, 1
    )
    player.play_area.swap(card, choice[0])
    game.log.add("swap", card=card.name, iid=card.instance_id, other=choice[0].name, other_iid=choice[0].instance_id)


def sergeant_zakiel_play(game, card):
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


def staunch_knight_register(game, card):
    def power_mod(g, target):
        if target is not card:
            return 0
        area = g.find_play_area(card)
        return 2 if area is not None and area.is_flank(card) else 0

    game.active_effects.add(ModifierEffect(card, card.controller, "power", power_mod))


def gatekeeper(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 7:
        steps.capture(game, card, opponent.aember - 5)
    else:
        steps.shortfall(game, card, f"captures nothing: {{pos:{opponent.id}}} Æmber is below 7", "Opponent below 7")
    return
    yield


def the_vaultkeeper_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, card.controller, "CannotBeStolenFrom", "=", True))


def veemos_lightbringer(game, card):
    targets = [c for c in game.all_creatures("any", card) if "elusive" in game.get_keywords(c)]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there is no elusive creature in play", "No elusive creature")
        return
    yield from game.destroy_cards(targets)


# -------------------------------------------------------------- upgrades --

def _armageddon_cloak_instead(cloak_card):
    def handler(game, would_be_destroyed):
        host = cloak_card.type_object.host
        if host is not would_be_destroyed:
            return False
        steps.fully_heal(game, host)
        game.destroy_upgrade(cloak_card)
        return True

    return handler


def armageddon_cloak_register(game, card):
    game.active_effects.add(InsteadEffect(card, card.controller, "would_be_destroyed", _armageddon_cloak_instead(card)))


def shoulder_armor_register(game, card):
    def power_mod(g, target):
        host = card.type_object.host
        if target is not host or host is None:
            return 0
        area = g.find_play_area(host)
        return 2 if area is not None and area.is_flank(host) else 0

    def armor_mod(g, target):
        host = card.type_object.host
        if target is not host or host is None:
            return 0
        area = g.find_play_area(host)
        return 2 if area is not None and area.is_flank(host) else 0

    game.active_effects.add(ModifierEffect(card, card.controller, "power", power_mod))
    game.active_effects.add(ModifierEffect(card, card.controller, "armor", armor_mod))

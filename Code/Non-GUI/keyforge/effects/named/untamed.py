"""Named (card-specific) effects for the Untamed house (Phase 3, Milestone D.4).
See Code/PHASE_3_CARD_POOL.md for canonical text and Code/PHASE_3_CARD_RULINGS.md
for the FAQ rulings behind each implementation."""

from __future__ import annotations

from ...enums import Affects, CardType, DecisionIntent, House
from ..effect_object import DurationEffect, INFINITE, ModifierEffect, TriggerEffect
from .. import steps
from ..generic import controller_of, opponent_of


# ------------------------------------------------------------- actions --

def cooperative_hunting(game, card):
    player = controller_of(game, card)
    n = len(player.play_area.creatures)
    if n == 0:
        steps.shortfall(game, card, "deals no damage: you have no friendly creature in play", "No friendly creature")
        return
    hit = []
    for i in range(n):
        options = game.all_creatures("any", card)
        if not options:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: deal 1 damage to a creature ({i + 1}/{n})", options, 1, 1,
            source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
        )
        steps.deal_damage(game, choice[0], 1)
        hit.append(choice[0])
    yield from game.check_destroyed(hit)


def curiosity(game, card):
    targets = [c for c in game.all_creatures("any", card) if "Scientist" in c.tags]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there is no Scientist creature in play", "No Scientist creature")
        return
    yield from game.destroy_cards(targets)


def fertility_chant(game, card):
    steps.gain(game, opponent_of(game, card), 2)
    return
    yield


def full_moon(game, card):
    def handler(g, event):
        if event["card"].type == CardType.CREATURE:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", handler, remaining_duration=1))
    return
    yield


def grasping_vines(game, card):
    player = controller_of(game, card)
    options = list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
    if not options:
        steps.shortfall(game, card, "returns nothing: there is no artifact in play", "No artifacts")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose up to 3 artifacts to return to their owners' hands", options, 0, min(3, len(options)),
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.ANY, optional=True,
    )
    for a in choice:
        steps.return_to_hand(game, a)


def _lose_one_then_maybe_forge(game, card):
    player = controller_of(game, card)
    if not steps.lose(game, player, 1):
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: forge a key at current cost?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    yield from game.forge_key(player.id, source=card)


def key_charge(game, card):
    yield from _lose_one_then_maybe_forge(game, card)


def lifeweb(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    n = game.creatures_played_on_turn(opponent.id, game.turn_number - 1)
    if n >= 3:
        steps.steal(game, opponent, player, 2, source=card)
    else:
        steps.shortfall(game, card, "steals nothing: your opponent played fewer than 3 creatures on their previous turn", "Not enough creatures played")
    return
    yield


def lost_in_the_woods(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    friendly_options = list(player.play_area.creatures)
    friendly_choice = []
    if friendly_options:
        friendly_choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose up to 2 friendly creatures", friendly_options, 0, min(2, len(friendly_options)),
            source_card=card, intent=DecisionIntent.SHUFFLE_IN, affects=Affects.FRIENDLY, optional=True,
        )
    enemy_options = list(opponent.play_area.creatures)
    enemy_choice = []
    if enemy_options:
        enemy_choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose up to 2 enemy creatures", enemy_options, 0, min(2, len(enemy_options)),
            source_card=card, intent=DecisionIntent.SHUFFLE_IN, affects=Affects.ENEMY, optional=True,
        )
    for c in friendly_choice + enemy_choice:
        owner = game.players[c.owner]
        area = game.find_play_area(c)
        if area is not None:
            area.remove(c)
            game.leave_play(c)
            owner.deck.shuffle_in([c], game.event_rng("reshuffle", owner.id))
            game.log.add("shuffle_into_deck", card=c.name, iid=c.instance_id, owner=owner.id)
    if not friendly_choice and not enemy_choice:
        steps.shortfall(game, card, "does nothing: there is no creature in play", "No creature")


def mimicry_play(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    # Excludes other copies of Mimicry itself -- copying one would just
    # recurse into "copy an action from the discard pile" again, forever.
    options = [c for c in opponent.discard.cards() if c.type == CardType.ACTION and c.name != "Mimicry"]
    if not options:
        steps.shortfall(game, card, "does nothing: there is no action card in your opponent's discard pile", "No action card")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose an action card in your opponent's discard pile to copy", options, 1, 1,
        source_card=card, intent=DecisionIntent.COPY, affects=Affects.ENEMY,
    )
    copied = choice[0]
    game.log.add("mimicry_copy", card=card.name, iid=card.instance_id, copied=copied.name)
    # "Treat it as a copy" for this one resolution: take on the copied
    # card's house (for any house-matching check during this resolution)
    # and run the copied card's own Play effect, as Mimicry itself
    # (Milestone C.15 -- a dynamic CardDef substitution scoped to this
    # resolution; the card is discarded right after anyway).
    card.house_override = copied.house
    if copied.card_def.on_play is not None:
        yield from copied.card_def.on_play(game, card)


def natures_call(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "returns nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose up to 3 creatures to return to their owners' hands", options, 0, min(3, len(options)),
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.ANY, optional=True,
    )
    for c in choice:
        steps.return_to_hand(game, c)


def nocturnal_maneuver(game, card):
    options = [c for c in game.all_creatures("any", card) if not c.Exhausted]
    if not options:
        steps.shortfall(game, card, "does nothing: there is no ready creature in play", "No ready creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose up to 3 creatures to exhaust", options, 0, min(3, len(options)),
        source_card=card, intent=DecisionIntent.EXHAUST, affects=Affects.ANY, optional=True,
    )
    for c in choice:
        steps.exhaust(game, c)


def perilous_wild(game, card):
    targets = [c for c in game.all_creatures("any", card) if "elusive" in game.get_keywords(c)]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: there is no elusive creature in play", "No elusive creature")
        return
    yield from game.destroy_cards(targets)


def regrowth(game, card):
    player = controller_of(game, card)
    options = [c for c in player.discard.cards() if c.type == CardType.CREATURE]
    if not options:
        steps.shortfall(game, card, "returns nothing: your discard pile has no creature", "No creature in discard")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature to return to your hand", options, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    player.discard.remove(target)
    player.hand.add(target)
    game.log.add("return_to_hand", card=target.name, iid=target.instance_id, owner=player.id)


def save_the_pack(game, card):
    player = controller_of(game, card)
    targets = [c for c in game.all_creatures("any", card) if c.type_object.damage > 0]
    yield from game.destroy_cards(targets)
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature in play has any damage", "No damaged creature")
    steps.gain_chains(game, player, card, 1)


def scout(game, card):
    player = controller_of(game, card)
    options = player.play_area.creatures
    if not options:
        steps.shortfall(game, card, "does nothing: there is no friendly creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose up to 2 friendly creatures to gain skirmish", options, 0, min(2, len(options)),
        source_card=card, intent=DecisionIntent.MODIFY, affects=Affects.FRIENDLY, optional=True,
    )
    if not choice:
        return
    chosen_ids = frozenset(c.instance_id for c in choice)

    def mod(g, target):
        return frozenset({"skirmish"}) if target.instance_id in chosen_ids else frozenset()

    game.active_effects.add(ModifierEffect(card, player.id, "keywords", mod))
    game._end_of_turn_cleanups.append(("remove_effects_from_source", card.instance_id))
    for c in choice:
        yield from game.ready_and_fight(c)


def stampede(game, card):
    player = controller_of(game, card)
    # Approximates "3 or more creatures used" with total use-events this
    # turn (used_this_turn is keyed by name, not by instance, so it can't
    # perfectly distinguish "3 different creatures" from "one creature used
    # 3 times" -- a documented, narrow simplification).
    total_uses = sum(player.used_this_turn.values())
    if total_uses >= 3:
        steps.steal(game, opponent_of(game, card), player, 2, source=card)
    else:
        steps.shortfall(game, card, "steals nothing: you have not used 3 or more creatures this turn", "Not enough creatures used")
    return
    yield


def the_common_cold(game, card):
    targets = game.all_creatures("any", card)
    for t in targets:
        steps.deal_damage(game, t, 1)
    destroyed = yield from game.check_destroyed(targets)
    remaining_mars = [c for c in game.all_creatures("any", card) if c.house == House.MARS and c not in destroyed]
    if not remaining_mars:
        return
    do_it = yield from game.yes_no(
        card.controller, f"{card.name}: destroy all Mars creatures?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if do_it:
        yield from game.destroy_cards(remaining_mars)


def troop_call(game, card):
    player = controller_of(game, card)
    from_discard = [c for c in player.discard.cards() if "Niffle" in c.tags]
    from_play = [c for c in player.play_area.creatures if "Niffle" in c.tags]
    if not from_discard and not from_play:
        steps.shortfall(game, card, "returns nothing: you have no friendly Niffle creature in play or in your discard pile", "No Niffle creature")
        return
    for c in from_discard:
        player.discard.remove(c)
        player.hand.add(c)
        game.log.add("return_to_hand", card=c.name, iid=c.instance_id, owner=player.id)
    for c in from_play:
        steps.return_to_hand(game, c)
    return
    yield


def vigor(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "heals nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature to heal up to 3 damage from", options, 1, 1,
        source_card=card, intent=DecisionIntent.HEAL, affects=Affects.ANY,
    )
    healed = steps.heal(game, choice[0], 3)
    if healed == 3:
        steps.gain(game, controller_of(game, card), 1)


def word_of_returning(game, card):
    opponent = opponent_of(game, card)
    targets = [c for c in opponent.play_area.creatures if c.aember_captured > 0]
    if not targets:
        steps.shortfall(game, card, "does nothing: no enemy creature has Æmber on it", "No Æmber to return")
        return
    total_returned = 0
    for t in targets:
        n = t.aember_captured
        steps.deal_damage(game, t, n)
        total_returned += n
        t.aember_captured = 0
    if total_returned:
        steps.gain(game, controller_of(game, card), total_returned)
    yield from game.check_destroyed(targets)


# ------------------------------------------------------------ artifacts --

def bear_flute(game, card):
    player = controller_of(game, card)
    bears_in_play = [c for c in player.play_area.creatures if c.name == "Ancient Bear"]
    if bears_in_play:
        target = bears_in_play[0]
        if len(bears_in_play) > 1:
            choice = yield from game.choose_cards(
                player.id, f"{card.name}: choose an Ancient Bear to fully heal", bears_in_play, 1, 1,
                source_card=card, intent=DecisionIntent.HEAL, affects=Affects.FRIENDLY,
            )
            target = choice[0]
        steps.fully_heal(game, target)
        return
    found = [c for c in player.deck.cards() if c.name == "Ancient Bear"] + [c for c in player.discard.cards() if c.name == "Ancient Bear"]
    if not found:
        steps.shortfall(game, card, "finds no Ancient Bear in your deck, discard pile, or play", "No Ancient Bear")
        return
    for c in found:
        # `found` mixes deck (hidden) and discard (public) cards, and by
        # here it's already removed from whichever one it was in -- err
        # toward not leaking rather than tracking which zone matched.
        if not player.deck.remove(c):
            player.discard.remove(c)
        player.hand.add(c)
        game.log.add("return_to_hand", card=c.name, iid=c.instance_id, owner=player.id, visible_to={player.id})
    remaining_discard = player.discard.take_all()
    player.deck.shuffle_in(remaining_discard, game.event_rng("reshuffle", player.id))


def nepenthe_seed(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    options = player.discard.cards()
    if not options:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a card to return to your hand", options, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    player.discard.remove(target)
    player.hand.add(target)
    game.log.add("return_to_hand", card=target.name, iid=target.instance_id, owner=player.id)


def ritual_of_balance(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember >= 6:
        steps.steal(game, opponent, controller_of(game, card), 1, source=card)
    else:
        steps.shortfall(game, card, f"steals nothing: {{pos:{opponent.id}}} Æmber is below 6", "Opponent below 6")
    return
    yield


def ritual_of_the_hunt(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "UsePermittedHouse", "add", House.UNTAMED))
    for c in player.play_area.creatures:
        if c.house == House.UNTAMED:
            c.CanBeUsed = True


def world_tree(game, card):
    player = controller_of(game, card)
    options = [c for c in player.discard.cards() if c.type == CardType.CREATURE]
    if not options:
        steps.shortfall(game, card, "returns nothing: your discard pile has no creature", "No creature in discard")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature to return to the top of your deck", options, 1, 1,
        source_card=card, intent=DecisionIntent.SHUFFLE_IN, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    player.discard.remove(target)
    player.deck.put_on_top(target)
    game.log.add("put_on_top", player=player.id, card=target.name, iid=target.instance_id)


# ------------------------------------------------------------ creatures --

def bigtwig_after_reap(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "does nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature to stun and exhaust", options, 1, 1,
        source_card=card, intent=DecisionIntent.STUN, affects=Affects.ANY,
    )
    target = choice[0]
    steps.stun(game, target)
    steps.exhaust(game, target)


def witch_of_the_wilds_register(game, card):
    def handler(g, event):
        if event["player"] == card.controller and event["house"] != House.UNTAMED:
            # Grants the same "extra off-house play" allowance Phase Shift
            # uses, rather than a new Untamed-only counter -- a documented
            # simplification (slightly broader than "one Untamed card").
            g.players[card.controller].NonLogosCardsPlayable += 1
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "house_chosen", handler))


def chota_hazri(game, card):
    yield from _lose_one_then_maybe_forge(game, card)


def flaxia(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    if len(player.play_area.creatures) > len(opponent.play_area.creatures):
        steps.gain(game, player, 2)
    else:
        steps.shortfall(game, card, "gains nothing: you do not control more creatures than your opponent", "Not enough creatures")
    return
    yield


def giant_sloth_register(game, card):
    def handler(g, event):
        if event["player"] == card.controller and event["card"].house == House.UNTAMED:
            g.players[card.controller].discarded_untamed_this_turn = True
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_discarded_from_hand", handler))


def _giant_sloth_restriction(game, card):
    return game.players[card.controller].discarded_untamed_this_turn


def giant_sloth_action(game, card):
    steps.gain(game, controller_of(game, card), 3)
    return
    yield


def halacor_register(game, card):
    def mod(g, target):
        area = g.find_play_area(card)
        if area is None or target.controller != card.controller or not area.is_flank(target):
            return frozenset()
        return frozenset({"skirmish"})

    game.active_effects.add(ModifierEffect(card, card.controller, "keywords", mod))


def inka_the_spider_effect(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "stuns nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature to stun", options, 1, 1,
        source_card=card, intent=DecisionIntent.STUN, affects=Affects.ANY,
    )
    steps.stun(game, choice[0])


def kindrith_longshot_after_reap(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: deal 2 damage to a creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    steps.deal_damage(game, choice[0], 2)
    yield from game.check_destroyed(choice)


def lupo_the_scarred_play(game, card):
    options = opponent_of(game, card).play_area.creatures
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no enemy creature in play", "No enemy creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose an enemy creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ENEMY,
    )
    steps.deal_damage(game, choice[0], 2)
    yield from game.check_destroyed(choice)


def murmook_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "KeyForgeCost", "+", 1))


def mushroom_man_register(game, card):
    def mod(g, target):
        if target is not card:
            return 0
        return 3 * max(0, 3 - g.players[card.controller].keys)

    game.active_effects.add(ModifierEffect(card, card.controller, "power", mod))


def niffle_queen_register(game, card):
    def mod(g, target):
        if target is card or target.controller != card.controller:
            return 0
        bonus = 0
        if "Beast" in target.tags:
            bonus += 1
        if "Niffle" in target.tags:
            bonus += 1
        return bonus

    game.active_effects.add(ModifierEffect(card, card.controller, "power", mod))


def piranha_monkeys_effect(game, card):
    targets = [c for c in game.all_creatures("any", card) if c is not card]
    if not targets:
        steps.shortfall(game, card, "deals no damage: there is no other creature in play", "No other creature")
        return
    for t in targets:
        steps.deal_damage(game, t, 2)
    yield from game.check_destroyed(targets)


def teliga_register(game, card):
    def handler(g, event):
        if event["player"] != card.controller and event["card"].type == CardType.CREATURE:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "any_card_played", handler))


def hunting_witch_register(game, card):
    def handler(g, event):
        played = event["card"]
        if event["player"] == card.controller and played is not card and played.type == CardType.CREATURE:
            steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", handler))


def witch_of_the_eye_after_reap(game, card):
    player = controller_of(game, card)
    options = [c for c in player.discard.cards() if c is not card]
    if not options:
        steps.shortfall(game, card, "returns nothing: your discard pile is empty", "Discard pile is empty")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a card to return to your hand", options, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    player.discard.remove(target)
    player.hand.add(target)
    game.log.add("return_to_hand", card=target.name, iid=target.instance_id, owner=player.id)

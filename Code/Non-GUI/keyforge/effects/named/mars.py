"""Named (card-specific) effects for the Mars house (Phase 3, Milestone D.3).
See Code/PHASE_3_CARD_POOL.md for canonical text and Code/PHASE_3_CARD_RULINGS.md
for the FAQ rulings behind each implementation."""

from __future__ import annotations

from ...enums import Affects, CardType, DecisionIntent, House
from ..effect_object import DurationEffect, INFINITE, InsteadEffect, ModifierEffect, TriggerEffect, register_cleanup_operation
from .. import steps
from ..generic import controller_of, opponent_of, reveal_from_hand


# ------------------------------------------------------------- actions --

def ammonia_clouds(game, card):
    targets = game.all_creatures("any", card)
    for t in targets:
        steps.deal_damage(game, t, 3)
    yield from game.check_destroyed(targets)


def battle_fleet(game, card):
    player = controller_of(game, card)
    revealed = yield from reveal_from_hand(
        game, player, lambda c: c.house == House.MARS, f"{card.name}: reveal any number of Mars cards from your hand",
        source_card=card,
    )
    steps.draw(game, player, len(revealed), source=card)


def deep_probe(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    chosen = yield from game.choose_house(player.id, f"{card.name}: choose a house", list(House))
    game.reveal_hand(opponent.id, player.id, source=card)
    targets = [c for c in opponent.hand.cards() if c.house == chosen and c.type == CardType.CREATURE]
    for t in targets:
        yield from steps.discard_from_hand(game, opponent, t)


def emp_blast(game, card):
    stun_targets = [c for c in game.all_creatures("any", card) if c.house == House.MARS or "Robot" in c.tags]
    for t in stun_targets:
        steps.stun(game, t)
    artifacts = list(game.players[1].play_area.artifacts) + list(game.players[2].play_area.artifacts)
    yield from game.destroy_cards(artifacts)


def hypnotic_command(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    n = sum(1 for c in player.play_area.creatures if game.get_effective_house(c) == House.MARS)
    if n == 0 or not opponent.play_area.creatures:
        steps.shortfall(game, card, "does nothing: no friendly Mars creature or no enemy creature", "Nothing to capture")
        return
    for i in range(n):
        options = list(opponent.play_area.creatures)
        if not options:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose an enemy creature to capture 1Æ from its own side ({i + 1}/{n})", options, 1, 1,
            source_card=card, intent=DecisionIntent.CAPTURE, affects=Affects.ENEMY,
        )
        steps.capture_from_own_side(game, choice[0], 1)


def irradiated_aember(game, card):
    opponent = opponent_of(game, card)
    if opponent.aember < 6:
        steps.shortfall(game, card, f"deals no damage: {{pos:{opponent.id}}} Æmber is below 6", "Opponent below 6")
        return
    targets = list(opponent.play_area.creatures)
    for t in targets:
        steps.deal_damage(game, t, 3)
    yield from game.check_destroyed(targets)


def key_abduction(game, card):
    player = controller_of(game, card)
    targets = [c for c in game.all_creatures("any", card) if c.house == House.MARS]
    for t in targets:
        steps.return_to_hand(game, t)
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: forge a key at +9Æ current cost, reduced by 1Æ per card in hand?",
        source_card=card, intent=DecisionIntent.OPTIONAL_COST,
    )
    if not do_it:
        return
    # `forge_key` itself floors the final cost at 0 -- don't pre-clamp the
    # modifier here, or a hand of 10+ cards (plausible after this very card
    # just returned every Mars creature) couldn't push the cost below the
    # printed current cost the way "reduced by 1Æ per card" implies.
    modifier = 9 - len(player.hand)
    yield from game.forge_key(player.id, cost_modifier=modifier, source=card)


def martian_hounds(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "does nothing: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.MODIFY, affects=Affects.ANY,
    )
    target = choice[0]
    damaged = sum(1 for c in game.all_creatures("any", card) if c.type_object.damage > 0)
    target.power_counters += 2 * damaged


def martians_make_bad_allies(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    game.reveal_hand(player.id, opponent.id, source=card)
    targets = [c for c in player.hand.cards() if c.type == CardType.CREATURE and c.house != House.MARS]
    for t in targets:
        steps.purge(game, t)
    if targets:
        steps.gain(game, player, len(targets))
    else:
        steps.shortfall(game, card, "purges nothing: your hand has no non-Mars creature", "No non-Mars creature")
    return
    yield


def mass_abduction(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    options = [c for c in opponent.play_area.creatures if c.type_object.damage > 0]
    if not options:
        steps.shortfall(game, card, "archives nothing: there is no damaged enemy creature in play", "No damaged enemy creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose up to 3 damaged enemy creatures to archive", options, 0, min(3, len(options)),
        source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.ENEMY, optional=True,
    )
    for c in choice:
        game.archive_from_play(c, player.id, return_to_owner_after=True)


def mating_season(game, card):
    targets = [c for c in game.all_creatures("any", card) if c.house == House.MARS]
    per_player = {1: 0, 2: 0}
    for t in targets:
        owner = game.players[t.owner]
        area = game.find_play_area(t)
        if area is not None:
            area.remove(t)
            game.leave_play(t)
            owner.deck.shuffle_in([t], game.event_rng("reshuffle", owner.id))
            game.log.add("shuffle_into_deck", card=t.name, iid=t.instance_id, owner=owner.id)
            per_player[t.owner] += 1
    for pid, n in per_player.items():
        if n:
            steps.gain(game, game.players[pid], n)
    return
    yield


def mothership_support(game, card):
    player = controller_of(game, card)
    n = sum(1 for c in player.play_area.creatures if game.get_effective_house(c) == House.MARS and not c.Exhausted)
    if n == 0:
        steps.shortfall(game, card, "deals no damage: there is no friendly ready Mars creature", "No ready Mars creature")
        return
    for i in range(n):
        options = game.all_creatures("any", card)
        if not options:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: deal 2 damage to a creature ({i + 1}/{n})", options, 1, 1,
            source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
        )
        steps.deal_damage(game, choice[0], 2)
        yield from game.check_destroyed(choice)


def orbital_bombardment(game, card):
    player = controller_of(game, card)
    revealed = yield from reveal_from_hand(
        game, player, lambda c: c.house == House.MARS, f"{card.name}: reveal any number of Mars cards from your hand",
        source_card=card,
    )
    for i in range(len(revealed)):
        options = game.all_creatures("any", card)
        if not options:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: deal 2 damage to a creature ({i + 1}/{len(revealed)})", options, 1, 1,
            source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
        )
        steps.deal_damage(game, choice[0], 2)
        yield from game.check_destroyed(choice)


def phosphorus_stars(game, card):
    player = controller_of(game, card)
    targets = [c for c in game.all_creatures("any", card) if c.house != House.MARS]
    for t in targets:
        steps.stun(game, t)
    steps.gain_chains(game, player, card, 2)
    return
    yield


def psychic_network(game, card):
    player = controller_of(game, card)
    n = sum(1 for c in player.play_area.creatures if game.get_effective_house(c) == House.MARS and not c.Exhausted)
    steps.steal(game, opponent_of(game, card), player, n, source=card)
    return
    yield


def sample_collection(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    n = opponent.keys
    if n == 0 or not opponent.play_area.creatures:
        steps.shortfall(game, card, "archives nothing: your opponent has forged no keys, or has no creature in play", "Nothing to archive")
        return
    for i in range(n):
        options = list(opponent.play_area.creatures)
        if not options:
            break
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose an enemy creature to archive ({i + 1}/{n})", options, 1, 1,
            source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.ENEMY,
        )
        game.archive_from_play(choice[0], player.id, return_to_owner_after=True)


def shatter_storm(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    lost = player.aember
    player.aember = 0
    if lost:
        game.log.add("lose", player=player.id, amount=lost)
    steps.lose(game, opponent, lost * 3)
    return
    yield


def soft_landing(game, card):
    player = controller_of(game, card)
    player.next_entry_ready = True
    return
    yield


def squawker(game, card):
    player = controller_of(game, card)
    mars_ready_targets = [c for c in player.play_area.creatures if c.house == House.MARS and c.Exhausted]
    non_mars_targets = [c for c in game.all_creatures("any", card) if c.house != House.MARS]
    modes = []
    if mars_ready_targets:
        modes.append("Ready a Mars creature")
    if non_mars_targets:
        modes.append("Stun a non-Mars creature")
    if not modes:
        steps.shortfall(game, card, "does nothing: no exhausted friendly Mars creature and no non-Mars creature", "Nothing to do")
        return
    mode = yield from game.choose_mode(player.id, f"{card.name}: choose one", modes, source_card=card)
    if mode == "Ready a Mars creature":
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose a Mars creature to ready", mars_ready_targets, 1, 1,
            source_card=card, intent=DecisionIntent.READY, affects=Affects.FRIENDLY,
        )
        steps.ready(game, choice[0])
    else:
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose a non-Mars creature to stun", non_mars_targets, 1, 1,
            source_card=card, intent=DecisionIntent.STUN, affects=Affects.ANY,
        )
        steps.stun(game, choice[0])


def total_recall(game, card):
    player = controller_of(game, card)
    ready_count = sum(1 for c in player.play_area.creatures if not c.Exhausted)
    steps.gain(game, player, ready_count)
    for c in list(player.play_area.creatures):
        steps.return_to_hand(game, c)
    return
    yield


# ------------------------------------------------------------ artifacts --

def combat_pheromones(game, card):
    # Simplified from "up to 2 other Mars cards" to "any number, for the
    # remainder of the turn" -- a per-use counter would need consumption
    # hooks threaded through _reap/_fight/_use_action/_use_omni for one
    # card's nuance; this reuses the existing UsePermittedHouse grant
    # (Sigil of Brotherhood) instead, which is a documented, deliberate
    # scope trim (slightly more generous than the printed cap).
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    game.active_effects.add(DurationEffect(card, player.id, 1, player.id, "UsePermittedHouse", "add", House.MARS))
    for c in player.play_area.creatures:
        if c.house == House.MARS:
            c.CanBeUsed = True


def commpod(game, card):
    player = controller_of(game, card)
    revealed = yield from reveal_from_hand(
        game, player, lambda c: c.house == House.MARS, f"{card.name}: reveal any number of Mars cards from your hand",
        source_card=card,
    )
    for _ in range(len(revealed)):
        options = [c for c in player.play_area.creatures if c.house == House.MARS and c.Exhausted]
        if not options:
            break
        do_it = yield from game.yes_no(
            player.id, f"{card.name}: ready a Mars creature?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
        )
        if not do_it:
            continue
        choice = yield from game.choose_cards(
            player.id, f"{card.name}: choose a Mars creature to ready", options, 1, 1,
            source_card=card, intent=DecisionIntent.READY, affects=Affects.FRIENDLY,
        )
        steps.ready(game, choice[0])


def crystal_hive_register(game, card):
    def handler(g, event):
        steps.gain(g, g.players[card.controller], 1)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "reap_resolved", handler, remaining_duration=1))


def crystal_hive(game, card):
    crystal_hive_register(game, card)
    return
    yield


def custom_virus(game, card):
    player = controller_of(game, card)
    yield from steps.sacrifice(game, card)
    creature_options = [c for c in player.hand.cards() if c.type == CardType.CREATURE]
    if not creature_options:
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: purge a creature from your hand?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature to purge from your hand", creature_options, 1, 1,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.FRIENDLY,
    )
    purged_card = choice[0]
    steps.purge(game, purged_card)
    targets = [c for c in game.all_creatures("any", card) if set(c.tags) & set(purged_card.tags)]
    yield from game.destroy_cards(targets)


def feeding_pit(game, card):
    player = controller_of(game, card)
    options = [c for c in player.hand.cards() if c.type == CardType.CREATURE]
    if not options:
        steps.shortfall(game, card, "discards nothing: your hand has no creature", "No creature in hand")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature to discard", options, 1, 1,
        source_card=card, intent=DecisionIntent.DISCARD, affects=Affects.FRIENDLY,
    )
    if (yield from steps.discard_from_hand(game, player, choice[0])):
        steps.gain(game, player, 1)


def invasion_portal(game, card):
    player = controller_of(game, card)
    discarded = []
    found = None
    while not player.deck.is_empty():
        top = player.deck.draw_top()
        player.discard.push(top)
        discarded.append(top)
        game.log.add("discard", player=player.id, card=top.name, iid=top.instance_id)
        if top.type == CardType.CREATURE and top.house == House.MARS:
            found = top
            break
    if found is not None:
        player.discard.remove(found)
        player.hand.add(found)
        game.log.add("return_to_hand", card=found.name, iid=found.instance_id, owner=player.id)
    elif not discarded:
        steps.shortfall(game, card, f"discards nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
    else:
        steps.shortfall(game, card, f"discards {len(discarded)} card(s) but finds no Mars creature", "No Mars creature found")
    return
    yield


def incubation_chamber(game, card):
    player = controller_of(game, card)
    options = [c for c in player.hand.cards() if c.house == House.MARS and c.type == CardType.CREATURE]
    if not options:
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: reveal a Mars creature from your hand?",
        source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a Mars creature to reveal and archive", options, 1, 1,
        source_card=card, intent=DecisionIntent.REVEAL, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    game.log.add("reveal", player=player.id, cards=[target.name], iids=[target.instance_id])
    steps.archive_card(game, player, target)


def mothergun(game, card):
    player = controller_of(game, card)
    revealed = yield from reveal_from_hand(
        game, player, lambda c: c.house == House.MARS, f"{card.name}: reveal any number of Mars cards from your hand",
        source_card=card,
    )
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    steps.deal_damage(game, choice[0], len(revealed))
    yield from game.check_destroyed(choice)


def _clear_elusive_suppressed_op(game, iid):
    game._elusive_suppressed = False


register_cleanup_operation("mars.clear_elusive_suppressed", _clear_elusive_suppressed_op)


def sniffer_action(game, card):
    game._elusive_suppressed = True
    game._end_of_turn_cleanups.append(("mars.clear_elusive_suppressed", None))
    return
    yield


def swap_widget(game, card):
    player = controller_of(game, card)
    ready_mars = [c for c in player.play_area.creatures if c.house == House.MARS and not c.Exhausted]
    if not ready_mars:
        steps.shortfall(game, card, "does nothing: there is no ready friendly Mars creature", "No ready Mars creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a ready friendly Mars creature to return", ready_mars, 1, 1,
        source_card=card, intent=DecisionIntent.RETURN_TO_HAND, affects=Affects.FRIENDLY,
    )
    returning = choice[0]
    # "Return ... to your hand" is mandatory (no "may"); "If you do, put ..."
    # only conditions the second half on whether the return happened, not on
    # whether a differently-named replacement is available in hand.
    if not steps.return_to_hand(game, returning):
        return
    hand_options = [c for c in player.hand.cards() if c.house == House.MARS and c.type == CardType.CREATURE and c.name != returning.name]
    if not hand_options:
        steps.shortfall(game, card, "puts nothing into play: your hand has no differently-named Mars creature", "No replacement")
        return
    choice2 = yield from game.choose_cards(
        player.id, f"{card.name}: choose a differently-named Mars creature to put into play", hand_options, 1, 1,
        source_card=card, intent=DecisionIntent.PLAY, affects=Affects.FRIENDLY,
    )
    game.put_creature_into_play_from_hand(player.id, choice2[0], ready=True)


# ------------------------------------------------------------ creatures --

def blypyp_after_reap(game, card):
    player = controller_of(game, card)
    player.next_mars_creature_ready = True
    return
    yield


def chuff_ape_register(game, card):
    card.stunned = True


def chuff_ape_after(game, card):
    player = controller_of(game, card)
    options = [c for c in player.play_area.creatures if c is not card]
    if not options:
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: sacrifice another friendly creature to fully heal {card.name}?",
        source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly creature to sacrifice", options, 1, 1,
        source_card=card, intent=DecisionIntent.SACRIFICE, affects=Affects.FRIENDLY,
    )
    sacrificed = yield from steps.sacrifice(game, choice[0])
    if sacrificed:
        steps.fully_heal(game, card)


def ether_spider_register(game, card):
    def handler(g, player, amount):
        opponent = g.players[3 - card.controller]
        if player is not opponent:
            return False
        card.aember_captured += amount
        g.log.add("capture", card=card.name, iid=card.instance_id, amount=amount)
        return True

    game.active_effects.add(InsteadEffect(card, card.controller, "aember_gain", handler))


def grabber_jammer_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "KeyForgeCost", "+", 1))


def grabber_jammer_after(game, card):
    steps.capture(game, card, 1)
    return
    yield


def grommid_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, card.controller, "CanPlayCreatures", "=", False))


def grommid_on_destroyed_fighting(game, survivor, victim):
    steps.lose(game, game.players[victim.controller], 1)
    return
    yield


def john_smyth_after(game, card):
    player = controller_of(game, card)
    options = [c for c in player.play_area.creatures if c.house == House.MARS and "Agent" not in c.tags]
    if not options:
        steps.shortfall(game, card, "readies nothing: there is no non-Agent friendly Mars creature", "No non-Agent Mars creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a non-Agent Mars creature to ready", options, 1, 1,
        source_card=card, intent=DecisionIntent.READY, affects=Affects.FRIENDLY,
    )
    steps.ready(game, choice[0])


def mindwarper_action(game, card):
    opponent = opponent_of(game, card)
    options = list(opponent.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "does nothing: there is no enemy creature in play", "No enemy creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose an enemy creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.CAPTURE, affects=Affects.ENEMY,
    )
    steps.capture_from_own_side(game, choice[0], 1)


def phylyx_the_disintegrator_action(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    n = sum(1 for c in player.play_area.creatures if game.get_effective_house(c) == House.MARS and c is not card)
    steps.lose(game, opponent, n)
    return
    yield


def qyxxlyx_plague_master_after(game, card):
    targets = [c for c in game.all_creatures("any", card) if "Human" in c.tags]
    if not targets:
        steps.shortfall(game, card, "deals no damage: there is no Human creature in play", "No Human creature")
        return
    for t in targets:
        steps.deal_damage(game, t, 3, ignore_armor=True)
    yield from game.check_destroyed(targets)


def tunk_register(game, card):
    def handler(g, event):
        played = event["card"]
        if played is not card and played.house == House.MARS and played.type == CardType.CREATURE:
            steps.fully_heal(g, card)
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", handler))


def ulyq_megamouth_after(game, card):
    player = controller_of(game, card)
    # Only creatures that can actually be used: picking an exhausted one (or
    # one past the rule of six) would silently waste the trigger, same as
    # Dominator Bauble.
    options = [
        c for c in player.play_area.creatures
        if c.house != House.MARS and not c.Exhausted and game._rule_of_six_ok(player, c.name)
    ]
    if not options:
        if any(c.house != House.MARS for c in player.play_area.creatures):
            steps.shortfall(game, card, "does nothing: every friendly non-Mars creature is exhausted (or already used 6 times this turn)", "No usable non-Mars creature")
        else:
            steps.shortfall(game, card, "does nothing: there is no friendly non-Mars creature in play", "No non-Mars creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly non-Mars creature to use", options, 1, 1,
        source_card=card, intent=DecisionIntent.USE_TARGET, affects=Affects.FRIENDLY,
    )
    yield from game.use_creature_ability(choice[0])


def uxlyx_the_zookeeper_after(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    options = list(opponent.play_area.creatures)
    if not options:
        steps.shortfall(game, card, "does nothing: there is no enemy creature in play", "No enemy creature")
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose an enemy creature to archive", options, 1, 1,
        source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.ENEMY,
    )
    game.archive_from_play(choice[0], player.id, return_to_owner_after=True)


def vezyma_thinkdrone_after(game, card):
    player = controller_of(game, card)
    # Printed text has no "another" qualifier -- Vezyma Thinkdrone may
    # archive itself.
    options = list(player.play_area.creatures) + list(player.play_area.artifacts)
    if not options:
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: archive a friendly creature or artifact from play?",
        source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a friendly card to archive", options, 1, 1,
        source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    area = game.find_play_area(target)
    if area is not None:
        area.remove(target)
        game.leave_play(target)
        player.archive.add(target)
        game.log.add("archive", player=player.id, card=target.name, iid=target.instance_id)


def yxili_marauder_register(game, card):
    def mod(g, target):
        return target.aember_captured if target is card else 0

    game.active_effects.add(ModifierEffect(card, card.controller, "power", mod))


def yxili_marauder_play(game, card):
    player = controller_of(game, card)
    n = sum(1 for c in player.play_area.creatures if game.get_effective_house(c) == House.MARS and not c.Exhausted)
    if n == 0:
        steps.shortfall(game, card, "captures nothing: there is no friendly ready Mars creature", "No ready Mars creature")
        return
    steps.capture(game, card, n)
    return
    yield


def yxilo_bolter_after(game, card):
    options = game.all_creatures("any", card)
    if not options:
        steps.shortfall(game, card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, f"{card.name}: choose a creature", options, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    # `deal_damage` returns whoever actually took the hit -- a redirect
    # (Shadow Self) can mean that isn't `choice[0]` -- so the purge check
    # and the destroy check below must both key off the return value.
    hit = steps.deal_damage(game, choice[0], 2)
    if hit is None:
        return
    if hit.type_object.damage >= game.get_power(hit):
        hit.destined_zone = "purged"
    yield from game.check_destroyed([hit])


def yxilx_dominator_register(game, card):
    card.stunned = True


def zorg_register(game, card):
    card.stunned = True


def zorg_before_fight(game, card, target):
    area = game.find_play_area(target)
    neighbors = area.neighbors(target) if area is not None else []
    steps.stun(game, target)
    for n in neighbors:
        steps.stun(game, n)
    return
    yield


def zyzzix_the_many_after(game, card):
    player = controller_of(game, card)
    options = [c for c in player.hand.cards() if c.type == CardType.CREATURE]
    if not options:
        return
    do_it = yield from game.yes_no(
        player.id, f"{card.name}: reveal a creature from your hand?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
    )
    if not do_it:
        return
    choice = yield from game.choose_cards(
        player.id, f"{card.name}: choose a creature to reveal", options, 1, 1,
        source_card=card, intent=DecisionIntent.REVEAL, affects=Affects.FRIENDLY,
    )
    target = choice[0]
    game.log.add("reveal", player=player.id, cards=[target.name], iids=[target.instance_id])
    steps.archive_card(game, player, target)
    card.power_counters += 3


# -------------------------------------------------------------- upgrades --

def _biomatrix_backup_effect(game, host_card):
    host_card.destined_zone = "archive"
    return
    yield


def biomatrix_backup_register(game, card):
    host = card.type_object.host
    if host is not None:
        host.extra_triggers["destroyed"].append(_biomatrix_backup_effect)


def biomatrix_backup_unregister(game, card):
    host = card.type_object.host
    if host is not None and _biomatrix_backup_effect in host.extra_triggers["destroyed"]:
        host.extra_triggers["destroyed"].remove(_biomatrix_backup_effect)


def _clear_house_override_op(game, iid):
    game.card_by_id(iid).house_override = None


register_cleanup_operation("mars.clear_house_override", _clear_house_override_op)


def brain_stem_antenna_register(game, card):
    host = card.type_object.host

    def handler(g, event):
        played = event["card"]
        if host is not None and played is not host and played.house == House.MARS and played.type == CardType.CREATURE:
            steps.ready(g, host)
            host.house_override = House.MARS
            player = g.players[host.controller]
            host.CanBeUsed = player.selected_house is not None and (
                g.get_effective_house(host) == player.selected_house or "versatile" in g.get_keywords(host)
            )
            g._end_of_turn_cleanups.append(("mars.clear_house_override", host.instance_id))
        return
        yield

    game.active_effects.add(TriggerEffect(card, card.controller, "card_played", handler))


def jammer_pack_register(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, INFINITE, 3 - card.controller, "KeyForgeCost", "+", 2))


def _red_planet_ray_gun_effect(game, host_card):
    options = game.all_creatures("any", host_card)
    if not options:
        steps.shortfall(game, host_card, "deals no damage: there is no creature in play", "No creature")
        return
    choice = yield from game.choose_cards(
        host_card.controller, f"{host_card.name}: choose a creature", options, 1, 1,
        source_card=host_card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    n = sum(1 for c in game.all_creatures("any", host_card) if game.get_effective_house(c) == House.MARS)
    # `deal_damage` returns whoever actually took the hit -- a redirect
    # (Shadow Self) can mean that isn't `choice[0]` -- so the destroy check
    # must key off the return value.
    hit = steps.deal_damage(game, choice[0], n)
    yield from game.check_destroyed([hit] if hit is not None else [])


def red_planet_ray_gun_register(game, card):
    host = card.type_object.host
    if host is not None:
        host.extra_triggers["after_reap"].append(_red_planet_ray_gun_effect)


def red_planet_ray_gun_unregister(game, card):
    host = card.type_object.host
    if host is not None and _red_planet_ray_gun_effect in host.extra_triggers["after_reap"]:
        host.extra_triggers["after_reap"].remove(_red_planet_ray_gun_effect)

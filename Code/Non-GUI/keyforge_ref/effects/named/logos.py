"""Bespoke per-card effects for Logos cards that don't fit the generic factories."""

from __future__ import annotations

from ...enums import Affects, DecisionIntent, House
from ...zones import Deck
from ..effect_object import DurationEffect, TriggerEffect, INFINITE, register_cleanup_operation
from .. import steps
from ..generic import controller_of, opponent_of


def help_from_future_self(game, card):
    player = controller_of(game, card)
    found = None
    for c in player.deck.cards():
        if c.name == "Timetraveller":
            found = c
            break
    if found is not None:
        player.deck.remove(found)
    else:
        for c in player.discard.cards():
            if c.name == "Timetraveller":
                found = c
                break
        if found is not None:
            player.discard.remove(found)
    if found is not None:
        player.hand.add(found)
        game.log.add("help_from_future_self", player=player.id, found=True, iid=found.instance_id)
    else:
        steps.shortfall(game, card, f"finds no Timetraveller in {{pos:{player.id}}} deck or discard pile (the discard is still shuffled in)", "No Timetraveller")
    discard_cards = player.discard.take_all()
    if discard_cards:
        player.deck.shuffle_in(discard_cards, game.event_rng("reshuffle", player.id))
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
    steps.draw(game, player, 1, source="Library Access")
    return
    yield


def phase_shift(game, card):
    controller_of(game, card).NonLogosCardsPlayable += 1
    return
    yield


def mother_register(game, card):
    effect = DurationEffect(card, card.controller, INFINITE, card.controller, "DrawUpToLimit", "+", 1)
    game.active_effects.add(effect)


def quixo_after_fight(game, card):
    steps.draw(game, controller_of(game, card), 1, source=card)
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
        player.deck.shuffle_in([card], game.event_rng("reshuffle", player.id))
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
        steps.shortfall(game, card, f"plays nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
        return
    yield from game.play_card_from_deck_top(player, top, ignore_house=True, source=card)


def sloppy_labwork(game, card):
    player = controller_of(game, card)
    if player.hand.cards():
        options = player.hand.cards()
        choice = yield from game.choose_cards(
            player.id, "Sloppy Labwork: archive a card", options, 1, 1,
            source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.FRIENDLY,
        )
        steps.archive_card(game, player, choice[0])
    else:
        steps.shortfall(game, card, f"archives and discards nothing: {{pos:{player.id}}} hand is empty", "Hand is empty")
        return
    if player.hand.cards():
        options = player.hand.cards()
        choice = yield from game.choose_cards(
            player.id, "Sloppy Labwork: discard a card", options, 1, 1,
            source_card=card, intent=DecisionIntent.DISCARD, affects=Affects.FRIENDLY,
        )
        yield from steps.discard_from_hand(game, player, choice[0])
    else:
        steps.shortfall(game, card, f"discards nothing: {{pos:{player.id}}} hand was empty after archiving", "Nothing left to discard")


# ---------------------------------------------------- v2.0 Milestone C ----

def bouncing_deathquark(game, card):
    player = controller_of(game, card)
    while True:
        enemy_targets = game.all_creatures("enemy", card)
        friendly_targets = game.all_creatures("friendly", card)
        if not enemy_targets or not friendly_targets:
            if not enemy_targets and not friendly_targets:
                steps.shortfall(game, card, "destroys nothing: there are no creatures in play", "No creatures")
            else:
                steps.shortfall(game, card, "stops: it needs both an enemy and a friendly creature to destroy", "Nothing left to destroy")
            return
        choice_e = yield from game.choose_cards(
            player.id, "Bouncing Deathquark: destroy an enemy creature", enemy_targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ENEMY,
        )
        choice_f = yield from game.choose_cards(
            player.id, "Bouncing Deathquark: destroy a friendly creature", friendly_targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.FRIENDLY,
        )
        yield from game.destroy_cards(choice_e + choice_f)
        if not game.all_creatures("enemy", card) or not game.all_creatures("friendly", card):
            return
        again = yield from game.yes_no(
            player.id, "Bouncing Deathquark: repeat the effect?", source_card=card, intent=DecisionIntent.OPTIONAL_TRIGGER,
        )
        if not again:
            return


def dimension_door(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, 1, card.controller, "ReapGainBecomesSteal", "=", True))
    return
    yield


def effervescent_principle(game, card):
    for pid in (1, 2):
        player = game.players[pid]
        steps.lose(game, player, player.aember // 2)
    player = controller_of(game, card)
    before = player.chains
    player.chains = min(24, player.chains + 1)
    game.log.add("gain_chains", player=player.id, n=player.chains - before, total=player.chains, card=card.name, iid=card.instance_id)
    return
    yield


def foggify(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, 2, 3 - card.controller, "CanFight", "=", False))
    return
    yield


def interdimensional_graft(game, card):
    effect = TriggerEffect(card, card.controller, "key_forged", _interdimensional_graft_handler(card), remaining_duration=2)
    game.active_effects.add(effect)
    return
    yield


def _interdimensional_graft_handler(source_card):
    def handler(game, event):
        forger_pid = event["player"]
        if forger_pid == source_card.controller:
            return
            yield
        forger = game.players[forger_pid]
        if forger.aember > 0:
            steps.steal(game, forger, game.players[source_card.controller], forger.aember, source=source_card)
        else:
            steps.shortfall(game, source_card, f"gains nothing: {{pos:{forger_pid}}} had no Æmber left after forging", "No Æmber left")
        game.active_effects.remove_from_source(source_card)
    return handler


def knowledge_is_power(game, card):
    player = controller_of(game, card)
    mode = yield from game.choose_mode(
        player.id, "Knowledge is Power: choose one", ["Archive a card", "Gain 1Æ per archived card"], source_card=card,
    )
    if mode == "Archive a card":
        if not player.hand.cards():
            steps.shortfall(game, card, f"archives nothing: {{pos:{player.id}}} hand is empty", "Hand is empty")
            return
        choice = yield from game.choose_cards(
            player.id, "Knowledge is Power: archive a card", player.hand.cards(), 1, 1,
            source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.FRIENDLY,
        )
        steps.archive_card(game, player, choice[0])
    else:
        steps.gain(game, player, len(player.archive))


def neuro_syphon(game, card):
    player = controller_of(game, card)
    opponent = opponent_of(game, card)
    if opponent.aember > player.aember:
        steps.steal(game, opponent, player, 1, source=card)
        steps.draw(game, player, 1, source=card)
    else:
        steps.shortfall(
            game, card,
            f"does nothing: {{pos:{opponent.id}}} Æmber ({opponent.aember}) isn't more than {{mine:{player.id}}} ({player.aember})",
            "Enemy doesn't have more Æmber",
        )
    return
    yield


def positron_bolt(game, card):
    flank_creatures = [
        c for pid in (1, 2) for c in game.players[pid].play_area.creatures if game.players[pid].play_area.is_flank(c)
    ]
    if not flank_creatures:
        steps.shortfall(game, card, "deals no damage: there are no flank creatures in play", "No flank creature")
        return
    choice = yield from game.choose_cards(
        card.controller, "Positron Bolt: deal 3 damage to a flank creature", flank_creatures, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    first = choice[0]
    steps.deal_damage(game, first, 3)
    area = game.players[first.controller].play_area
    neighbors = area.neighbors(first)
    affected = [first]
    if neighbors:
        second = neighbors[0]
        steps.deal_damage(game, second, 2)
        affected.append(second)
        further = [n for n in area.neighbors(second) if n is not first]
        if further:
            third = further[0]
            steps.deal_damage(game, third, 1)
            affected.append(third)
    yield from game.check_destroyed(affected)


def random_access_archives(game, card):
    player = controller_of(game, card)
    top = player.deck.draw_top()
    if top is None:
        steps.shortfall(game, card, f"archives nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
        return
    player.archive.add(top)
    # From the deck (hidden) -- the opponent never legitimately saw this
    # card, so the log entry isn't visible to them either.
    game.log.add("archive", player=player.id, card=top.name, iid=top.instance_id, visible_to={player.id})
    return
    yield


def remote_access(game, card):
    opponent = opponent_of(game, card)
    targets = [
        a for a in opponent.play_area.artifacts
        if (a.card_def.on_action is not None or a.card_def.on_omni is not None) and not a.Exhausted
    ]
    if not targets:
        steps.shortfall(game, card, "uses nothing: the opponent has no usable artifact in play", "No usable enemy artifact")
        return
    choice = yield from game.choose_cards(
        card.controller, "Remote Access: use an opponent's artifact as if it were yours", targets, 1, 1,
        source_card=card, intent=DecisionIntent.USE_TARGET, affects=Affects.ENEMY,
    )
    yield from game.use_artifact_ability(choice[0], card.controller)


def reverse_time(game, card):
    player = controller_of(game, card)
    old_deck_cards = player.deck.cards()
    old_discard_cards = player.discard.take_all()
    player.deck = Deck(old_discard_cards)
    for c in old_deck_cards:
        player.discard.push(c)
    player.deck.shuffle(game.event_rng("deck_shuffle", player.id))
    game.log.add("swap", player=player.id, card=card.name, iid=card.instance_id, swap_kind="deck_discard")
    return
    yield


def twin_bolt_emission(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "deals no damage: there are no creatures in play", "No creatures")
        return
    choice1 = yield from game.choose_cards(
        card.controller, "Twin Bolt Emission: deal 2 damage to a creature", targets, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    first = choice1[0]
    remaining = [c for c in targets if c is not first]
    if not remaining:
        steps.deal_damage(game, first, 2)
        yield from game.check_destroyed([first])
        steps.shortfall(game, card, "deals damage to only 1 creature: it was the only one in play", "Only 1 creature in play")
        return
    choice2 = yield from game.choose_cards(
        card.controller, "Twin Bolt Emission: deal 2 damage to a different creature", remaining, 1, 1,
        source_card=card, intent=DecisionIntent.DAMAGE, affects=Affects.ANY,
    )
    second = choice2[0]
    steps.deal_damage(game, first, 2)
    steps.deal_damage(game, second, 2)
    yield from game.check_destroyed([first, second])


def anomaly_exploiter(game, card):
    targets = [c for c in game.all_creatures("any", card) if c.type_object.damage > 0]
    if not targets:
        steps.shortfall(game, card, "destroys nothing: no creature in play is damaged", "No damaged creature")
        return
    choice = yield from game.choose_cards(
        card.controller, "Anomaly Exploiter: destroy a damaged creature", targets, 1, 1,
        source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ANY,
    )
    yield from game.destroy_cards(choice)


def chaos_portal(game, card):
    player = controller_of(game, card)
    houses = game.player_houses(player.id)
    chosen = yield from game.choose_house(player.id, "Chaos Portal: choose a house", houses)
    top = player.deck.peek_top()
    if top is None:
        steps.shortfall(game, card, f"reveals nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
        return
    game.log.add("reveal_top", player=player.id, card=top.name, iid=top.instance_id)
    if top.house != chosen:
        steps.shortfall(game, card, f"doesn't play {top.name}: it is not {chosen.value}", f"Not {chosen.value}")
        return
    yield from game.play_card_from_deck_top(player, top, ignore_house=True, source=card)


def crazy_killing_machine(game, card):
    not_destroyed_count = 0
    for pid in (1, 2):
        player = game.players[pid]
        top = player.deck.draw_top()
        if top is None:
            not_destroyed_count += 1
            continue
        player.discard.push(top)
        game.log.add("discard", player=pid, card=top.name, iid=top.instance_id)
        targets = [
            c for c in (
                list(game.players[1].play_area.creatures) + list(game.players[1].play_area.artifacts)
                + list(game.players[2].play_area.creatures) + list(game.players[2].play_area.artifacts)
            )
            if game.get_effective_house(c) == top.house
        ]
        if not targets:
            not_destroyed_count += 1
            continue
        choice = yield from game.choose_cards(
            card.controller, f"Crazy Killing Machine: destroy a {top.house.value} creature or artifact", targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ANY,
        )
        yield from game.destroy_cards(choice)
    if not_destroyed_count >= 2:
        yield from game.destroy_cards([card])


def mobius_scroll(game, card):
    # Mobius Scroll itself always archives to its own OWNER (a specific
    # named card follows normal owner-based zone rules, even when used "as
    # if yours" -- Poltergeist, Remote Access, Nexus); only "your hand"
    # below refers to whoever is using the ability.
    player = controller_of(game, card)
    owner = game.players[card.owner]
    area = game.find_play_area(card)
    if area is not None:
        area.remove(card)
        game.leave_play(card)
    owner.archive.add(card)
    game.log.add("archive", player=owner.id, card=card.name, iid=card.instance_id)
    options = player.hand.cards()
    n_max = min(2, len(options))
    if n_max > 0:
        choice = yield from game.choose_cards(
            player.id, "Mobius Scroll: archive up to 2 cards from your hand", options, 0, n_max,
            source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.FRIENDLY, optional=True,
        )
        for c in choice:
            steps.archive_card(game, player, c)
    return
    yield


def spangler_box(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "purges nothing: there are no creatures in play", "No creature to purge")
        return
    choice = yield from game.choose_cards(
        card.controller, "Spangler Box: purge a creature", targets, 1, 1,
        source_card=card, intent=DecisionIntent.PURGE, affects=Affects.ANY,
    )
    target = choice[0]
    steps.purge(game, target)
    target.purged_by = card
    yield from game.take_control(card, 3 - card.controller)


def spectral_tunneler(game, card):
    targets = game.all_creatures("any", card)
    if not targets:
        steps.shortfall(game, card, "chooses nothing: there are no creatures in play", "No creature")
        return
    choice = yield from game.choose_cards(
        card.controller, "Spectral Tunneler: choose a creature", targets, 1, 1,
        source_card=card, intent=DecisionIntent.MODIFY, affects=Affects.ANY,
    )
    target = choice[0]
    target.forced_flank = True
    target.extra_triggers["after_reap"].append(_spectral_tunneler_draw)
    game.log.add(
        "duration_effect", card=card.name, iid=card.instance_id, variable="FlankAndAfterReapDraw", op="=",
        value=True, player=card.controller, affected=[target.controller],
    )
    game._end_of_turn_cleanups.append(("logos.spectral_tunneler_revert", target.instance_id))


def _spectral_tunneler_draw(game, host_card):
    steps.draw(game, game.players[host_card.controller], 1, source="Spectral Tunneler")
    return
    yield


def _spectral_tunneler_revert_op(game, iid):
    target = game.card_by_id(iid)
    target.forced_flank = False
    if _spectral_tunneler_draw in target.extra_triggers["after_reap"]:
        target.extra_triggers["after_reap"].remove(_spectral_tunneler_draw)


register_cleanup_operation("logos.spectral_tunneler_revert", _spectral_tunneler_revert_op)


def strange_gizmo_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "key_forged", _strange_gizmo_handler(card)))


def _strange_gizmo_handler(gizmo_card):
    def handler(game, event):
        if event["player"] != gizmo_card.controller:
            return
            yield
        targets = (
            list(game.players[1].play_area.creatures) + list(game.players[1].play_area.artifacts)
            + list(game.players[2].play_area.creatures) + list(game.players[2].play_area.artifacts)
        )
        yield from game.destroy_cards(targets)
    return handler


def batdrone_fight(game, card):
    steps.steal(game, opponent_of(game, card), controller_of(game, card), 1, source=card)
    return
    yield


def brain_eater_on_destroyed_fighting(game, survivor, victim):
    steps.draw(game, game.players[survivor.controller], 1, source=survivor)
    return
    yield


def dextre_destroyed(game, card):
    card.destined_zone = "deck_top"
    return
    yield


def dr_escotera(game, card):
    opponent = opponent_of(game, card)
    if opponent.keys > 0:
        steps.gain(game, controller_of(game, card), opponent.keys)
    else:
        steps.shortfall(game, card, f"gains nothing: {{pos:{opponent.id}}} has no forged keys", "No forged keys")
    return
    yield


def dysania(game, card):
    opponent = opponent_of(game, card)
    archived = opponent.archive.take_all()
    if not archived:
        steps.shortfall(game, card, f"discards nothing: {{pos:{opponent.id}}} archive is empty", "Archive is empty")
        return
    for c in archived:
        opponent.discard.push(c)
        game.log.add("discard", player=opponent.id, card=c.name, iid=c.instance_id)
    steps.gain(game, controller_of(game, card), len(archived))
    return
    yield


def harland_mindlock(game, card):
    opponent = opponent_of(game, card)
    targets = [c for c in opponent.play_area.creatures if opponent.play_area.is_flank(c)]
    if not targets:
        steps.shortfall(game, card, "takes control of nothing: the opponent has no flank creature", "No enemy flank creature")
        return
    choice = yield from game.choose_cards(
        card.controller, "Harland Mindlock: take control of an enemy flank creature", targets, 1, 1,
        source_card=card, intent=DecisionIntent.TAKE_CONTROL, affects=Affects.ENEMY,
    )
    yield from game.take_control(choice[0], card.controller, until_source=card)


def neutron_shark(game, card):
    while True:
        player = controller_of(game, card)
        opponent = opponent_of(game, card)
        enemy_targets = list(opponent.play_area.creatures) + list(opponent.play_area.artifacts)
        friendly_targets = list(player.play_area.creatures) + list(player.play_area.artifacts)
        if not enemy_targets or not friendly_targets:
            steps.shortfall(game, card, "destroys nothing more: it needs both an enemy and a friendly creature/artifact in play", "Nothing left to destroy")
            return
        choice_e = yield from game.choose_cards(
            player.id, "Neutron Shark: destroy an enemy creature or artifact", enemy_targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.ENEMY,
        )
        choice_f = yield from game.choose_cards(
            player.id, "Neutron Shark: destroy a friendly creature or artifact", friendly_targets, 1, 1,
            source_card=card, intent=DecisionIntent.DESTROY, affects=Affects.FRIENDLY,
        )
        yield from game.destroy_cards(choice_e + choice_f)
        if card not in player.play_area.creatures:
            return  # Neutron Shark stops once it leaves play
        top = player.deck.draw_top()
        if top is None:
            return
        player.discard.push(top)
        game.log.add("discard", player=player.id, card=top.name, iid=top.instance_id)
        if top.house == House.LOGOS:
            return


def novu_archaeologist(game, card):
    player = controller_of(game, card)
    options = player.discard.cards()
    if not options:
        steps.shortfall(game, card, f"archives nothing: {{pos:{player.id}}} discard pile is empty", "Discard pile is empty")
        return
    choice = yield from game.choose_cards(
        player.id, "Novu Archaeologist: archive a card from your discard pile", options, 1, 1,
        source_card=card, intent=DecisionIntent.ARCHIVE, affects=Affects.FRIENDLY,
    )
    c = choice[0]
    player.discard.remove(c)
    player.archive.add(c)
    game.log.add("archive", player=player.id, card=c.name, iid=c.instance_id)


def ozmo(game, card):
    targets = [c for c in game.all_creatures("any", card) if "Mars" in c.tags]
    if not targets:
        steps.shortfall(game, card, "does nothing: there are no Mars creatures in play", "No Mars creature")
        return
    mode = yield from game.choose_mode(
        card.controller, "Ozmo: heal or stun a Mars creature", ["Heal 3", "Stun"], source_card=card,
    )
    # The mode is already decided here, so the target decision can carry its
    # real intent directly -- unlike `HeuristicBot`'s old `_last_mode` hack,
    # nothing downstream needs to remember what an identically-worded prior
    # decision resolved to (Milestone B's motivating example).
    target_intent = DecisionIntent.HEAL if mode == "Heal 3" else DecisionIntent.STUN
    choice = yield from game.choose_cards(
        card.controller, "Ozmo: choose a Mars creature", targets, 1, 1,
        source_card=card, intent=target_intent, affects=Affects.ANY,
    )
    target = choice[0]
    if mode == "Heal 3":
        steps.heal(game, target, 3)
    else:
        target.stunned = True
        game.log.add("stun", card=target.name, iid=target.instance_id)


def psychic_bug(game, card):
    opponent = opponent_of(game, card)
    game.reveal_hand(opponent.id, card.controller, source=card)
    return
    yield


def replicator(game, card):
    player = controller_of(game, card)
    candidates = [
        c for c in (list(game.players[1].play_area.creatures) + list(game.players[2].play_area.creatures))
        if c is not card
        # Excludes other Replicators specifically: its own reap effect is
        # "trigger another creature's reap effect", so chaining into a
        # second one lets that choice chain back here forever.
        and c.card_def.on_reap is not replicator
        and (c.card_def.on_reap is not None or c.extra_triggers.get("after_reap"))
    ]
    if not candidates:
        steps.shortfall(game, card, "copies nothing: no other creature in play has a reap effect", "No reap effect to copy")
        return
    choice = yield from game.choose_cards(
        player.id, "Replicator: choose a creature's reap effect to trigger", candidates, 1, 1,
        source_card=card, intent=DecisionIntent.COPY, affects=Affects.ANY,
    )
    target = choice[0]
    original_controller = target.controller
    target.controller = player.id
    try:
        if target.card_def.on_reap is not None:
            yield from target.card_def.on_reap(game, target)
        for extra in list(target.extra_triggers.get("after_reap", [])):
            yield from extra(game, target)
    finally:
        target.controller = original_controller


def research_smoko_destroyed(game, card):
    # Unqualified "Destroyed:" abilities benefit the controller at the
    # moment of destruction, not the owner (see dust_imp_destroyed).
    player = controller_of(game, card)
    top = player.deck.draw_top()
    if top is None:
        steps.shortfall(game, card, f"archives nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
        return
    player.archive.add(top)
    # From the deck (hidden) -- the opponent never legitimately saw this
    # card, so the log entry isn't visible to them either.
    game.log.add("archive", player=player.id, card=top.name, iid=top.instance_id, visible_to={player.id})
    return
    yield


def skippy_timehog(game, card):
    game.active_effects.add(DurationEffect(card, card.controller, 2, 3 - card.controller, "CannotUseCards", "=", True))
    return
    yield


def vespilon_theorist(game, card):
    player = controller_of(game, card)
    houses = game.player_houses(player.id)
    chosen = yield from game.choose_house(player.id, "Vespilon Theorist: choose a house", houses)
    top = player.deck.draw_top()
    if top is None:
        steps.shortfall(game, card, f"reveals nothing: {{pos:{player.id}}} deck is empty", "Deck is empty")
        return
    game.log.add("reveal_top", player=player.id, card=top.name, iid=top.instance_id)
    if top.house == chosen:
        player.archive.add(top)
        game.log.add("archive", player=player.id, card=top.name, iid=top.instance_id)
        steps.gain(game, player, 1)
    else:
        player.discard.push(top)
        game.log.add("discard", player=player.id, card=top.name, iid=top.instance_id)


def veylan_analyst_register(game, card):
    game.active_effects.add(TriggerEffect(card, card.controller, "artifact_used", _veylan_handler(card)))


def _veylan_handler(veylan_card):
    def handler(game, event):
        if event["player"] == veylan_card.controller:
            steps.gain(game, game.players[veylan_card.controller], 1)
        return
        yield
    return handler


def experimental_therapy(game, card):
    host = card.type_object.host
    host.stunned = True
    host.Exhausted = True
    game.log.add("stun", card=host.name, iid=host.instance_id)
    return
    yield


def rocket_boots_register(game, card):
    host = card.type_object.host
    if host is None:
        return
    host.extra_triggers["after_fight"].append(_rocket_boots_effect)
    host.extra_triggers["after_reap"].append(_rocket_boots_effect)


def rocket_boots_unregister(game, card):
    host = card.type_object.host
    if host is None:
        return
    if _rocket_boots_effect in host.extra_triggers["after_fight"]:
        host.extra_triggers["after_fight"].remove(_rocket_boots_effect)
    if _rocket_boots_effect in host.extra_triggers["after_reap"]:
        host.extra_triggers["after_reap"].remove(_rocket_boots_effect)


def _rocket_boots_effect(game, host_card):
    player = game.players[host_card.controller]
    if player.uses_of(host_card.name) <= 1:
        steps.ready(game, host_card)
    return
    yield


def transposition_sandals_register(game, card):
    host = card.type_object.host
    if host is None:
        return
    host.granted_action = _transposition_sandals_action


def transposition_sandals_unregister(game, card):
    host = card.type_object.host
    if host is not None and host.granted_action is _transposition_sandals_action:
        host.granted_action = None


def _transposition_sandals_action(game, host_card):
    player = game.players[host_card.controller]
    others = [c for c in player.play_area.creatures if c is not host_card]
    if not others:
        steps.shortfall(game, host_card, "swaps with nothing: there is no other friendly creature", "No other friendly creature")
        return
    choice = yield from game.choose_cards(
        player.id, "Transposition Sandals: swap with another friendly creature", others, 1, 1,
        source_card=host_card, intent=DecisionIntent.SWAP, affects=Affects.FRIENDLY,
    )
    other = choice[0]
    area = player.play_area
    i, j = area.creatures.index(host_card), area.creatures.index(other)
    area.creatures[i], area.creatures[j] = area.creatures[j], area.creatures[i]
    game.log.add("swap", player=player.id, card=host_card.name, iid=host_card.instance_id)
    if not other.Exhausted:
        may_use = yield from game.yes_no(
            player.id, f"Transposition Sandals: use {other.name} this turn?",
            source_card=host_card, intent=DecisionIntent.OPTIONAL_TRIGGER,
        )
        if may_use:
            yield from game.use_creature_ability(other)
    else:
        steps.shortfall(game, host_card, f"{other.name} is exhausted and can't be used", f"{other.name} is exhausted")

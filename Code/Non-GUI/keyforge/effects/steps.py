"""Basic effect steps.

Each step operates on already-chosen targets and returns True if it did
something, False if it did nothing at all (a step that only partly works
still counts as success, per the rules). Steps that additionally need a
player choice (destroy pipeline, using a creature's ability) are generator
functions delegating to methods on `game`; the rest are plain functions.
"""

from __future__ import annotations

from ..cards.card import CreatureType
from ..keyed_random import portable_choice


def shortfall(game, source, reason: str, short: str = "") -> None:
    """Log that an effect couldn't be carried out in full, and why.

    `source` is the Card whose effect fell short (or a plain name for
    things like the draw step). `reason` is a predicate that reads after
    the card's name ("steals nothing: {pos:2} Æmber pool is empty").
    Player references are placeholders the UI resolves for whoever is
    reading: {pos:N} -> "your" / "your opponent's" / "Player N's",
    {obj:N} -> "you" / "your opponent" / "Player N". `short` is a few words
    for an on-board popup ("Nothing to steal")."""
    name = getattr(source, "name", source)
    iid = getattr(source, "instance_id", None)
    controller = getattr(source, "controller", None)
    game.log.add("shortfall", card=name, iid=iid, player=controller, reason=reason, short=short)


def gain(game, player, amount: int) -> bool:
    if amount <= 0:
        return False
    for e in game.active_effects.insteads_for("aember_gain"):
        if e.handler(game, player, amount):
            return True
    player.aember += amount
    game.log.add("gain", player=player.id, amount=amount)
    return True


def lose(game, player, amount: int) -> bool:
    n = min(amount, player.aember)
    if n <= 0:
        return False
    player.aember -= n
    game.log.add("lose", player=player.id, amount=n)
    return True


def steal(game, from_player, to_player, amount: int, source=None) -> bool:
    if amount > 0 and from_player.get_cannot_be_stolen_from(game):
        if source is not None:
            shortfall(game, source, f"steals nothing: {{pos:{from_player.id}}} Æmber cannot be stolen", "Cannot be stolen")
        return False
    n = min(amount, from_player.aember)
    if source is not None and 0 < n < amount:
        shortfall(game, source, f"steals only {n} of {amount} Æmber: that was all of {{pos:{from_player.id}}} Æmber", f"Only {n} Æmber to steal")
    if n <= 0:
        if source is not None and amount > 0:
            shortfall(game, source, f"steals nothing: {{pos:{from_player.id}}} Æmber pool is empty", "Nothing to steal")
        return False
    from_player.aember -= n
    to_player.aember += n
    game.log.add("steal", frm=from_player.id, to=to_player.id, amount=n)
    return True


def capture(game, card, amount: int) -> bool:
    if amount <= 0:
        return False
    opponent = game.players[3 - card.controller]
    n = min(amount, opponent.aember)
    if n <= 0:
        shortfall(game, card, f"captures nothing: {{pos:{opponent.id}}} Æmber pool is empty", "Nothing to capture")
        return False
    if n < amount:
        shortfall(game, card, f"captures only {n} of {amount} Æmber: that was all of {{pos:{opponent.id}}} Æmber", f"Only {n} Æmber to capture")
    opponent.aember -= n
    card.aember_captured += n
    game.log.add("capture", card=card.name, iid=card.instance_id, amount=n)
    return True


def capture_from_own_side(game, card, amount: int) -> bool:
    """Like `capture`, but `card` captures Æmber from its OWN controller's
    pool, not the opponent's (Mindwarper: an enemy creature "captures 1
    from its own side"). The captured Æmber still releases to `card`'s
    controller's opponent when it leaves play, same as any other capture."""
    if amount <= 0:
        return False
    own_player = game.players[card.controller]
    n = min(amount, own_player.aember)
    if n <= 0:
        shortfall(game, card, f"captures nothing: {{pos:{own_player.id}}} Æmber pool is empty", "Nothing to capture")
        return False
    if n < amount:
        shortfall(game, card, f"captures only {n} of {amount} Æmber: that was all of {{pos:{own_player.id}}} Æmber", f"Only {n} Æmber to capture")
    own_player.aember -= n
    card.aember_captured += n
    game.log.add("capture", card=card.name, iid=card.instance_id, amount=n)
    return True


def draw(game, player, n: int, source=None) -> bool:
    if n <= 0:
        return False
    drawn_iids = []
    for _ in range(n):
        if player.deck.is_empty():
            if player.discard.is_empty():
                break
            cards = player.discard.take_all()
            player.deck.shuffle_in(cards, game.event_rng("reshuffle", player.id))
            game.log.add("reshuffle", player=player.id)
        card = player.deck.draw_top()
        if card is None:
            break
        player.hand.add(card)
        drawn_iids.append(card.instance_id)
    if drawn_iids:
        game.log.add("draw", player=player.id, n=len(drawn_iids), iids=drawn_iids)
    if source is not None and len(drawn_iids) < n:
        why = f"{{pos:{player.id}}} deck and discard pile are both empty"
        if drawn_iids:
            shortfall(game, source, f"draws only {len(drawn_iids)} of {n} cards: {why}", "Deck ran out")
        else:
            shortfall(game, source, f"draws nothing: {why}", "No cards left to draw")
    return len(drawn_iids) > 0


def archive_card(game, player, card) -> bool:
    if not player.hand.remove(card):
        if not player.deck.remove(card):
            return False
    player.archive.add(card)
    # Always from hand or deck (both hidden) -- see keyforge/log.py.
    game.log.add("archive", player=player.id, card=card.name, iid=card.instance_id, visible_to={player.id})
    return True


def discard_from_hand(game, player, card):
    if not player.hand.remove(card):
        return False
    player.discard.push(card)
    game.log.add("discard", player=player.id, card=card.name, iid=card.instance_id)
    yield from game._fire_event("card_discarded_from_hand", {"player": player.id, "card": card})
    return True


def discard_random(game, player, source=None):
    cards = player.hand.cards()
    if not cards:
        if source is not None:
            shortfall(game, source, f"discards nothing: {{pos:{player.id}}} hand is empty", "Hand is empty")
        return False
    card = portable_choice(game.event_rng("random_discard", player.id), cards)
    player.hand.remove(card)
    player.discard.push(card)
    game.log.add("discard_random", player=player.id, card=card.name, iid=card.instance_id)
    yield from game._fire_event("card_discarded_from_hand", {"player": player.id, "card": card})
    return True


def purge(game, card) -> bool:
    owner = game.players[card.owner]
    # A card in play sits in whichever play area actually contains it
    # (looked up by membership, not by `card.controller` -- see
    # `Game.find_play_area`); every other zone is always the owner's.
    area = game.find_play_area(card)
    if area is not None:
        area.remove(card)
        game.leave_play(card)
    elif not (owner.hand.remove(card) or owner.discard.remove(card) or owner.archive.remove(card) or owner.deck.remove(card)):
        return False
    owner.purged.add(card)
    game.log.add("purge", card=card.name, iid=card.instance_id, owner=owner.id)
    return True


def put_on_top(game, player, card) -> bool:
    player.deck.put_on_top(card)
    game.log.add("put_on_top", player=player.id, card=card.name, iid=card.instance_id)
    return True


def put_on_bottom(game, player, card) -> bool:
    player.deck.put_on_bottom(card)
    game.log.add("put_on_bottom", player=player.id, card=card.name, iid=card.instance_id)
    return True


def deal_damage(game, creature, amount: int, ignore_armor: bool = False):
    """Deals `amount` damage to `creature`. A "cannot be dealt damage"
    prevention (Shield of Justice, Potion of Invulnerability, Protectrix)
    is checked first, ahead of armor -- it's a full replacement, not a
    reduction, but the attempt still counts as having happened for other
    triggers (MRB 18.3 FAQ, Shoulder Id). Armor absorbs next (unless
    `ignore_armor`, Qyxxlyx Plague Master: "cannot be prevented by armor"),
    on `creature` itself; only the leftover (if any) is subject to a
    redirect effect (Shadow Self) -- "redirect after armor". Returns the
    creature that actually took nonzero damage (the redirect target, if any
    damage got past prevention and armor), or None otherwise -- callers
    that need to apply "this fight's damage" logic (poison) to the right
    creature should key off this return value, not the original target."""
    if amount <= 0:
        return None
    if not isinstance(creature.type_object, CreatureType):
        return None
    if creature.damage_prevented or game.players[creature.controller].get_cannot_be_dealt_damage(game):
        game.log.add("damage_prevented", card=creature.name, iid=creature.instance_id, amount=amount)
        return None
    to = creature.type_object
    available_armor = 0 if ignore_armor else max(0, game.get_armor(creature) - to.armor_used_this_turn)
    absorbed = min(available_armor, amount)
    to.armor_used_this_turn += absorbed
    remaining = amount - absorbed
    game.log.add("damage", card=creature.name, iid=creature.instance_id, amount=amount, absorbed=absorbed)
    if remaining <= 0:
        return None
    target = game.resolve_damage_target(creature)
    if target is not creature:
        game.log.add(
            "damage_redirected", card=creature.name, iid=creature.instance_id,
            to=target.name, to_iid=target.instance_id, amount=remaining,
        )
        game._redirected_hits.append(target)
    target.type_object.damage += remaining
    return target


def stun(game, card) -> bool:
    if not isinstance(card.type_object, CreatureType):
        return False
    card.stunned = True
    game.log.add("stun", card=card.name, iid=card.instance_id)
    return True


def gain_chains(game, player, source, n: int) -> int:
    """Adds `n` chains to `player`, capped at 24 (Gateway to Dis, Coward's
    End). Returns how many were actually added."""
    if n <= 0:
        return 0
    before = player.chains
    player.chains = min(24, player.chains + n)
    added = player.chains - before
    game.log.add(
        "gain_chains", player=player.id, n=added, total=player.chains,
        card=getattr(source, "name", source), iid=getattr(source, "instance_id", None),
    )
    return added


def place_aember(game, card, amount: int) -> bool:
    """Places `amount` Æmber from the common supply directly onto `card`
    (Blood Money) -- unlike `capture`, this doesn't come out of either
    player's pool. It behaves exactly like captured Æmber from then on:
    released to the *card's controller's opponent* if the card leaves
    play (see `Game.leave_play`)."""
    if amount <= 0:
        return False
    card.aember_captured += amount
    game.log.add("place_aember", card=card.name, iid=card.instance_id, amount=amount)
    return True


def heal(game, creature, amount: int) -> int:
    if amount <= 0 or not isinstance(creature.type_object, CreatureType):
        return 0
    to = creature.type_object
    healed = min(amount, to.damage)
    to.damage -= healed
    if healed:
        game.log.add("heal", card=creature.name, iid=creature.instance_id, amount=healed)
    return healed


def fully_heal(game, creature) -> int:
    """Heals all of `creature`'s current damage (Duma the Martyr, Mugwump,
    Yo Mama Mastery, Armageddon Cloak) -- Milestone C.4."""
    if not isinstance(creature.type_object, CreatureType):
        return 0
    return heal(game, creature, creature.type_object.damage)


def sacrifice(game, card):
    """Destroy a card in play (sacrifice uses the same Destroy pipeline)."""
    if game.find_play_area(card) is None:
        return False
    destroyed = yield from game.destroy_cards([card])
    return card in destroyed


def return_to_hand(game, card) -> bool:
    """Returns `card` from play to its OWNER's hand -- always, even when the
    card text says "your hand" (Wardrummer, Total Recall): MRB 18.3
    "Movement between zones" puts a card leaving play in its owner's zone
    unless the ability explicitly names another zone, and the Faygin FAQ
    applies that to exactly this wording."""
    receiver = game.players[card.owner]
    area = game.find_play_area(card)
    if area is None or not area.remove(card):
        return False
    game.leave_play(card)
    receiver.hand.add(card)
    game.log.add("return_to_hand", card=card.name, iid=card.instance_id, owner=receiver.id)
    return True


def shuffle_into_deck(game, card) -> bool:
    owner = game.players[card.owner]
    found = (
        owner.hand.remove(card)
        or owner.discard.remove(card)
        or owner.archive.remove(card)
    )
    if not found:
        return False
    owner.deck.shuffle_in([card], game.event_rng("reshuffle", owner.id))
    # From hand, discard, or the owner's own archive -- discard is public,
    # but hand and archive aren't, and there's no cheap way here to tell
    # which one matched, so this errs toward not leaking.
    game.log.add("shuffle_into_deck", card=card.name, iid=card.instance_id, owner=owner.id, visible_to={owner.id})
    return True


def ready(game, card) -> bool:
    if not card.Exhausted:
        return False
    card.Exhausted = False
    game.log.add("ready", card=card.name, iid=card.instance_id)
    return True


def exhaust(game, card) -> bool:
    if card.Exhausted:
        return False
    card.Exhausted = True
    game.log.add("exhaust", card=card.name, iid=card.instance_id)
    return True


def use_creature(game, card):
    """Reap, fight, or use the Action of a friendly creature, whichever is
    possible and legal (ignoring house). Generator: may need a fight target."""
    yield from game.use_creature_ability(card)
    return True

"""BoardSnapshot: a plain-data picture of the whole `Game` object at one
instant, from one viewer's perspective.

Every physical card the engine has ever created gets a `CardState` (even
ones sitting in a hidden zone), keyed by `instance_id`, so the animation
layer can track a stable sprite for every card all game long. `face_up`
is the *only* thing that gates whether that sprite is allowed to show its
front: it is computed strictly from the spec's visibility rules, so a
hidden card's name/house/image are present in the data model (needed to
draw *something* and to size zones correctly) but the rendering layer
must never draw a face for a card whose `face_up` is False for the
current viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from keyforge.cards.card import CreatureType
from keyforge.enums import CardType

ZONE_DECK = "deck"
ZONE_HAND = "hand"
ZONE_DISCARD = "discard"
ZONE_ARCHIVE = "archive"
ZONE_PURGED = "purged"
ZONE_CREATURE = "play_creature"
ZONE_ARTIFACT = "play_artifact"
ZONE_UPGRADE = "upgrade"

# An upgrade is attached to a creature already in play, so it's public
# information exactly like the creature itself -- omitting it here made
# every upgrade render as a face-down card-back glued to its host (see
# UX_FIX_PLAN.md's second-pass findings).
PUBLIC_ZONES = {ZONE_DISCARD, ZONE_PURGED, ZONE_CREATURE, ZONE_ARTIFACT, ZONE_UPGRADE}


@dataclass
class CardState:
    iid: int
    name: str
    house: str
    type: str
    image: Optional[str]
    owner: int
    controller: int
    zone: str
    index: int
    zone_count: int
    exhausted: bool = False
    power: int = 0
    armor: int = 0
    armor_used: int = 0
    damage: int = 0
    aember_captured: int = 0
    elusive: bool = False
    skirmish: bool = False
    taunt: bool = False
    poison: bool = False
    hazardous: int = 0
    assault: int = 0
    versatile: bool = False
    cannot_be_dealt_damage: bool = False
    stunned: bool = False
    power_counters: int = 0
    aember_stored: int = 0
    host_iid: Optional[int] = None
    upgrade_iids: List[int] = field(default_factory=list)
    under_count: int = 0
    face_up: bool = True


@dataclass
class PlayerSnapshot:
    id: int
    aember: int
    keys: int
    chains: int
    selected_house: Optional[str]
    hand_count: int
    deck_count: int
    discard_count: int
    archive_count: int
    purged_count: int
    key_cost: int = 6
    can_forge: bool = True


@dataclass
class BoardSnapshot:
    viewer: int
    active_player: int
    turn_number: int
    is_over: bool
    result: Optional[dict]
    players: Dict[int, PlayerSnapshot]
    cards: Dict[int, CardState]
    active_effects: List[dict]
    first_player: int = 1

    def zone_cards(self, pid: int, zone: str) -> List[CardState]:
        out = [c for c in self.cards.values() if c.owner == pid and c.zone == zone]
        out.sort(key=lambda c: c.index)
        return out

    def creatures_and_artifacts(self, pid: int) -> List[CardState]:
        return self.zone_cards(pid, ZONE_CREATURE) + self.zone_cards(pid, ZONE_ARTIFACT)


def _card_state(game, card, owner: int, zone: str, index: int, zone_count: int, viewer: int) -> CardState:
    cdef = card.card_def
    to = card.type_object
    is_creature = isinstance(to, CreatureType)
    power = game.get_power(card) if is_creature else getattr(to, "base_power", 0)
    armor = game.get_armor(card) if is_creature else getattr(to, "base_armor", 0)
    armor_used = getattr(to, "armor_used_this_turn", 0)
    damage = getattr(to, "damage", 0)
    keywords = game.get_keywords(card)
    cannot_be_dealt_damage = is_creature and (
        card.damage_prevented or game.players[card.controller].get_cannot_be_dealt_damage(game)
    )
    host_iid = None
    if hasattr(to, "host") and to.host is not None:
        host_iid = to.host.instance_id
    upgrade_iids = []
    if isinstance(to, CreatureType):
        upgrade_iids = [u.instance_id for u in to.upgrades]

    if zone in PUBLIC_ZONES:
        face_up = True
    elif zone in (ZONE_HAND, ZONE_ARCHIVE):
        face_up = owner == viewer
    else:  # deck
        face_up = False

    return CardState(
        iid=card.instance_id,
        name=card.name,
        house=card.house.value,
        type=card.type.value,
        image=cdef.image,
        owner=owner,
        controller=card.controller,
        zone=zone,
        index=index,
        zone_count=zone_count,
        exhausted=card.Exhausted,
        power=power,
        armor=armor,
        armor_used=armor_used,
        damage=damage,
        aember_captured=card.aember_captured,
        elusive="elusive" in keywords,
        skirmish="skirmish" in keywords,
        taunt="taunt" in keywords,
        poison="poison" in keywords,
        hazardous=game.get_hazardous(card) if is_creature else 0,
        assault=game.get_assault(card) if is_creature else 0,
        versatile="versatile" in keywords,
        cannot_be_dealt_damage=cannot_be_dealt_damage,
        stunned=card.stunned,
        power_counters=card.power_counters,
        aember_stored=card.aember_stored,
        host_iid=host_iid,
        upgrade_iids=upgrade_iids,
        under_count=len(card.under_cards),
        face_up=face_up,
    )


def build_snapshot(game, viewer: int) -> BoardSnapshot:
    cards: Dict[int, CardState] = {}
    players: Dict[int, PlayerSnapshot] = {}

    for pid, player in game.players.items():
        zones = [
            (ZONE_DECK, player.deck.cards()),
            (ZONE_HAND, player.hand.cards()),
            (ZONE_DISCARD, player.discard.cards()),
            (ZONE_ARCHIVE, player.archive.cards()),
            (ZONE_PURGED, player.purged.cards()),
            (ZONE_CREATURE, list(player.play_area.creatures)),
            (ZONE_ARTIFACT, list(player.play_area.artifacts)),
        ]
        for zone, zone_cards in zones:
            if zone == ZONE_HAND and pid == viewer:
                # The viewer's own hand is shown sorted -- active house first,
                # then by house, type, name. The opponent's hand keeps engine
                # order: sorting it would leak each drawn card's house through
                # where its (face-down) card lands in the row.
                active = player.selected_house
                zone_cards = sorted(
                    zone_cards,
                    key=lambda c: (active is not None and c.house != active, c.house.value, c.type.value, c.name),
                )
            n = len(zone_cards)
            for i, card in enumerate(zone_cards):
                cards[card.instance_id] = _card_state(game, card, pid, zone, i, n, viewer)
            # upgrades attached to creatures aren't in any top-level zone list;
            # surface them too, positioned by (tucked under their host).
            if zone == ZONE_CREATURE:
                for i, creature in enumerate(zone_cards):
                    for u_idx, upg in enumerate(creature.type_object.upgrades):
                        # An upgrade's owner is whoever's decklist it came
                        # from, not whichever side its host creature is on --
                        # an upgrade can attach to an enemy creature (Collar
                        # of Subordination), and `pid` here is the host's
                        # side.
                        cards[upg.instance_id] = _card_state(
                            game, upg, upg.owner, "upgrade", u_idx, len(creature.type_object.upgrades), viewer
                        )

        players[pid] = PlayerSnapshot(
            id=pid,
            aember=player.aember,
            keys=player.keys,
            chains=player.chains,
            selected_house=player.selected_house.value if player.selected_house else None,
            hand_count=len(player.hand),
            deck_count=len(player.deck),
            discard_count=len(player.discard),
            archive_count=len(player.archive),
            purged_count=len(player.purged),
            key_cost=player.get_key_forge_cost(game),
            can_forge=player.get_can_key_forge(game),
        )

    active_effects = [
        {
            "source_name": e.source_card.name if e.source_card is not None else "?",
            "source_iid": e.source_card.instance_id if e.source_card is not None else -1,
            "_house": e.source_card.house.value if e.source_card is not None else None,
            "player_affected": e.player_affected,
            "variable": e.variable,
            "op": e.op,
            "value": e.value,
            "remaining_duration": e.remaining_duration,
        }
        for e in game.active_effects.duration_effects
    ]

    return BoardSnapshot(
        viewer=viewer,
        active_player=game.active_player_id,
        turn_number=game.turn_number,
        is_over=game.is_over,
        result=game.result,
        players=players,
        cards=cards,
        active_effects=active_effects,
        first_player=game.first_player,
    )

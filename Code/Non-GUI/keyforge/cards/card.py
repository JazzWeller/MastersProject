"""Card and TypeObject definitions."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..enums import CardType, House

_instance_counter = itertools.count(1)


class TypeObject:
    """Base for the per-type behaviour object attached to a Card."""

    def __init__(self):
        self.card: Optional["Card"] = None


class ActionType(TypeObject):
    pass


class ArtifactType(TypeObject):
    pass


class CreatureType(TypeObject):
    def __init__(self, power: int, armor: int = 0):
        super().__init__()
        self.base_power = power
        self.base_armor = armor
        self.damage = 0
        self.armor_used_this_turn = 0
        self.upgrades: List["Card"] = []


class UpgradeType(TypeObject):
    def __init__(self):
        super().__init__()
        self.host: Optional["Card"] = None


@dataclass
class CardDef:
    """Static definition of a printed card, shared by every physical copy."""

    id: int  # the pool catalog number (Code/PHASE_2_CARD_POOL.md's "#" column); not the id printed on the card
    name: str
    house: House
    type: CardType
    aember_on_play: int = 0
    aember_captured: int = 0
    tags: tuple = ()  # traits (creature races/classes) and artifact/upgrade subtypes (Weapon, Location, ...)
    keywords: tuple = ()  # elusive, skirmish, taunt, poison, versatile (own printed keywords)
    text: str = ""  # canonical rules text (errata applied), for the card inspector
    errata: Optional[str] = None  # note on which official errata (if any) canonical text applies
    image: Optional[str] = None
    power: int = 0
    armor: int = 0
    # upgrade-only: what this upgrade grants its host while attached (Flame-Wreathed, Ring of Invisibility)
    power_bonus: int = 0
    armor_bonus: int = 0  # Protect the Weak; Shoulder Armor's flank-conditional bonus uses ModifierEffect instead
    grants_keywords: tuple = ()
    hazardous: int = 0
    spendable_for_keys: bool = False  # Pocket Universe, Safe Place: aember_stored may pay forge costs
    play_cost_aember: int = 0  # Truebaru: must lose this much Æmber in order to play
    min_aember_to_play: int = 0  # Kelifi Dragon: needs this much Æmber in the pool to play, but doesn't spend it
    cannot_reap: bool = False  # Tireless Crocag
    on_play: Optional[Callable] = None          # generator effect function(game, card) or None
    on_reap: Optional[Callable] = None
    on_before_fight: Optional[Callable] = None  # Firespitter: resolves as the attacker, before the fight itself
    on_fight: Optional[Callable] = None
    on_action: Optional[Callable] = None
    on_omni: Optional[Callable] = None
    on_destroyed: Optional[Callable] = None
    # (game, survivor, victim) -> generator; fires on the creature that
    # *survived* a fight when the other one died (Overlord Greking, Stealer
    # of Souls, Brain Eater). Not fired if both die.
    on_destroyed_fighting: Optional[Callable] = None
    register_passive: Optional[Callable] = None  # (game, card) -> None, called on enter play
    unregister_passive: Optional[Callable] = None  # (game, card) -> None, called on leave play


class Card:
    """A single physical instance of a card."""

    def __init__(self, card_def: CardDef, owner: int):
        self.instance_id = next(_instance_counter)
        self.id = card_def.id
        self.name = card_def.name
        self.house = card_def.house
        self.type = card_def.type
        self.tags = card_def.tags
        self.keywords = card_def.keywords
        self.aember_on_play = card_def.aember_on_play
        self.aember_captured = card_def.aember_captured
        self.image = card_def.image
        self.card_def = card_def
        self.owner = owner
        self.controller = owner

        if card_def.type == CardType.CREATURE:
            self.type_object: TypeObject = CreatureType(card_def.power, card_def.armor)
        elif card_def.type == CardType.ARTIFACT:
            self.type_object = ArtifactType()
        elif card_def.type == CardType.UPGRADE:
            self.type_object = UpgradeType()
        else:
            self.type_object = ActionType()
        self.type_object.card = self

        # Elusive/Skirmish/Taunt/Poison/Versatile are computed from
        # `game.get_keywords(card)` (printed + upgrade-granted), not stored
        # here -- see Code/PHASE_2_PLAN.md Milestone B.1.
        self.IgnoreElusive = False
        self.CanBeUsed = False
        self.Exhausted = True
        self.destroyed = False
        self.fought_this_turn = False  # was Target of a fight (for Elusive)
        self.flank: Optional[str] = None
        # extra abilities granted by upgrades: event name -> list of (game, card) -> generator
        self.extra_triggers: dict = {"after_reap": [], "after_fight": [], "destroyed": []}
        # overrides where a destroyed card goes instead of the discard pile (Bad Penny)
        self.destined_zone: Optional[str] = None
        self.power_counters = 0  # +1 power counters (Eater of the Dead)
        self.stunned = False  # Experimental Therapy, Ozmo: next use only exhausts, no effect
        self.house_override: Optional[House] = None  # Sneklifter-style re-housing
        self.forced_flank = False  # Spectral Tunneler: considered a flank creature for the rest of the turn
        self.granted_action: Optional[Callable] = None  # Transposition Sandals: an Action ability granted dynamically
        self.aember_stored = 0  # Pocket Universe, Safe Place: spendable at forge time, lost (not released) on leaving play
        self.under_cards: List["Card"] = []  # Masterplan: cards placed facedown beneath this one
        self.purged_by: Optional["Card"] = None  # Spangler Box: tracks what purged this, to return it later
        self.redirect_fight_damage_to: Optional["Card"] = None  # Gabos Longarms: this fight's damage goes here instead
        self.armor_negated = False  # Red-Hot Armor: loses all of its armor until the end of the turn
        self.damage_prevented = False  # Protectrix: cannot be dealt damage until the end of the turn

    def __repr__(self):
        return f"<Card {self.name} #{self.instance_id}>"

    def reset_on_leave_play(self):
        if isinstance(self.type_object, CreatureType):
            self.type_object.damage = 0
            self.type_object.armor_used_this_turn = 0
            self.type_object.upgrades = []
        self.Exhausted = True
        self.destroyed = False
        self.fought_this_turn = False
        self.CanBeUsed = False
        self.flank = None
        self.destined_zone = None
        self.power_counters = 0
        self.stunned = False
        self.house_override = None
        self.aember_stored = 0
        self.forced_flank = False
        self.granted_action = None
        self.armor_negated = False
        self.damage_prevented = False

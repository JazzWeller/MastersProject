"""Definitions for all 49 Phase 1 cards."""

from __future__ import annotations

from ..enums import CardType, House
from ..effects import generic, named
from .card import CardDef

DIS = House.DIS
LOGOS = House.LOGOS
SHADOWS = House.SHADOWS
ACTION = CardType.ACTION
ARTIFACT = CardType.ARTIFACT
CREATURE = CardType.CREATURE
UPGRADE = CardType.UPGRADE


def _passive(register, unregister):
    return register, unregister


CARD_DEFS = {}


def _define(**kwargs) -> CardDef:
    cd = CardDef(**kwargs)
    CARD_DEFS[cd.name] = cd
    return cd


# ------------------------------------------------------------------ Dis --

_define(id=94, name="Arise", house=DIS, image="Dis/arise.png", type=ACTION, aember_on_play=0, on_play=named.arise)

_define(id=55, name="Control the Weak", house=DIS, image="Dis/control-the-weak.png", type=ACTION, aember_on_play=1, on_play=named.control_the_weak)

_define(id=56, name="Creeping Oblivion", house=DIS, image="Dis/creeping-oblivion.png", type=ACTION, aember_on_play=1, on_play=named.creeping_oblivion)

_define(
    id=73,
    name="Dominator Bauble",
    house=DIS, image="Dis/dominator-bauble.png",
    type=ARTIFACT,
    aember_on_play=0,
    tags=("Item",),
    on_action=named.dominator_bauble,
)

_define(
    id=83,
    name="Dust Imp",
    house=DIS, image="Dis/dust-imp.png",
    type=CREATURE,
    power=2,
    armor=0,
    tags=("Imp",),
    on_destroyed=named.dust_imp_destroyed,
)

_define(
    id=85,
    name="Ember Imp",
    house=DIS, image="Dis/ember-imp.png",
    type=CREATURE,
    power=2,
    armor=0,
    tags=("Imp",),
    register_passive=named.ember_imp_register,
    unregister_passive=named.ember_imp_unregister,
)

_define(id=59, name="Gateway to Dis", house=DIS, image="Dis/gateway-to-dis.png", type=ACTION, aember_on_play=0, on_play=named.gateway_to_dis)

_define(
    id=84,
    name="Guardian Demon",
    house=DIS, image="Dis/guardian-demon.png",
    type=CREATURE,
    power=4,
    armor=0,
    tags=("Demon",),
    on_play=named.guardian_demon,
    on_reap=named.guardian_demon,
    on_fight=named.guardian_demon,
)

_define(
    id=75,
    name="Lash of Broken Dreams",
    house=DIS, image="Dis/lash-of-broken-dreams.png",
    type=ARTIFACT,
    tags=("Weapon",),
    on_action=generic.duration_effect("KeyForgeCost", "+", 3, 2, "enemy"),
)

_define(
    id=76,
    name="Library of the Damned",
    house=DIS, image="Dis/library-of-the-damned.png",
    type=ARTIFACT,
    tags=("Location",),
    on_action=generic.archive_n(1),
)

_define(
    id=77,
    name="Lifeward",
    house=DIS, image="Dis/lifeward.png",
    type=ARTIFACT,
    aember_on_play=1,
    tags=("Power",),
    on_omni=named.lifeward_omni,
)

_define(
    id=92,
    name="Pit Demon",
    house=DIS, image="Dis/pit-demon.png",
    type=CREATURE,
    power=5,
    armor=0,
    tags=("Demon",),
    on_action=generic.steal_n(1),
)

_define(
    id=96,
    name="Shooler",
    house=DIS, image="Dis/shooler.png",
    type=CREATURE,
    power=5,
    armor=0,
    tags=("Demon",),
    on_play=named.shooler_play,
)

_define(
    id=97,
    name="Snudge",
    house=DIS, image="Dis/snudge.png",
    type=CREATURE,
    power=4,
    armor=0,
    tags=("Demon",),
    on_reap=named.snudge,
    on_fight=named.snudge,
)

_define(
    id=99,
    name="Succubus",
    house=DIS, image="Dis/succubus.png",
    type=CREATURE,
    power=3,
    armor=0,
    tags=("Demon",),
    register_passive=named.succubus_register,
    unregister_passive=lambda game, card: game.active_effects.remove_from_source(card),
)

_define(
    id=101,
    name="The Terror",
    house=DIS, image="Dis/the-terror.png",
    type=CREATURE,
    power=5,
    armor=0,
    tags=("Demon", "Knight"),
    on_play=named.the_terror_play,
)

_define(id=88, name="Three Fates", house=DIS, image="Dis/three-fates.png", type=ACTION, aember_on_play=0, on_play=named.three_fates)


# ---------------------------------------------------------------- Logos --

_define(
    id=311,
    name="Doc Bookton",
    house=LOGOS, image="Logos/doc-bookton.png",
    type=CREATURE,
    power=5,
    armor=0,
    tags=("Human", "Scientist"),
    on_reap=generic.draw_n(1),
)

_define(
    id=111,
    name="Help From Future Self",
    house=LOGOS, image="Logos/help-from-future-self.png",
    type=ACTION,
    aember_on_play=1,
    on_play=named.help_from_future_self,
)

_define(id=114, name="Labwork", house=LOGOS, image="Logos/labwork.png", type=ACTION, aember_on_play=1, on_play=generic.archive_n(1))

_define(id=317, name="Library Access", house=LOGOS, image="Logos/library-access.png", type=ACTION, aember_on_play=0, on_play=named.library_access_play)

_define(
    id=129,
    name="Library of Babble",
    house=LOGOS, image="Logos/library-of-babble.png",
    type=ARTIFACT,
    tags=("Location",),
    on_action=generic.draw_n(1),
)

_define(
    id=156,
    name="Mother",
    house=LOGOS, image="Logos/mother.png",
    type=CREATURE,
    power=5,
    armor=0,
    tags=("Robot", "Scientist"),
    register_passive=named.mother_register,
    unregister_passive=lambda game, card: game.active_effects.remove_from_source(card),
)

_define(id=157, name="Phase Shift", house=LOGOS, image="Logos/phase-shift.png", type=ACTION, aember_on_play=0, on_play=named.phase_shift)

_define(
    id=144,
    name="Quixo the Adventurer",
    house=LOGOS, image="Logos/quixo-the-adventurer.png",
    type=CREATURE,
    power=3,
    armor=0,
    tags=("Human", "Scientist"),
    on_fight=named.quixo_after_fight,
    register_passive=named.quixo_register,
    unregister_passive=named.quixo_unregister,
)

_define(
    id=122,
    name="Scrambler Storm",
    house=LOGOS, image="Logos/scrambler-storm.png",
    type=ACTION,
    aember_on_play=1,
    on_play=generic.duration_effect("CanPlayActions", "=", False, 2, "enemy"),
)

_define(id=123, name="Sloppy Labwork", house=LOGOS, image="Logos/sloppy-labwork.png", type=ACTION, aember_on_play=1, on_play=named.sloppy_labwork)

_define(
    id=135,
    name="The Howling Pit",
    house=LOGOS, image="Logos/the-howling-pit.png",
    type=ARTIFACT,
    aember_on_play=1,
    tags=("Location",),
    register_passive=named.the_howling_pit_register,
    unregister_passive=lambda game, card: game.active_effects.remove_from_source(card),
)

_define(
    id=153,
    name="Timetraveler",
    house=LOGOS, image="Logos/timetraveller.png",
    type=CREATURE,
    power=2,
    armor=0,
    aember_on_play=1,
    tags=("Human", "Scientist"),
    on_play=generic.draw_n(2),
    on_action=named.timetraveler_action,
)

_define(
    id=154,
    name="Titan Mechanic",
    house=LOGOS, image="Logos/titan-mechanic.png",
    type=CREATURE,
    power=6,
    armor=0,
    tags=("Cyborg", "Scientist"),
    register_passive=named.titan_mechanic_register,
    unregister_passive=lambda game, card: game.active_effects.remove_from_source(card),
)

_define(id=125, name="Wild Wormhole", house=LOGOS, image="Logos/wild-wormhole.png", type=ACTION, aember_on_play=1, on_play=named.wild_wormhole)


# -------------------------------------------------------------- Shadows --

_define(
    id=296,
    name="Bad Penny",
    house=SHADOWS, image="Shadows/bad-penny.png",
    type=CREATURE,
    power=1,
    armor=0,
    tags=("Human", "Thief"),
    on_destroyed=named.bad_penny_destroyed,
)

_define(id=267, name="Bait and Switch", house=SHADOWS, image="Shadows/bait-and-switch.png", type=ACTION, aember_on_play=0, on_play=named.bait_and_switch)

_define(id=268, name="Booby Trap", house=SHADOWS, image="Shadows/booby-trap.png", type=ACTION, aember_on_play=1, on_play=named.booby_trap)

_define(
    id=316,
    name="Duskrunner",
    house=SHADOWS, image="Shadows/duskrunner.png",
    type=UPGRADE,
    aember_on_play=0,
    register_passive=named.duskrunner_register,
    unregister_passive=named.duskrunner_unregister,
)

_define(id=270, name="Ghostly Hand", house=SHADOWS, image="Shadows/ghostly-hand.png", type=ACTION, aember_on_play=2, on_play=named.ghostly_hand)

_define(id=274, name="Lights Out", house=SHADOWS, image="Shadows/lights-out.png", type=ACTION, aember_on_play=1, on_play=named.lights_out)

_define(
    id=275,
    name="Miasma",
    house=SHADOWS, image="Shadows/miasma.png",
    type=ACTION,
    aember_on_play=1,
    on_play=generic.duration_effect("CanKeyForge", "=", False, 2, "enemy"),
)

_define(id=276, name="Nerve Blast", house=SHADOWS, image="Shadows/nerve-blast.png", type=ACTION, aember_on_play=0, on_play=named.nerve_blast)

_define(
    id=306,
    name="Noddy the Thief",
    house=SHADOWS, image="Shadows/noddy-the-thief.png",
    type=CREATURE,
    power=2,
    armor=0,
    tags=("Elf", "Thief"),
    on_action=generic.steal_n(1),
    register_passive=named.elusive_register,
    unregister_passive=named.elusive_unregister,
)

_define(
    id=307,
    name="Old Bruno",
    house=SHADOWS, image="Shadows/old-bruno.png",
    type=CREATURE,
    power=3,
    armor=0,
    tags=("Elf", "Thief"),
    on_play=named.old_bruno_play,
    register_passive=named.elusive_register,
    unregister_passive=named.elusive_unregister,
)

_define(id=277, name="One Last Job", house=SHADOWS, image="Shadows/one-last-job.png", type=ACTION, aember_on_play=1, on_play=named.one_last_job)

_define(id=278, name="Oubliette", house=SHADOWS, image="Shadows/oubliette.png", type=ACTION, aember_on_play=0, on_play=named.oubliette)

_define(id=279, name="Pawn Sacrifice", house=SHADOWS, image="Shadows/pawn-sacrifice.png", type=ACTION, aember_on_play=1, on_play=named.pawn_sacrifice)

_define(
    id=281,
    name="Relentless Whispers",
    house=SHADOWS, image="Shadows/relentless-whispers.png",
    type=ACTION,
    aember_on_play=1,
    on_play=named.relentless_whispers,
)

_define(
    id=308,
    name="Silvertooth",
    house=SHADOWS, image="Shadows/silvertooth.png",
    type=CREATURE,
    power=2,
    armor=0,
    tags=("Elf", "Thief"),
    on_play=named.silvertooth_play,
)

_define(
    id=294,
    name="Subtle Maul",
    house=SHADOWS, image="Shadows/subtle-maul.png",
    type=ARTIFACT,
    tags=("Weapon",),
    on_action=named.subtle_maul,
)

_define(
    id=283,
    name="Too Much To Protect",
    house=SHADOWS, image="Shadows/too-much-to-protect.png",
    type=ACTION,
    aember_on_play=1,
    on_play=named.too_much_to_protect,
)

_define(
    id=584,
    name="Urchin",
    house=SHADOWS, image="Shadows/urchin.png",
    type=CREATURE,
    power=1,
    armor=0,
    tags=("Thief", "Elf"),
    on_play=named.urchin_play,
    register_passive=named.elusive_register,
    unregister_passive=named.elusive_unregister,
)


def get_card_def(name: str) -> CardDef:
    return CARD_DEFS[name]

"""Card definitions for the full 370-card CotA pool (all 7 houses), loaded
from the generated `cards/data/cota_pool.json` (built by
`Code/Non-GUI/tools/build_card_data.py` from `Code/PHASE_2_CARD_POOL.md` and
`Code/PHASE_3_CARD_POOL.md` -- no network access at runtime). Base stats,
traits, keywords and canonical text come from that JSON; gameplay hooks
(on_play, on_reap, register_passive, ...) are wired up here by card name, in
`HOOKS`. A card not yet in `HOOKS` still gets a fully-formed `CardDef` (so
art/text/registry tests pass), just with no scripted behavior yet -- see
Code/PHASE_2_PLAN.md Milestone C and Code/PHASE_3_PLAN.md Milestone D."""

from __future__ import annotations

import json
import os
from typing import Dict

from ..enums import CardType, House
from ..effects import generic
from ..effects.named import brobnar as named_brobnar
from ..effects.named import dis as named_dis
from ..effects.named import logos as named_logos
from ..effects.named import mars as named_mars
from ..effects.named import sanctum as named_sanctum
from ..effects.named import shadows as named_shadows
from ..effects.named import untamed as named_untamed
from .card import CardDef

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POOL_JSON_PATH = os.path.join(_DATA_DIR, "cota_pool.json")

HOOKS: Dict[str, dict] = {}


def hook(name: str, **kwargs) -> None:
    if name in HOOKS:
        raise ValueError(f"duplicate hook registration for {name!r}")
    HOOKS[name] = kwargs


# ------------------------------------------------------------------ Dis --

hook("Arise!", on_play=named_dis.arise)
hook("Control the Weak", on_play=named_dis.control_the_weak)
hook("Creeping Oblivion", on_play=named_dis.creeping_oblivion)
hook("Dominator Bauble", on_action=named_dis.dominator_bauble)
hook("Dust Imp", on_destroyed=named_dis.dust_imp_destroyed)
hook("Ember Imp", register_passive=named_dis.ember_imp_register, unregister_passive=named_dis.ember_imp_unregister)
hook("Gateway to Dis", on_play=named_dis.gateway_to_dis)
hook("Guardian Demon", on_play=named_dis.guardian_demon, on_reap=named_dis.guardian_demon, on_fight=named_dis.guardian_demon)
hook("Lash of Broken Dreams", on_action=generic.duration_effect("KeyForgeCost", "+", 3, 2, "enemy"))
hook("Library of the Damned", on_action=generic.archive_n(1))
hook("Lifeward", on_omni=named_dis.lifeward_omni)
hook("Pit Demon", on_action=generic.steal_n(1))
hook("Shooler", on_play=named_dis.shooler_play)
hook("Snudge", on_reap=named_dis.snudge, on_fight=named_dis.snudge)
hook("Succubus", register_passive=named_dis.succubus_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("The Terror", on_play=named_dis.the_terror_play)
hook("Three Fates", on_play=named_dis.three_fates)

hook("A Fair Game", on_play=named_dis.a_fair_game)
hook("Dance of Doom", on_play=named_dis.dance_of_doom)
hook("Fear", on_play=named_dis.fear)
hook("Gongoozle", on_play=named_dis.gongoozle)
hook("Guilty Hearts", on_play=named_dis.guilty_hearts)
hook("Hand of Dis", on_play=named_dis.hand_of_dis)
hook("Hecatomb", on_play=named_dis.hecatomb)
hook("Tendrils of Pain", on_play=named_dis.tendrils_of_pain)
hook("Hysteria", on_play=named_dis.hysteria)
hook("Key Hammer", on_play=named_dis.key_hammer)
hook("Mind Barb", on_play=named_dis.mind_barb)
hook("Pandemonium", on_play=named_dis.pandemonium)
hook("Poltergeist", on_play=named_dis.poltergeist)
hook("Red-Hot Armor", on_play=named_dis.red_hot_armor)
hook("Annihilation Ritual", register_passive=named_dis.annihilation_ritual_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Key to Dis", on_omni=named_dis.key_to_dis)
hook("Sacrificial Altar", on_action=named_dis.sacrificial_altar)
hook("Screaming Cave", on_action=named_dis.screaming_cave)
hook("Soul Snatcher", register_passive=named_dis.soul_snatcher_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Charette", on_play=generic.capture_n(3))
hook("Drumble", on_play=named_dis.drumble_play)
hook("Eater of the Dead", on_fight=named_dis.eater_of_the_dead, on_reap=named_dis.eater_of_the_dead)
hook("Gabos Longarms", register_passive=named_dis.gabos_longarms_register, unregister_passive=named_dis.gabos_longarms_unregister)
hook("Overlord Greking", on_destroyed_fighting=named_dis.overlord_greking_on_destroyed_fighting)
hook("Stealer of Souls", on_destroyed_fighting=named_dis.stealer_of_souls_on_destroyed_fighting)
hook("Master of 1", on_reap=named_dis.master_of_n(1))
hook("Master of 2", on_reap=named_dis.master_of_n(2))
hook("Master of 3", on_reap=named_dis.master_of_n(3))
hook("Pitlord", register_passive=named_dis.pitlord_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Restringuntus", on_play=named_dis.restringuntus, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Shaffles", register_passive=named_dis.shaffles_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Tentacus", register_passive=named_dis.tentacus_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Tocsin", on_reap=named_dis.tocsin)
hook("Tolas", register_passive=named_dis.tolas_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Truebaru", play_cost_aember=3, on_destroyed=named_dis.truebaru_destroyed)
hook("Collar of Subordination", register_passive=named_dis.collar_of_subordination_register, unregister_passive=named_dis.collar_of_subordination_unregister)
hook("Flame-Wreathed", power_bonus=2, hazardous=2)


# ---------------------------------------------------------------- Logos --

hook("Doc Bookton", on_reap=generic.draw_n(1))
hook("Help from Future Self", on_play=named_logos.help_from_future_self)
hook("Labwork", on_play=generic.archive_n(1))
hook("Library Access", on_play=named_logos.library_access_play)
hook("Library of Babble", on_action=generic.draw_n(1))
hook("Mother", register_passive=named_logos.mother_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Phase Shift", on_play=named_logos.phase_shift)
hook("Quixo the “Adventurer”", on_fight=named_logos.quixo_after_fight)
hook("Scrambler Storm", on_play=generic.duration_effect("CanPlayActions", "=", False, 2, "enemy"))
hook("Sloppy Labwork", on_play=named_logos.sloppy_labwork)
hook("The Howling Pit", register_passive=named_logos.the_howling_pit_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Timetraveller", on_play=generic.draw_n(2), on_action=named_logos.timetraveler_action)
hook("Titan Mechanic", register_passive=named_logos.titan_mechanic_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Wild Wormhole", on_play=named_logos.wild_wormhole)

hook("Bouncing Deathquark", on_play=named_logos.bouncing_deathquark)
hook("Dimension Door", on_play=named_logos.dimension_door)
hook("Effervescent Principle", on_play=named_logos.effervescent_principle)
hook("Foggify", on_play=named_logos.foggify)
hook("Interdimensional Graft", on_play=named_logos.interdimensional_graft)
hook("Knowledge is Power", on_play=named_logos.knowledge_is_power)
hook("Neuro Syphon", on_play=named_logos.neuro_syphon)
hook("Positron Bolt", on_play=named_logos.positron_bolt)
hook("Random Access Archives", on_play=named_logos.random_access_archives)
hook("Remote Access", on_play=named_logos.remote_access)
hook("Reverse Time", on_play=named_logos.reverse_time)
hook("Twin Bolt Emission", on_play=named_logos.twin_bolt_emission)
hook("Anomaly Exploiter", on_action=named_logos.anomaly_exploiter)
hook("Chaos Portal", on_action=named_logos.chaos_portal)
hook("Crazy Killing Machine", on_action=named_logos.crazy_killing_machine)
hook("Mobius Scroll", on_action=named_logos.mobius_scroll)
hook("Pocket Universe", spendable_for_keys=True, on_action=generic.move_aember_to_card(1))
hook("Spangler Box", on_action=named_logos.spangler_box)
hook("Spectral Tunneler", on_action=named_logos.spectral_tunneler)
hook("Strange Gizmo", register_passive=named_logos.strange_gizmo_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Batdrone", on_fight=named_logos.batdrone_fight)
hook("Brain Eater", on_destroyed_fighting=named_logos.brain_eater_on_destroyed_fighting)
hook("Dextre", on_play=generic.capture_n(1), on_destroyed=named_logos.dextre_destroyed)
hook("Dr. Escotera", on_play=named_logos.dr_escotera)
hook("Dysania", on_play=named_logos.dysania)
hook("Ganymede Archivist", on_reap=generic.archive_n(1))
hook("Harland Mindlock", on_play=named_logos.harland_mindlock)
hook("Neutron Shark", on_play=named_logos.neutron_shark, on_fight=named_logos.neutron_shark, on_reap=named_logos.neutron_shark)
hook("Novu Archaeologist", on_action=named_logos.novu_archaeologist)
hook("Ozmo, Martianologist", on_fight=named_logos.ozmo, on_reap=named_logos.ozmo)
hook("Psychic Bug", on_play=named_logos.psychic_bug, on_reap=named_logos.psychic_bug)
hook("Replicator", on_reap=named_logos.replicator)
hook("Research Smoko", on_destroyed=named_logos.research_smoko_destroyed)
hook("Skippy Timehog", on_play=named_logos.skippy_timehog)
hook("Vespilon Theorist", on_reap=named_logos.vespilon_theorist)
hook("Veylan Analyst", register_passive=named_logos.veylan_analyst_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Experimental Therapy", grants_keywords=("versatile",), on_play=named_logos.experimental_therapy)
hook("Rocket Boots", register_passive=named_logos.rocket_boots_register, unregister_passive=named_logos.rocket_boots_unregister)
hook("Transposition Sandals", register_passive=named_logos.transposition_sandals_register, unregister_passive=named_logos.transposition_sandals_unregister)


# -------------------------------------------------------------- Shadows --

hook("Bad Penny", on_destroyed=named_shadows.bad_penny_destroyed)
hook("Bait and Switch", on_play=named_shadows.bait_and_switch)
hook("Booby Trap", on_play=named_shadows.booby_trap)
hook("Duskrunner", register_passive=named_shadows.duskrunner_register, unregister_passive=named_shadows.duskrunner_unregister)
hook("Ghostly Hand", on_play=named_shadows.ghostly_hand)
hook("Lights Out", on_play=named_shadows.lights_out)
hook("Miasma", on_play=generic.duration_effect("CanKeyForge", "=", False, 2, "enemy"))
hook("Nerve Blast", on_play=named_shadows.nerve_blast)
hook("Noddy the Thief", on_action=named_shadows.noddy_action)
hook("Old Bruno", on_play=named_shadows.old_bruno_play)
hook("One Last Job", on_play=named_shadows.one_last_job)
hook("Oubliette", on_play=named_shadows.oubliette)
hook("Pawn Sacrifice", on_play=named_shadows.pawn_sacrifice)
hook("Relentless Whispers", on_play=named_shadows.relentless_whispers)
hook("Silvertooth", on_play=named_shadows.silvertooth_play)
hook("Subtle Maul", on_action=named_shadows.subtle_maul)
hook("Too Much to Protect", on_play=named_shadows.too_much_to_protect)
hook("Urchin", on_play=named_shadows.urchin_play)

hook("Finishing Blow", on_play=named_shadows.finishing_blow)
hook("Hidden Stash", on_play=generic.archive_n(1))
hook("Imperial Traitor", on_play=named_shadows.imperial_traitor)
hook("Key of Darkness", on_play=named_shadows.key_of_darkness)
hook("Poison Wave", on_play=named_shadows.poison_wave)
hook("Routine Job", on_play=named_shadows.routine_job)
hook("Treasure Map", on_play=named_shadows.treasure_map)
hook("Customs Office", register_passive=named_shadows.customs_office_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Evasion Sigil", register_passive=named_shadows.evasion_sigil_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Longfused Mines", on_omni=named_shadows.longfused_mines)
hook("Masterplan", on_play=named_shadows.masterplan_play, on_omni=named_shadows.masterplan_omni)
hook("Safe Place", spendable_for_keys=True, on_action=generic.move_aember_to_card(1))
hook("Seeker Needle", on_action=named_shadows.seeker_needle)
hook("Skeleton Key", on_action=named_shadows.skeleton_key)
hook("Special Delivery", on_omni=named_shadows.special_delivery)
hook("Speed Sigil", register_passive=named_shadows.speed_sigil_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("The Sting", register_passive=named_shadows.the_sting_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card), on_action=named_shadows.the_sting_action)
hook("Bulleteye", on_reap=named_shadows.bulleteye)
hook("Carlo Phantom", register_passive=named_shadows.carlo_phantom_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Deipno Spymaster", on_omni=named_shadows.deipno_spymaster)
hook("Faygin", on_reap=named_shadows.faygin)
hook("Mack the Knife", extra_keywords=("versatile",), on_action=named_shadows.mack_the_knife)
hook("Magda the Rat", on_play=named_shadows.magda_the_rat_play, unregister_passive=named_shadows.magda_the_rat_unregister)
hook("Mooncurser", on_fight=generic.steal_n(1))
hook("Nexus", on_reap=named_shadows.nexus)
hook("Dodger", on_fight=generic.steal_n(1))
hook("Selwyn the Fence", on_fight=named_shadows.selwyn_the_fence, on_reap=named_shadows.selwyn_the_fence)
hook("Shadow Self", extra_keywords=("no_fight_damage",), register_passive=named_shadows.shadow_self_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Smiling Ruth", on_reap=named_shadows.smiling_ruth)
hook("Sneklifter", on_play=named_shadows.sneklifter)
hook("Umbra", on_fight=generic.steal_n(1))
hook("Ring of Invisibility", grants_keywords=("elusive", "skirmish"))
hook("Silent Dagger", register_passive=named_shadows.silent_dagger_register, unregister_passive=named_shadows.silent_dagger_unregister)


# -------------------------------------------------------------- Brobnar --

hook("Anger", on_play=named_brobnar.anger)
hook("Barehanded", on_play=named_brobnar.barehanded)
hook("Blood Money", on_play=named_brobnar.blood_money)
hook("Brothers in Battle", on_play=named_brobnar.brothers_in_battle)
hook("Burn the Stockpile", on_play=named_brobnar.burn_the_stockpile)
hook("Champion’s Challenge", on_play=named_brobnar.champions_challenge)
hook("Coward’s End", on_play=named_brobnar.cowards_end)
hook("Follow the Leader", on_play=named_brobnar.follow_the_leader)
hook("Lava Ball", on_play=generic.deal_damage_to_chosen_with_splash(4, 2))
hook("Loot the Bodies", on_play=named_brobnar.loot_the_bodies)
hook("Take that, Smartypants", on_play=named_brobnar.take_that_smartypants)
hook("Punch", on_play=generic.deal_damage_to_chosen(3))
hook("Relentless Assault", on_play=named_brobnar.relentless_assault)
hook("Smith", on_play=named_brobnar.smith)
hook("Sound the Horns", on_play=named_brobnar.sound_the_horns)
hook("Tremor", on_play=named_brobnar.tremor)
hook("Unguarded Camp", on_play=named_brobnar.unguarded_camp)
hook("Warsong", on_play=named_brobnar.warsong)

hook("Autocannon", register_passive=named_brobnar.autocannon_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Banner of Battle", register_passive=named_brobnar.banner_of_battle_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Cannon", on_action=generic.deal_damage_to_chosen(2))
hook("Gauntlet of Command", on_action=named_brobnar.gauntlet_of_command)
hook("Iron Obelisk", register_passive=named_brobnar.iron_obelisk_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Mighty Javelin", on_omni=named_brobnar.mighty_javelin)
hook("Pile of Skulls", register_passive=named_brobnar.pile_of_skulls_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Screechbomb", on_omni=named_brobnar.screechbomb)
hook("The Warchest", on_action=named_brobnar.the_warchest)

hook("Bilgum Avalanche", register_passive=named_brobnar.bilgum_avalanche_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Valdr", register_passive=named_brobnar.valdr_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Bumpsy", on_play=generic.lose_n(1))
hook("Earthshaker", on_play=named_brobnar.earthshaker)
hook("Firespitter", on_before_fight=named_brobnar.firespitter_before_fight)
hook("Ganger Chieftain", on_play=named_brobnar.ganger_chieftain_play)
hook("Grenade Snib", on_destroyed=generic.lose_n(2))
hook("Headhunter", on_fight=generic.gain_n(1))
hook("Hebe the Huge", on_play=named_brobnar.hebe_the_huge)
hook("Kelifi Dragon", min_aember_to_play=7, on_fight=named_brobnar.kelifi_dragon_after, on_reap=named_brobnar.kelifi_dragon_after)
hook("King of the Crag", register_passive=named_brobnar.king_of_the_crag_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Krump", on_destroyed_fighting=named_brobnar.krump_on_destroyed_fighting)
hook("Lomir Flamefist", on_play=named_brobnar.lomir_flamefist)
hook("Looter Goblin", on_reap=named_brobnar.loot_the_bodies)
hook("Mugwump", on_destroyed_fighting=named_brobnar.mugwump_on_destroyed_fighting)
hook("Pingle Who Annoys", register_passive=named_brobnar.pingle_who_annoys_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Rock-Hurling Giant", register_passive=named_brobnar.rock_hurling_giant_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Rogue Ogre", register_passive=named_brobnar.rogue_ogre_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Smaaash", on_play=named_brobnar.smaaash)
hook("Tireless Crocag", cannot_reap=True, extra_keywords=("versatile",), on_play=named_brobnar.tireless_crocag_play, register_passive=named_brobnar.tireless_crocag_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Troll", on_reap=generic.heal_self_n(3))
hook("Wardrummer", on_play=named_brobnar.wardrummer)

hook("Blood of Titans", power_bonus=5)
hook("Phoenix Heart", register_passive=named_brobnar.phoenix_heart_register, unregister_passive=named_brobnar.phoenix_heart_unregister)
hook("Yo Mama Mastery", grants_keywords=("taunt",), on_play=named_brobnar.yo_mama_mastery_play)


# -------------------------------------------------------------- Sanctum --

hook("Begone!", on_play=named_sanctum.begone)
hook("Blinding Light", on_play=named_sanctum.blinding_light)
hook("Charge!", on_play=named_sanctum.charge)
hook("Cleansing Wave", on_play=named_sanctum.cleansing_wave)
hook("Clear Mind", on_play=named_sanctum.clear_mind)
hook("Doorstep to Heaven", on_play=named_sanctum.doorstep_to_heaven)
hook("Glorious Few", on_play=named_sanctum.glorious_few)
hook("Honorable Claim", on_play=named_sanctum.honorable_claim)
hook("Inspiration", on_play=named_sanctum.inspiration)
hook("Mighty Lance", on_play=named_sanctum.mighty_lance)
hook("Oath of Poverty", on_play=named_sanctum.oath_of_poverty)
hook("One Stood Against Many", on_play=named_sanctum.one_stood_against_many)
hook("Radiant Truth", on_play=named_sanctum.radiant_truth)
hook("Shield of Justice", on_play=generic.duration_effect("CannotBeDealtDamage", "=", True, 1, "self"))
hook("Take Hostages", on_play=named_sanctum.take_hostages)
hook("Terms of Redress", on_play=named_sanctum.terms_of_redress)
hook("The Harder They Come", on_play=named_sanctum.the_harder_they_come)
hook("The Spirit’s Way", on_play=named_sanctum.the_spirits_way)
# Virtuous Works is vanilla (aember_on_play only, from the pool data) -- no hook.

hook("Epic Quest", on_play=named_sanctum.epic_quest_play, on_omni=named_sanctum.epic_quest_omni)
hook("Gorm of Omm", on_omni=named_sanctum.gorm_of_omm)
hook("Hallowed Blaster", on_action=named_sanctum.hallowed_blaster)
hook("Potion of Invulnerability", on_omni=named_sanctum.potion_of_invulnerability)
hook("Round Table", register_passive=named_sanctum.round_table_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Sigil of Brotherhood", on_omni=named_sanctum.sigil_of_brotherhood)
hook("Whispering Reliquary", on_action=named_sanctum.whispering_reliquary)

hook("Bulwark", register_passive=named_sanctum.bulwark_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
# Champion Anaphiel is taunt-only (from the pool's Keywords column) -- no hook.
hook("Champion Tabris", on_fight=generic.capture_n(1))
hook("Commander Remiel", on_reap=named_sanctum.commander_remiel)
hook("Duma the Martyr", on_destroyed=named_sanctum.duma_the_martyr_destroyed)
hook("Francus", on_destroyed_fighting=named_sanctum.francus_on_destroyed_fighting)
hook("Grey Monk", register_passive=named_sanctum.grey_monk_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card), on_reap=named_sanctum.grey_monk_after_reap)
hook("Hayyel the Merchant", register_passive=named_sanctum.hayyel_the_merchant_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Horseman of Death", on_play=named_sanctum.horseman_of_death)
hook("Horseman of Famine", on_play=named_sanctum.horseman_of_famine, on_fight=named_sanctum.horseman_of_famine, on_reap=named_sanctum.horseman_of_famine)
hook("Horseman of Pestilence", on_play=named_sanctum.horseman_of_pestilence, on_fight=named_sanctum.horseman_of_pestilence, on_reap=named_sanctum.horseman_of_pestilence)
hook("Horseman of War", on_play=named_sanctum.horseman_of_war)
hook("Jehu the Bureaucrat", register_passive=named_sanctum.jehu_the_bureaucrat_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Lady Maxena", on_play=named_sanctum.lady_maxena_play, on_action=named_sanctum.lady_maxena_action)
hook("Lord Golgotha", on_before_fight=named_sanctum.lord_golgotha_before_fight)
hook("Numquid the Fair", on_play=named_sanctum.numquid_the_fair)
hook("Protectrix", on_reap=named_sanctum.protectrix_after_reap)
hook("Raiding Knight", on_play=generic.capture_n(1))
hook("Sanctum Guardian", on_fight=named_sanctum.sanctum_guardian_after, on_reap=named_sanctum.sanctum_guardian_after)
hook("Sequis", on_reap=generic.capture_n(1))
hook("Sergeant Zakiel", on_play=named_sanctum.sergeant_zakiel_play)
hook("Staunch Knight", register_passive=named_sanctum.staunch_knight_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Gatekeeper", on_play=named_sanctum.gatekeeper)
hook("The Vaultkeeper", register_passive=named_sanctum.the_vaultkeeper_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Veemos Lightbringer", on_play=named_sanctum.veemos_lightbringer)

hook("Armageddon Cloak", hazardous=2, register_passive=named_sanctum.armageddon_cloak_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Mantle of the Zealot", grants_keywords=("versatile",))
hook("Protect the Weak", armor_bonus=1, grants_keywords=("taunt",))
hook("Shoulder Armor", register_passive=named_sanctum.shoulder_armor_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))


# ----------------------------------------------------------------- Mars --

hook("Ammonia Clouds", on_play=named_mars.ammonia_clouds)
hook("Battle Fleet", on_play=named_mars.battle_fleet)
hook("Deep Probe", on_play=named_mars.deep_probe)
hook("EMP Blast", on_play=named_mars.emp_blast)
hook("Hypnotic Command", on_play=named_mars.hypnotic_command)
hook("Irradiated Æmber", on_play=named_mars.irradiated_aember)
hook("Key Abduction", on_play=named_mars.key_abduction)
hook("Martian Hounds", on_play=named_mars.martian_hounds)
hook("Martians Make Bad Allies", on_play=named_mars.martians_make_bad_allies)
hook("Mass Abduction", on_play=named_mars.mass_abduction)
hook("Mating Season", on_play=named_mars.mating_season)
hook("Mothership Support", on_play=named_mars.mothership_support)
hook("Orbital Bombardment", on_play=named_mars.orbital_bombardment)
hook("Phosphorus Stars", on_play=named_mars.phosphorus_stars)
hook("Psychic Network", on_play=named_mars.psychic_network)
hook("Sample Collection", on_play=named_mars.sample_collection)
hook("Shatter Storm", on_play=named_mars.shatter_storm)
hook("Soft Landing", on_play=named_mars.soft_landing)
hook("Squawker", on_play=named_mars.squawker)
hook("Total Recall", on_play=named_mars.total_recall)

hook("Combat Pheromones", on_omni=named_mars.combat_pheromones)
hook("Commpod", on_action=named_mars.commpod)
hook("Crystal Hive", on_action=named_mars.crystal_hive)
hook("Custom Virus", on_omni=named_mars.custom_virus)
hook("Feeding Pit", on_action=named_mars.feeding_pit)
hook("Invasion Portal", on_action=named_mars.invasion_portal)
hook("Incubation Chamber", on_omni=named_mars.incubation_chamber)
hook("Mothergun", on_action=named_mars.mothergun)
hook("Sniffer", on_action=named_mars.sniffer_action)
hook("Swap Widget", on_action=named_mars.swap_widget)

hook("Blypyp", on_reap=named_mars.blypyp_after_reap)
hook("Chuff Ape", register_passive=named_mars.chuff_ape_register, on_fight=named_mars.chuff_ape_after, on_reap=named_mars.chuff_ape_after)
hook("Ether Spider", extra_keywords=("no_fight_damage",), register_passive=named_mars.ether_spider_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Grabber Jammer", register_passive=named_mars.grabber_jammer_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card), on_fight=named_mars.grabber_jammer_after, on_reap=named_mars.grabber_jammer_after)
hook("Grommid", register_passive=named_mars.grommid_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card), on_destroyed_fighting=named_mars.grommid_on_destroyed_fighting)
hook("“John Smyth”", on_fight=named_mars.john_smyth_after, on_reap=named_mars.john_smyth_after)
hook("Mindwarper", on_action=named_mars.mindwarper_action)
hook("Phylyx the Disintegrator", on_action=named_mars.phylyx_the_disintegrator_action)
hook("Qyxxlyx Plague Master", on_fight=named_mars.qyxxlyx_plague_master_after, on_reap=named_mars.qyxxlyx_plague_master_after)
hook("Tunk", register_passive=named_mars.tunk_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Ulyq Megamouth", on_fight=named_mars.ulyq_megamouth_after, on_reap=named_mars.ulyq_megamouth_after)
hook("Uxlyx the Zookeeper", on_reap=named_mars.uxlyx_the_zookeeper_after)
hook("Vezyma Thinkdrone", on_reap=named_mars.vezyma_thinkdrone_after)
hook("Yxili Marauder", register_passive=named_mars.yxili_marauder_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card), on_play=named_mars.yxili_marauder_play)
hook("Yxilo Bolter", on_fight=named_mars.yxilo_bolter_after, on_reap=named_mars.yxilo_bolter_after)
hook("Yxilx Dominator", register_passive=named_mars.yxilx_dominator_register)
hook("Zorg", register_passive=named_mars.zorg_register, on_before_fight=named_mars.zorg_before_fight)
hook("Zyzzix the Many", on_fight=named_mars.zyzzix_the_many_after, on_reap=named_mars.zyzzix_the_many_after)

hook("Biomatrix Backup", register_passive=named_mars.biomatrix_backup_register, unregister_passive=named_mars.biomatrix_backup_unregister)
hook("Brain Stem Antenna", register_passive=named_mars.brain_stem_antenna_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Jammer Pack", register_passive=named_mars.jammer_pack_register, unregister_passive=lambda game, card: game.active_effects.remove_from_source(card))
hook("Red Planet Ray Gun", register_passive=named_mars.red_planet_ray_gun_register, unregister_passive=named_mars.red_planet_ray_gun_unregister)


# ------------------------------------------------------------- Untamed --

# See Code/PHASE_3_PLAN.md Milestone D.4.


# -------------------------------------------------------- build CARD_DEFS --

def _load_pool() -> list:
    with open(POOL_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_card_defs() -> Dict[str, CardDef]:
    defs = {}
    for entry in _load_pool():
        name = entry["name"]
        kwargs = dict(HOOKS.get(name, {}))
        # A few cards' own keywords aren't captured structurally in the pool
        # table (e.g. Mack the Knife's "you may use ... as if it belonged to
        # the active house" prose) -- `extra_keywords` lets a hook add to the
        # card's printed keywords without having to repeat the JSON ones.
        extra_keywords = kwargs.pop("extra_keywords", ())
        keywords = tuple(entry["keywords"]) + tuple(extra_keywords)
        defs[name] = CardDef(
            id=entry["number"],
            name=name,
            house=House(entry["house"]),
            type=CardType(entry["type"]),
            aember_on_play=entry["aember_on_play"],
            tags=tuple(entry["traits"]),
            keywords=keywords,
            text=entry["text"],
            errata=entry["errata"],
            image=entry["image"],
            power=entry["power"] or 0,
            armor=entry["armor"],
            **kwargs,
        )
    return defs


CARD_DEFS: Dict[str, CardDef] = _build_card_defs()


def get_card_def(name: str) -> CardDef:
    return CARD_DEFS[name]

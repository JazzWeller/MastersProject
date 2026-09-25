# Phase 2 Card Rulings — Dis, Logos, Shadows

One entry per Phase 2 pool card: the 159 *Call of the Archons* Dis, Logos and Shadows cards. Canonical text is in `Code/PHASE_2_CARD_POOL.md`. This file adds each card's errata and every ruling in the official KeyForge Master Rulebook v18.3 (Feb 2026) that bears on it: the Card-Specific FAQ, plus General FAQ and glossary examples that name the card. Rulings are paraphrased as test cases, and each names the engine test (under `Code/Non-GUI/`) that checks it. A ruling marked *(no test)* is one of three kinds: it involves a card outside this project's pool, the engine can't reach it (mavericks), or it follows from the general rules without a card-specific path. A card with no ruling gets a one-line placeholder: standard rules apply.

## Dis

### 053 A Fair Game

No card-specific MRB 18.3 ruling; standard rules apply.

### 054 Arise!

No card-specific MRB 18.3 ruling; standard rules apply.

### 055 Control the Weak

No card-specific MRB 18.3 ruling; standard rules apply.

### 056 Creeping Oblivion

No card-specific MRB 18.3 ruling; standard rules apply.

### 057 Dance of Doom

No card-specific MRB 18.3 ruling; standard rules apply.

### 058 Fear

No card-specific MRB 18.3 ruling; standard rules apply.

### 059 Gateway to Dis

**Rulings (MRB 18.3):**
- Tolas destroyed by the same Gateway to Dis gives no one Æmber: Tolas has already left play when the other creatures reach the discard pile (Timing FAQ). Tested by `tests/test_cards_dis_phase2.py::test_tolas_destroyed_in_the_same_batch_gains_nothing`.
- Æmber on a creature destroyed by Gateway goes to its opponent as the creature leaves play, before Magda the Rat's leaves-play steal resolves (Leaves Play FAQ). Tested by `tests/test_rulings_phase2.py::test_magda_the_rat_aember_on_creatures_moves_before_her_steal`.

### 060 Gongoozle

No card-specific MRB 18.3 ruling; standard rules apply.

### 061 Guilty Hearts

No card-specific MRB 18.3 ruling; standard rules apply.

### 062 Hand of Dis

No card-specific MRB 18.3 ruling; standard rules apply.

### 063 Hecatomb

No card-specific MRB 18.3 ruling; standard rules apply.

### 064 Tendrils of Pain

**Errata:** MRB 18.3 errata (the pool table's canonical text already includes it).


### 065 Hysteria

No card-specific MRB 18.3 ruling; standard rules apply.

### 066 Key Hammer

**Rulings (MRB 18.3):**
- If the opponent forged two keys on their previous turn, Key Hammer unforges only one of them. Tested by `tests/test_rulings_phase2.py::test_key_hammer_unforges_only_one_of_two_keys`.

### 067 Mind Barb

No card-specific MRB 18.3 ruling; standard rules apply.

### 068 Pandemonium

No card-specific MRB 18.3 ruling; standard rules apply.

### 069 Poltergeist

**Rulings (MRB 18.3):**
- Poltergeist may target an artifact that can't currently be used: resolve as much as possible, which here means just destroying it. Tested by `tests/test_cards_dis_phase2.py::test_poltergeist_can_use_an_omni_only_artifact`.

### 070 Red-Hot Armor

No card-specific MRB 18.3 ruling; standard rules apply.

### 071 Three Fates

**Rulings (MRB 18.3):**
- "The 3 most powerful" is a group: fill it from the highest power down; when a tie straddles the last place, the active player chooses among the tied creatures (Glossary, Groups of "Most Powerful"). Tested by `tests/test_cards_dis.py::test_three_fates_destroys_three_most_powerful`.

### 072 Annihilation Ritual

No card-specific MRB 18.3 ruling; standard rules apply.

### 073 Dominator Bauble

No card-specific MRB 18.3 ruling; standard rules apply.

### 074 Key to Dis

No card-specific MRB 18.3 ruling; standard rules apply.

### 075 Lash of Broken Dreams

No card-specific MRB 18.3 ruling; standard rules apply.

### 076 Library of the Damned

No card-specific MRB 18.3 ruling; standard rules apply.

### 077 Lifeward

No card-specific MRB 18.3 ruling; standard rules apply.

### 078 Sacrificial Altar

No card-specific MRB 18.3 ruling; standard rules apply.

### 079 Screaming Cave

No card-specific MRB 18.3 ruling; standard rules apply.

### 080 Soul Snatcher

No card-specific MRB 18.3 ruling; standard rules apply.

### 081 Charette

No card-specific MRB 18.3 ruling; standard rules apply.

### 082 Drumble

No card-specific MRB 18.3 ruling; standard rules apply.

### 083 Dust Imp

No card-specific MRB 18.3 ruling; standard rules apply.

### 084 Eater of the Dead

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 085 Ember Imp

No card-specific MRB 18.3 ruling; standard rules apply.

### 086 Gabos Longarms

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Its "Before Fight:" damage can go to an elusive creature: elusive only prevents damage when that creature is attacked. Tested by `tests/test_cards_dis_phase2.py::test_gabos_longarms_redirects_its_own_fight_damage`.
- Attacking an elusive creature still lets Gabos deal its damage to another creature. Tested by `tests/test_cards_dis_phase2.py::test_gabos_longarms_redirect_target_is_actually_destroy_checked`.

### 087 Overlord Greking

**Rulings (MRB 18.3):**
- Greking only takes creatures from the opponent's discard pile; if the destroyed creature went to a hidden zone instead (hand, deck, archives), nothing happens. Tested by `tests/test_cards_dis_phase2.py::test_overlord_greking_takes_control_of_creature_it_kills`.
- A Tolas taken by Greking does not trigger on its own destruction: its constant ability is only active while it is in play (Timing FAQ). Tested by `tests/test_rulings_phase2.py::test_tolas_taken_by_overlord_greking_does_not_trigger_on_itself`.

### 088 Guardian Demon

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 089 Master of 1

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 090 Master of 2

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 091 Master of 3

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 092 Pit Demon

No card-specific MRB 18.3 ruling; standard rules apply.

### 093 Pitlord

**Rulings (MRB 18.3):**
- Pitlord (must choose Dis) against Restringuntus naming Dis (cannot choose Dis): cannot wins, so the player chooses one of their other houses. Tested by `tests/test_rules_phase2.py::test_cannot_beats_must_and_falls_back_to_normal_choice`.
- A player can never be forced to choose a house that isn't on their identity card and that they control no card of. The engine can't reach this: mavericks are out of scope, so a deck with Pitlord always has Dis. *(no test)*

### 094 Restringuntus

**Rulings (MRB 18.3):**
- See Pitlord: "cannot" takes precedence over "must". Tested by `tests/test_rules_phase2.py::test_cannot_beats_must_and_falls_back_to_normal_choice`.

### 095 Shaffles

No card-specific MRB 18.3 ruling; standard rules apply.

### 096 Shooler

No card-specific MRB 18.3 ruling; standard rules apply.

### 097 Snudge

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 098 Stealer of Souls

**Rulings (MRB 18.3):**
- If Stealer of Souls and the creature it fights are destroyed together, its ability does not resolve: it has left play at the same time. Tested by `tests/test_cards_dis_phase2.py::test_stealer_of_souls_does_not_trigger_if_it_also_dies`.
- A victim whose own "Destroyed:" ability moves it elsewhere (Bad Penny back to hand) was still destroyed, so the 1Æ is gained; it just can't be purged from a zone it never reached (following the Tolas/Bad Penny and Yxilo Bolter/Bad Penny FAQs). Tested by `tests/test_cards_dis_phase2.py::test_stealer_of_souls_still_gains_1_but_cannot_purge_a_bad_penny`.

### 099 Succubus

No card-specific MRB 18.3 ruling; standard rules apply.

### 100 Tentacus

No card-specific MRB 18.3 ruling; standard rules apply.

### 101 The Terror

No card-specific MRB 18.3 ruling; standard rules apply.

### 102 Tocsin

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 103 Tolas

**Rulings (MRB 18.3):**
- A Bad Penny that returns to its owner's hand when destroyed still counts as destroyed, so Tolas gives its opponent 1Æ. Tested by `tests/test_rulings_phase2.py::test_a_bad_penny_returned_to_hand_still_counts_as_destroyed`.
- Tolas destroyed in the same batch as other creatures (Gateway to Dis) triggers for none of them. Tested by `tests/test_cards_dis_phase2.py::test_tolas_destroyed_in_the_same_batch_gains_nothing`.

### 104 Truebaru

No card-specific MRB 18.3 ruling; standard rules apply.

### 105 Collar of Subordination

**Rulings (MRB 18.3):**
- When two control effects conflict, the most recently applied one takes precedence (Control of Cards FAQ). *(no test)*

### 106 Flame-Wreathed

**Rulings (MRB 18.3):**
- An upgrade stays under the control of the player who played it, even when attached to an enemy creature (Pitlord FAQ). *(no test)*

## Logos

### 107 Bouncing Deathquark

**Rulings (MRB 18.3):**
- The effect may repeat as long as there is a friendly creature to apply it to, even if the previous friendly target wasn't actually destroyed. Tested by `tests/test_cards_logos_phase2.py::test_bouncing_deathquark_repeats_while_possible`.

### 108 Dimension Door

**Rulings (MRB 18.3):**
- Reaping replaces the Æmber gain with a steal, so effects that capture reaped Æmber (e.g. Sir Marrows) have nothing to capture (Replacement Effects FAQ). Tested by `tests/test_cards_logos_phase2.py::test_dimension_door_makes_reap_gain_a_steal`.

### 109 Effervescent Principle

No card-specific MRB 18.3 ruling; standard rules apply.

### 110 Foggify

No card-specific MRB 18.3 ruling; standard rules apply.

### 111 Help from Future Self

No card-specific MRB 18.3 ruling; standard rules apply.

### 112 Interdimensional Graft

**Rulings (MRB 18.3):**
- Its after-forge steal and other after-forge effects share a timing window; the active player orders them. Tested by `tests/test_cards_logos_phase2.py::test_interdimensional_graft_steals_forgers_remaining_aember`.

### 113 Knowledge is Power

**Rulings (MRB 18.3):**
- "For each archived card you have" means for each card in your archives. Tested by `tests/test_cards_logos_phase2.py::test_knowledge_is_power_archive_mode`.

### 114 Labwork

No card-specific MRB 18.3 ruling; standard rules apply.

### 115 Library Access

**Errata:** MRB 18.3 errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Library Access then Wild Wormhole: gain Wild Wormhole's Æmber bonus; Library Access's draw and Wild Wormhole's play effect are simultaneous (either order); then the top card's bonus; then its play effect, simultaneous with Library Access's draw. Tested by `tests/test_wild_wormhole.py::test_library_access_trigger_fires_on_next_card_played`.
- If Wild Wormhole plays Library Access, you draw for Wild Wormhole: you are still in its after-play window. Tested by `tests/test_wild_wormhole.py::test_wormhole_plays_library_access_and_draws`.

### 116 Neuro Syphon

No card-specific MRB 18.3 ruling; standard rules apply.

### 117 Phase Shift

**Rulings (MRB 18.3):**
- Two Phase Shifts allow two non-Logos cards. Tested by `tests/test_rulings_phase2.py::test_two_phase_shifts_allow_two_non_logos_cards`.
- A non-Logos card played by Wild Wormhole uses up Phase Shift's allowance. Tested by `tests/test_rulings_phase2.py::test_wild_wormhole_playing_a_non_logos_card_uses_phase_shifts_allowance`.
- Playing Mimicry as a copy of a Logos card still uses the allowance, since Mimicry could only be played because of it. *(no test)*
- On the first player's first turn, Phase Shift lets them play another card despite the First Turn Rule (General FAQ, First Turn Rule). Tested by `tests/test_rulings_phase2.py::test_phase_shift_lets_the_first_player_play_another_card`.

### 118 Positron Bolt

**Rulings (MRB 18.3):**
- On a creature that is a flank only because of Spectral Tunneler: 3 to it, 2 to a neighbor of your choice, 1 to that neighbor's other neighbor. Tested by `tests/test_cards_logos_phase2.py::test_positron_bolt_hits_flank_then_neighbor_chain`.

### 119 Random Access Archives

No card-specific MRB 18.3 ruling; standard rules apply.

### 120 Remote Access

**Rulings (MRB 18.3):**
- Using an enemy card never changes who controls it. Tested by `tests/test_cards_logos_phase2.py::test_remote_access_uses_enemy_artifact_as_if_yours`.

### 121 Reverse Time

**Rulings (MRB 18.3):**
- The deck is turned over as a whole onto the discard pile: the old top card of the deck becomes the bottom of the discard pile. Tested by `tests/test_rulings_phase2.py::test_reverse_time_turns_the_deck_over_onto_the_discard_pile`.

### 122 Scrambler Storm

No card-specific MRB 18.3 ruling; standard rules apply.

### 123 Sloppy Labwork

No card-specific MRB 18.3 ruling; standard rules apply.

### 124 Twin Bolt Emission

No card-specific MRB 18.3 ruling; standard rules apply.

### 125 Wild Wormhole

**Rulings (MRB 18.3):**
- If the top card can't be played (Kelifi Dragon without enough Æmber), it goes back to the top of the deck. Tested by `tests/test_cards_logos.py::test_wild_wormhole_returns_uncastable_card_to_top`.
- Wild Wormhole's effect resolves normally under the First Turn Rule, which only limits cards played from hand. *(no test)*

### 126 Anomaly Exploiter

No card-specific MRB 18.3 ruling; standard rules apply.

### 127 Chaos Portal

No card-specific MRB 18.3 ruling; standard rules apply.

### 128 Crazy Killing Machine

No card-specific MRB 18.3 ruling; standard rules apply.

### 129 Library of Babble

No card-specific MRB 18.3 ruling; standard rules apply.

### 130 Mobius Scroll

No card-specific MRB 18.3 ruling; standard rules apply.

### 131 Pocket Universe

No card-specific MRB 18.3 ruling; standard rules apply.

### 132 Spangler Box

**Rulings (MRB 18.3):**
- Purged cards return by being put into play, which ignores play restrictions (Kelifi Dragon returns with 0 Æmber). Tested by `tests/test_cards_logos_phase2.py::test_spangler_box_returns_purged_cards_when_it_leaves_play`.
- A card moved from the purged zone into archives no longer returns when Spangler Box leaves play. *(no test)*

### 133 Spectral Tunneler

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- An "After Reap:" ability Nexus gains from an enemy Spectral Tunneler still resolves: the after-reap window is still open. Tested by `tests/test_cards_logos_phase2.py::test_spectral_tunneler_grants_flank_status_and_after_reap_draw`.

### 134 Strange Gizmo

No card-specific MRB 18.3 ruling; standard rules apply.

### 135 The Howling Pit

No card-specific MRB 18.3 ruling; standard rules apply.

### 136 Batdrone

No card-specific MRB 18.3 ruling; standard rules apply.

### 137 Brain Eater

No card-specific MRB 18.3 ruling; standard rules apply.

### 138 Dextre

**Rulings (MRB 18.3):**
- Its "Destroyed:" ability moves it before it is destroyed, so its neighbors change immediately (Scowly Caper FAQ). *(no test)*

### 139 Doc Bookton

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 140 Dr. Escotera

No card-specific MRB 18.3 ruling; standard rules apply.

### 141 Dysania

**Rulings (MRB 18.3):**
- "Each of their archived cards" means each card in their archives. Tested by `tests/test_cards_logos_phase2.py::test_dysania_discards_archive_and_gains`.
- Cards that leave the archives to their owner's hand instead (Sample Collection) were not discarded, so they give no Æmber. *(no test)*

### 142 Ganymede Archivist

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 143 Harland Mindlock

No card-specific MRB 18.3 ruling; standard rules apply.

### 144 Quixo the “Adventurer”

No card-specific MRB 18.3 ruling; standard rules apply.

### 145 Mother

No card-specific MRB 18.3 ruling; standard rules apply.

### 146 Neutron Shark

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Once Neutron Shark destroys itself it stops: its ability can't repeat after it has left play, even if the top card isn't Logos. Tested by `tests/test_rulings_phase2.py::test_neutron_shark_that_destroys_itself_does_not_repeat`.

### 147 Novu Archaeologist

No card-specific MRB 18.3 ruling; standard rules apply.

### 148 Ozmo, Martianologist

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 149 Psychic Bug

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 150 Replicator

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- With several "After Reap:" abilities on the target, the active player chooses which one resolves. Tested by `tests/test_cards_logos_phase2.py::test_replicator_triggers_another_creatures_reap`.
- Triggering an enemy Sequis captures 1 from your opponent's pool: you use it as if you controlled it. Tested by `tests/test_rulings_phase2.py::test_replicator_on_an_enemy_sequis_captures_from_its_owners_pool`.
- Triggering an enemy Sanctum Guardian does nothing: no creature moves to the other player's battleline without a control change. Tested by `tests/test_rulings_phase2.py::test_replicator_on_an_enemy_sanctum_guardian_moves_nothing`.

### 151 Research Smoko

No card-specific MRB 18.3 ruling; standard rules apply.

### 152 Skippy Timehog

No card-specific MRB 18.3 ruling; standard rules apply.

### 153 Timetraveller

No card-specific MRB 18.3 ruling; standard rules apply.

### 154 Titan Mechanic

No card-specific MRB 18.3 ruling; standard rules apply.

### 155 Vespilon Theorist

No card-specific MRB 18.3 ruling; standard rules apply.

### 156 Veylan Analyst

No card-specific MRB 18.3 ruling; standard rules apply.

### 157 Experimental Therapy

**Errata:** MRB 18.3 errata (the pool table's canonical text already includes it).


### 158 Rocket Boots

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 159 Transposition Sandals

No card-specific MRB 18.3 ruling; standard rules apply.

## Shadows

### 267 Bait and Switch

**Errata:** MRB 18.3 errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Repeats at most once: steal 1 if behind, then check again and steal once more if still behind. Tested by `tests/test_cards_shadows.py::test_bait_and_switch_repeats_at_most_once_even_with_a_large_gap`.

### 268 Booby Trap

No card-specific MRB 18.3 ruling; standard rules apply.

### 269 Finishing Blow

No card-specific MRB 18.3 ruling; standard rules apply.

### 270 Ghostly Hand

No card-specific MRB 18.3 ruling; standard rules apply.

### 271 Hidden Stash

No card-specific MRB 18.3 ruling; standard rules apply.

### 272 Imperial Traitor

No card-specific MRB 18.3 ruling; standard rules apply.

### 273 Key of Darkness

No card-specific MRB 18.3 ruling; standard rules apply.

### 274 Lights Out

No card-specific MRB 18.3 ruling; standard rules apply.

### 275 Miasma

No card-specific MRB 18.3 ruling; standard rules apply.

### 276 Nerve Blast

**Rulings (MRB 18.3):**
- A steal whose source is replaced still counts as a steal for "if you do". *(no test)*

### 277 One Last Job

No card-specific MRB 18.3 ruling; standard rules apply.

### 278 Oubliette

No card-specific MRB 18.3 ruling; standard rules apply.

### 279 Pawn Sacrifice

No card-specific MRB 18.3 ruling; standard rules apply.

### 280 Poison Wave

**Rulings (MRB 18.3):**
- Creatures tagged for destruction by the same damage are destroyed together; healing one of them afterwards (Duma the Martyr) doesn't save it. *(no test)*

### 281 Relentless Whispers

No card-specific MRB 18.3 ruling; standard rules apply.

### 282 Routine Job

No card-specific MRB 18.3 ruling; standard rules apply.

### 283 Too Much to Protect

No card-specific MRB 18.3 ruling; standard rules apply.

### 284 Treasure Map

No card-specific MRB 18.3 ruling; standard rules apply.

### 285 Customs Office

No card-specific MRB 18.3 ruling; standard rules apply.

### 286 Evasion Sigil

No card-specific MRB 18.3 ruling; standard rules apply.

### 287 Longfused Mines

No card-specific MRB 18.3 ruling; standard rules apply.

### 288 Masterplan

**Rulings (MRB 18.3):**
- The card beneath is played before Masterplan is sacrificed, so it can still interact with Masterplan (e.g. return it to hand). Tested by `tests/test_cards_shadows_phase2.py::test_masterplan_stores_a_card_then_omni_plays_it_and_sacrifices`.

### 289 Safe Place

No card-specific MRB 18.3 ruling; standard rules apply.

### 290 Seeker Needle

**Rulings (MRB 18.3):**
- If the damage is redirected (Shadow Self), the creature that took it is destroyed normally, but the chosen creature wasn't, so no Æmber is gained. Tested by `tests/test_rulings_phase2.py::test_seeker_needle_redirect_destroys_shadow_self_without_the_gain`.

### 291 Skeleton Key

No card-specific MRB 18.3 ruling; standard rules apply.

### 292 Special Delivery

**Rulings (MRB 18.3):**
- See Shadow Self: the purge applies only to the chosen creature. Tested by `tests/test_rulings_phase2.py::test_special_delivery_destroys_shadow_self_but_does_not_purge_it`.

### 293 Speed Sigil

No card-specific MRB 18.3 ruling; standard rules apply.

### 294 Subtle Maul

No card-specific MRB 18.3 ruling; standard rules apply.

### 295 The Sting

No card-specific MRB 18.3 ruling; standard rules apply.

### 296 Bad Penny

**Rulings (MRB 18.3):**
- Its "Destroyed:" ability returns it to hand before other pending effects (Yxilo Bolter's purge) can reach it. Tested by `tests/test_cards_shadows.py::test_bad_penny_destroyed_goes_to_hand`.
- It still counts as destroyed for other cards (Tolas). Tested by `tests/test_rulings_phase2.py::test_a_bad_penny_returned_to_hand_still_counts_as_destroyed`.

### 297 Bulleteye

**Rulings (MRB 18.3):**
- Destroyed by assault damage, it was not destroyed in a fight (The Warchest doesn't count it). Tested by `tests/test_rulings_phase2.py::test_bulleteye_killed_by_assault_was_not_destroyed_in_a_fight`.

### 298 Carlo Phantom

No card-specific MRB 18.3 ruling; standard rules apply.

### 299 Deipno Spymaster

No card-specific MRB 18.3 ruling; standard rules apply.

### 300 Faygin

**Rulings (MRB 18.3):**
- Returning an enemy Urchin sends it to its owner's hand: a card leaving play always goes to its owner's zone unless the ability names another zone. The same rule applies to every "return ... to your hand" in the pool. Tested by `tests/test_cards_shadows_phase2.py::test_faygin_can_return_an_enemy_urchin_to_its_owners_hand`.

### 301 Macis Asp

**Rulings (MRB 18.3):**
- Poison applies to damage from its power even when that damage lands on a different creature (Drecker FAQ, and likewise a Shadow Self redirect). Tested by `tests/test_cards_shadows_phase2.py::test_shadow_self_redirect_via_a_real_fight_moves_poisons_kill_with_it`.

### 302 Mack the Knife

**Rulings (MRB 18.3):**
- It may deal the 1 damage to itself and still gain 1Æ: an ability that has started resolving finishes even if its card leaves play. Tested by `tests/test_rulings_phase2.py::test_mack_the_knife_can_destroy_itself_and_still_gain`.

### 303 Magda the Rat

**Errata:** MRB 18.3 errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Æmber on creatures destroyed along with Magda moves as they leave play, before her leaves-play steal. Tested by `tests/test_rulings_phase2.py::test_magda_the_rat_aember_on_creatures_moves_before_her_steal`.

### 304 Mooncurser

No card-specific MRB 18.3 ruling; standard rules apply.

### 305 Nexus

**Rulings (MRB 18.3):**
- Using an enemy artifact "as if it were yours" makes it count as friendly for its own ability. Tested by `tests/test_cards_shadows_phase2.py::test_nexus_uses_an_enemy_artifact_as_if_it_were_its_own`.
- An "After Reap:" ability gained from Spectral Tunneler resolves after Nexus reaps. Tested by `tests/test_cards_logos_phase2.py::test_spectral_tunneler_grants_flank_status_and_after_reap_draw`.

### 306 Noddy the Thief

No card-specific MRB 18.3 ruling; standard rules apply.

### 307 Old Bruno

No card-specific MRB 18.3 ruling; standard rules apply.

### 308 Dodger

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 309 Selwyn the Fence

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).


### 310 Shadow Self

**Rulings (MRB 18.3):**
- Armor on the neighbor prevents damage first; only what gets past armor is redirected (Raiding Knight example). Tested by `tests/test_rulings_phase2.py::test_armor_prevents_before_the_redirect`.
- Redirected damage is not reduced again by Shadow Self's own armor (Bulwark example). Tested by `tests/test_rulings_phase2.py::test_bulwark_armor_does_not_protect_shadow_self_from_redirected_damage`.
- Shadow Self destroyed by damage redirected from a fight was destroyed in a fight. Tested by `tests/test_cards_shadows_phase2.py::test_shadow_self_redirect_via_a_real_fight_moves_poisons_kill_with_it`.
- Special Delivery on its neighbor: Shadow Self is destroyed, not purged ("that creature" is the chosen one). Tested by `tests/test_rulings_phase2.py::test_special_delivery_destroys_shadow_self_but_does_not_purge_it`.

### 311 Silvertooth

**Rulings (MRB 18.3):**
- Enters play ready: abilities that change how a card enters play apply as it enters. Tested by `tests/test_cards_shadows.py::test_silvertooth_readies_itself`.

### 312 Smiling Ruth

No card-specific MRB 18.3 ruling; standard rules apply.

### 313 Sneklifter

**Rulings (MRB 18.3):**
- Sneklifter's house change lasts until the artifact leaves play; if control changes away and back, the artifact is Shadows again. Tested by `tests/test_cards_shadows_phase2.py::test_sneklifter_takes_control_of_an_enemy_artifact`.

### 314 Umbra

No card-specific MRB 18.3 ruling; standard rules apply.

### 315 Urchin

**Rulings (MRB 18.3):**
- See Faygin. Tested by `tests/test_cards_shadows_phase2.py::test_faygin_can_return_an_enemy_urchin_to_its_owners_hand`.

### 316 Duskrunner

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

**Rulings (MRB 18.3):**
- Abilities an upgrade grants count as part of the creature's text box. *(no test)*

### 317 Ring of Invisibility

No card-specific MRB 18.3 ruling; standard rules apply.

### 318 Silent Dagger

**Errata:** general Fight/Reap errata (the pool table's canonical text already includes it).

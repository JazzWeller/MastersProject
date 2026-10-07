"""The coverage registry (Agent Observation Plan, Milestone O4; invariant
I4): every attribute of every engine state object, classified.

- ENCODED: the v2 extract (`keyforge/infoset_v2.py`) reads it;
- DERIVED: it follows from what is encoded, or is the engine's machinery
  over it (an index, a cache, the journal behind the history);
- HIDDEN: no viewer may read it, by the rule given;
- BOOKKEEPING: carries no information -- and so must always hold its
  default, which the registry test checks.

`tests/test_agent_observation_o4.py` introspects live objects across fuzz
games (every pool) and fails on an attribute missing here, or a
BOOKKEEPING one that ever differs from its default. A new attribute in the
engine is a decision about what the agent sees, made here.
"""

from __future__ import annotations

ENCODED, DERIVED, HIDDEN, BOOKKEEPING = "encoded", "derived", "hidden", "bookkeeping"

_E = (ENCODED, "")


def _bk(default, why):
    return (BOOKKEEPING, why, default)


REGISTRY = {
    "Game": {
        "players": _E,
        "active_effects": _E,
        "turn_number": _E,
        "active_player_id": _E,
        "_first_player": _E,
        "is_over": _E,
        "result": (DERIVED, "the outcome, once over: turn.is_over plus the players' keys"),
        "pending_decision": (ENCODED, "the decision block (v1's, kept)"),
        "_elusive_suppressed": _E,
        "_end_of_turn_cleanups": (ENCODED, "cleanup tokens"),
        "_temp_control": (ENCODED, "temporary-control tokens"),
        "_redirected_hits": _E,
        "_machine": (ENCODED, "resolution tokens (the explicit stack, R7)"),
        "_driver": (DERIVED, "the native/compiled driver over the same stack"),
        "_causes": (DERIVED, "the sources of the resolving frames, which the resolution tokens carry"),
        "config": (ENCODED, "max_turns; the rest is the decklists and seed"),
        "journal": (DERIVED, "the history (O6), through the projection"),
        "log": (DERIVED, "the history (O6), through the projection"),
        "choice_record": (DERIVED, "the projection's decision events"),
        "choice_log": (DERIVED, "the same choices, as live objects"),
        "_rng_counters": (HIDDEN, "how much randomness was drawn: the future's keys, never shown"),
        "_instance_counter": (DERIVED, "fixed after the deal: the decklists"),
        "_cards_by_id": (DERIVED, "an index of the entities"),
        "_fork_snapshot": (DERIVED, "a cache of this game, for forks"),
        "_player_houses_cache": (DERIVED, "a cache of each player's houses"),
        "execution": (DERIVED, "an engine setting with no game meaning"),
        "_container_index": (DERIVED, "a cache for copies"),
        "_projectors": (DERIVED, "per-viewer caches of the projection"),
        "_trackers": (DERIVED, "per-viewer caches of the knowledge tracker"),
        "_trackers2": (DERIVED, "per-viewer caches of the second-order tracker"),
        "_chance": (DERIVED, "per-viewer caches of the chance filter"),
        "_history": (DERIVED, "per-viewer caches of the history encoding (O6)"),
        "_world_cut": (DERIVED, "where a determinized world began (O3)"),
    },
    "Player": {
        "id": (DERIVED, "the side"),
        "deck": (ENCODED, "entities: knowledge, and sizes"),
        "hand": (ENCODED, "entities"),
        "discard": (ENCODED, "entities"),
        "archive": (ENCODED, "entities"),
        "purged": (ENCODED, "entities"),
        "play_area": (ENCODED, "entities' positions"),
        "all_cards": (DERIVED, "the decklist: the entity order"),
        "aember": _E, "keys": _E, "chains": _E,
        "CanKeyForge": _E, "CanPlayActions": _E, "CanPlayCreatures": _E, "CardDrawModifier": _E,
        "CardPlayedLimit": _E, "CardsPlayed": _E, "DrawUpToLimit": _E, "KeyForgeCost": _E,
        "NonLogosCardsPlayable": _E, "ExtraHousePlayable": _E, "HouseSelection": _E, "ReapGainBecomesSteal": _E,
        "CanFight": _E, "CannotUseCards": _E, "CanPlayCards": _E, "FirstCreatureEntersReady": _E,
        "CannotBeDealtDamage": _E, "CanOnlyFight": _E, "CannotBeStolenFrom": _E,
        "next_entry_ready": _E, "next_mars_creature_ready": _E, "discarded_untamed_this_turn": _E,
        "creatures_played_this_turn": _E, "selected_house": _E, "cards_played_or_discarded_this_turn": _E,
        "hand_plays_this_turn": _E, "used_this_turn": _E, "hand_revealed_to": _E,
    },
    "Card": {
        "instance_id": (DERIVED, "the entity's index"),
        "card_def": (DERIVED, "static: by the card's name"),
        "type_object": (ENCODED, "its attributes, below"),
        "id": _E, "name": _E, "house": _E, "type": _E, "tags": _E, "keywords": _E, "aember_on_play": _E,
        "aember_captured": _E, "image": (DERIVED, "static: the art"),
        "owner": _E, "controller": _E, "CanBeUsed": _E, "Exhausted": _E, "destroyed": _E,
        "fought_this_turn": _E, "flank": _E, "extra_triggers": (ENCODED, "the hooks it holds, by name"),
        "destined_zone": _E, "power_counters": _E, "stunned": _E, "house_override": _E, "forced_flank": _E,
        "granted_action": _E, "aember_stored": _E, "under_cards": (ENCODED, "each points to its host"),
        "purged_by": _E, "redirect_fight_damage_to": _E, "armor_negated": _E, "damage_prevented": _E,
        "archive_return_to_owner": _E,
        "IgnoreElusive": _bk(False, "never set to True by any card"),
        "_ember_imp_effect": (DERIVED, "holds its effect so it can be unregistered; the effect is an effect token"),
    },
    "CreatureType": {
        "base_power": _E, "base_armor": _E, "damage": _E, "armor_used_this_turn": _E,
        "upgrades": (ENCODED, "each upgrade points to its host"),
    },
    "UpgradeType": {"host": (ENCODED, "the host pointer")},
    "ArtifactType": {},
    "ActionType": {},
    "DurationEffect": {
        "source_card": _E, "controller": _E, "remaining_duration": _E, "player_affected": _E,
        "variable": _E, "op": _E, "value": (ENCODED, "evaluated now"), "conditional": (ENCODED, "evaluated now"),
    },
    "TriggerEffect": {
        "source_card": _E, "controller": _E, "event": _E, "remaining_duration": _E,
        "handler": (ENCODED, "by name"),
    },
    "InsteadEffect": {"source_card": _E, "controller": _E, "kind": _E, "handler": (ENCODED, "by name")},
    "ModifierEffect": {"source_card": _E, "controller": _E, "kind": _E, "handler": (ENCODED, "by name")},
    "ActiveEffectList": {
        "duration_effects": _E, "trigger_effects": _E, "instead_effects": _E, "modifier_effects": _E,
        "_duration_by_key": (DERIVED, "an index of the duration effects"),
    },
    "Deck": {"_cards": (ENCODED, "entities: knowledge, known positions, size"), "_j": (DERIVED, "the journal"),
             "zone": (DERIVED, "the zone's name")},
    "Hand": {"_cards": (ENCODED, "entities"), "_j": (DERIVED, "the journal"), "zone": (DERIVED, "the zone's name")},
    "DiscardPile": {"_cards": (ENCODED, "entities"), "_j": (DERIVED, "the journal"), "zone": (DERIVED, "the zone's name")},
    "Archive": {"_cards": (ENCODED, "entities"), "_j": (DERIVED, "the journal"), "zone": (DERIVED, "the zone's name")},
    "PurgedZone": {"_cards": (ENCODED, "entities"), "_j": (DERIVED, "the journal"), "zone": (DERIVED, "the zone's name")},
    "PlayArea": {"creatures": (ENCODED, "entities' positions"), "artifacts": (ENCODED, "entities' positions"),
                 "_j": (DERIVED, "the journal"), "pid": (DERIVED, "the side"),
                 "_bl": (DERIVED, "the zone's name"), "_ar": (DERIVED, "the zone's name")},
}


def attributes(obj):
    """The attribute names of a live object (slots, then instance dict)."""
    names = []
    for k in type(obj).__mro__:
        names.extend(getattr(k, "__slots__", ()))
    names.extend(getattr(obj, "__dict__", {}).keys())
    return [n for n in dict.fromkeys(names) if hasattr(obj, n)]


def check(obj, problems: set) -> None:
    """Adds to `problems` every attribute of `obj` not in the registry, and
    every BOOKKEEPING one away from its default."""
    cls = type(obj).__name__
    table = REGISTRY.get(cls)
    if table is None:
        problems.add(f"{cls}: not in the registry")
        return
    for name in attributes(obj):
        entry = table.get(name)
        if entry is None:
            problems.add(f"{cls}.{name}: unclassified")
        elif entry[0] == BOOKKEEPING and getattr(obj, name) != entry[2]:
            problems.add(f"{cls}.{name}: bookkeeping, but {getattr(obj, name)!r} (default {entry[2]!r})")


def check_game(game, problems: set) -> None:
    check(game, problems)
    check(game.active_effects, problems)
    for e in (game.active_effects.duration_effects + game.active_effects.trigger_effects
              + game.active_effects.instead_effects + game.active_effects.modifier_effects):
        check(e, problems)
    for p in game.players.values():
        check(p, problems)
        for z in ("deck", "hand", "discard", "archive", "purged", "play_area"):
            check(getattr(p, z), problems)
    for c in game._cards_by_id.values():
        check(c, problems)
        check(c.type_object, problems)

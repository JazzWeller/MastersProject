"""The v2 tensor specification (Agent Observation Plan, Milestone O5):
FEATURE_VERSION 2, lossless.

v1 (`agent/spec.py`, `agent/features_v1.py`) is frozen; v1 checkpoints load
only with it. v2 has typed token sets, each a padded array of rows with a
mask, and nothing capped:

| Block | Rows | What |
|---|---|---|
| GLOBAL | 1 | the turn, the cap and turns left, both players' every counter and flag, the decision |
| ENTITY | 72 | each card: identity, the viewer's knowledge (O2) and priors (O3), and, where visible, its whole state |
| EFFECT | any | each lasting effect: kind, name, op, value now, duration, active, sides, source |
| RESOLUTION | any | each frame of the resolution stack (R7): routine, site, depth, ability kind, event, source |
| RES_POINTER | any | (frame, role, entity) for each card a frame's locals hold |
| RES_OP | any | (frame, operation) for each operation still reachable from a frame |
| CLEANUP | any | each pending end-of-turn operation: op, target |
| MATCH | any | each previous game of the match |
| OPTION | any | each option: verb, pointer, payloads (no hashing: mode names and events are ids) |

Columns are integers (`I`: ids, pointers, codes, bit sets) or floats
(`F`). Every id comes from a vocabulary in `agent/vocab/` (an unknown entry
raises). A pointer is an entity index, `NO_POINTER` for none, or
`HIDDEN_MINE` / `HIDDEN_THEIRS` for a card the viewer can't see.

Static card meaning is three tables, all built (`agent/static_v2.py`):
attributes (v1's, with traits from the vocabulary), parsed text rows, and
engine signatures; plus a text embedding when its table exists.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

FEATURE_VERSION = 2
FEATURE_MINOR = 0

NO_POINTER = -1
HIDDEN_MINE = -2
HIDDEN_THEIRS = -3

N_ENTITIES = 72

ZONE_KINDS = ("hand", "deck", "discard", "archive", "purged", "battleline", "artifacts", "attached", "under",
              "limbo", "setup")
# A zone code: 1 + kind * 3 + side (0 mine, 1 theirs, 2 neither); 0 = none.
N_ZONE_CODES = 1 + 3 * len(ZONE_KINDS)


def zone_code(kind: str, side) -> int:
    return 1 + ZONE_KINDS.index(kind) * 3 + (2 if side is None else side)


class Columns:
    """A token block's columns: (name, I/F), in order."""

    def __init__(self, name: str, columns: List[Tuple[str, str]]):
        self.name = name
        self.columns = list(columns)
        self.int_names = [c for c, t in self.columns if t == "I"]
        self.float_names = [c for c, t in self.columns if t == "F"]
        self.i = {c: k for k, c in enumerate(self.int_names)}
        self.f = {c: k for k, c in enumerate(self.float_names)}
        self.n_int = len(self.int_names)
        self.n_float = len(self.float_names)

    def describe(self):
        return [list(c) for c in self.columns]


ENTITY = Columns("entity", [
    ("card", "I"), ("owner", "I"),
    # knowledge (O2)
    ("exact", "I"), ("zone", "I"), ("mask", "I"), ("deck_pos", "I"), ("group", "I"), ("group_size", "I"),
    ("since_seen", "I"), ("last_seen", "I"), ("exit_op", "I"), ("opp_exact", "I"), ("opp_mask", "I"),
    # priors (O3), for the opponent's hidden cards
    ("p_hand", "F"), ("p_archive", "F"), ("p_deck", "F"), ("p_elsewhere", "F"), ("p_next_draw", "F"),
    # visible state
    ("visible", "I"), ("controller", "I"), ("position", "I"), ("index", "I"), ("of", "I"), ("flank", "I"),
    ("host", "I"), ("under_index", "I"), ("house", "I"), ("house_override", "I"), ("destined_zone", "I"),
    ("temp_original", "I"), ("redirect_to", "I"), ("purged_by", "I"), ("flags", "I"), ("keywords", "I"),
    ("extra_triggers", "I"), ("selected", "I"), ("uses_this_turn", "I"), ("played_this_turn", "I"),
    ("damage", "F"), ("power", "F"), ("armor", "F"), ("armor_used", "F"), ("base_power", "F"), ("base_armor", "F"),
    ("captured", "F"), ("stored", "F"), ("power_counters", "F"), ("hazardous", "F"), ("assault", "F"),
    ("n_upgrades", "F"), ("n_under", "F"),
])
# ENTITY.flags bits
ENTITY_FLAGS = ("exhausted", "stunned", "can_be_used", "destroyed", "fought_this_turn", "forced_flank",
                "granted_action", "archive_return_to_owner", "damage_prevented", "armor_negated", "is_creature",
                "ignore_elusive", "subject")  # subject: the deck a match-level bid is about
FLANKS = ("left", "right", "both", "center")

PLAYER_COUNTERS = (
    "aember", "keys", "chains", "CardDrawModifier", "DrawUpToLimit", "KeyForgeCost", "NonLogosCardsPlayable",
    "creatures_played_this_turn", "cards_played_or_discarded_this_turn", "hand_plays_this_turn",
    "key_cost_now", "n_hand", "n_deck", "n_discard", "n_archive", "n_purged", "n_creatures", "n_artifacts",
    "card_played_limit", "uses_this_turn", "cards_played_kinds",
)
PLAYER_FLAGS = (
    "CanKeyForge", "CanPlayActions", "CanPlayCreatures", "ReapGainBecomesSteal", "CanFight", "CannotUseCards",
    "CanPlayCards", "FirstCreatureEntersReady", "CannotBeDealtDamage", "CanOnlyFight", "CannotBeStolenFrom",
    "next_entry_ready", "next_mars_creature_ready", "discarded_untamed_this_turn", "hand_revealed_to_other",
    "has_card_played_limit",
)
GLOBAL = Columns("global", [
    ("turn", "F"), ("is_my_turn", "I"), ("i_went_first", "I"), ("max_turns", "F"), ("has_max_turns", "I"),
    ("turns_left", "F"), ("elusive_suppressed", "I"), ("is_over", "I"),
    ("active_house", "I"), ("my_houses", "I"), ("their_houses", "I"),
    ("my_selected_house", "I"), ("their_selected_house", "I"), ("my_house_selection", "I"),
    ("their_house_selection", "I"), ("my_extra_house_playable", "I"), ("their_extra_house_playable", "I"),
    ("format", "I"), ("game_number", "F"), ("my_score", "F"), ("their_score", "F"), ("my_start_chains", "F"),
    ("their_start_chains", "F"), ("swapped", "I"),
    ("kind", "I"), ("intent", "I"), ("affects", "I"), ("optional", "I"), ("min_n", "F"), ("max_n", "F"),
    ("n_options", "F"), ("prefix", "F"), ("decision_source", "I"), ("decider", "I"),
] + [(f"my_{c}", "F") for c in PLAYER_COUNTERS] + [(f"their_{c}", "F") for c in PLAYER_COUNTERS]
  + [(f"my_{c}", "I") for c in PLAYER_FLAGS] + [(f"their_{c}", "I") for c in PLAYER_FLAGS])

EFFECT = Columns("effect", [
    ("kind", "I"), ("variable", "I"), ("event", "I"), ("effect_kind", "I"), ("op", "I"), ("value", "F"), ("value_kind", "I"), ("remaining", "F"),
    ("infinite", "I"), ("active", "I"), ("conditional", "I"), ("affected", "I"), ("controller", "I"),
    ("source", "I"),
])
EFFECT_KINDS = ("duration", "trigger", "instead", "modifier")
VALUE_KINDS = ("none", "number", "bool", "house", "other")

RESOLUTION = Columns("resolution", [
    ("routine", "I"), ("site", "I"), ("depth", "I"), ("kind", "I"), ("event", "I"), ("source", "I"),
    ("pc", "I"), ("n_locals", "I"),
])
# A card a frame's local holds: the entity (visible), or HIDDEN_* with its
# zone and how many such cards the local holds (hidden cards have no place).
RES_POINTER = Columns("res_pointer", [("frame", "I"), ("role", "I"), ("entity", "I"), ("zone", "I"), ("count", "I"),
                                      ("position", "I")])
RES_OP = Columns("res_op", [("frame", "I"), ("op", "I"), ("order", "I")])
CLEANUP = Columns("cleanup", [("op", "I"), ("target", "I")])
MATCH = Columns("match", [
    ("winner", "I"), ("my_keys", "F"), ("their_keys", "F"), ("turns", "F"), ("my_chains", "F"),
    ("their_chains", "F"), ("first", "I"), ("swapped", "I"),
])
OPTION = Columns("option", [
    ("verb", "I"), ("pointer", "I"), ("number", "F"), ("bool", "I"), ("house", "I"), ("flank", "I"),
    ("first_player", "I"), ("mode", "I"), ("event", "I"), ("tag", "I"),
    ("hand_of_house", "F"), ("creatures_of_house", "F"), ("ready_of_house", "F"), ("artifacts_of_house", "F"),
])

BLOCKS = (GLOBAL, ENTITY, EFFECT, RESOLUTION, RES_POINTER, RES_OP, CLEANUP, MATCH, OPTION)
BLOCK_BY_NAME = {b.name: b for b in BLOCKS}


def layout() -> dict:
    return {"feature_version": FEATURE_VERSION, "blocks": {b.name: b.describe() for b in BLOCKS},
            "zone_kinds": list(ZONE_KINDS), "entity_flags": list(ENTITY_FLAGS), "flanks": list(FLANKS)}


LAYOUT_HASH = hashlib.sha256(json.dumps(layout(), sort_keys=True).encode("utf-8")).hexdigest()


def stamp() -> dict:
    """What every v2 checkpoint and shard carries: the layout and every
    vocabulary's version (as one hash, plus the card vocabulary's)."""
    from keyforge.cards.vocabulary import VOCAB_HASH
    from keyforge.version import ENGINE_VERSION

    from .vocab import vocab_hash

    return {"feature_version": FEATURE_VERSION, "feature_minor": FEATURE_MINOR, "layout_hash": LAYOUT_HASH,
            "vocab_hash": VOCAB_HASH, "vocab_v2_hash": vocab_hash(), "engine_version": ENGINE_VERSION}

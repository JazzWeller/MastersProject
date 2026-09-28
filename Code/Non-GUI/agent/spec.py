"""The tensor specification (Agent Training Plan, Milestone M1): every
feature layout constant, and the version numbers stamped into checkpoints
and training shards.

**Five blocks.** Each is a list of named fields with widths, ending in
explicitly reserved, always-zero slots:

- `STATIC` -- per card *definition*, a row of `static_table()` indexed by
  vocabulary id: house, type, printed stats, keywords, traits, ability
  presence, rules-text verbs. These never change during a game, so they're
  never re-encoded per decision; the network looks them up by `card_id`
  (`ml/encode.py`). This is the "attribute path" -- what carries a card
  the network has never trained on (M2, M11).
- `ENTITY` -- per entity, per decision: zone, knowledge/ownership flags,
  in-play state, decision-relative flags.
- `INPLAY` -- the in-play slice of `ENTITY`, stored sparsely (only for
  entities actually on the board) by the compact encoding.
- `GLOBAL` -- one vector per decision.
- `OPTION` -- one row per legal option.

The dense entity row the network sees is `STATIC[card_id] ++ ENTITY`; see
`entity_width()`.

**The reserved-slot rule** (enforced by `tests/test_agent_training_m1.py`):
adding a feature *into a reserved slot* bumps `FEATURE_MINOR` only, and every
existing checkpoint keeps loading with the new dimension zero. Adding one
anywhere else shifts indices and bumps `FEATURE_VERSION`, invalidating every
checkpoint and every encoded tensor. `reserved_report()` states how many
reserved slots each block has left; a block reaching zero is a deliberate
decision point, not a surprise.

To consume a reserved slot: rename one `reserved_*` field (keeping its
width) and bump `FEATURE_MINOR`. Never insert a field before an existing one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

from keyforge.enums import Affects, CardType, DecisionIntent, DecisionKind
from keyforge.infoset import FLAG_BITS, HOUSES, VERB_NAMES, ZONE_NAMES

FEATURE_VERSION = 1  # major: any non-reserved layout change
FEATURE_MINOR = 1  # a reserved slot put to use -- 1: house-option context (OPTION.house_context)

# Reserved slots put to use, per minor version: (block name, field name).
# A checkpoint from an older minor version never trained on these inputs,
# so `ml/checkpoints.py` zeroes their input weights on load -- the new
# dimension then starts out exactly inert, and the old network's behaviour
# is preserved until it's fine-tuned.
MINOR_ADDITIONS = {
    1: (("option", "house_context"),),
}

N_ENTITIES = 72
CARDS_PER_PLAYER = 36
VOCAB_CAPACITY = 370  # the whole CotA registry, from day one (M11)

# ---------------------------------------------------------- categoricals --
# All append-only, like the infoset vocabularies they mirror.
KEYWORDS: Tuple[str, ...] = ("elusive", "skirmish", "taunt", "poison", "versatile", "no_fight_damage", "assault", "hazardous")
KEYWORD_SLOTS = 24  # 8 used, 16 spare keyword bits (the plan's reserve for new keywords)
TRAIT_BUCKETS = 32  # traits hashed (crc32) into buckets, no trait vocabulary
TEXT_VERBS: Tuple[str, ...] = (
    "destroy", "damage", "capture", "steal", "gain", "lose", "draw", "discard", "archive", "purge",
    "ready", "exhaust", "stun", "heal", "return", "shuffle", "play", "use", "fight", "reap",
    "forge", "key", "control", "power",
)
ABILITIES: Tuple[str, ...] = ("on_play", "on_reap", "on_fight", "on_action", "on_omni", "on_destroyed", "passive", "on_before_fight")
CARD_TYPES: Tuple[CardType, ...] = (CardType.ACTION, CardType.ARTIFACT, CardType.CREATURE, CardType.UPGRADE)
DECISION_KINDS: Tuple[DecisionKind, ...] = tuple(DecisionKind)  # 13 today
KIND_SLOTS = 16
INTENTS: Tuple[DecisionIntent, ...] = tuple(DecisionIntent)  # index 0 of the block = "no intent"
INTENT_SLOTS = 1 + len(INTENTS) + 8  # none + today's + 8 spare intent slots
AFFECTS: Tuple[Affects, ...] = tuple(Affects)  # index 0 of the block = "not given"
FORMATS: Tuple[str, ...] = ("archon", "reversal", "adaptive")
FLANKS: Tuple[str, ...] = ("left", "right", "both", "neither")
# Duration-effect variables the engine reads (Player.get_*, duration_effects_for).
EFFECT_VARIABLES: Tuple[str, ...] = (
    "ArtifactPlayToll", "ArtifactUseToll", "CanFight", "CanKeyForge", "CanOnlyFight", "CanPlayActions",
    "CanPlayCards", "CanPlayCreatures", "CannotBeDealtDamage", "CannotBeStolenFrom", "CannotChooseHouse",
    "CannotUseCards", "CardDrawModifier", "CardPlayedLimit", "DrawUpToLimit", "FightPermittedHouse",
    "FirstCreatureEntersReady", "HouseSelection", "KeyForgeCost", "NonLogosCardsPlayable",
    "ReapGainBecomesSteal", "RedirectsForgePayment", "UsePermittedHouse", "FlankAndAfterReapDraw",
)
EFFECT_SLOTS = 32  # per side; the last slot is "other" (an unlisted variable)
STRING_BUCKETS = 8  # CHOOSE_MODE names, trigger events: crc32-hashed

# Normalizers: divide by these so typical values land in ~[0, 1].
SCALE = {
    "aember": 10.0, "keys": 3.0, "chains": 6.0, "hand": 10.0, "deck": 36.0, "pile": 36.0,
    "small_pile": 10.0, "board": 10.0, "key_cost": 10.0, "turn": 50.0, "power": 10.0,
    "armor": 5.0, "tokens": 5.0, "n_options": 20.0, "n": 10.0, "match_game": 3.0,
    "score": 2.0, "bid": 24.0, "number": 10.0,
}


class Block:
    """A named-field feature block: `offset(name)`, `width`, and how many of
    its trailing `reserved_*` slots are still unused."""

    def __init__(self, name: str, fields: List[Tuple[str, int]]):
        self.name = name
        self.fields = list(fields)
        self._offsets: Dict[str, Tuple[int, int]] = {}
        at = 0
        for fname, w in self.fields:
            if fname in self._offsets:
                raise ValueError(f"{name}: duplicate field {fname!r}")
            self._offsets[fname] = (at, w)
            at += w
        self.width = at

    def offset(self, field: str) -> int:
        return self._offsets[field][0]

    def span(self, field: str) -> Tuple[int, int]:
        return self._offsets[field]

    def reserved_remaining(self) -> int:
        return sum(w for f, w in self.fields if f.startswith("reserved"))

    def describe(self) -> list:
        return [[f, w] for f, w in self.fields]


STATIC = Block("static", [
    ("house", len(HOUSES)),
    ("type", len(CARD_TYPES)),
    ("power", 1),
    ("armor", 1),
    ("aember", 1),
    ("keywords", KEYWORD_SLOTS),
    ("assault", 1),
    ("hazardous", 1),
    ("traits", TRAIT_BUCKETS),
    ("abilities", len(ABILITIES)),
    ("upgrade_power_bonus", 1),
    ("upgrade_armor_bonus", 1),
    ("grants_keywords", len(KEYWORDS)),
    ("text_verbs", len(TEXT_VERBS)),
    ("reserved_static", 8),
])

INPLAY = Block("inplay", [
    ("damage", 1),
    ("power", 1),
    ("armor", 1),
    ("armor_used", 1),
    ("captured", 1),
    ("stored", 1),
    ("power_counters", 1),
    ("stunned", 1),
    ("ready", 1),
    ("flank", len(FLANKS)),
    ("battleline_index", 1),
    ("n_upgrades", 1),
    ("n_under", 1),
    ("damage_prevented", 1),
    ("armor_negated", 1),
    ("fought_this_turn", 1),
    ("effective_house", len(HOUSES)),
    ("keywords", len(KEYWORDS)),
    ("hazardous", 1),
    ("assault", 1),
    ("is_creature", 1),
])

ENTITY = Block("entity", [
    ("zone", len(ZONE_NAMES) + 2),  # +2 spare zone codes
    ("flags", len(FLAG_BITS) + 1),  # +1 spare flag bit
    ("inplay", INPLAY.width),
    ("selected_in_prefix", 1),  # sequential multi-select: already picked
    ("reserved_entity", 16),
])

GLOBAL = Block("global", [
    ("me", 17),
    ("them", 17),
    ("turn", 1),
    ("is_my_turn", 1),
    ("i_went_first", 1),
    ("active_house", len(HOUSES) + 1),  # +1: no active house
    ("my_houses", len(HOUSES)),
    ("their_houses", len(HOUSES)),
    ("effects_me", EFFECT_SLOTS),
    ("effects_them", EFFECT_SLOTS),
    ("format", len(FORMATS) + 1),
    ("match", 6),  # game number, my score, their score, my/their starting chains, deck swapped
    ("kind", KIND_SLOTS),
    ("intent", INTENT_SLOTS),
    ("affects", 1 + len(AFFECTS)),
    ("decision_scalars", 5),  # optional, min_n, max_n, n_options, prefix length
    ("reserved_global", 24),
])

OPTION = Block("option", [
    ("verb", len(VERB_NAMES) + 3),  # +3 spare verbs; slot len(VERB_NAMES) = sequential "stop"
    ("has_pointer", 1),
    ("number", 1),
    ("bool", 1),
    ("house", len(HOUSES)),
    ("flank", 2),
    ("first_player", 2),
    ("string", STRING_BUCKETS),
    # Minor 1: a house option (CHOOSE_HOUSE / CHOOSE_HOUSE_FOR_EFFECT) has
    # no entity to point at, so it carries what choosing it unlocks: my hand
    # cards, creatures, ready creatures and artifacts of that house.
    ("house_context", 4),
    ("reserved_option", 4),
])
STOP_VERB = len(VERB_NAMES)  # the sequential multi-select "stop here" pseudo-option

BLOCKS = (STATIC, ENTITY, INPLAY, GLOBAL, OPTION)


def entity_width() -> int:
    """The dense per-entity row the trunk sees: static attributes (looked up
    by card id) followed by the per-decision entity block."""
    return STATIC.width + ENTITY.width


def layout() -> dict:
    """The full layout, as data -- hashed into `LAYOUT_HASH` and stamped
    into checkpoints, so a checkpoint can be checked against more than
    just the version integers."""
    return {
        "feature_version": FEATURE_VERSION,
        "blocks": {b.name: b.describe() for b in BLOCKS},
        "zones": list(ZONE_NAMES),
        "verbs": list(VERB_NAMES),
        "keywords": list(KEYWORDS),
        "text_verbs": list(TEXT_VERBS),
        "effect_variables": list(EFFECT_VARIABLES),
        "kinds": [k.name for k in DECISION_KINDS],
        "intents": [i.name for i in INTENTS],
    }


def _layout_hash(ignore_reserved_names: bool) -> str:
    data = layout()
    if ignore_reserved_names:
        # A MINOR change renames reserved fields and appends to the
        # vocabularies; the major hash is over the index-bearing shape only.
        data = {"feature_version": FEATURE_VERSION, "widths": {b.name: b.width for b in BLOCKS}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


LAYOUT_HASH = _layout_hash(False)
SHAPE_HASH = _layout_hash(True)


def reserved_report() -> Dict[str, int]:
    """Reserved slots left, per block -- plus the spare keyword, intent,
    kind, zone and verb capacity."""
    return {
        "static": STATIC.reserved_remaining(),
        "entity": ENTITY.reserved_remaining(),
        "global": GLOBAL.reserved_remaining(),
        "option": OPTION.reserved_remaining(),
        "keyword_bits": KEYWORD_SLOTS - len(KEYWORDS),
        "intents": INTENT_SLOTS - 1 - len(INTENTS),
        "kinds": KIND_SLOTS - len(DECISION_KINDS),
        "zones": ENTITY.span("zone")[1] - len(ZONE_NAMES),
        "verbs": OPTION.span("verb")[1] - len(VERB_NAMES) - 1,
        "effect_variables": EFFECT_SLOTS - 1 - len(EFFECT_VARIABLES),
    }


def stamp() -> dict:
    """What every checkpoint and encoded shard carries."""
    from keyforge.cards.vocabulary import VOCAB_HASH
    from keyforge.version import ENGINE_VERSION

    return {
        "feature_version": FEATURE_VERSION,
        "feature_minor": FEATURE_MINOR,
        "shape_hash": SHAPE_HASH,
        "layout_hash": LAYOUT_HASH,
        "vocab_hash": VOCAB_HASH,
        "engine_version": ENGINE_VERSION,
    }


class FeatureVersionMismatch(ValueError):
    pass


def check_stamp(data: dict, what: str = "artifact", *, allow_minor: bool = True) -> None:
    """Refuses `data` (a `stamp()` dict) made against a different major
    layout. A different minor version is fine when `allow_minor` (the new
    reserved dimensions are zero in the old artifact); a different
    vocabulary hash is fine too as long as it only appended rows, which
    `ml/checkpoints.py` checks separately (it needs the vocabularies)."""
    if data.get("feature_version") != FEATURE_VERSION or data.get("shape_hash") != SHAPE_HASH:
        raise FeatureVersionMismatch(
            f"{what} has feature_version={data.get('feature_version')!r} shape={str(data.get('shape_hash'))[:12]!r}; "
            f"this code is feature_version={FEATURE_VERSION} shape={SHAPE_HASH[:12]}"
        )
    if not allow_minor and data.get("feature_minor") != FEATURE_MINOR:
        raise FeatureVersionMismatch(f"{what} has feature_minor={data.get('feature_minor')!r}, need {FEATURE_MINOR}")

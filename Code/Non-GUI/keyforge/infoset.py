"""A fast, redacted information-set extract for learned agents (Agent
Training Plan, Milestone M1).

`keyforge/observation.py`'s `Observation` is the audited, serializable,
everything-a-viewer-may-know record -- but building one costs ~640 us per
decision (every visible card's full canonical state, plus the whole
redacted history), ~25x the engine's own per-decision cost. A network
encoder needs a small, fixed subset of that, every decision, in the hot
path of search and self-play. `InfoSet` is that subset, built directly
from the live game under exactly the same visibility rules
`build_observation` applies:

- my hand and my archive are visible to me; my deck is visible only as an
  unordered set (by elimination -- both decklists are public);
- the opponent's hand is visible only while `Player.hand_revealed_to`
  covers me; their archive and deck never are;
- discard piles, purged zones and everything in play are public;
- a card placed facedown beneath another (Masterplan) is visible only to
  its own owner, who put it there from their own hand.

**Every card is always one of the 72 entities** (both 36-card decklists are
public), so hidden information is expressed by *where* an entity is said to
be, never by omitting it: an opponent card I'm not entitled to locate is
`opp_unseen` ("somewhere in their hand, deck or archive"), one of mine in my
deck is `my_deck_unordered`. The set of unseen cards is known exactly; only
the split is hidden.

Nothing here holds a live `Card` (only names, instance ids and plain
values), and nothing a viewer isn't entitled to is ever read into it: the
leak test for this module (`tests/test_agent_training_m1.py`) asserts that
the InfoSet of a game and of every determinized fork of it
(`Game.fork_determinized`, which re-deals exactly the hidden cards) encode
byte-identically -- i.e. the extract is a function of the information set
alone.

Zone codes, option verbs and flag bits are **append-only**: an agent's
feature layout (`agent/spec.py`) indexes into them, and a checkpoint trained
against one layout must keep meaning the same thing later.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card, CreatureType
from .effects.effect_object import TriggerEffect
from .enums import DecisionKind, House

# ------------------------------------------------------------ vocabulary --
# Append-only. "my"/"opp" is the SIDE whose zone it is, from the viewer's
# perspective -- not the card's owner, which is a separate flag (a card I
# own can sit in my opponent's battleline after they take control of it).
ZONE_NAMES: Tuple[str, ...] = (
    "my_hand",
    "my_deck_unordered",
    "my_discard",
    "my_archive",
    "my_purged",
    "my_creature",
    "my_artifact",
    "my_upgrade_attached",
    "my_under",
    "my_limbo",  # in no zone at all: being played/resolved right now (public)
    "opp_hand",
    "opp_deck_unordered",
    "opp_discard",
    "opp_archive",
    "opp_purged",
    "opp_creature",
    "opp_artifact",
    "opp_upgrade_attached",
    "opp_under",
    "opp_limbo",
    "opp_unseen",
    "none",  # between games of a match: no zone at all
)
ZONE = {name: i for i, name in enumerate(ZONE_NAMES)}
_UNLOCATED = frozenset({ZONE["my_deck_unordered"], ZONE["opp_unseen"], ZONE["none"]})

# Per-entity flag bits.
KNOWN_TO_ME = 1
KNOWN_TO_THEM = 2
OWNER_IS_ME = 4
CONTROLLER_IS_ME = 8
LEGAL_OPTION = 16  # referenced by an option of the pending decision
SOURCE_CARD = 32  # the card that asked the pending decision
SUBJECT = 64  # what a match-level decision is about (the deck being bid on)
FLAG_BITS = (KNOWN_TO_ME, KNOWN_TO_THEM, OWNER_IS_ME, CONTROLLER_IS_ME, LEGAL_OPTION, SOURCE_CARD, SUBJECT)

# Option verbs, append-only.
VERB_NAMES: Tuple[str, ...] = (
    "play",
    "discard",
    "use_action",
    "use_omni",
    "reap",
    "fight",
    "end_turn",
    "card",  # select a card (CHOOSE_CARDS, a Card in ORDER_EFFECTS)
    "trigger",  # an ORDER_EFFECTS trigger
    "house",
    "flank",
    "bool",
    "number",
    "mode",  # a named string choice (CHOOSE_MODE, a named ORDER_EFFECTS step)
    "first_player",
    "bid",
    "pass",
)
VERB = {name: i for i, name in enumerate(VERB_NAMES)}

_ACTION_VERBS = {
    PlayCard: VERB["play"],
    DiscardCard: VERB["discard"],
    UseAction: VERB["use_action"],
    UseOmni: VERB["use_omni"],
    Reap: VERB["reap"],
    Fight: VERB["fight"],
}

HOUSES: Tuple[House, ...] = tuple(House)
HOUSE_INDEX = {h: i for i, h in enumerate(HOUSES)}

FLANK_NAMES = ("left", "right", "both", "neither")

# In-play tuple layout (`InfoSet.inplay[entity_index]`).
IP_DAMAGE = 0
IP_POWER = 1
IP_ARMOR = 2
IP_ARMOR_USED = 3
IP_CAPTURED = 4
IP_STORED = 5
IP_POWER_COUNTERS = 6
IP_STUNNED = 7
IP_EXHAUSTED = 8
IP_FLANK = 9  # index into FLANK_NAMES
IP_BL_INDEX = 10  # battleline position (host's, for an upgrade/under card)
IP_BL_SIZE = 11
IP_N_UPGRADES = 12
IP_N_UNDER = 13
IP_DAMAGE_PREVENTED = 14
IP_ARMOR_NEGATED = 15
IP_FOUGHT = 16
IP_EFF_HOUSE = 17  # House
IP_KEYWORDS = 18  # frozenset of keyword strings
IP_HAZARDOUS = 19
IP_ASSAULT = 20
IP_IS_CREATURE = 21

# Per-player scalar tuple layout (`InfoSet.players[0]` = me, `[1]` = them).
PL_AEMBER = 0
PL_KEYS = 1
PL_CHAINS = 2
PL_HAND = 3
PL_DECK = 4
PL_DISCARD = 5
PL_ARCHIVE = 6
PL_PURGED = 7
PL_CREATURES = 8
PL_ARTIFACTS = 9
PL_KEY_COST = 10
PL_HAND_REVEALED = 11  # my hand is currently revealed to the other side
PL_CARDS_PLAYED = 12  # cards played or discarded this turn
PL_CREATURES_PLAYED = 13
PL_DESTROYED_IN_FIGHT = 14  # this side's creatures destroyed in fights this turn
PL_FORGED_THIS_TURN = 15
PL_FORGED_RECENTLY = 16  # forged a key in either of the two previous turns (Key Hammer's "their previous turn")


class InfoSet:
    """What one viewer may know, as plain data. See the module docstring.

    `entity_iids[i]`/`entity_names[i]`/`zones[i]`/`flags[i]` describe entity
    `i`: indices 0-35 are the viewer's own decklist, 36-71 the opponent's,
    each in instance-id (deck-build, never shuffle) order. `inplay` maps an
    entity index to its in-play tuple (`IP_*`), for cards on the board,
    upgrades attached to a creature, and cards beneath another card.
    `options` is one `(verb, entity_index_or_-1, payload)` per option of
    the pending decision, in `Decision.options` order -- empty if the
    pending decision belongs to the other player."""

    __slots__ = (
        "viewer", "entity_iids", "entity_names", "zones", "flags", "inplay",
        "players", "turn_number", "is_my_turn", "i_went_first", "active_house",
        "my_houses", "their_houses", "effects", "match", "decision", "options",
        "index_of_iid",
    )

    def __init__(self):
        self.viewer: int = 0
        self.entity_iids: List[int] = []
        self.entity_names: List[str] = []
        self.zones: List[int] = []
        self.flags: List[int] = []
        self.inplay: Dict[int, tuple] = {}
        self.players: Tuple[tuple, tuple] = ((), ())
        self.turn_number: int = 0
        self.is_my_turn: bool = False
        self.i_went_first: bool = False
        self.active_house: Optional[House] = None
        self.my_houses: Tuple[House, ...] = ()
        self.their_houses: Tuple[House, ...] = ()
        # (variable, affects_me) per active duration effect.
        self.effects: Tuple[Tuple[str, bool], ...] = ()
        # (format, game_number, my_score, their_score, my_start_chains,
        # their_start_chains, deck_swapped)
        self.match: tuple = ("archon", 1, 0, 0, 0, 0, False)
        # (kind: DecisionKind, intent, affects, optional, min_n, max_n) or None
        self.decision: Optional[tuple] = None
        self.options: Tuple[tuple, ...] = ()
        self.index_of_iid: Dict[int, int] = {}


def _match_context(game, viewer: int, match) -> tuple:
    starting = game.config.starting_chains or {}
    if match is None:
        return ("archon", 1, 0, 0, starting.get(viewer, 0), starting.get(3 - viewer, 0), False)
    from .match import deck_label  # local: match.py imports game.py

    swapped = deck_label(game.config.decks[viewer - 1]) != deck_label(match.decks[viewer - 1])
    return (
        match.format,
        len(match.games) + 1,
        match.score.get(viewer, 0),
        match.score.get(3 - viewer, 0),
        starting.get(viewer, 0),
        starting.get(3 - viewer, 0),
        swapped,
    )


def _player_scalars(game, player, other_pid: int) -> tuple:
    turn = game.turn_number
    pid = player.id
    return (
        player.aember,
        player.keys,
        player.chains,
        len(player.hand),
        len(player.deck),
        len(player.discard),
        len(player.archive),
        len(player.purged),
        len(player.play_area.creatures),
        len(player.play_area.artifacts),
        player.get_key_forge_cost(game),
        other_pid in player.hand_revealed_to,
        player.cards_played_or_discarded_this_turn,
        player.creatures_played_this_turn,
        game.creatures_destroyed_in_fight_this_turn(pid),
        game.forged_key_on_turn(pid, turn),
        game.forged_key_on_turn(pid, turn - 1) or game.forged_key_on_turn(pid, turn - 2),
    )


def _option_entry(decision, option, index_of_iid: Dict[int, int]) -> tuple:
    verb = _ACTION_VERBS.get(type(option))
    if verb is not None:
        return (verb, index_of_iid.get(option.card.instance_id, -1), None)
    if isinstance(option, EndTurn):
        return (VERB["end_turn"], -1, None)
    if isinstance(option, Card):
        return (VERB["card"], index_of_iid.get(option.instance_id, -1), None)
    if isinstance(option, TriggerEffect):
        src = option.source_card
        return (VERB["trigger"], index_of_iid.get(src.instance_id, -1) if src is not None else -1, option.event)
    if isinstance(option, tuple) and len(option) >= 2 and isinstance(option[0], str) and isinstance(option[1], Card):
        # An ad hoc ORDER_EFFECTS item: ("extra", card, handler) is an
        # upgrade-granted Destroyed trigger on `card` (Game.destroy_cards).
        # The handler's __name__ says which ability it is -- stable across
        # processes, unlike the function object itself.
        tag = option[0]
        if len(option) > 2 and callable(option[2]):
            tag = f"{tag}:{getattr(option[2], '__name__', '?')}"
        return (VERB["trigger"], index_of_iid.get(option[1].instance_id, -1), tag)
    if isinstance(option, House):
        return (VERB["house"], -1, option)
    if isinstance(option, bool):
        return (VERB["bool"], -1, option)
    kind = decision.kind
    if kind == DecisionKind.CHOOSE_FLANK:
        return (VERB["flank"], -1, option)
    if kind == DecisionKind.CHOOSE_FIRST_PLAYER:
        return (VERB["first_player"], -1, option)
    if kind == DecisionKind.BID_CHAINS:
        if option == "pass":
            return (VERB["pass"], -1, None)
        return (VERB["bid"], -1, int(option))
    if isinstance(option, int):
        return (VERB["number"], -1, option)
    if isinstance(option, str):
        return (VERB["mode"], -1, option)
    # A new option type is a hard error, never a silent zero vector -- a
    # silently mis-encoded option would corrupt training data invisibly.
    raise ValueError(f"infoset: no encoding for option {option!r} of type {type(option).__name__} ({kind.name})")


def _decision_summary(decision) -> tuple:
    return (
        decision.kind,
        decision.intent,
        decision.affects,
        bool(decision.optional),
        decision.min_n,
        decision.max_n,
    )


def _inplay_tuple(game, card: Card, flank: int, bl_index: int, bl_size: int) -> tuple:
    to = card.type_object
    is_creature = isinstance(to, CreatureType)
    if is_creature:
        return (
            to.damage, game.get_power(card), game.get_armor(card), to.armor_used_this_turn,
            card.aember_captured, card.aember_stored, card.power_counters, card.stunned, card.Exhausted,
            flank, bl_index, bl_size, len(to.upgrades), len(card.under_cards), card.damage_prevented,
            card.armor_negated, card.fought_this_turn, game.get_effective_house(card), game.get_keywords(card),
            game.get_hazardous(card), game.get_assault(card), True,
        )
    return (
        0, 0, 0, 0, card.aember_captured, card.aember_stored, 0, card.stunned, card.Exhausted,
        flank, bl_index, bl_size, 0, len(card.under_cards), False, False, False,
        game.get_effective_house(card), frozenset(), 0, 0, False,
    )


def _attached_tuple(host_tuple: tuple, card: Card) -> tuple:
    """An upgrade on (or a card beneath) `host`: its own identity, the
    host's position -- so the encoder can say *where* it is without a
    second pointer."""
    return (
        0, 0, 0, 0, 0, 0, 0, False, card.Exhausted,
        host_tuple[IP_FLANK], host_tuple[IP_BL_INDEX], host_tuple[IP_BL_SIZE], 0, 0, False, False, False,
        card.house, frozenset(), 0, 0, False,
    )


def _flank_code(creatures: list, i: int, card: Card) -> int:
    n = len(creatures)
    if n == 1:
        return 2  # both
    if i == 0:
        return 0
    if i == n - 1:
        return 1
    return 2 if card.forced_flank else 3


def build_infoset(game, viewer: int, *, match=None) -> InfoSet:
    """The redacted extract for `viewer` at `game`'s current point. `match`,
    if given, supplies format/score/deck-swap context (a standalone game is
    a single Archon game)."""
    me = game.players[viewer]
    opp = game.players[3 - viewer]
    opp_pid = opp.id

    info = InfoSet()
    info.viewer = viewer
    entity_cards = me.all_cards + opp.all_cards
    iids = [c.instance_id for c in entity_cards]
    index_of_iid = {iid: i for i, iid in enumerate(iids)}
    info.entity_iids = iids
    info.entity_names = [c.name for c in entity_cards]
    info.index_of_iid = index_of_iid
    n = len(entity_cards)
    zones = [-1] * n
    flags = [0] * n
    inplay: Dict[int, tuple] = {}

    for i, c in enumerate(entity_cards):
        if c.owner == viewer:
            flags[i] = OWNER_IS_ME

    def place(card: Card, zone: int, known_to_them: bool, controller_is_me: bool) -> int:
        i = index_of_iid[card.instance_id]
        zones[i] = zone
        f = flags[i] | KNOWN_TO_ME
        if known_to_them:
            f |= KNOWN_TO_THEM
        if controller_is_me:
            f |= CONTROLLER_IS_ME
        flags[i] = f
        return i

    for side, player in ((0, me), (1, opp)):
        mine = side == 0
        prefix = "my_" if mine else "opp_"
        for c in player.discard.cards():
            place(c, ZONE[prefix + "discard"], True, mine)
        for c in player.purged.cards():
            place(c, ZONE[prefix + "purged"], True, mine)
        creatures = player.play_area.creatures
        bl_size = len(creatures)
        for bl, c in enumerate(creatures):
            i = place(c, ZONE[prefix + "creature"], True, c.controller == viewer)
            t = _inplay_tuple(game, c, _flank_code(creatures, bl, c), bl, bl_size)
            inplay[i] = t
            for u in c.type_object.upgrades:
                u_mine = u.controller == viewer
                j = place(u, ZONE[("my_" if u_mine else "opp_") + "upgrade_attached"], True, u_mine)
                inplay[j] = _attached_tuple(t, u)
            _place_under(c, t, viewer, place, inplay, prefix, mine)
        for c in player.play_area.artifacts:
            i = place(c, ZONE[prefix + "artifact"], True, c.controller == viewer)
            t = _inplay_tuple(game, c, 3, -1, 0)
            inplay[i] = t
            _place_under(c, t, viewer, place, inplay, prefix, mine)

    hand_known_to_them = opp_pid in me.hand_revealed_to
    for c in me.hand.cards():
        place(c, ZONE["my_hand"], hand_known_to_them, True)
    for c in me.archive.cards():
        place(c, ZONE["my_archive"], False, True)
    if viewer in opp.hand_revealed_to:
        for c in opp.hand.cards():
            place(c, ZONE["opp_hand"], True, False)

    my_deck_iids = {c.instance_id for c in me.deck.cards()}
    hidden_iids = None
    for i, z in enumerate(zones):
        if z != -1:
            continue
        # Not visible to the viewer anywhere. One of mine that's in my
        # own deck is known to be there by elimination; one that's in no
        # zone at all is mid-resolution (being played), which is public;
        # anything else -- the opponent's hand/deck/archive, a card
        # beneath one of their cards, or a card of mine they archived
        # from play -- is only "somewhere hidden on their side".
        if iids[i] in my_deck_iids:
            zones[i] = ZONE["my_deck_unordered"]
            flags[i] |= CONTROLLER_IS_ME
            continue
        if hidden_iids is None:
            hidden_iids = _hidden_iids(me, opp)
        if iids[i] in hidden_iids:
            zones[i] = ZONE["opp_unseen"]
        else:
            card = entity_cards[i]
            limbo_mine = card.controller == viewer
            zones[i] = ZONE["my_limbo" if limbo_mine else "opp_limbo"]
            flags[i] |= KNOWN_TO_ME | KNOWN_TO_THEM | (CONTROLLER_IS_ME if limbo_mine else 0)

    info.zones = zones
    info.flags = flags
    info.inplay = inplay
    info.players = (_player_scalars(game, me, opp_pid), _player_scalars(game, opp, viewer))
    info.turn_number = game.turn_number
    info.is_my_turn = (not game.is_over) and game.active_player_id == viewer
    info.i_went_first = game.first_player == viewer
    active = game.players[game.active_player_id].selected_house if not game.is_over else None
    info.active_house = active
    info.my_houses = tuple(game.player_houses(viewer))
    info.their_houses = tuple(game.player_houses(opp_pid))
    info.effects = tuple(
        (e.variable, e.player_affected == viewer) for e in game.active_effects.duration_effects
    )
    info.match = _match_context(game, viewer, match)

    d = game.pending_decision
    if d is not None:
        info.decision = _decision_summary(d)
        if d.player == viewer:
            info.options = tuple(_option_entry(d, o, index_of_iid) for o in d.options)
            for _verb, ptr, _payload in info.options:
                if ptr >= 0:
                    flags[ptr] |= LEGAL_OPTION
        src = d.source_card
        if isinstance(src, Card):
            j = index_of_iid.get(src.instance_id)
            if j is not None and zones[j] not in _UNLOCATED:
                flags[j] |= SOURCE_CARD
    return info


def _hidden_iids(me, opp) -> set:
    """Every card in a zone the viewer can't see into (the opponent's hand,
    deck and archive, and anything beneath a card) -- only ever used to
    tell "hidden" apart from "in no zone at all" (limbo, which is public),
    never to locate a hidden card."""
    out = {c.instance_id for c in opp.hand.cards()}
    out.update(c.instance_id for c in opp.deck.cards())
    out.update(c.instance_id for c in opp.archive.cards())
    for player in (me, opp):
        for c in player.play_area.creatures:
            out.update(u.instance_id for u in c.under_cards)
        for c in player.play_area.artifacts:
            out.update(u.instance_id for u in c.under_cards)
    return out


def _place_under(host: Card, host_tuple: tuple, viewer: int, place, inplay, prefix: str, host_mine: bool) -> None:
    """Cards facedown beneath `host` (Masterplan): the owner put each one
    there from their own hand and knows it; nobody else does -- the host's
    `IP_N_UNDER` count is public, the identity isn't, so an unseen one is
    left for the "not visible anywhere" pass (`opp_unseen`)."""
    for u in host.under_cards:
        if u.owner != viewer:
            continue
        j = place(u, ZONE[prefix + "under"], False, host_mine)
        inplay[j] = _attached_tuple(host_tuple, u)


def build_match_infoset(match, viewer: int) -> InfoSet:
    """For `BID_CHAINS`/`CHOOSE_FIRST_PLAYER`, which arrive between games
    (no live `Game` -- see `observation.build_match_observation`). Entities
    are the two decks of the game just finished, the viewer's first, all
    zoned `none`; the deck being bid on (if any) is flagged `SUBJECT`."""
    from .match import deck_label  # local: match.py imports game.py

    last_game = match.finished_games[-1]
    last_record = match.games[-1]
    me = last_game.players[viewer]
    opp = last_game.players[3 - viewer]
    info = InfoSet()
    info.viewer = viewer
    entity_cards = me.all_cards + opp.all_cards
    info.entity_iids = [c.instance_id for c in entity_cards]
    info.entity_names = [c.name for c in entity_cards]
    info.index_of_iid = {iid: i for i, iid in enumerate(info.entity_iids)}
    info.zones = [ZONE["none"]] * len(entity_cards)
    info.flags = [OWNER_IS_ME if c.owner == viewer else 0 for c in entity_cards]
    if match.bidding_deck is not None:
        label = deck_label(match.bidding_deck)
        for seat, deck in last_record.seat_decks.items():
            if deck_label(deck) == label:
                lo = 0 if seat == viewer else len(me.all_cards)
                for i in range(lo, lo + len(last_game.players[seat].all_cards)):
                    info.flags[i] |= SUBJECT
    zero = (0,) * (PL_FORGED_RECENTLY + 1)
    info.players = (
        (0, last_record.final_keys.get(viewer, 0), last_record.final_chains.get(viewer, 0)) + zero[3:],
        (0, last_record.final_keys.get(3 - viewer, 0), last_record.final_chains.get(3 - viewer, 0)) + zero[3:],
    )
    info.my_houses = tuple(last_game.player_houses(viewer))
    info.their_houses = tuple(last_game.player_houses(3 - viewer))
    info.match = (
        match.format, len(match.games) + 1, match.score.get(viewer, 0), match.score.get(3 - viewer, 0), 0, 0, False,
    )
    d = match.pending_decision
    if d is not None:
        info.decision = _decision_summary(d)
        if d.player == viewer:
            info.options = tuple(_option_entry(d, o, info.index_of_iid) for o in d.options)
    return info


def infoset_for(obj, viewer: int) -> InfoSet:
    """`build_infoset`/`build_match_infoset` for a `Game` or a `Match`,
    whichever `obj` is and whichever point it's at."""
    from .match import Match  # local: match.py imports game.py

    if isinstance(obj, Match):
        live = obj.current_game
        if live is not None and not live.is_over:
            return build_infoset(live, viewer, match=obj)
        return build_match_infoset(obj, viewer)
    return build_infoset(obj, viewer)

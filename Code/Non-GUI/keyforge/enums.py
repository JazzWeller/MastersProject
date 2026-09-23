"""Enumerations shared across the engine."""

from enum import Enum, auto


class House(Enum):
    BROBNAR = "Brobnar"
    DIS = "Dis"
    LOGOS = "Logos"
    MARS = "Mars"
    SANCTUM = "Sanctum"
    SHADOWS = "Shadows"
    UNTAMED = "Untamed"


class CardType(Enum):
    ACTION = "Action"
    ARTIFACT = "Artifact"
    CREATURE = "Creature"
    UPGRADE = "Upgrade"


class Trigger(Enum):
    PLAY = auto()
    AFTER_REAP = auto()
    AFTER_FIGHT = auto()
    BEFORE_FIGHT = auto()
    FIGHTING = auto()
    DESTROYED = auto()
    OMNI = auto()
    ACTION = auto()


class Zone(Enum):
    DECK = auto()
    HAND = auto()
    DISCARD = auto()
    ARCHIVE = auto()
    PURGED = auto()
    PLAY_CREATURE = auto()
    PLAY_ARTIFACT = auto()


class Flank(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class Affects(Enum):
    """Whose cards/board a decision's options draw from, relative to the
    player being asked -- independent of `DecisionIntent`, since e.g. a
    DAMAGE choice can target friendly, enemy, or either creatures depending
    on the card. Set on `Decision.affects` (Milestone B)."""

    FRIENDLY = "friendly"
    ENEMY = "enemy"
    ANY = "any"
    NONE = "none"  # options aren't cards at all (modes, numbers, bools, ...)


class DecisionIntent(Enum):
    """What a decision is FOR, independent of its `DecisionKind` and option
    type -- see Code/AGENT_INTERFACE_PLAN.md, Milestone B. The same
    `(DecisionKind, option type)` pair means different things depending only
    on which card asked (Ozmo's Heal-3-or-Stun target: both `CHOOSE_CARDS`
    over `List[Card]`); an agent conditioned on state and options alone
    can't tell them apart without this tag. Values were added as the
    141-call-site audit reached them, not invented up front, so this list
    mirrors what the card pool actually needs.

    Append-only: a new card needing a genuinely new kind of decision adds a
    value here rather than overloading an existing one, and an unrecognized
    intent reaching a dispatcher (`HeuristicBot._choose_cards`) is a hard
    error, never a silent default -- a silently mislabelled decision would
    corrupt training data invisibly.
    """

    # Targeting a creature/artifact for an effect that hurts or removes it.
    DESTROY = auto()
    DAMAGE = auto()
    STUN = auto()
    EXHAUST = auto()
    PURGE = auto()
    SACRIFICE = auto()
    SPARE = auto()  # choose which one is exempted from an otherwise-destroy-everyone effect

    # Targeting a creature/artifact for an effect that helps it.
    HEAL = auto()
    READY = auto()
    CAPTURE = auto()  # place/move captured Aember onto a card
    DRAIN = auto()  # move Aember off of a card, back to a player's pool

    # Zone moves and attachment.
    DISCARD = auto()
    ARCHIVE = auto()
    RETURN_TO_HAND = auto()
    SHUFFLE_IN = auto()
    ATTACH = auto()  # choose an upgrade's host, or what to attach something to
    PLAY = auto()  # choose a card to play from an unusual zone (discard, deck top)

    # Combat.
    FIGHT_TARGET = auto()

    # Control and borrowed use.
    TAKE_CONTROL = auto()
    USE_TARGET = auto()  # choose a creature/artifact to activate -- "as if it were yours" (Poltergeist) or an off-house/off-turn use of your own (Dominator Bauble)

    # Reading, copying, or redirecting, without moving/damaging/destroying.
    REVEAL = auto()
    COPY = auto()
    REDIRECT = auto()
    SWAP = auto()
    MODIFY = auto()  # target for an ongoing effect with no more specific intent (Spectral Tunneler's flank-and-draw)

    # Structural / bookkeeping decisions.
    ORDER = auto()  # ORDER_EFFECTS: which of several triggers/effects resolves first
    MODE = auto()  # CHOOSE_MODE: which of a card's printed modes
    NUMBER = auto()  # CHOOSE_NUMBER: pick a numeric value

    # YES_NO framings -- distinct because they carry different stakes for an
    # agent (a cost changes what you can afford next; a trigger doesn't).
    OPTIONAL_TRIGGER = auto()  # "may" resolve an optional ability
    OPTIONAL_COST = auto()  # pay an optional cost for a bigger effect


class PrivilegeLevel(Enum):
    """What capability object an agent is handed by the driver (Agent
    Interface Plan, Milestone G) -- enforced by construction: an agent
    given `OBSERVATION`'s capability object has no `fork`/`fork_
    determinized` method to call at all, not merely a policy asking it not
    to."""

    OBSERVATION = "observation"  # default: observations only
    SEARCH = "search"  # adds fork_determinized, fork_many, run_branches, apply, run_until
    PRIVILEGED = "privileged"  # adds fork, full_state_observation


class Resample(Enum):
    """What `Game.fork_determinized`/`fork_many` resample, relative to a
    viewer -- Agent Interface Plan, Milestone D. Named so the
    hidden-information diagnostic (Milestone J) can separate "the cost of
    not knowing my own draws" from "the cost of not knowing their hand"."""

    OWN_DECK = "own_deck"  # only the viewer's own remaining deck order (chance only)
    OPPONENT_PRIVATE = "opponent_private"  # the opponent's hand + archive + deck, redistributed among themselves
    ALL = "all"  # both of the above


class DecisionKind(Enum):
    MULLIGAN = auto()
    CHOOSE_HOUSE = auto()
    TAKE_ARCHIVE = auto()
    CHOOSE_ACTION = auto()
    CHOOSE_CARDS = auto()
    CHOOSE_FLANK = auto()
    CHOOSE_HOUSE_FOR_EFFECT = auto()
    ORDER_EFFECTS = auto()
    YES_NO = auto()
    BID_CHAINS = auto()  # Adaptive match format: "pass" or an integer chain bid
    CHOOSE_FIRST_PLAYER = auto()  # match formats: "first" or "second"
    CHOOSE_NUMBER = auto()  # Dance of Doom: pick an integer from a list of choices
    CHOOSE_MODE = auto()  # Knowledge is Power: pick one of several named modes


# The three decision kinds where the generator stack is only `Game._run`'s
# turn loop and one step of `Game._take_turn` -- Step 1 (forge a key) and
# any card resolution have already fully unwound by the time one of these
# is yielded, so `Game.copy()` (Milestone E2) is valid here and only here
# (see its own docstring for why: not at any of these means a suspended,
# unreconstructable generator frame). The single canonical definition --
# sim/simulate.py's own invariant-check boundaries and Game.copy()'s
# precondition must never quietly drift apart.
BOUNDARY_KINDS = (DecisionKind.CHOOSE_ACTION, DecisionKind.CHOOSE_HOUSE, DecisionKind.TAKE_ARCHIVE)

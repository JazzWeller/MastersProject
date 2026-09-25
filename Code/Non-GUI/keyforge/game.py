"""The rules engine: setup, turn order, legal actions, and resolution."""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import types
from typing import Any, Dict, List, Optional, Tuple

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import ActionType, ArtifactType, Card, CreatureType, TypeObject, UpgradeType
from .cards.decks import build_deck
from .config import GameConfig
from .decision import Decision
from .effects import steps
from .effects.effect_object import ActiveEffectList, DurationEffect, InsteadEffect, ModifierEffect, TriggerEffect, resolve_cleanup
from .enums import Affects, BOUNDARY_KINDS, CardType, DecisionIntent, DecisionKind, House, Resample
from .keyed_random import derive_rng, portable_choice, portable_shuffle
from .log import GameLog
from .player import Player
from .replay import decode_choice, encode_choice, replay
from .state_hash import compute_state_hash
from .view import build_view
from .zones import Deck


@dataclasses.dataclass(frozen=True)
class _PlayGate:
    """The player-wide facts behind `why_not_playable`/`_legal_actions`'s
    hand-card checks, computed once per `_legal_actions` call instead of
    once per card -- see `Game._build_play_gate`."""

    can_play_cards: bool
    house: Optional[House]
    non_logos_allowance: int
    extra_house_playable: Dict[House, int]
    card_played_limit: Optional[int]
    can_play_creatures: bool
    can_play_actions: bool
    any_creature_in_play: bool
    artifact_toll: Optional[tuple]


# --------------------------------------------- Game.copy() (Milestone E2) ----
# Hand-written, not `copy.deepcopy`: deepcopy treats functions as atomic
# (shared by reference, never copied), so a deepcopied game's
# DurationEffect.value/TriggerEffect.handler/etc. would still close over
# the ORIGINAL game's Card objects wherever that closure captured one --
# and empirically (across hundreds of games' worth of active effects, every
# house), a Card is the ONLY kind of object any such closure ever captures
# in this codebase. `_rebind_closure` fixes exactly that, generically,
# rather than requiring every card file to avoid closures entirely (the
# six `_end_of_turn_cleanups` sites still moved to plain data, since a
# *pending* cleanup has no `game.active_effects` entry of its own to visit
# and rebind post hoc).


def _make_cell(value):
    return (lambda: value).__closure__[0]


def _rebind_closure(fn, card_remap: Dict[int, Card]):
    """`fn`, or a function behaviorally identical to it, except any `Card`
    in its closure that's a key in `card_remap` (by `id()` of the OLD card)
    is replaced with the corresponding new one. Plain data (True, an int, a
    House, ...) and closures that capture nothing pass through unchanged."""
    if fn is None or not isinstance(fn, types.FunctionType) or fn.__closure__ is None:
        return fn
    new_cells = []
    changed = False
    for cell in fn.__closure__:
        try:
            value = cell.cell_contents
        except ValueError:  # pragma: no cover -- an empty cell (a function referencing itself); never seen here
            new_cells.append(cell)
            continue
        if isinstance(value, Card) and id(value) in card_remap:
            value = card_remap[id(value)]
            changed = True
        new_cells.append(_make_cell(value))
    if not changed:
        return fn
    new_fn = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, tuple(new_cells))
    new_fn.__kwdefaults__ = fn.__kwdefaults__
    return new_fn


def _copy_type_object(old: TypeObject) -> TypeObject:
    """A fresh `TypeObject`, `card` left `None` -- the caller sets it once
    the new `Card` it belongs to exists (a `Card` and its `type_object`
    are constructed together, but that back-reference is only meaningful
    once both new objects exist)."""
    if isinstance(old, CreatureType):
        new = CreatureType.__new__(CreatureType)
        new.base_power = old.base_power
        new.base_armor = old.base_armor
        new.damage = old.damage
        new.armor_used_this_turn = old.armor_used_this_turn
        new.upgrades = list(old.upgrades)  # Card refs fixed up in _relink_card
    elif isinstance(old, UpgradeType):
        new = UpgradeType.__new__(UpgradeType)
        new.host = old.host  # Card ref fixed up in _relink_card
    elif isinstance(old, ArtifactType):
        new = ArtifactType.__new__(ArtifactType)
    else:
        new = ActionType.__new__(ActionType)
    new.card = None
    return new


def _copy_card(old: Card) -> Card:
    """A fresh `Card` with the same instance_id and every current field
    value, sharing `card_def` (immutable, printed-card data). Cross-card
    references (`type_object.upgrades`/`.host`, `purged_by`,
    `redirect_fight_damage_to`, `under_cards`) still point at the OLD
    game's cards until `_relink_card` runs -- deferred because the card
    they need to point at instead might not exist yet during this pass."""
    new = Card.__new__(Card)
    for slot in Card.__slots__:
        setattr(new, slot, getattr(old, slot))
    new.type_object = _copy_type_object(old.type_object)
    new.type_object.card = new
    new.extra_triggers = {k: list(v) for k, v in old.extra_triggers.items()}
    new.under_cards = list(old.under_cards)
    return new


def _relink_card(new_card: Card, card_remap: Dict[int, Card]) -> None:
    """Fixes up `new_card`'s references to OTHER cards (set to the old
    game's cards by `_copy_card`) to point at their new-game counterparts.
    Run only after every card has been copied, so every reference this
    looks up is guaranteed to already be in `card_remap`."""
    new_card.purged_by = card_remap.get(id(new_card.purged_by), new_card.purged_by) if new_card.purged_by is not None else None
    if new_card.redirect_fight_damage_to is not None:
        new_card.redirect_fight_damage_to = card_remap[id(new_card.redirect_fight_damage_to)]
    new_card.under_cards = [card_remap[id(c)] for c in new_card.under_cards]
    if isinstance(new_card.type_object, CreatureType):
        new_card.type_object.upgrades = [card_remap[id(c)] for c in new_card.type_object.upgrades]
    elif isinstance(new_card.type_object, UpgradeType) and new_card.type_object.host is not None:
        new_card.type_object.host = card_remap[id(new_card.type_object.host)]


def _copy_zone_cards(cards: List[Card], card_remap: Dict[int, Card]) -> List[Card]:
    return [card_remap[id(c)] for c in cards]


def _copy_player(old, card_remap: Dict[int, Card]) -> "Player":
    new = Player.__new__(Player)
    new.__dict__.update(old.__dict__)
    new.all_cards = _copy_zone_cards(old.all_cards, card_remap)
    new.deck = Deck(_copy_zone_cards(old.deck.cards(), card_remap))
    new.hand = type(old.hand)()
    for c in _copy_zone_cards(old.hand.cards(), card_remap):
        new.hand.add(c)
    new.discard = type(old.discard)()
    for c in _copy_zone_cards(old.discard.cards(), card_remap):
        new.discard.push(c)
    new.archive = type(old.archive)()
    for c in _copy_zone_cards(old.archive.cards(), card_remap):
        new.archive.add(c)
    new.purged = type(old.purged)()
    for c in _copy_zone_cards(old.purged.cards(), card_remap):
        new.purged.add(c)
    new.play_area = type(old.play_area)()
    new.play_area.creatures = _copy_zone_cards(old.play_area.creatures, card_remap)
    new.play_area.artifacts = _copy_zone_cards(old.play_area.artifacts, card_remap)
    new.CardsPlayed = dict(old.CardsPlayed)
    new.used_this_turn = dict(old.used_this_turn)
    new.ExtraHousePlayable = dict(old.ExtraHousePlayable)
    new.hand_revealed_to = set(old.hand_revealed_to)
    return new


def _copy_active_effects(old: ActiveEffectList, card_remap: Dict[int, Card]) -> ActiveEffectList:
    def remapped_source(e):
        return card_remap[id(e.source_card)] if e.source_card is not None else None

    new = ActiveEffectList()
    for e in old.duration_effects:
        new.add(DurationEffect(
            source_card=remapped_source(e), controller=e.controller, remaining_duration=e.remaining_duration,
            player_affected=e.player_affected, variable=e.variable, op=e.op,
            value=_rebind_closure(e.value, card_remap) if callable(e.value) else e.value,
            conditional=_rebind_closure(e.conditional, card_remap),
        ))
    for e in old.trigger_effects:
        new.add(TriggerEffect(
            source_card=remapped_source(e), controller=e.controller, event=e.event,
            handler=_rebind_closure(e.handler, card_remap), remaining_duration=e.remaining_duration,
        ))
    for e in old.instead_effects:
        new.add(InsteadEffect(
            source_card=remapped_source(e), controller=e.controller, kind=e.kind,
            handler=_rebind_closure(e.handler, card_remap),
        ))
    for e in old.modifier_effects:
        new.add(ModifierEffect(
            source_card=remapped_source(e), controller=e.controller, kind=e.kind,
            handler=_rebind_closure(e.handler, card_remap),
        ))
    return new


class Game:
    def __init__(self, config: GameConfig):
        self.config = config
        # Per-game, assigned in deck-build order (see `_setup`): same seed
        # -> same instance ids, in any process, in any fork (Milestone A).
        # Replaces the old module-level counter in cards/card.py, whose
        # value depended on how many games (and ad hoc test cards) had
        # already run in this process.
        self._instance_counter = itertools.count(1)
        # Per-`(player, event kind)` draw counters backing `event_rng` --
        # see keyforge/keyed_random.py. This dict *is* "the RNG state" for
        # `state_hash`: every draw is a pure function of it plus the seed.
        self._rng_counters: dict = {}
        self.players = {1: Player(1), 2: Player(2)}
        self.active_effects = ActiveEffectList()
        self.log = GameLog()
        self.choice_log: list = []
        # The same choices as option indices (see keyforge/replay.py): with
        # the config's seed, this reproduces the game exactly.
        self.choice_record: list = []
        self.is_over = False
        self.result = None
        self.active_player_id = 1
        self.turn_number = 0
        self._first_player = 1
        # Harland Mindlock-style "until this leaves play" control changes:
        # source card instance_id -> [(controlled_card, original_pid), ...]
        self._temp_control: dict = {}
        # `(operation, instance_id)` pairs resolved via `effect_object.
        # resolve_cleanup` at the active player's next cleanup, for one-off
        # "remainder of the turn" effects that don't fit the DurationEffect
        # model (Spectral Tunneler). Data, not closures (Milestone E2:
        # `Game.copy()` can copy a plain tuple; it can't rebind a closure
        # over a specific Card the way replaying naturally would).
        self._end_of_turn_cleanups: List[Tuple[str, Optional[int]]] = []
        # Creatures that took redirected damage (Shadow Self) since the last
        # `check_destroyed`: that call destroy-checks them too, so an effect
        # that only checks the creature it chose still destroys the one that
        # actually took the damage (MRB 18.3 FAQ, Shadow Self / Special
        # Delivery).
        self._redirected_hits: List[Card] = []
        self._elusive_suppressed = False  # Sniffer: "for the remainder of the turn, each creature loses elusive"
        # Populated once in _setup: the decklist never changes mid-game, so
        # player_houses() doesn't need to recompute it on every call.
        self._player_houses_cache: Dict[int, List[House]] = {}
        # instance_id -> Card, for every card either player was ever dealt
        # (Milestone E2's cleanup-as-data resolvers and Game.copy() both
        # need to look a card up by id rather than close over it directly).
        # No card is ever created after _setup with a fresh instance_id
        # (see new_instance_id's own callers), so this never needs updating.
        self._cards_by_id: Dict[int, Card] = {}
        self._driver = self._run()
        self.pending_decision: Optional[Decision] = None
        self._prime()

    # ---------------------------------------------------------- driver ----

    def _prime(self):
        try:
            self.pending_decision = next(self._driver)
        except StopIteration:
            self.is_over = True
            self.pending_decision = None

    def submit(self, choice):
        if self.is_over or self.pending_decision is None:
            raise RuntimeError("No pending decision to submit a choice for")
        if not self.pending_decision.validate(choice):
            raise ValueError(f"Invalid choice {choice!r} for decision {self.pending_decision}")
        self.choice_log.append(choice)
        self.choice_record.append(encode_choice(self.pending_decision, choice))
        try:
            self.pending_decision = self._driver.send(choice)
        except StopIteration:
            self.is_over = True
            self.pending_decision = None

    def submit_index(self, encoded) -> None:
        """Fast path for a TRUSTED, already-encoded choice -- an int (a list
        of ints for CHOOSE_CARDS/ORDER_EFFECTS), `choice_record`'s own shape
        -- from `replay()`, `apply()`, or the driver's forced-decision
        step (Milestone L). Skips `Decision.validate`'s and `encode_choice`'s
        O(len(options)) scans: the caller already knows the index is legal,
        since it either came from a prior `submit`/`validate` (a stored
        record) or from a `Decision` the engine itself built with exactly
        one option. Skips `choice_log` too (kept only by `submit`, for a
        caller that wants live `Card` references in the log -- not needed
        to replay, since `choice_record` alone is sufficient).

        Never call this with an untrusted or hand-constructed index --
        there is no check here that `encoded` actually names a legal
        option. Human and GUI input must keep using `submit`."""
        if self.is_over or self.pending_decision is None:
            raise RuntimeError("No pending decision to submit a choice for")
        choice = decode_choice(self.pending_decision, encoded)
        # A copy, not `encoded` itself, matching `encode_choice`'s own
        # contract of always handing `choice_record` a fresh list -- `encoded`
        # here is usually an element of the CALLER's own record/list, and
        # `choice_record` must never alias it.
        self.choice_record.append(list(encoded) if isinstance(encoded, list) else encoded)
        try:
            self.pending_decision = self._driver.send(choice)
        except StopIteration:
            self.is_over = True
            self.pending_decision = None

    def view_for(self, pid: int):
        return build_view(self, pid)

    def new_instance_id(self) -> int:
        """The next id in this game's own sequence -- every `Card` built for
        this game (deck-build in `_setup`; a future `GameConfig.setup_script`,
        Milestone D) should get its `instance_id` from here, never from
        `cards/card.py`'s process-wide fallback counter, so that the same
        seed produces the same ids in any process or fork."""
        return next(self._instance_counter)

    def card_by_id(self, instance_id: int) -> Card:
        """The `Card` with this instance id, wherever it currently sits."""
        return self._cards_by_id[instance_id]

    def event_rng(self, kind: str, player: Optional[int] = None):
        """A fresh, independent `random.Random` for one instance of
        `(player, kind)` -- see keyforge/keyed_random.py for why randomness
        is keyed per event instead of drawn from one shared stream."""
        key = (player, kind)
        counter = self._rng_counters.get(key, 0)
        self._rng_counters[key] = counter + 1
        return derive_rng(self.config.seed, player, kind, counter)

    def state_hash(self) -> str:
        """A canonical hash over everything that affects future play --
        zones and their order, per-card state, players, active effects,
        pending decision, log, and keyed-RNG counters. Two independently
        replayed games that reached this point via the same choices hash
        equal; see keyforge/state_hash.py for exactly what's covered and
        two known, accepted approximations around closures."""
        return compute_state_hash(self)

    def outcome_for(self, pid: int) -> int:
        """+1 / 0 / -1 from `pid`'s perspective. 0 covers both an explicit
        draw (the `max_turns` limit) and any other winner-less result --
        callers that need to tell those apart should read `self.result`
        directly; this is the terminal training/eval signal, which doesn't
        care which kind of non-win it was."""
        if self.result is None:
            raise RuntimeError("Game.outcome_for() called before the game is over")
        winner = self.result.get("winner")
        if winner is None:
            return 0
        return 1 if winner == pid else -1

    # ---------------------------------------------------------- forking ----
    # Agent Interface Plan, Milestone D. `fork`/`fork_determinized` are
    # "branch-and-hold": they return a `Game` in the caller's own process,
    # which only this (replay) backend and Milestone E's snapshot-copy
    # backend can provide. Search code that should also run under
    # Milestone E's process-fork backend should be written against
    # `keyforge.branching.run_branches` ("branch-and-run") instead.
    #
    # Tree-node guidance: a search tree should store the PATH of choice
    # indices from the root (a list of encoded choices, `Game.choice_
    # record`'s own shape), never a live `Game`. A node is regenerated on
    # demand by `root.fork()` (or `fork_determinized()`) followed by
    # `Game.apply(path)`. That bounds memory to the tree's own shape, and
    # keeps the tree itself serializable.

    def fork(self) -> "Game":
        """An independent `Game` at this exact point, carrying the TRUE
        hidden state and RNG future: replaying the same config and
        `choice_record` reproduces a game byte-for-byte (Milestone A), so
        this is just that. PRIVILEGED -- a normal agent handed this could
        search its own future draws and the opponent's true hand."""
        return replay(self.config, self.choice_record)

    def fork_determinized(self, viewer: int, rng: random.Random, resample: Resample = Resample.ALL) -> "Game":
        """`fork()`, then resamples what `viewer` doesn't know, then
        reseeds the fork's own future randomness from `rng`. Without the
        reseed, every future `event_rng` draw in the fork would exactly
        match the true game's -- Milestone A's keyed randomness is a pure
        function of `(seed, player, kind, counter)`, and a fork inherits
        both the seed and the counters -- which is exactly the "future
        leakage" this class's acceptance test checks for.

        `resample` (keyforge/enums.py):
        - `OWN_DECK`: only `viewer`'s own remaining deck order.
        - `OPPONENT_PRIVATE`: the opponent's hand + archive + deck,
          redistributed among themselves. Any card currently revealed to
          `viewer` (`Player.hand_revealed_to`) is left exactly where it
          is -- today's coarse version of "card location knowledge": a
          card's hidden position is known to `viewer` only while an active
          reveal covers it, matching `keyforge/view.py`'s own visibility
          rule for the non-privileged `PlayerView`.
        - `ALL`: both.
        """
        fork = self.fork()
        opponent = 3 - viewer
        if resample in (Resample.OWN_DECK, Resample.ALL):
            fork._resample_own_deck(viewer, rng)
        if resample in (Resample.OPPONENT_PRIVATE, Resample.ALL):
            fork._resample_hidden_pool(opponent, viewer, rng)
        # A fresh, independent seed drawn from the caller's own `rng` --
        # portability doesn't matter here (this value is never itself
        # replayed as an engine-internal draw), only that it decorrelates
        # the fork's future from the true game's. `replay()` hands the
        # fork the SAME `GameConfig` object `self` uses (it doesn't copy
        # it), so this must replace it rather than mutate it in place --
        # otherwise `self.config.seed` changes too.
        fork.config = dataclasses.replace(fork.config, seed=rng.getrandbits(63))
        return fork

    def _resample_own_deck(self, pid: int, rng: random.Random) -> None:
        player = self.players[pid]
        cards = player.deck.cards()
        if len(cards) > 1:
            portable_shuffle(rng, cards)
        player.deck = Deck(cards)

    def _resample_hidden_pool(self, pid: int, viewer: int, rng: random.Random) -> None:
        """Redistributes `pid`'s hand + archive + deck among themselves,
        preserving each zone's count, except any hand card `viewer`
        currently has revealed to them (`hand_revealed_to`)."""
        player = self.players[pid]
        hand_cards = player.hand.cards()
        revealed = viewer in player.hand_revealed_to
        fixed = hand_cards if revealed else []
        pool = ([] if revealed else list(hand_cards)) + player.archive.cards() + player.deck.cards()
        portable_shuffle(rng, pool)
        n_hand = len(hand_cards) - len(fixed)
        n_archive = len(player.archive)
        new_hand, new_archive, new_deck = fixed + pool[:n_hand], pool[n_hand : n_hand + n_archive], pool[n_hand + n_archive :]
        for c in list(player.hand.cards()):
            player.hand.remove(c)
        for c in new_hand:
            player.hand.add(c)
        for c in list(player.archive.cards()):
            player.archive.remove(c)
        for c in new_archive:
            player.archive.add(c)
        player.deck = Deck(new_deck)

    def _branch_seed(self, index: int) -> int:
        """A deterministic per-branch seed derived from this game's own
        seed, so `fork_many` is reproducible without the caller having to
        invent a seed list."""
        return derive_rng(self.config.seed, None, "branch_seed", index).getrandbits(63)

    def fork_many(
        self, n: int, *, viewer: Optional[int] = None, resample: Resample = Resample.ALL, seeds: Optional[List[int]] = None
    ) -> List["Game"]:
        """n branches from this state, each with its own determinization
        and RNG seed -- the batch primitive search agents should be
        written against, so a faster backend (Milestone E) can serve it at
        a fraction of the cost with no agent-side change. Omitting
        `viewer` gives `n` exact (privileged) forks instead."""
        if seeds is None:
            seeds = [self._branch_seed(i) for i in range(n)]
        if viewer is None:
            return [self.fork() for _ in seeds]
        return [self.fork_determinized(viewer, random.Random(s), resample=resample) for s in seeds]

    def apply(self, choice_indices) -> "Game":
        """Advances this game along a recorded path of *encoded* choices
        (`Game.choice_record`'s own shape) -- typically called on a fresh
        fork. Returns `self`, for chaining onto `fork()`/`fork_determinized`.
        Uses `submit_index` (Milestone L): `choice_indices` is trusted,
        already-encoded input, same as `replay()`'s own record."""
        for encoded in choice_indices:
            if self.is_over or self.pending_decision is None:
                raise ValueError("Game.apply: more choices than the game has pending decisions for")
            self.submit_index(encoded)
        return self

    def copy(self) -> "Game":
        """An independent snapshot of this game (Milestone E2's snapshot-
        copy backend) -- exact and PRIVILEGED, the fast analog of `fork()`.
        See this module's `_copy_card`/`_copy_player`/`_copy_active_effects`/
        `_rebind_closure` for how; the short version: hand-written, not
        `copy.deepcopy` (which treats functions as atomic/shared by
        reference, so a deepcopied game's effect handlers would still close
        over THIS game's cards, not the copy's own).

        ONLY VALID AT A BOUNDARY DECISION (`enums.BOUNDARY_KINDS`) -- raises
        otherwise. Not-a-boundary means part of the "generator stack" (a
        card effect mid-resolution, Step 1's key-forge payment) is
        suspended state this can't reconstruct the way an OS-level fork
        (Milestone E1) or a full replay (`fork()`, valid at ANY decision)
        does. `is_over` is the other valid case (nothing left to resume)."""
        if not self.is_over and (self.pending_decision is None or self.pending_decision.kind not in BOUNDARY_KINDS):
            kind = self.pending_decision.kind.name if self.pending_decision is not None else None
            raise ValueError(
                f"Game.copy() is only valid at a boundary decision "
                f"({', '.join(k.name for k in BOUNDARY_KINDS)}) or a finished game, not {kind} -- use fork() instead"
            )

        new = Game.__new__(Game)
        new.config = self.config
        new._instance_counter = self._instance_counter  # never advances past _setup; safe to share
        new._rng_counters = dict(self._rng_counters)
        new.active_player_id = self.active_player_id
        new.turn_number = self.turn_number
        new._first_player = self._first_player
        new.is_over = self.is_over
        new.result = dict(self.result) if self.result is not None else None
        new._elusive_suppressed = self._elusive_suppressed
        new._player_houses_cache = dict(self._player_houses_cache)

        card_remap: Dict[int, Card] = {}
        new._cards_by_id = {}
        for iid, old_card in self._cards_by_id.items():
            new_card = _copy_card(old_card)
            card_remap[id(old_card)] = new_card
            new._cards_by_id[iid] = new_card
        for new_card in new._cards_by_id.values():
            _relink_card(new_card, card_remap)

        new.players = {pid: _copy_player(p, card_remap) for pid, p in self.players.items()}

        new.log = GameLog()
        new.log.events = list(self.log.events)
        for kind_, events in self.log.by_kind.items():
            new.log.by_kind[kind_] = list(events)

        new.active_effects = _copy_active_effects(self.active_effects, card_remap)
        new._end_of_turn_cleanups = list(self._end_of_turn_cleanups)
        new._redirected_hits = [card_remap.get(id(c), c) for c in self._redirected_hits]
        new._temp_control = {
            source_iid: [(card_remap.get(id(c), c), orig_pid) for c, orig_pid in entries]
            for source_iid, entries in self._temp_control.items()
        }

        new.choice_log = list(self.choice_log)
        new.choice_record = [list(e) if isinstance(e, list) else e for e in self.choice_record]

        if self.is_over:
            new._driver = iter(())
            new.pending_decision = None
        else:
            new.pending_decision = self.pending_decision  # _resume dispatches on .kind alone; replaced below
            new._driver = new._resume()
            try:
                new.pending_decision = next(new._driver)
            except StopIteration:
                new.is_over = True
                new.pending_decision = None
        return new

    def run_until(self, predicate, policy) -> "Game":
        """Plays this game forward with `policy(game, decision) -> choice`
        (a plain decoded choice, as `submit` expects) until `predicate
        (game)` holds or the game ends. Returns `self`. See
        `until_end_of_turn`/`until_player_decides`/`until_boundary` below
        for the conditions the plan names; `run_until(until_end_of_turn
        (game), policy)` is the turn-bounded search leaf."""
        while not self.is_over and not predicate(self):
            self.submit(policy(self, self.pending_decision))
        return self

    # -------------------------------------------------- choice helpers ----

    def choose_cards(
        self, player, prompt, options, min_n=0, max_n=1, *, source_card=None, intent=None, affects=None, optional=False
    ):
        options = list(options)
        if not options:
            return []
        if min_n == max_n and len(options) == min_n:
            return options
        choice = yield Decision(
            player, DecisionKind.CHOOSE_CARDS, prompt, options, min_n, max_n,
            source_card=source_card, intent=intent, affects=affects, optional=optional,
        )
        return choice

    def choose_house(self, player, prompt, houses):
        houses = list(houses)
        if not houses:
            return None
        if len(houses) == 1:
            return houses[0]
        choice = yield Decision(player, DecisionKind.CHOOSE_HOUSE_FOR_EFFECT, prompt, houses, 1, 1)
        return choice

    def yes_no(self, player, prompt, *, source_card=None, intent=None, optional=True):
        choice = yield Decision(
            player, DecisionKind.YES_NO, prompt, [True, False], 1, 1,
            source_card=source_card, intent=intent, affects=Affects.NONE, optional=optional,
        )
        return choice

    def order_effects(self, player, items, prompt="Choose the order these resolve", *, source_card=None):
        items = list(items)
        if len(items) <= 1:
            return items
        choice = yield Decision(
            player, DecisionKind.ORDER_EFFECTS, prompt, items, len(items), len(items),
            source_card=source_card, intent=DecisionIntent.ORDER, affects=Affects.NONE,
        )
        return choice

    def choose_number(self, player, prompt, numbers, *, source_card=None):
        numbers = list(numbers)
        if not numbers:
            return None
        if len(numbers) == 1:
            return numbers[0]
        choice = yield Decision(
            player, DecisionKind.CHOOSE_NUMBER, prompt, numbers, 1, 1,
            source_card=source_card, intent=DecisionIntent.NUMBER, affects=Affects.NONE,
        )
        return choice

    def choose_mode(self, player, prompt, modes, *, source_card=None):
        modes = list(modes)
        if not modes:
            return None
        if len(modes) == 1:
            return modes[0]
        choice = yield Decision(
            player, DecisionKind.CHOOSE_MODE, prompt, modes, 1, 1,
            source_card=source_card, intent=DecisionIntent.MODE, affects=Affects.NONE,
        )
        return choice

    # ------------------------------------------------------- get funcs ----

    def get_power(self, card: Card) -> int:
        bonus = card.power_counters
        if isinstance(card.type_object, CreatureType):
            bonus += sum(upg.card_def.power_bonus for upg in card.type_object.upgrades)
            for m in self.active_effects.modifiers_for("power"):
                bonus += m.handler(self, card)
        return max(0, card.type_object.base_power + bonus)

    def get_armor(self, card: Card) -> int:
        if card.armor_negated:
            return 0
        bonus = 0
        if isinstance(card.type_object, CreatureType):
            bonus += sum(upg.card_def.armor_bonus for upg in card.type_object.upgrades)
            for m in self.active_effects.modifiers_for("armor"):
                bonus += m.handler(self, card)
        return max(0, card.type_object.base_armor + bonus)

    def get_keywords(self, card: Card) -> frozenset:
        """Printed keywords plus anything granted by attached upgrades
        (Ring of Invisibility, Experimental Therapy) or another card's
        passive (Halacor): elusive, skirmish, taunt, poison, versatile.
        Hazardous has a value, not just a presence/absence, so it's
        computed separately by `get_hazardous`."""
        kw = set(card.card_def.keywords)
        if isinstance(card.type_object, CreatureType):
            for upg in card.type_object.upgrades:
                kw |= set(upg.card_def.grants_keywords)
            for m in self.active_effects.modifiers_for("keywords"):
                kw |= m.handler(self, card)
        if self._elusive_suppressed:
            kw.discard("elusive")
        return frozenset(kw)

    def get_fight_damage_bonus(self, attacker: Card, target: Card) -> int:
        """Extra damage `attacker` deals in this fight specifically (Valdr:
        +2 while attacking a flank creature) -- unlike `get_power`, this
        never applies when the card is on the *defending* side of a fight,
        so it's summed at the one call site in `_fight` rather than folded
        into `get_power`."""
        return sum(m.handler(self, attacker, target) for m in self.active_effects.modifiers_for("fight_damage"))

    def get_hazardous(self, card: Card) -> int:
        # Hazardous is summed across every source on the creature (per the
        # project's own Phase 1.1 plan), not capped at the single largest
        # one -- deck-building allows duplicates, so a creature could carry
        # more than one hazardous-granting upgrade (e.g. two Flame-Wreathed).
        # Includes the creature's own printed hazardous (Briar Grubbling).
        if not isinstance(card.type_object, CreatureType):
            return 0
        return card.card_def.hazardous + sum(upg.card_def.hazardous for upg in card.type_object.upgrades)

    def get_assault(self, card: Card) -> int:
        """The before-fight mirror of hazardous: an attacker with assault X
        deals X damage to the defender before the fight itself (Ancient
        Bear; Way of the Bear grants it via an upgrade)."""
        if not isinstance(card.type_object, CreatureType):
            return 0
        return card.card_def.assault + sum(upg.card_def.assault for upg in card.type_object.upgrades)

    def get_effective_house(self, card: Card) -> House:
        """The house a card counts as for playability/CanBeUsed purposes
        (Sneklifter's re-housing -- moot in this pool, since every deck
        already has all three houses, but implemented for correctness)."""
        return card.house_override or card.house

    def player_houses(self, pid: int) -> List[House]:
        # The decklist never changes mid-game, so this is computed once in
        # _setup rather than re-building the set/sort (and re-hashing every
        # House enum member) on each of the ~7.7k calls/game the profiler
        # measured (Milestone L). A copy, not the cached list itself, so a
        # caller can never mutate the cache.
        return list(self._player_houses_cache[pid])

    def find_play_area(self, card: Card):
        """Whichever player's `PlayArea` currently physically contains
        `card`, or None if it isn't in play. Looked up by membership, not by
        `card.controller` -- `use_artifact_ability` temporarily repoints
        `controller` to the borrowing player for "as if it were yours"
        effects (Poltergeist, Remote Access, Nexus), while the card itself
        never actually moves out of its real controller's play area."""
        for pid in (1, 2):
            area = self.players[pid].play_area
            if card in area.creatures or card in area.artifacts:
                return area
        return None

    def all_creatures(self, scope: str, source_card: Card) -> List[Card]:
        pid = source_card.controller
        if scope == "friendly":
            return list(self.players[pid].play_area.creatures)
        if scope == "enemy":
            return list(self.players[3 - pid].play_area.creatures)
        return list(self.players[1].play_area.creatures) + list(self.players[2].play_area.creatures)

    def legal_fight_targets(self, attacker: Card, exclude=frozenset()) -> List[Card]:
        """Enemy creatures `attacker` may fight: all of them, minus any
        creature that is the *neighbor* of a Taunt creature (unless that
        neighbor itself has Taunt) -- Taunt protects neighbors, not itself
        -- and minus any instance_id in `exclude` (One Stood Against Many:
        a different enemy creature each time). Empty if `attacker`'s
        controller can't fight at all (Foggify)."""
        if not self.players[attacker.controller].get_can_fight(self):
            return []
        opponent = self.players[3 - attacker.controller]
        creatures = list(opponent.play_area.creatures)
        protected = set()
        if not attacker.card_def.ignores_taunt:
            for c in creatures:
                if "taunt" not in self.get_keywords(c):
                    continue
                for neighbor in opponent.play_area.neighbors(c):
                    if "taunt" not in self.get_keywords(neighbor):
                        protected.add(neighbor.instance_id)
        result = [c for c in creatures if c.instance_id not in protected and c.instance_id not in exclude]
        if attacker.card_def.can_only_fight_stunned:
            result = [c for c in result if c.stunned]
        return result

    def _can_fight_off_house(self, card: Card) -> bool:
        """True if `card` may fight this turn despite not being of the
        active house (Brothers in Battle, Follow the Leader) -- checked as
        an OR alongside the normal `card.CanBeUsed` gate, for Fight only;
        Reap/Action/Omni are unaffected."""
        extra = self.players[card.controller].get_fight_permitted_extra(self)
        if not extra:
            return False
        return True in extra or self.get_effective_house(card) in extra

    def _can_use_off_house(self, card: Card) -> bool:
        """True if `card` may be used (reap/action/fight) this turn despite
        not being of the active house (Sigil of Brotherhood: 'you may use
        friendly Sanctum creatures'). Broader than `_can_fight_off_house`:
        this also covers Reap and Action, not just Fight."""
        extra = self.players[card.controller].get_use_permitted_extra(self)
        if not extra:
            return False
        return True in extra or self.get_effective_house(card) in extra

    # ------------------------------------------------------------ setup ----

    def _setup(self):
        p1_deck_name, p2_deck_name = self.config.decks
        for pid, deck_name in ((1, p1_deck_name), (2, p2_deck_name)):
            cards = build_deck(deck_name, pid)
            for card in cards:
                card.instance_id = self.new_instance_id()
            self.players[pid].all_cards = list(cards)
            self._player_houses_cache[pid] = sorted({c.house for c in cards}, key=lambda h: h.value)
            for c in cards:
                self._cards_by_id[c.instance_id] = c
            self.players[pid].deck = Deck(cards)
            self.players[pid].deck.shuffle(self.event_rng("deck_shuffle", pid))
            if self.config.starting_chains:
                self.players[pid].chains = self.config.starting_chains.get(pid, 0)
        if self.config.first_player in (1, 2):
            first = self.config.first_player
        else:
            first = portable_choice(self.event_rng("first_player"), [1, 2])
        self._first_player = first
        second = 3 - first
        self._draw_opening_hand(first, 7)
        self._draw_opening_hand(second, 6)
        for pid in (first, second):
            yield from self._maybe_mulligan(pid)

    def _draw_opening_hand(self, pid: int, base_size: int) -> None:
        """Opening hand size, reduced by starting chains the same way a
        normal draw step is (see `_draw_step`): 1 fewer card per 6 chains
        (rounded up), then shed one chain. Used by the Reversal/Adaptive
        match formats, where a bid-winner can start a game with chains."""
        player = self.players[pid]
        n = base_size
        if player.chains > 0:
            penalty = math.ceil(player.chains / 6)
            n = max(0, n - penalty)
            player.chains -= 1
            self.log.add("shed_chain", player=pid, fewer=penalty, total=player.chains, source="opening hand")
        steps.draw(self, player, n)

    def _maybe_mulligan(self, pid: int):
        player = self.players[pid]
        choice = yield Decision(pid, DecisionKind.MULLIGAN, "Mulligan your hand?", [True, False], 1, 1)
        # A mulligan decision is public either way -- unlike "mulligan"
        # below (fired only when it happens, and long-relied on by the GUI's
        # animation/text), this always fires so the observation layer
        # (Milestone C) can reconstruct the full public history, including
        # a decline that otherwise leaves no trace.
        self.log.add("mulligan_decision", player=pid, took=choice)
        if choice:
            n = len(player.hand)
            cards = player.hand.take_all()
            player.deck.shuffle_in(cards, self.event_rng("reshuffle", pid))
            steps.draw(self, player, max(0, n - 1))
            self.log.add("mulligan", player=pid)

    # --------------------------------------------- constructed positions ----

    def _take_named_card(self, pid: int, name: str) -> Card:
        """Removes and returns a card named `name` from wherever it
        currently sits among player `pid`'s own dealt cards (hand, deck,
        discard, archive, or purged, checked in that order) -- never a
        freshly conjured card, so `setup_script` can't violate the 36-card
        pool invariant `sim.simulate.check_invariants` checks for."""
        player = self.players[pid]
        for zone in (player.hand, player.deck, player.discard, player.archive, player.purged):
            for c in zone.cards():
                if c.name == name:
                    zone.remove(c)
                    return c
        raise ValueError(f"setup_script: no {name!r} among player {pid}'s own cards (already placed elsewhere?)")

    def _apply_setup_script(self) -> None:
        """Applies `self.config.setup_script` (Agent Interface Plan,
        Milestone D: "positions as replayable data") after normal setup and
        before the first decision. Each op is `(op, ...)`; unknown ops are a
        hard error rather than a silent no-op, since a typo here should
        never quietly produce a different position than intended."""
        for op in self.config.setup_script or []:
            kind = op[0]
            if kind == "put_creature":
                _, pid, name, flank = op
                card = self._take_named_card(pid, name)
                card.controller = pid
                self.players[pid].play_area.add_creature(card, flank)
                card.Exhausted = False
                if card.card_def.register_passive:
                    card.card_def.register_passive(self, card)
            elif kind == "put_artifact":
                _, pid, name = op
                card = self._take_named_card(pid, name)
                card.controller = pid
                self.players[pid].play_area.add_artifact(card)
                card.Exhausted = False
                if card.card_def.register_passive:
                    card.card_def.register_passive(self, card)
            elif kind == "hand_card":
                _, pid, name = op
                self.players[pid].hand.add(self._take_named_card(pid, name))
            elif kind == "discard_card":
                _, pid, name = op
                self.players[pid].discard.push(self._take_named_card(pid, name))
            elif kind == "archive_card":
                _, pid, name = op
                self.players[pid].archive.add(self._take_named_card(pid, name))
            elif kind == "set_aember":
                _, pid, amount = op
                self.players[pid].aember = amount
            elif kind == "set_keys":
                _, pid, amount = op
                self.players[pid].keys = amount
            elif kind == "set_chains":
                _, pid, amount = op
                self.players[pid].chains = amount
            elif kind == "damage":
                _, pid, name, amount = op
                card = next((c for c in self.players[pid].play_area.creatures if c.name == name), None)
                if card is None:
                    raise ValueError(f"setup_script: no {name!r} in player {pid}'s play area to damage")
                card.type_object.damage = amount
            else:
                raise ValueError(f"setup_script: unknown op {kind!r}")

    # -------------------------------------------------------- main loop ----

    def _run(self):
        yield from self._setup()
        self._apply_setup_script()
        self.active_player_id = self._first_player
        self.turn_number = 0
        while True:
            self.turn_number += 1
            over = yield from self._take_turn(self.active_player_id)
            if over:
                return
            if self.config.max_turns is not None and self.turn_number >= self.config.max_turns:
                self.result = {"winner": None, "turns": self.turn_number, "reason": "turn limit"}
                return
            self.active_player_id = 3 - self.active_player_id

    def _take_turn(self, pid: int):
        player = self.players[pid]
        self.active_player_id = pid
        self.log.add("turn_start", player=pid, turn=self.turn_number)

        # Step 1: forge a key. A routine "not enough Æmber yet" isn't logged
        # (that's normal, expected state almost every early turn); only a
        # notable block -- affordable but disabled (Miasma) -- is. Step 1
        # never itself pauses at a boundary decision, so `Game.copy()`
        # (Milestone E2) never needs to resume mid-Step-1 -- everything
        # from Step 2 on is factored into `_take_turn_from_house` so
        # `_resume` can re-enter there directly.
        cost = player.get_key_forge_cost(self)
        if player.aember >= cost:
            if player.get_can_key_forge(self):
                yield from self._pay_and_forge_key(pid, cost)
                if self.result is not None:
                    return True
            else:
                source = self._effect_source("CanKeyForge", pid)
                self.log.add("forge_skipped", player=pid, aember=player.aember, cost=cost, source=source)

        return (yield from self._take_turn_from_house(pid))

    def _take_turn_from_house(self, pid: int):
        """Steps 2-6 of `_take_turn`: choose a house, optional archive
        pickup, the main action loop, cleanup, draw. Split out so
        `Game._resume` (Milestone E2) can re-enter turn resolution here
        directly for a copy taken at a `CHOOSE_HOUSE` boundary, without
        redoing Step 1's already-resolved key-forge payment."""
        # Step 2: choose a house
        yield from self._choose_house_step(pid)
        return (yield from self._take_turn_from_archive(pid))

    def _take_turn_from_archive(self, pid: int):
        """Steps 3-6: as `_take_turn_from_house`, for a copy taken at a
        `TAKE_ARCHIVE` boundary."""
        player = self.players[pid]
        # Step 3: optional archive pickup
        if len(player.archive) > 0:
            take = yield Decision(
                pid, DecisionKind.TAKE_ARCHIVE, "Take your archive into your hand?", [True, False], 1, 1
            )
            # See _maybe_mulligan: public either way, always recorded so the
            # observation layer can see a decline too.
            self.log.add("archive_decision", player=pid, took=take)
            if take:
                for c in player.archive.take_all():
                    if c.archive_return_to_owner:
                        c.archive_return_to_owner = False
                        self.players[c.owner].hand.add(c)
                    else:
                        player.hand.add(c)
                self.log.add("take_archive", player=pid)

        return (yield from self._take_turn_action_loop(pid))

    def _take_turn_action_loop(self, pid: int):
        """Steps 4-6: as `_take_turn_from_house`, for a copy taken at a
        `CHOOSE_ACTION` boundary -- the common case (64% of all decisions,
        Milestone L's own baseline)."""
        player = self.players[pid]
        # Step 4: main action loop
        while True:
            options = self._legal_actions(pid)
            choice = yield Decision(pid, DecisionKind.CHOOSE_ACTION, "Choose an action", options, 1, 1)
            if isinstance(choice, EndTurn):
                break
            yield from self._resolve_action(pid, choice)
            if self.result is not None:  # a card played/used mid-turn forged a winning key
                return True

        # Step 5: cleanup
        yield from self._cleanup_turn(pid)
        # Step 6: draw
        self._draw_step(pid)
        return False

    def _resume(self):
        """Re-enters turn resolution at exactly the boundary decision
        `self.pending_decision` already holds (Milestone E2's snapshot-copy
        backend) -- the driver a `Game.copy()` gets in place of the live
        generator `deepcopy` can't duplicate. Dispatches on `pending_decision
        .kind` alone (CHOOSE_HOUSE / TAKE_ARCHIVE / CHOOSE_ACTION each map
        to exactly one of the `_take_turn_from_*` entry points above, and
        `Game.copy()` only ever runs at one of these three kinds), then
        continues the outer turn loop exactly like `_run` does."""
        pid = self.active_player_id
        kind = self.pending_decision.kind
        if kind == DecisionKind.CHOOSE_HOUSE:
            over = yield from self._take_turn_from_house(pid)
        elif kind == DecisionKind.TAKE_ARCHIVE:
            over = yield from self._take_turn_from_archive(pid)
        elif kind == DecisionKind.CHOOSE_ACTION:
            over = yield from self._take_turn_action_loop(pid)
        else:
            raise AssertionError(f"Game._resume: {kind.name} is not a boundary kind")
        if over:
            return
        while True:
            if self.config.max_turns is not None and self.turn_number >= self.config.max_turns:
                self.result = {"winner": None, "turns": self.turn_number, "reason": "turn limit"}
                return
            self.active_player_id = 3 - self.active_player_id
            self.turn_number += 1
            over = yield from self._take_turn(self.active_player_id)
            if over:
                return

    def _choose_house_step(self, pid: int):
        player = self.players[pid]
        forced = player.get_house_selection(self)
        cannot = player.get_cannot_choose_houses(self)
        if forced is not None and forced not in cannot:
            house = forced
            self.log.add("house_forced", player=pid, house=house.value, source=self._effect_source("HouseSelection", pid))
        else:
            options = [h for h in self.player_houses(pid) if h not in cannot]
            house = None
            if options:
                house = yield Decision(pid, DecisionKind.CHOOSE_HOUSE, "Choose your house", options, 1, 1)
        player.selected_house = house
        for c in player.play_area.all_cards():
            c.CanBeUsed = house is not None and (self.get_effective_house(c) == house or "versatile" in self.get_keywords(c))
        self.log.add("choose_house", player=pid, house=house.value if house else None)
        if house is not None:
            yield from self._fire_event("house_chosen", {"player": pid, "house": house})

    def reveal_hand(self, target_pid: int, viewer_pid: int, source=None) -> None:
        """Psychic Bug, Imperial Traitor, A Fair Game: lets `viewer_pid`
        see `target_pid`'s hand through `view.build_view`, until the end
        of the current turn."""
        self.players[target_pid].hand_revealed_to.add(viewer_pid)
        self.log.add(
            "reveal_hand", player=target_pid, viewer=viewer_pid,
            source=(source.name if source is not None else None),
        )

    def _cleanup_turn(self, pid: int):
        player = self.players[pid]
        yield from self._fire_event("end_of_turn", {"player": pid})
        for c in player.play_area.all_cards():
            c.Exhausted = False
            c.CanBeUsed = False
            c.fought_this_turn = False
        for p in self.players.values():
            for c in p.play_area.creatures:
                c.type_object.armor_used_this_turn = 0
            p.hand_revealed_to.clear()
        player.reset_turn_counters()
        self.active_effects.end_of_turn_tick()
        cleanups, self._end_of_turn_cleanups = self._end_of_turn_cleanups, []
        for operation, iid in cleanups:
            resolve_cleanup(self, operation, iid)

    def _draw_step(self, pid: int):
        player = self.players[pid]
        limit = player.get_draw_up_to_limit(self)
        base = max(0, limit - len(player.hand))
        if base == 0:
            return
        n = base + player.get_card_draw_modifier(self)
        if player.chains > 0:
            penalty = math.ceil(player.chains / 6)
            n -= penalty
            player.chains -= 1
            self.log.add("shed_chain", player=pid, fewer=penalty, total=player.chains, source="draw step")
        steps.draw(self, player, max(0, n), source="Hand refill")

    # ---------------------------------------------------------- forging ----

    def _stored_key_sources(self, player: Player) -> List[Card]:
        return [c for c in player.play_area.artifacts if c.card_def.spendable_for_keys and c.aember_stored > 0]

    def _pay_forge_cost(self, pid: int, cost: int):
        """Pays `cost` from pool Æmber, plus stored-on-artifact Æmber
        (Pocket Universe, Safe Place) if the pool falls short. Returns False
        without effect if `pid` can't afford it even with those sources."""
        player = self.players[pid]
        if player.aember >= cost:
            player.aember -= cost
        else:
            shortfall_amt = cost - player.aember
            sources = self._stored_key_sources(player)
            if sum(c.aember_stored for c in sources) < shortfall_amt:
                return False
            order = sources
            if len(sources) > 1:
                order = yield from self.order_effects(
                    pid, sources, "Choose which cards to spend stored Æmber from first, toward the shortfall"
                )
            player.aember = 0
            remaining = shortfall_amt
            for c in order:
                take = min(remaining, c.aember_stored)
                c.aember_stored -= take
                remaining -= take
                self.log.add("spend_stored_aember", player=pid, card=c.name, iid=c.instance_id, amount=take)
                if remaining <= 0:
                    break
        self._credit_forge_payment(pid, cost)
        return True

    def _credit_forge_payment(self, pid: int, cost: int) -> None:
        """The Sting: redirects Æmber `pid` spends forging a key to The
        Sting's controller instead of letting it vanish into the supply."""
        if cost <= 0:
            return
        redirects = [e for e in self.active_effects.duration_effects_for("RedirectsForgePayment", pid) if e.is_active(self)]
        if not redirects:
            return
        receiver = self.players[redirects[0].value]
        receiver.aember += cost
        self.log.add("gain", player=receiver.id, amount=cost, source="forge payment")

    def _pay_and_forge_key(self, pid: int, cost: int, source=None):
        """Pays `cost` and forges a key, checking the 3-keys win condition.
        Returns False (no effect) if `pid` can't afford it."""
        ok = yield from self._pay_forge_cost(pid, cost)
        if not ok:
            return False
        player = self.players[pid]
        player.keys += 1
        self.log.add(
            "forge_key", player=pid, keys=player.keys, cost=cost, turn=self.turn_number,
            source=(source.name if source is not None else None),
        )
        yield from self._fire_event("key_forged", {"player": pid})
        if player.keys >= 3:
            self.result = {"winner": pid, "turns": self.turn_number, "reason": "3 keys"}
        return True

    def forge_key(self, pid: int, cost_modifier: int = 0, source=None):
        """Forges a key outside the normal turn step, at a modified current
        cost (Key of Darkness). Does NOT check CanKeyForge: Miasma and The
        Sting say "skip your forge a key step", not "you cannot forge a
        key" -- that only disables Step 1 of the turn (`_take_turn`), and
        must not block a card effect that forges one directly. Logs a
        shortfall and returns False if forging is unaffordable."""
        player = self.players[pid]
        cost = max(0, player.get_key_forge_cost(self) + cost_modifier)
        ok = yield from self._pay_and_forge_key(pid, cost, source=source)
        if not ok and source is not None:
            steps.shortfall(
                self, source,
                f"can't forge a key: {{pos:{pid}}} Æmber is {player.aember}, and it costs {cost}",
                "Not enough Æmber",
            )
        return ok

    def forged_key_on_turn(self, pid: int, turn_number: int) -> bool:
        """Whether `pid` forged a key on their turn numbered `turn_number`
        (Key Hammer, Tendrils of Pain: 'forged a key on their previous turn')."""
        return any(
            e.data.get("player") == pid and e.data.get("turn") == turn_number
            for e in self.log.by_kind.get("forge_key", ())
        )

    def creatures_destroyed_in_fight_this_turn(self, controller_pid: int) -> int:
        """How many of `controller_pid`'s creatures were destroyed
        specifically by a fight's own damage exchange (not before-fight,
        hazardous, assault, or any card effect) so far this turn (The
        Warchest)."""
        return sum(
            1 for e in self.log.by_kind.get("destroyed_in_fight", ())
            if e.data.get("controller") == controller_pid and e.data.get("turn") == self.turn_number
        )

    def creatures_played_on_turn(self, pid: int, turn_number: int) -> int:
        """How many creatures `pid` played on their turn numbered
        `turn_number` (Lifeweb: "if your opponent played 3 or more
        creatures on their previous turn")."""
        return sum(
            1 for e in self.log.by_kind.get("play_card", ())
            if e.data.get("player") == pid and e.data.get("type") == "Creature" and e.data.get("turn") == turn_number
        )

    # ----------------------------------------------------- legal actions ----

    def _rule_of_six_ok(self, player: Player, name: str) -> bool:
        return player.CardsPlayed.get(name, 0) + player.used_this_turn.get(name, 0) < 6

    def _effect_source(self, variable: str, pid: int) -> Optional[str]:
        """Name of the card behind the active effect on `variable` for `pid`, if any."""
        for e in self.active_effects.duration_effects_for(variable, pid):
            if e.is_active(self) and e.source_card is not None:
                return e.source_card.name
        return None

    @property
    def first_player(self) -> int:
        return self._first_player

    def first_turn_limited(self, pid: int) -> bool:
        """True while `pid` is the first player on the game's first turn and has
        already used their one play-or-discard (spec: First Turn Rule)."""
        return (
            self.turn_number == 1
            and pid == self._first_player
            and self.players[pid].cards_played_or_discarded_this_turn >= 1
        )

    @staticmethod
    def _uses_play_allowance(gate: "_PlayGate", card: Card) -> bool:
        """True if playing `card` would spend an off-house allowance (Phase
        Shift: any non-Logos card; Witch of the Wilds: an Untamed card).
        Such an allowance also lifts the First Turn Rule for the card it
        permits (MRB 18.3 General FAQ, "First Turn Rule": playing Phase
        Shift lets the first player play another card)."""
        if card.house == gate.house:
            return False
        return (
            (card.house != House.LOGOS and gate.non_logos_allowance > 0)
            or gate.extra_house_playable.get(card.house, 0) > 0
        )

    def _has_play_allowance(self, player: Player) -> bool:
        return player.get_non_logos_cards_playable(self) > 0 or any(v > 0 for v in player.ExtraHousePlayable.values())

    def _build_play_gate(self, pid: int) -> "_PlayGate":
        """The player-wide facts `why_not_playable`/`_is_playable_fast` both
        check, computed once instead of once per hand card (Milestone L:
        ~35 calls/game with ~4-8 cards each -> ~4-8x fewer `duration_effects_
        for` scans, which is where most of the profiled cost was). No
        `_effect_source` lookups here -- those only matter for a rejection
        message, which `why_not_playable` builds lazily, on its own slow
        path, only for the one card a caller actually asked about."""
        player = self.players[pid]
        return _PlayGate(
            can_play_cards=player.get_can_play_cards(self),
            house=player.selected_house,
            non_logos_allowance=player.get_non_logos_cards_playable(self),
            extra_house_playable=player.ExtraHousePlayable,
            card_played_limit=player.get_card_played_limit(self),
            can_play_creatures=player.get_can_play_creatures(self),
            can_play_actions=player.get_can_play_actions(self),
            any_creature_in_play=bool(self.players[1].play_area.creatures or self.players[2].play_area.creatures),
            artifact_toll=player.get_artifact_play_toll(self),
        )

    def _is_playable_fast(self, gate: "_PlayGate", card: Card, player: "Player") -> bool:
        """Same conditions as `why_not_playable`'s reject checks (from
        `can_play_cards` on -- the three checks before that, "not in hand"/
        "not your turn"/"first turn limited", hold by construction for every
        card `_legal_actions` calls this on), but boolean-only, and reading
        `gate` instead of re-querying every effect-derived flag per card."""
        if not gate.can_play_cards:
            return False
        if not (
            card.house == gate.house
            or (card.house != House.LOGOS and gate.non_logos_allowance > 0)
            or gate.extra_house_playable.get(card.house, 0) > 0
        ):
            return False
        if gate.card_played_limit is not None and player.hand_plays_this_turn >= gate.card_played_limit:
            return False
        if card.type == CardType.CREATURE and not gate.can_play_creatures:
            return False
        if card.type == CardType.ACTION and not gate.can_play_actions:
            return False
        if card.type == CardType.UPGRADE and not gate.any_creature_in_play:
            return False
        if not self._rule_of_six_ok(player, card.name):
            return False
        if card.card_def.play_cost_aember > player.aember:
            return False
        if card.card_def.min_aember_to_play > player.aember:
            return False
        if card.type == CardType.ARTIFACT:
            toll = gate.artifact_toll
            if toll is not None and toll[0] > player.aember:
                return False
        return True

    def why_not_playable(self, pid: int, card: Card) -> Optional[str]:
        """Why `card` in `pid`'s hand can't be played right now, or None if it can.
        Uses exactly the checks `_legal_actions` uses, so the two can't disagree."""
        player = self.players[pid]
        if card not in player.hand.cards():
            return "Not in hand"
        if pid != self.active_player_id or player.selected_house is None:
            return "Not your turn"
        gate = self._build_play_gate(pid)
        if self.first_turn_limited(pid) and not self._uses_play_allowance(gate, card):
            return "First turn: you may play or discard only one card"
        if self._is_playable_fast(gate, card, player):
            return None
        if not gate.can_play_cards:
            source = self._effect_source("CanPlayCards", pid)
            return "You cannot play cards this turn" + (f" ({source})" if source else "")
        house = gate.house
        if not (
            card.house == house
            or (card.house != House.LOGOS and gate.non_logos_allowance > 0)
            or gate.extra_house_playable.get(card.house, 0) > 0
        ):
            return f"Not of your active house ({house.value})"
        if gate.card_played_limit is not None and player.hand_plays_this_turn >= gate.card_played_limit:
            source = self._effect_source("CardPlayedLimit", pid)
            return f"Card play limit reached ({gate.card_played_limit}{', ' + source if source else ''})"
        if card.type == CardType.CREATURE and not gate.can_play_creatures:
            source = self._effect_source("CanPlayCreatures", pid)
            return "You cannot play creatures this turn" + (f" ({source})" if source else "")
        if card.type == CardType.ACTION and not gate.can_play_actions:
            source = self._effect_source("CanPlayActions", pid)
            return "You cannot play actions this turn" + (f" ({source})" if source else "")
        if card.type == CardType.UPGRADE and not gate.any_creature_in_play:
            return "No creature to attach it to"
        if not self._rule_of_six_ok(player, card.name):
            return "Rule of six: already played or used 6 times this turn"
        if card.card_def.play_cost_aember > player.aember:
            return f"Costs {card.card_def.play_cost_aember} Æmber to play (you have {player.aember})"
        if card.card_def.min_aember_to_play > player.aember:
            return f"Needs {card.card_def.min_aember_to_play} Æmber in your pool to play (you have {player.aember})"
        if card.type == CardType.ARTIFACT:
            toll = gate.artifact_toll
            if toll is not None and toll[0] > player.aember:
                return f"Costs {toll[0]} Æmber to play (Customs Office) and you have {player.aember}"
        raise AssertionError(  # pragma: no cover -- would mean _is_playable_fast disagrees with the checks above it
            f"why_not_playable: no reason found for {card.name!r} despite _is_playable_fast rejecting it"
        )

    def why_not_usable(self, pid: int, card: Card) -> Optional[str]:
        """Why `card` (a creature or artifact already in `pid`'s play area)
        has no usable action at all right now (no Reap, Fight, Action, or
        Omni), or None if it has at least one. Uses exactly the checks
        `_legal_actions` uses, so the two can't disagree."""
        player = self.players[pid]
        cannot_use = player.get_cannot_use_cards(self)

        if card.type == CardType.ARTIFACT:
            toll = player.get_artifact_use_toll(self)
            can_afford_toll = toll is None or player.aember >= toll[0]
            has_action = card.CanBeUsed and card.card_def.on_action is not None
            has_omni = card.card_def.on_omni is not None
            if not cannot_use and not card.Exhausted and can_afford_toll and self._rule_of_six_ok(player, card.name) and (has_action or has_omni):
                return None
            if cannot_use:
                source = self._effect_source("CannotUseCards", pid)
                return "Your cards can't be used this turn" + (f" ({source})" if source else "")
            if card.Exhausted:
                return "Exhausted: already used or played this turn"
            if not can_afford_toll:
                return f"Costs {toll[0]} Æmber to use (you have {player.aember})"
            if not self._rule_of_six_ok(player, card.name):
                return "Rule of six: already used 6 times this turn"
            if not card.CanBeUsed and card.card_def.on_action is not None and not has_omni:
                return f"Not of your active house ({player.selected_house.value})" if player.selected_house else "No active house chosen"
            return "No usable ability right now"

        # creature
        only_fight = player.get_can_only_fight(self)
        usable = (
            not cannot_use and not card.Exhausted and self._rule_of_six_ok(player, card.name)
            and (card.card_def.use_restriction is None or card.card_def.use_restriction(self, card))
        )
        can_use_house = card.CanBeUsed or self._can_use_off_house(card)
        has_fight_target = bool(self.legal_fight_targets(card))
        can_fight = usable and (can_use_house or self._can_fight_off_house(card)) and has_fight_target
        can_reap_or_action = usable and can_use_house and not only_fight and (
            not card.card_def.cannot_reap or card.card_def.on_action is not None or card.granted_action is not None
        )
        can_omni = usable and not only_fight and card.card_def.on_omni is not None
        if can_reap_or_action or can_fight or can_omni:
            return None

        if cannot_use:
            source = self._effect_source("CannotUseCards", pid)
            return "Your cards can't be used this turn" + (f" ({source})" if source else "")
        if card.Exhausted:
            return "Exhausted: already used or played this turn"
        if not self._rule_of_six_ok(player, card.name):
            return "Rule of six: already used 6 times this turn"
        if card.card_def.use_restriction is not None and not card.card_def.use_restriction(self, card):
            return "This creature's ability can't be used right now"
        if not can_use_house and not self._can_fight_off_house(card):
            return f"Not of your active house ({player.selected_house.value})" if player.selected_house else "No active house chosen"
        if only_fight and not can_fight:
            source = self._effect_source("CanOnlyFight", pid)
            base = "You can only fight this turn" + (f" ({source})" if source else "")
            return base + ("" if has_fight_target else ", and this creature has no legal fight target")
        if card.card_def.cannot_reap and card.card_def.on_action is None and card.granted_action is None and not can_omni and not can_fight:
            return "Cannot reap, and no legal fight target right now"
        return "No usable ability right now"

    def _legal_actions(self, pid: int):
        player = self.players[pid]
        house = player.selected_house
        actions = []
        limited_now = self.first_turn_limited(pid)

        if not limited_now or self._has_play_allowance(player):
            gate = self._build_play_gate(pid)
            for card in player.hand.cards():
                if limited_now and not self._uses_play_allowance(gate, card):
                    continue
                if self._is_playable_fast(gate, card, player):
                    actions.append(PlayCard(card))
                if not limited_now and house is not None and card.house == house:
                    actions.append(DiscardCard(card))

        cannot_use = player.get_cannot_use_cards(self)
        only_fight = player.get_can_only_fight(self)
        for card in player.play_area.creatures:
            usable = (
                not cannot_use
                and not card.Exhausted
                and self._rule_of_six_ok(player, card.name)
                and (card.card_def.use_restriction is None or card.card_def.use_restriction(self, card))
            )
            can_use_house = card.CanBeUsed or self._can_use_off_house(card)
            if usable and can_use_house and not only_fight:
                if not card.card_def.cannot_reap:
                    actions.append(Reap(card))
                if card.card_def.on_action is not None or card.granted_action is not None:
                    actions.append(UseAction(card))
            if usable and (can_use_house or self._can_fight_off_house(card)) and self.legal_fight_targets(card):
                actions.append(Fight(card))
            if usable and not only_fight and card.card_def.on_omni is not None:
                actions.append(UseOmni(card))

        toll = player.get_artifact_use_toll(self)
        can_afford_toll = toll is None or player.aember >= toll[0]
        for card in player.play_area.artifacts:
            if not cannot_use and not card.Exhausted and can_afford_toll:
                if card.CanBeUsed and card.card_def.on_action is not None and self._rule_of_six_ok(player, card.name):
                    actions.append(UseAction(card))
                if card.card_def.on_omni is not None and self._rule_of_six_ok(player, card.name):
                    actions.append(UseOmni(card))

        actions.append(EndTurn())
        return actions

    def _resolve_action(self, pid: int, action):
        if isinstance(action, PlayCard):
            yield from self._play_card(pid, action.card)
        elif isinstance(action, DiscardCard):
            yield from self._discard_card(pid, action.card)
        elif isinstance(action, UseAction):
            yield from self._use_action(pid, action.card)
        elif isinstance(action, UseOmni):
            yield from self._use_omni(pid, action.card)
        elif isinstance(action, Reap):
            yield from self._reap(pid, action.card)
        elif isinstance(action, Fight):
            yield from self._fight(pid, action.card)

    def _discard_card(self, pid: int, card: Card):
        player = self.players[pid]
        player.hand.remove(card)
        player.discard.push(card)
        player.cards_played_or_discarded_this_turn += 1
        self.log.add("discard_from_hand", player=pid, card=card.name, iid=card.instance_id)
        yield from self._fire_event("card_discarded_from_hand", {"player": pid, "card": card})

    # ------------------------------------------------------- playing a card ----

    def _choose_flank(self, pid: int):
        player = self.players[pid]
        if len(player.play_area.creatures) == 0:
            return "left"
        choice = yield Decision(pid, DecisionKind.CHOOSE_FLANK, "Choose a flank", ["left", "right"], 1, 1)
        return choice

    def _card_is_somewhere(self, card: Card) -> bool:
        for p in self.players.values():
            if card in p.hand.cards() or card in p.discard.cards():
                return True
            if card in p.archive.cards() or card in p.purged.cards():
                return True
            if card in p.deck.cards():
                return True
            if card in p.play_area.creatures or card in p.play_area.artifacts:
                return True
        return False

    def _play_card(self, pid: int, card: Card, from_deck_top: bool = False):
        player = self.players[pid]
        if not player.get_can_play_cards(self):
            return False
        if not self._rule_of_six_ok(player, card.name):
            return False
        if card.type == CardType.CREATURE and not player.get_can_play_creatures(self):
            return False
        if card.type == CardType.ACTION and not player.get_can_play_actions(self):
            return False
        if card.type == CardType.UPGRADE:
            targets = list(self.players[1].play_area.creatures) + list(self.players[2].play_area.creatures)
            if not targets:
                return False
        if card.card_def.play_cost_aember > player.aember:
            return False
        if card.card_def.min_aember_to_play > player.aember:
            return False
        if card.type == CardType.ARTIFACT:
            toll = player.get_artifact_play_toll(self)
            if toll is not None and toll[0] > player.aember:
                return False

        house = player.selected_house
        if house is not None and card.house != house and card.house != House.LOGOS:
            # A house-scoped allowance (Witch of the Wilds: Untamed only) is
            # more specific than the blanket one (Phase Shift: any house), so
            # it's consumed first when both could cover this play.
            if player.ExtraHousePlayable.get(card.house, 0) > 0:
                player.ExtraHousePlayable[card.house] -= 1
            elif player.NonLogosCardsPlayable > 0:
                player.NonLogosCardsPlayable -= 1

        # Resolve the flank choice (which may require a player decision) before
        # touching any zone, so the card is never "in transit" between hand and
        # play area while a decision is pending.
        flank = None
        host = None
        if card.type == CardType.CREATURE:
            flank = yield from self._choose_flank(pid)
        elif card.type == CardType.UPGRADE:
            targets = list(self.players[1].play_area.creatures) + list(self.players[2].play_area.creatures)
            choice = yield from self.choose_cards(
                pid, f"Attach {card.name} to a creature", targets, 1, 1,
                source_card=card, intent=DecisionIntent.ATTACH, affects=Affects.ANY,
            )
            host = choice[0]

        if not from_deck_top:
            player.hand.remove(card)
            player.hand_plays_this_turn += 1
            player.cards_played_or_discarded_this_turn += 1
        player.CardsPlayed[card.name] = player.CardsPlayed.get(card.name, 0) + 1

        cdef = card.card_def
        card.controller = pid
        self.log.add(
            "play_card",
            player=pid,
            card=card.name,
            iid=card.instance_id,
            type=card.type.value,
            flank=flank,
            from_deck_top=from_deck_top,
            turn=self.turn_number,
        )
        if cdef.play_cost_aember:
            player.aember -= cdef.play_cost_aember
            self.log.add("pay", player=pid, amount=cdef.play_cost_aember, card=card.name, iid=card.instance_id)
        if card.type == CardType.ARTIFACT:
            toll = player.get_artifact_play_toll(self)
            if toll is not None:
                amount, receiver_pid = toll
                player.aember -= amount
                self.players[receiver_pid].aember += amount
                self.log.add("pay", player=pid, amount=amount, card=card.name, iid=card.instance_id, to_player=receiver_pid)

        if card.type == CardType.CREATURE:
            player.play_area.add_creature(card, flank)
            # Speed Sigil: the first creature played each turn enters ready.
            first_creature_this_turn = player.creatures_played_this_turn == 0
            player.creatures_played_this_turn += 1
            enters_ready = (first_creature_this_turn and player.get_first_creature_enters_ready(self)) or player.next_entry_ready
            if player.next_mars_creature_ready and card.house == House.MARS:
                enters_ready = True
            card.Exhausted = not enters_ready
            player.next_entry_ready = False
            player.next_mars_creature_ready = False
            # Step 2 of the turn only flags cards already in play; anything
            # entering play later must be flagged too, or a card readied on
            # entry (Silvertooth) could never be used that turn.
            card.CanBeUsed = player.selected_house is not None and (
                self.get_effective_house(card) == player.selected_house or "versatile" in self.get_keywords(card)
            )
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._fire_event("creature_entered_play", {"card": card})
            yield from self._play_resolution(card)
        elif card.type == CardType.ARTIFACT:
            player.play_area.add_artifact(card)
            card.Exhausted = not player.next_entry_ready
            player.next_entry_ready = False
            card.CanBeUsed = player.selected_house is not None and (
                self.get_effective_house(card) == player.selected_house or "versatile" in self.get_keywords(card)
            )
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._play_resolution(card)
        elif card.type == CardType.UPGRADE:
            card.type_object.host = host
            host.type_object.upgrades.append(card)
            # `card.controller` stays `pid` (the player who attached it,
            # set above) rather than snapping to the host's controller --
            # an upgrade can attach to an enemy creature (Collar of
            # Subordination), and its own register_passive needs to see
            # that the two differ to know it should transfer control.
            # Every other upgrade's granted ability reads `host.controller`
            # directly, never the upgrade card's own, so this is safe.
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if cdef.on_play is not None:
                yield from cdef.on_play(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._run_play_trigger_check(card)
        else:  # Action
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._play_resolution(card)
            if not self._card_is_somewhere(card):
                self.players[card.owner].discard.push(card)
        # Unlike "card_played" (which only ever notifies the SAME player's
        # own listeners, per `_run_play_trigger_check`'s controller filter),
        # this fires for every registered listener regardless of whose turn
        # it is -- for "each time your OPPONENT plays a creature" effects
        # (Teliga), which the controller-scoped event can't express.
        yield from self._fire_event("any_card_played", {"player": pid, "card": card})
        return True

    def _play_resolution(self, card: Card):
        """Resolve a played card's Play effect and its play-trigger check, in the
        tied order described in the plan (3.5)."""
        # Only other cards' triggers compete with the Play effect -- the same
        # set `_run_play_trigger_check` fires. A creature's own passive
        # registered as it entered (Tunk) isn't one, and with no Play effect
        # there is nothing to order.
        pre_existing = [
            t for t in self.active_effects.triggers_for("card_played")
            if t.controller == card.controller and t.source_card is not card
        ]
        if pre_existing and card.card_def.on_play is not None:
            order = yield from self.order_effects(
                card.controller, ["effect", "check"], f"{card.name}: choose what resolves first", source_card=card,
            )
        else:
            order = ["effect", "check"]
        for step_name in order:
            if step_name == "effect":
                if card.card_def.on_play is not None:
                    yield from card.card_def.on_play(self, card)
            else:
                yield from self._run_play_trigger_check(card)

    def _run_play_trigger_check(self, card: Card):
        event_player = card.controller
        triggers = [
            t
            for t in self.active_effects.triggers_for("card_played")
            if t.source_card is not card and t.controller == event_player
        ]
        if not triggers:
            return
        if len(triggers) > 1:
            ordered = yield from self.order_effects(
                event_player, triggers, f"Playing {card.name} triggered these: choose their order", source_card=card,
            )
        else:
            ordered = triggers
        for trig in ordered:
            yield from trig.handler(self, {"player": event_player, "card": card})

    def play_card_from_deck_top(self, player: Player, top_card: Card, ignore_house: bool = True, source=None):
        reason = self._effect_play_refusal(player, top_card)
        player.deck.remove(top_card)
        ok = yield from self._play_card(player.id, top_card, from_deck_top=True)
        if not ok:
            player.deck.put_on_top(top_card)
            if source is not None:
                steps.shortfall(
                    self, source,
                    f"can't play {top_card.name} from the top of {{pos:{player.id}}} deck: {reason or 'it cannot be played right now'}. It stays on top of the deck",
                    f"Can't play {top_card.name}",
                )

    def _effect_play_refusal(self, player: Player, card: Card) -> Optional[str]:
        """Why `_play_card` would refuse `card` when an effect plays it (the
        house and hand-size limits don't apply there), or None."""
        if not player.get_can_play_cards(self):
            src = self._effect_source("CanPlayCards", player.id)
            return "cards can't be played this turn" + (f" ({src})" if src else "")
        if not self._rule_of_six_ok(player, card.name):
            return f"{card.name} has already been played or used 6 times this turn (rule of six)"
        if card.type == CardType.CREATURE and not player.get_can_play_creatures(self):
            src = self._effect_source("CanPlayCreatures", player.id)
            return "creatures can't be played this turn" + (f" ({src})" if src else "")
        if card.type == CardType.ACTION and not player.get_can_play_actions(self):
            src = self._effect_source("CanPlayActions", player.id)
            return "actions can't be played this turn" + (f" ({src})" if src else "")
        if card.type == CardType.UPGRADE and not (self.players[1].play_area.creatures or self.players[2].play_area.creatures):
            return "it is an upgrade and there is no creature to attach it to"
        if card.card_def.play_cost_aember > player.aember:
            return f"it costs {card.card_def.play_cost_aember} Æmber to play and {{pos:{player.id}}} Æmber is {player.aember}"
        if card.card_def.min_aember_to_play > player.aember:
            return f"it needs {card.card_def.min_aember_to_play} Æmber in the pool to play and {{pos:{player.id}}} Æmber is {player.aember}"
        if card.type == CardType.ARTIFACT:
            toll = player.get_artifact_play_toll(self)
            if toll is not None and toll[0] > player.aember:
                return f"Customs Office charges {toll[0]}Æ to play an artifact and {{pos:{player.id}}} Æmber is {player.aember}"
        return None

    # --------------------------------------------------- reap/fight/use ----

    def _consume_stun_if_present(self, card: Card) -> bool:
        """Experimental Therapy, Ozmo: the next use of a stunned creature
        only exhausts it and clears the stun -- no effect resolves."""
        if not card.stunned:
            return False
        card.stunned = False
        card.Exhausted = True
        self.log.add("stun_consumed", card=card.name, iid=card.instance_id)
        return True

    def _pay_artifact_use_toll(self, pid: int, card: Card) -> None:
        """Tentacus: the opponent must pay its controller 1Æ to use an
        artifact. Only `_legal_actions` offers this when it's affordable,
        so this always succeeds when reached."""
        if card.type != CardType.ARTIFACT:
            return
        toll = self.players[pid].get_artifact_use_toll(self)
        if toll is None:
            return
        amount, receiver_pid = toll
        self.players[pid].aember -= amount
        self.players[receiver_pid].aember += amount
        self.log.add("pay", player=pid, amount=amount, card=card.name, iid=card.instance_id, to_player=receiver_pid)

    def _use_action(self, pid: int, card: Card):
        player = self.players[pid]
        if self._consume_stun_if_present(card):
            player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
            return
        self._pay_artifact_use_toll(pid, card)
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        self.log.add("use_action", player=pid, card=card.name, iid=card.instance_id)
        effect = card.card_def.on_action or card.granted_action
        yield from effect(self, card)
        if card.type == CardType.ARTIFACT:
            yield from self._fire_event("artifact_used", {"player": pid, "card": card})

    def _use_omni(self, pid: int, card: Card):
        player = self.players[pid]
        if self._consume_stun_if_present(card):
            player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
            return
        self._pay_artifact_use_toll(pid, card)
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        self.log.add("use_omni", player=pid, card=card.name, iid=card.instance_id)
        yield from card.card_def.on_omni(self, card)
        if card.type == CardType.ARTIFACT:
            yield from self._fire_event("artifact_used", {"player": pid, "card": card})

    def _reap(self, pid: int, card: Card):
        player = self.players[pid]
        if self._consume_stun_if_present(card):
            player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
            return
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        if player.get_reap_gain_becomes_steal(self):
            steps.steal(self, self.players[3 - pid], player, 1, source=card)
        else:
            player.aember += 1
        self.log.add("reap", player=pid, card=card.name, iid=card.instance_id)
        yield from self._fire_event("reap_resolved", {"card": card})
        cdef = card.card_def
        if cdef.on_reap is not None:
            yield from cdef.on_reap(self, card)
        for extra in list(card.extra_triggers.get("after_reap", [])):
            yield from extra(self, card)

    def _fight(self, pid: int, attacker: Card, exclude=frozenset()):
        """Returns the creature `attacker` actually fought (even if the
        fight did nothing, e.g. elusive), or None if no fight happened at
        all (stunned, cancelled, or no legal target) -- One Stood Against
        Many uses this to exclude that target from the next of its 3
        fights."""
        player = self.players[pid]
        if self._consume_stun_if_present(attacker):
            player.used_this_turn[attacker.name] = player.used_this_turn.get(attacker.name, 0) + 1
            return None

        # A fight that has no legal target never happens at all -- it costs
        # nothing, so the creature is not exhausted (MRB 18.3 FAQ: Anger
        # used on Bumpsy with no enemy creatures in play leaves it ready).
        # This matters for "ready and fight with a creature" effects
        # (`ready_and_fight`); every other caller of `_fight` already only
        # calls it when `legal_fight_targets` is non-empty.
        if not self.legal_fight_targets(attacker, exclude):
            return None
        attacker.Exhausted = True
        player.used_this_turn[attacker.name] = player.used_this_turn.get(attacker.name, 0) + 1

        # before_fight triggers (Evasion Sigil) may cancel the fight outright.
        before = {"attacker": attacker, "cancelled": False}
        yield from self._fire_event("before_fight", before)
        if before["cancelled"]:
            return None

        targets = self.legal_fight_targets(attacker, exclude)
        if not targets:
            return None
        choice = yield from self.choose_cards(
            pid, f"Choose a target for {attacker.name} to fight", targets, 1, 1,
            source_card=attacker, intent=DecisionIntent.FIGHT_TARGET, affects=Affects.ENEMY,
        )
        target = choice[0]
        self.log.add(
            "fight",
            attacker=attacker.name,
            attacker_iid=attacker.instance_id,
            target=target.name,
            target_iid=target.instance_id,
        )
        # Fired once the fight is committed (a legal target was chosen),
        # regardless of how it plays out -- Warsong: "each time a friendly
        # creature fights, gain 1 Æmber for the remainder of the turn."
        yield from self._fire_event("fight_resolved", {"attacker": attacker, "target": target})

        destroyed = yield from self.check_destroyed([attacker, target])
        if attacker in destroyed or target in destroyed:
            return target

        # A printed "Before Fight:" ability (Firespitter, Lord Golgotha)
        # resolves as the attacker, before hazardous and the fight itself.
        # It can hit creatures well beyond this fight's two participants, so
        # the destroyed-check after it covers everyone in play, not just
        # attacker/target.
        if attacker.card_def.on_before_fight is not None:
            yield from attacker.card_def.on_before_fight(self, attacker, target)
            destroyed = yield from self.check_destroyed(self.all_creatures("any", attacker))
            if attacker in destroyed or target in destroyed:
                return target

        # Assault: before the fight itself, the attacker deals its assault
        # damage to the defender (Ancient Bear) -- hazardous's mirror image.
        assault_n = self.get_assault(attacker)
        if assault_n > 0:
            steps.deal_damage(self, target, assault_n)
            destroyed = yield from self.check_destroyed([attacker, target])
            if attacker in destroyed or target in destroyed:
                return target

        # Hazardous: before the fight itself, the defender deals its
        # hazardous damage to the attacker (Flame-Wreathed).
        hazardous_n = self.get_hazardous(target)
        if hazardous_n > 0:
            steps.deal_damage(self, attacker, hazardous_n)
            destroyed = yield from self.check_destroyed([attacker, target])
            if attacker in destroyed:
                return target

        target_keywords = self.get_keywords(target)
        attacker_keywords = self.get_keywords(attacker)
        skip_fight = (
            "elusive" in target_keywords
            and not target.fought_this_turn
            and not target.IgnoreElusive
            and not attacker.card_def.ignores_elusive
        )
        target.fought_this_turn = True
        if skip_fight:
            steps.shortfall(
                self, target,
                "is elusive: the first time it is attacked each turn, no damage is dealt by either creature",
                "Elusive: no damage",
            )
        if not skip_fight:
            # Skirmish ("when you use this creature to fight, it is dealt no
            # damage in return") only protects a creature when IT is the one
            # doing the fighting -- i.e. it gates whether the attacker takes
            # the target's counter-damage, not whether the target takes the
            # attacker's damage. A skirmish creature being attacked (as the
            # target) has no special protection from that.
            # Track whichever creature actually took each hit -- a redirect
            # (Shadow Self) can mean that isn't `attacker`/`target` -- so
            # the destroy check below covers it too, not just the two
            # original fighters.
            hits = []
            if "skirmish" not in attacker_keywords:
                hit = steps.deal_damage(self, attacker, self.get_power(target))
                if hit is not None:
                    hits.append(hit)
                    # Poison kills whichever creature actually took the
                    # damage, and only if any damage got through armor.
                    if "poison" in target_keywords and hit in self.players[hit.controller].play_area.creatures:
                        hit.type_object.damage = max(hit.type_object.damage, self.get_power(hit))
            # Gabos Longarms: a before_fight choice can redirect the
            # attacker's own damage output to a third creature. Shadow Self
            # ("deals no damage when fighting") deals none at all.
            if "no_fight_damage" not in attacker_keywords:
                dmg_target = attacker.redirect_fight_damage_to or target
                attack_power = self.get_power(attacker) + self.get_fight_damage_bonus(attacker, target)
                hit = steps.deal_damage(self, dmg_target, attack_power)
                if hit is not None:
                    hits.append(hit)
                    if "poison" in attacker_keywords and hit in self.players[hit.controller].play_area.creatures:
                        hit.type_object.damage = max(hit.type_object.damage, self.get_power(hit))
            attacker.redirect_fight_damage_to = None
            destroyed = yield from self.check_destroyed([attacker, target] + hits)
            for c in destroyed:
                # Distinct from the plain "destroyed" log entry: this marks
                # specifically that the fight's own damage exchange (not
                # before-fight/hazardous/assault or any other effect) is
                # what did it (The Warchest).
                self.log.add("destroyed_in_fight", card=c.name, iid=c.instance_id, controller=c.controller, turn=self.turn_number)

            # "destroyed fighting X" triggers (Overlord Greking, Stealer of
            # Souls, Brain Eater): fire on whichever side survived, naming
            # the side that died. Not fired if both die.
            survivor, victim = None, None
            if attacker in destroyed and target not in destroyed:
                survivor, victim = target, attacker
            elif target in destroyed and attacker not in destroyed:
                survivor, victim = attacker, target
            if survivor is not None and survivor.card_def.on_destroyed_fighting is not None:
                yield from survivor.card_def.on_destroyed_fighting(self, survivor, victim)

            if attacker in destroyed:
                return target

        cdef = attacker.card_def
        if cdef.on_fight is not None:
            yield from cdef.on_fight(self, attacker)
        for extra in list(attacker.extra_triggers.get("after_fight", [])):
            yield from extra(self, attacker)
        return target

    def use_creature_ability(self, card: Card):
        """Reap, fight, or use the Action of a friendly creature, whichever is
        possible (Dominator Bauble). Ignores house restrictions."""
        pid = card.controller
        player = self.players[pid]
        if card.Exhausted or not self._rule_of_six_ok(player, card.name) or player.get_cannot_use_cards(self):
            return
        if card.card_def.use_restriction is not None and not card.card_def.use_restriction(self, card):
            return
        if card.stunned:
            # A stunned creature has exactly one "use" -- itself, consumed
            # for no effect -- so there's no menu of options to offer.
            yield from self._reap(pid, card)
            return
        only_fight = player.get_can_only_fight(self)
        options = []
        if not card.card_def.cannot_reap and not only_fight:
            options.append("reap")
        if self.legal_fight_targets(card):
            options.append("fight")
        if not only_fight and (card.card_def.on_action is not None or card.granted_action is not None):
            options.append("action")
        if not options:
            return
        if len(options) > 1:
            choice = yield from self.choose_cards(
                pid, f"Use {card.name}", options, 1, 1,
                source_card=card, intent=DecisionIntent.MODE, affects=Affects.NONE,
            )
            kind = choice[0]
        else:
            kind = options[0]
        if kind == "reap":
            yield from self._reap(pid, card)
        elif kind == "fight":
            yield from self._fight(pid, card)
        elif kind == "action":
            yield from self._use_action(pid, card)

    def ready_and_fight(self, card: Card, exclude=frozenset()):
        """Ready `card` and immediately attempt to fight with it, regardless
        of its current Exhausted state (Anger, Ganger Chieftain, Gauntlet of
        Command, Relentless Assault, One Stood Against Many). A stunned
        creature just has its stun cleared instead (MRB 18.3 FAQ: "if a card
        allows you to use a creature, and that creature is stunned, remove
        the stun instead of doing anything else"). If it ends up with no
        legal fight target, it simply stays ready -- `_fight` never exhausts
        a creature that had nothing to fight (MRB 18.3 FAQ, Anger used on
        Bumpsy). Returns the creature it fought, or None."""
        pid = card.controller
        player = self.players[pid]
        if self._consume_stun_if_present(card):
            player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
            return None
        steps.ready(self, card)
        return (yield from self._fight(pid, card, exclude))

    # ------------------------------------------------------------ destroy ----

    def check_destroyed(self, cards: List[Card]):
        if self._redirected_hits:
            cards = list(cards) + [c for c in self._redirected_hits if c not in cards]
            self._redirected_hits.clear()
        to_destroy = []
        for c in cards:
            if c.destroyed or not isinstance(c.type_object, CreatureType):
                continue
            if c not in self.players[c.controller].play_area.creatures:
                continue
            if c.type_object.damage >= self.get_power(c):
                to_destroy.append(c)
        # `destroy_cards` itself checks "would be destroyed, instead X"
        # replacements (Armageddon Cloak) -- every path that destroys a
        # card, damage-lethality or direct, funnels through it.
        destroyed = yield from self.destroy_cards(to_destroy)
        return destroyed

    def put_creature_into_play_from_hand(self, player_id: int, card: Card, flank: Optional[str] = None, ready: bool = False) -> bool:
        """Puts `card` directly into play from `player_id`'s hand, bypassing
        normal play restrictions (house, Æmber cost, rule of six, ...) --
        Swap Widget: "put a Mars creature ... from your hand into play".
        Unlike `_play_card`, this never resolves the card's own Play
        ability (real KeyForge distinguishes "play" from "put into play")."""
        player = self.players[player_id]
        if not player.hand.remove(card):
            return False
        card.controller = player_id
        player.play_area.add_creature(card, flank)
        card.Exhausted = not ready
        card.CanBeUsed = player.selected_house is not None and (
            self.get_effective_house(card) == player.selected_house or "versatile" in self.get_keywords(card)
        )
        if card.card_def.register_passive:
            card.card_def.register_passive(self, card)
        self.log.add("put_into_play", card=card.name, iid=card.instance_id, player=player_id, source=None)
        return True

    def archive_from_play(self, card: Card, archiving_player_id: int, return_to_owner_after: bool = False) -> bool:
        """Puts `card` (a creature or artifact currently in play, possibly
        the opponent's) into `archiving_player_id`'s archive -- Sample
        Collection, Mass Abduction: 'put into YOUR archives' names the
        archiving player's zone, not the card's owner's. If
        `return_to_owner_after`, the card is flagged so that when it later
        leaves that archive it goes to its own owner's hand instead of the
        archiving player's, per those cards' own text."""
        area = self.find_play_area(card)
        if area is None:
            return False
        area.remove(card)
        self.leave_play(card)
        card.archive_return_to_owner = return_to_owner_after
        self.players[archiving_player_id].archive.add(card)
        self.log.add("archive", player=archiving_player_id, card=card.name, iid=card.instance_id)
        return True

    def destroy_upgrade(self, upgrade_card: Card) -> bool:
        """Removes `upgrade_card` from its host and sends it to its owner's
        discard pile, without destroying (or otherwise affecting) the host
        itself (Armageddon Cloak's own destroyed-replacement)."""
        host = upgrade_card.type_object.host
        if host is None or upgrade_card not in host.type_object.upgrades:
            return False
        host.type_object.upgrades.remove(upgrade_card)
        if upgrade_card.card_def.unregister_passive is not None:
            upgrade_card.card_def.unregister_passive(self, upgrade_card)
        self.players[upgrade_card.owner].discard.push(upgrade_card)
        self.log.add("destroyed", card=upgrade_card.name, iid=upgrade_card.instance_id, destination="discard")
        return True

    def destroy_cards(self, cards: List[Card]):
        # "Would be destroyed, instead X" replacements (Armageddon Cloak) get
        # a chance to intercept before anything is actually destroyed, one
        # card at a time -- each registered handler decides for itself
        # whether it applies to that specific card, and returns whether it
        # fired. This runs here, not in `check_destroyed`, so it covers
        # EVERY path that destroys a card -- damage-lethality (check_
        # destroyed) as well as a direct `game.destroy_cards(...)` call from
        # dozens of card effects across every house (Begone!, EMP Blast, ...).
        insteads = self.active_effects.insteads_for("would_be_destroyed")
        if insteads:
            survivors = []
            for c in cards:
                intercepted = False
                for e in insteads:
                    if e.handler(self, c):
                        intercepted = True
                        break
                if not intercepted:
                    survivors.append(c)
            cards = survivors
        batch = []
        for c in cards:
            # `.destroyed` only dedupes within one batch that's still being
            # resolved -- it's reset by `reset_on_leave_play` as soon as a
            # card actually leaves play, so a card destroyed earlier this
            # turn (e.g. by its own sacrifice action, mid-resolution of an
            # effect like Poltergeist that destroys it again afterward)
            # needs this separate still-in-play check too.
            if c.destroyed or self.find_play_area(c) is None:
                continue
            c.destroyed = True
            batch.append(c)
        if not batch:
            return []
        to_resolve = [c for c in batch if c.card_def.on_destroyed is not None]
        for c in batch:
            for extra in list(c.extra_triggers.get("destroyed", [])):
                to_resolve.append(("extra", c, extra))
        if len(to_resolve) > 1:
            ordered = yield from self.order_effects(
                self.active_player_id, to_resolve, "Destroyed effects: choose the order they resolve"
            )
        else:
            ordered = to_resolve
        for item in ordered:
            if isinstance(item, tuple):
                _, c, extra = item
                yield from extra(self, c)
            else:
                yield from item.card_def.on_destroyed(self, item)
        # A "each time a creature is destroyed" source (Soul Snatcher, Tolas)
        # that is itself destroyed in this same batch does not trigger at
        # all for the batch -- it isn't around to see any of it happen,
        # including its own death (confirmed ruling: Tolas + Gateway to Dis
        # gains nothing).
        also_destroyed = set(batch)
        for c in batch:
            if isinstance(c.type_object, CreatureType):
                yield from self._fire_event("creature_destroyed", {"card": c}, exclude_sources=also_destroyed)
        for c in batch:
            self._move_destroyed_card(c)
        return batch

    def _fire_event(self, event_name: str, event_data: dict, exclude_sources=None):
        """Yields from every registered TriggerEffect for `event_name`
        (Soul Snatcher/Tolas's 'each time a creature is destroyed', Veylan
        Analyst's 'each time you use an artifact', Shaffles's end of turn,
        ...), letting the active player order them if more than one fires."""
        triggers = [t for t in self.active_effects.triggers_for(event_name)]
        if exclude_sources:
            triggers = [t for t in triggers if t.source_card not in exclude_sources]
        if not triggers:
            return
        if len(triggers) > 1:
            ordered = yield from self.order_effects(self.active_player_id, triggers, f"Choose the order these resolve")
        else:
            ordered = triggers
        for trig in ordered:
            yield from trig.handler(self, event_data)

    def _move_destroyed_card(self, card: Card):
        owner = self.players[card.owner]
        destination = card.destined_zone or "discard"
        if (
            destination == "discard"
            and isinstance(card.type_object, CreatureType)
            and self.active_effects.insteads_for("discard_destination")
        ):
            # Annihilation Ritual: a creature (not an artifact) that would
            # enter a discard pile from play is purged instead.
            destination = "purged"
        # Logged first so what leaving play causes (captured Æmber going
        # back) reads after "X is destroyed", not before it.
        self.log.add("destroyed", card=card.name, iid=card.instance_id, destination=destination)
        area = self.find_play_area(card)
        if area is not None:
            area.remove(card)
            self.leave_play(card)
        if destination == "hand":
            owner.hand.add(card)
        elif destination == "purged":
            owner.purged.add(card)
        elif destination == "deck_top":
            owner.deck.put_on_top(card)
        elif destination == "archive":
            owner.archive.add(card)
        else:
            owner.discard.push(card)

    def leave_play(self, card: Card):
        if card.card_def.unregister_passive is not None:
            card.card_def.unregister_passive(self, card)
        if isinstance(card.type_object, CreatureType):
            for upg in list(card.type_object.upgrades):
                if upg.card_def.unregister_passive is not None:
                    upg.card_def.unregister_passive(self, upg)
                owner = self.players[upg.owner]
                owner.discard.push(upg)
                upg.reset_on_leave_play()
        if card.aember_captured > 0:
            opponent = self.players[3 - card.controller]
            opponent.aember += card.aember_captured
            self.log.add("capture_released", card=card.name, iid=card.instance_id, amount=card.aember_captured, player=opponent.id)
            card.aember_captured = 0
        if card.aember_stored > 0:
            # Pocket Universe, Safe Place: stored Æmber vanishes into the
            # supply on leaving play, unlike captured Æmber (which the
            # opponent gets).
            self.log.add("aember_stored_lost", card=card.name, iid=card.instance_id, amount=card.aember_stored)
        if card.under_cards:
            for under in card.under_cards:
                self.players[under.owner].discard.push(under)
                self.log.add("destroyed", card=under.name, iid=under.instance_id, destination="discard")
            card.under_cards = []
        self._revert_temp_control(card)
        self._return_spangler_purged_cards(card)
        card.reset_on_leave_play()

    def _revert_temp_control(self, source_card: Card) -> None:
        """Harland Mindlock: creatures it took temporary control of revert
        to their original controller when it leaves play. The reverted
        creature always lands on the left flank -- a real flank choice
        would need `leave_play` to become a generator, which none of its
        many call sites support; a fixed flank is a deliberate scope trim."""
        entries = self._temp_control.pop(source_card.instance_id, [])
        for controlled_card, original_pid in entries:
            current_pid = controlled_card.controller
            if controlled_card.destroyed or controlled_card not in self.players[current_pid].play_area.creatures:
                continue
            self.players[current_pid].play_area.remove(controlled_card)
            self.players[original_pid].play_area.add_creature(controlled_card, "left")
            controlled_card.controller = original_pid
            self.log.add(
                "take_control", card=controlled_card.name, iid=controlled_card.instance_id,
                from_player=current_pid, to_player=original_pid, permanent=True, reverted=True,
            )

    def _return_spangler_purged_cards(self, source_card: Card) -> None:
        """Spangler Box: 'If Spangler Box leaves play, return to play all
        cards purged by Spangler Box.' Also lands on a fixed left flank --
        see `_revert_temp_control`."""
        returning = [c for c in self.players[1].purged.cards() + self.players[2].purged.cards() if c.purged_by is source_card]
        for c in returning:
            self.players[c.owner].purged.remove(c)
            c.purged_by = None
            if c.type == CardType.CREATURE:
                self.players[c.owner].play_area.add_creature(c, "left")
            elif c.type == CardType.ARTIFACT:
                self.players[c.owner].play_area.add_artifact(c)
            else:
                self.players[c.owner].discard.push(c)
                continue
            c.controller = c.owner
            c.Exhausted = True
            if c.card_def.register_passive:
                c.card_def.register_passive(self, c)
            self.log.add("put_into_play", card=c.name, iid=c.instance_id, player=c.owner, source=source_card.name)

    def resolve_damage_target(self, creature: Card) -> Card:
        """Applies any active damage-redirect effects (Shadow Self) to
        `creature`, returning whichever creature should actually take the
        damage. Called once, centrally, from `steps.deal_damage`."""
        for e in self.active_effects.insteads_for("damage_target"):
            redirect = e.handler(self, creature)
            if redirect is not None:
                return redirect
        return creature

    def take_control(self, card: Card, new_pid: int, until_source: Optional[Card] = None):
        """Moves `card` to `new_pid`'s control (Smiling Ruth, Sneklifter,
        Overlord Greking, Harland Mindlock). If `until_source` is given,
        control reverts to the original controller when that source card
        leaves play; otherwise the change is permanent. A creature's new
        controller picks its flank; an artifact just moves over. The old
        controller is found by physical location (`find_play_area`), not by
        `card.controller` -- that field is temporarily repointed to the
        borrower during a "use as if yours" effect (Poltergeist, Remote
        Access, Nexus), while the card itself hasn't actually moved."""
        old_area = self.find_play_area(card)
        if old_area is None:
            return
        old_pid = 1 if old_area is self.players[1].play_area else 2
        if old_pid == new_pid:
            return
        new_area = self.players[new_pid].play_area
        if isinstance(card.type_object, CreatureType):
            old_area.remove(card)
            flank = yield from self._choose_flank(new_pid)
            new_area.add_creature(card, flank)
        elif card.type == CardType.ARTIFACT:
            old_area.remove(card)
            new_area.add_artifact(card)
        else:
            return
        card.controller = new_pid
        if until_source is not None:
            self._temp_control.setdefault(until_source.instance_id, []).append((card, old_pid))
        self.log.add(
            "take_control", card=card.name, iid=card.instance_id,
            from_player=old_pid, to_player=new_pid, permanent=until_source is None, reverted=False,
        )

    def use_artifact_ability(self, card: Card, as_pid: int):
        """Uses `card`'s Action or Omni ability 'as if it were yours'
        (Poltergeist, Remote Access, Nexus): the effect resolves as though
        `as_pid` controlled it, then control reverts -- unlike
        `take_control`, this is a one-time use, not a lasting change.

        It's `as_pid` doing the using, so the same restrictions a normal
        use would face apply to them: the rule of six, Skippy Timehog's
        CannotUseCards, and Tentacus's Æmber toll (paid by `as_pid`, to
        whoever Tentacus's effect names)."""
        as_player = self.players[as_pid]
        ability = card.card_def.on_action or card.card_def.on_omni
        if (
            card.Exhausted
            or ability is None
            or as_player.get_cannot_use_cards(self)
            or not self._rule_of_six_ok(as_player, card.name)
        ):
            steps.shortfall(self, card, "can't be used: no Action or Omni, already exhausted, or restricted", "Can't use")
            return
        toll = as_player.get_artifact_use_toll(self)
        if toll is not None and toll[0] > as_player.aember:
            steps.shortfall(self, card, "can't be used: can't afford the Æmber toll to use an artifact", "Can't afford toll")
            return
        is_omni = card.card_def.on_action is None
        original_controller = card.controller
        card.controller = as_pid
        self._pay_artifact_use_toll(as_pid, card)
        card.Exhausted = True
        as_player.used_this_turn[card.name] = as_player.used_this_turn.get(card.name, 0) + 1
        self.log.add(
            "use_omni" if is_omni else "use_action",
            player=as_pid, card=card.name, iid=card.instance_id, as_if_yours=(as_pid != original_controller),
        )
        try:
            yield from ability(self, card)
        finally:
            card.controller = original_controller
        if card.type == CardType.ARTIFACT:
            yield from self._fire_event("artifact_used", {"player": as_pid, "card": card})


# --------------------------------------------------- run_until predicates ----
# Ready-made conditions for `Game.run_until` (Milestone D). Each factory
# takes whatever context it needs at creation time and returns a plain
# `predicate(game) -> bool`.


def until_game_over(game: "Game") -> bool:
    return game.is_over


def until_player_decides(pid: int):
    def predicate(game: "Game") -> bool:
        return game.is_over or (game.pending_decision is not None and game.pending_decision.player == pid)

    return predicate


def until_boundary():
    """The next `CHOOSE_ACTION`/`CHOOSE_HOUSE`/`TAKE_ARCHIVE` decision --
    `enums.BOUNDARY_KINDS`, where no card effect is mid-resolution (88.9%
    of all decisions, per the plan's own baseline)."""

    def predicate(game: "Game") -> bool:
        return game.is_over or (game.pending_decision is not None and game.pending_decision.kind in BOUNDARY_KINDS)

    return predicate


def until_end_of_turn(game: "Game"):
    """True once `game`'s turn number has advanced past its value when this
    predicate was created -- the turn-bounded search leaf: `game.run_until
    (until_end_of_turn(game), policy)`."""
    start_turn = game.turn_number

    def predicate(g: "Game") -> bool:
        return g.is_over or g.turn_number > start_turn

    return predicate

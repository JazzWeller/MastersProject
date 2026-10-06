"""The zone journal (Agent Observation Plan, Milestone O1).

Every move of a card between zones appends one entry to `game.journal`:

    (seq, turn, iid, owner, controller, from_zone, to_zone, op, deck_epoch, public_position, cause)

- A zone is `(kind, key)`: `kind` is hand, deck, discard, archive, purged,
  battleline, artifacts (keyed by player id), attached or under (keyed by
  the host card's instance id), limbo (between zones, key None), or setup
  (where a card comes from when it is dealt into its deck, keyed by owner;
  hidden).
- `op` names what moved it, as `leave/enter` (e.g. `draw_top/add`,
  `remove/push`), or a single word for an entry that isn't a move. Those
  have `from_zone == to_zone`: `shuffle` (iid None), `swap`, `reveal` (a
  card shown to both players), `search_reveal` (the same, found by a
  search), and `reveal_hand` / `unreveal_hand` (iid None: a hand is shown to
  the other player until the end of the turn, `Game.reveal_hand`).
- `deck_epoch` counts the shuffles of the deck involved, if any: a known
  position in a deck lasts until its next shuffle.
- `public_position` is "top" or "bottom" for a placement or draw at a known
  end of a deck, the partner's instance id for a battleline swap, the
  viewer for `reveal_hand`, else None.
- `cause` is the instance id of the card whose ability is resolving
  (innermost, `Game._caused`), or None for a rules move (a card played,
  discarded, drawn at the end of the turn, destroyed by fight damage).

**Pairing.** A card that leaves a zone gets an entry to limbo at once; when
it enters a zone before the next decision, that same entry is completed in
place (it keeps its `seq`), so a move is one entry. A card still between
zones when a decision comes (an action being played) keeps its "to limbo"
entry, and its later arrival is a "from limbo" entry. Pairing depends only
on the game -- never on when anyone looked at the journal -- so the journal
is deterministic, and every entry is final once a decision is pending.

Alongside the entries:
- `marks[i]` is how many log events preceded entry `i`, which places the
  journal and the log in one order (keyforge/projection.py);
- `decisions` holds one record per submitted decision (`decision()`), the
  input of the projection's decision events. A record is kept as the
  decision itself until something reads it (`records()`, a copy, a
  snapshot): writing it down costs nothing on the engine's hot path.

The journal is privileged, like `game.log`; it is excluded from
`state_hash`; `Game.copy()` copies it and a replay regenerates it. A
determinization re-deal is not a game event and is not journaled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card
from .effects.effect_object import TriggerEffect
from .enums import DecisionIntent, DecisionKind, House

LIMBO = ("limbo", None)
SETUP = {1: ("setup", 1), 2: ("setup", 2)}
# The option verbs whose payload is a card's instance id.
CARD_VERBS = frozenset({"card", "play", "discard", "action", "omni", "reap", "fight"})

Zone = Tuple[str, Optional[int]]

_VERBS = {PlayCard: "play", DiscardCard: "discard", UseAction: "action", UseOmni: "omni", Reap: "reap", Fight: "fight"}
_LIST_KINDS = frozenset({DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS})
_KIND_NAMES = {k: k.name for k in DecisionKind}
_INTENT_NAMES = {i: i.name for i in DecisionIntent}
# Zone kinds both players see into. A hand or an archive is its owner's
# (and a hand also whoever it is revealed to), a deck nobody's, a card
# under another its owner's.
PUBLIC_KINDS = frozenset({"discard", "purged", "battleline", "artifacts", "attached", "limbo"})


class Journal:
    __slots__ = ("_entries", "game", "_open", "_epochs", "_marks", "decisions", "_raw", "_events",
                 "_pc", "_pz", "_pop", "_ppos", "_pcause", "_pturn", "_pmark", "_pepoch")

    def __init__(self, game):
        self.game = game
        # the game's log events (`Game` sets it once its log exists): their
        # count is each entry's mark
        self._events: List = []
        self._entries: List[tuple] = []
        self._marks: List[int] = []
        self.decisions: List[tuple] = []
        # index of the first record in `decisions` still raw
        self._raw = 0
        # instance id -> index of its open "to limbo" entry, until the card
        # enters a zone or a decision comes
        self._open: Dict[int, int] = {}
        # player id -> how many times their deck has been shuffled
        self._epochs: Dict[int, int] = {1: 0, 2: 0}
        # The latest `leave`, not yet written: almost always the same card's
        # `enter` comes next, and the two make one entry. Anything else
        # writes it first (`_flush`), so the entries are exactly as if each
        # `leave` had been written at once.
        self._pc = self._pz = self._pop = self._ppos = self._pcause = self._pturn = self._pmark = self._pepoch = None

    # Readers see every entry, the pending one included.
    @property
    def entries(self) -> List[tuple]:
        if self._pc is not None:
            self._flush()
        return self._entries

    @property
    def marks(self) -> List[int]:
        if self._pc is not None:
            self._flush()
        return self._marks

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

    def __iter__(self):
        return iter(self.entries)

    def epoch(self, zone: Zone) -> Optional[int]:
        return self._epochs[zone[1]] if zone[0] == "deck" else None

    def _flush(self) -> None:
        """Writes the pending `leave` as an open entry to limbo."""
        card = self._pc
        self._pc = None
        entries = self._entries
        i = len(entries)
        entries.append((i, self._pturn, card.instance_id, card.owner, card.controller, self._pz, LIMBO, self._pop,
                        self._pepoch, self._ppos, self._pcause))
        self._marks.append(self._pmark)
        self._open[card.instance_id] = i

    def _append(self, entry: tuple) -> None:
        if self._pc is not None:
            self._flush()
        entry = (len(self._entries),) + entry
        self._entries.append(entry)
        self._marks.append(len(self._events))

    def leave(self, card, zone: Zone, op: str, position: Optional[str] = None) -> None:
        if self._pc is not None:
            self._flush()
        game = self.game
        causes = game._causes
        self._pc = card
        self._pz = zone
        self._pop = op
        self._ppos = position
        self._pcause = causes[-1] if causes else None
        self._pturn = game.turn_number
        self._pmark = len(self._events)
        self._pepoch = self._epochs[zone[1]] if zone[0] == "deck" else None

    def enter(self, card, zone: Zone, op: str, position=None) -> None:
        entries = self._entries
        if self._pc is card:  # the usual move: one entry
            self._pc = None
            epoch = self._pepoch
            if epoch is None and zone[0] == "deck":
                epoch = self._epochs[zone[1]]
            pos = self._ppos
            entries.append((len(entries), self._pturn, card.instance_id, card.owner, card.controller, self._pz, zone,
                            _move_op(self._pop, op), epoch, pos if pos is not None else position, self._pcause))
            self._marks.append(self._pmark)
            return
        if self._pc is not None:
            self._flush()
        iid = card.instance_id
        i = self._open.pop(iid, None)
        if i is not None:
            e = entries[i]
            epoch = e[8]
            if epoch is None and zone[0] == "deck":
                epoch = self._epochs[zone[1]]
            entries[i] = (e[0], e[1], iid, e[3], card.controller, e[5], zone, _move_op(e[7], op), epoch,
                          e[9] if e[9] is not None else position, e[10])
            return
        game = self.game
        causes = game._causes
        frm = SETUP[card.owner] if op == "deal" else LIMBO
        entries.append((len(entries), game.turn_number, iid, card.owner, card.controller, frm, zone, op,
                        self._epochs[zone[1]] if zone[0] == "deck" else None, position,
                        causes[-1] if causes else None))
        self._marks.append(len(self._events))

    def deal(self, cards, zone: Zone) -> None:
        """`cards` dealt into `zone` (a deck), in order: `enter` for each."""
        if self._pc is not None:
            self._flush()
        entries = self._entries
        n = len(entries)
        turn = self.game.turn_number
        epoch = self._epochs[zone[1]]
        entries.extend([(n + k, turn, c.instance_id, c.owner, c.controller, SETUP[c.owner], zone, "deal", epoch, None,
                         None) for k, c in enumerate(cards)])
        self._marks.extend([len(self._events)] * len(cards))

    def note(self, card, zone: Zone, op: str, position=None) -> None:
        """An entry that isn't a move (a swap of battleline places, a
        reveal)."""
        causes = self.game._causes
        self._append((self.game.turn_number, card.instance_id if card is not None else None,
                      card.owner if card is not None else None, card.controller if card is not None else None,
                      zone, zone, op, self.epoch(zone), position, causes[-1] if causes else None))

    def reveal(self, card, op: str = "reveal", position: Optional[str] = None) -> None:
        """`card` is shown to both players, where it is (`position` "top"
        for the top card of a deck)."""
        self.note(card, zone_of(self.game, card), op, position)

    def reveal_hand(self, pid: int, viewer: int, revealed: bool = True) -> None:
        zone = ("hand", pid)
        causes = self.game._causes
        self._append((self.game.turn_number, None, pid, None, zone, zone,
                      "reveal_hand" if revealed else "unreveal_hand", None, viewer, causes[-1] if causes else None))

    def shuffle(self, pid: int) -> None:
        if self._pc is not None:
            self._flush()
        self._epochs[pid] += 1
        zone = ("deck", pid)
        causes = self.game._causes
        self._append((self.game.turn_number, None, pid, None, zone, zone, "shuffle",
                      self._epochs[pid], None, causes[-1] if causes else None))

    # ------------------------------------------------------- decisions ----

    def decision(self, d, encoded) -> None:
        """A decision was submitted, with `encoded` its `choice_record`
        entry: cards still between zones stay in limbo, and the decision is
        recorded (raw, until `records()`)."""
        if self._pc is not None:
            self._flush()
        if self._open:
            self._open.clear()
        self.decisions.append((len(self._entries), len(self._events), d, encoded))

    def settle(self) -> None:
        """Writes everything down: the pending move, and every decision
        record in data form (before a snapshot)."""
        if self._pc is not None:
            self._flush()
        self._pz = self._pop = self._ppos = self._pcause = self._pturn = self._pmark = self._pepoch = None
        self.records()

    def records(self) -> List[tuple]:
        """Every decision record, each

            (n_entries, n_log, player, kind, intent, source, affects, optional, chosen, offered)

        `chosen` and `offered` (every option) are option forms: (verb, iid)
        for a card or an action on one (verb in `CARD_VERBS`), else (verb,
        payload). Where each card was at that moment is the journal's, up to
        `n_entries` (the projection folds it)."""
        decisions = self.decisions
        for i in range(self._raw, len(decisions)):
            n_e, n_l, d, encoded = decisions[i]
            options = d.options
            chosen = tuple([_form(options[j]) for j in encoded]) if d.kind in _LIST_KINDS else (_form(options[encoded]),)
            src, intent, affects = d.source_card, d.intent, d.affects
            decisions[i] = (n_e, n_l, d.player, _KIND_NAMES[d.kind],
                            None if intent is None else _INTENT_NAMES[intent],
                            src.instance_id if isinstance(src, Card) else None,
                            None if affects is None else affects.value,
                            d.optional, chosen, tuple([_form(o) for o in options]))
        self._raw = len(decisions)
        return decisions

    def copy_into(self, game) -> "Journal":
        new = Journal(game)
        new._entries = list(self.entries)
        new._marks = list(self.marks)
        new.decisions = list(self.records())  # a copy holds no decision of this game's
        new._raw = len(new.decisions)
        new._open = dict(self._open)
        new._epochs = dict(self._epochs)
        return new


_MOVE_OPS: Dict[Tuple[str, str], str] = {}


def _move_op(leave_op: str, enter_op: str) -> str:
    """"leave/enter", one string per pair."""
    s = _MOVE_OPS.get((leave_op, enter_op))
    if s is None:
        s = _MOVE_OPS[(leave_op, enter_op)] = leave_op + "/" + enter_op
    return s


def _form(o):
    t = type(o)
    if t is Card:
        return ("card", o.instance_id)
    verb = _VERBS.get(t)
    if verb is not None:
        return (verb, o.card.instance_id)
    if t is EndTurn:
        return ("end_turn", None)
    if isinstance(o, TriggerEffect):
        return ("trigger", (o.source_card.instance_id if o.source_card is not None else None, o.event))
    if t is House:
        return ("house", o.value)
    if t is bool:
        return ("bool", o)
    if t in (int, float, str) or o is None:
        return ("value", o)
    if isinstance(o, (list, tuple)):
        return ("seq", tuple([_form(x) for x in o]))
    if isinstance(o, Card):
        return ("card", o.instance_id)
    return ("opaque", t.__name__)


def zone_of(game, card) -> Zone:
    """Where `card` is, by looking (for the rare entry that needs it)."""
    pid = card.owner
    for p in (game.players[pid], game.players[3 - pid]):
        for kind in ("hand", "deck", "discard", "archive", "purged"):
            if card in getattr(p, kind)._cards:
                return (kind, p.id)
        if card in p.play_area.creatures:
            return ("battleline", p.id)
        if card in p.play_area.artifacts:
            return ("artifacts", p.id)
    return LIMBO

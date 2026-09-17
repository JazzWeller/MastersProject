"""The rules engine: setup, turn order, legal actions, and resolution."""

from __future__ import annotations

import math
import random
from typing import List, Optional

from .actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from .cards.card import Card, CreatureType
from .cards.decks import build_deck
from .config import GameConfig
from .decision import Decision
from .effects import steps
from .effects.effect_object import ActiveEffectList
from .enums import CardType, DecisionKind, House
from .log import GameLog
from .player import Player
from .replay import encode_choice
from .view import build_view
from .zones import Deck


class Game:
    def __init__(self, config: GameConfig):
        self.config = config
        self.rng = random.Random(config.seed)
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

    def view_for(self, pid: int):
        return build_view(self, pid)

    # -------------------------------------------------- choice helpers ----

    def choose_cards(self, player, prompt, options, min_n=0, max_n=1):
        options = list(options)
        if not options:
            return []
        if min_n == max_n and len(options) == min_n:
            return options
        choice = yield Decision(player, DecisionKind.CHOOSE_CARDS, prompt, options, min_n, max_n)
        return choice

    def choose_house(self, player, prompt, houses):
        houses = list(houses)
        if not houses:
            return None
        if len(houses) == 1:
            return houses[0]
        choice = yield Decision(player, DecisionKind.CHOOSE_HOUSE_FOR_EFFECT, prompt, houses, 1, 1)
        return choice

    def yes_no(self, player, prompt):
        choice = yield Decision(player, DecisionKind.YES_NO, prompt, [True, False], 1, 1)
        return choice

    def order_effects(self, player, items, prompt="Choose the order these resolve"):
        items = list(items)
        if len(items) <= 1:
            return items
        choice = yield Decision(player, DecisionKind.ORDER_EFFECTS, prompt, items, len(items), len(items))
        return choice

    # ------------------------------------------------------- get funcs ----

    def get_power(self, card: Card) -> int:
        return card.type_object.base_power

    def get_armor(self, card: Card) -> int:
        return card.type_object.base_armor

    def player_houses(self, pid: int) -> List[House]:
        return sorted({c.house for c in self.players[pid].all_cards}, key=lambda h: h.value)

    def all_creatures(self, scope: str, source_card: Card) -> List[Card]:
        pid = source_card.controller
        if scope == "friendly":
            return list(self.players[pid].play_area.creatures)
        if scope == "enemy":
            return list(self.players[3 - pid].play_area.creatures)
        return list(self.players[1].play_area.creatures) + list(self.players[2].play_area.creatures)

    # ------------------------------------------------------------ setup ----

    def _setup(self):
        p1_deck_name, p2_deck_name = self.config.decks
        for pid, deck_name in ((1, p1_deck_name), (2, p2_deck_name)):
            cards = build_deck(deck_name, pid)
            self.players[pid].all_cards = list(cards)
            self.players[pid].deck = Deck(cards)
            self.players[pid].deck.shuffle(self.rng)
        if self.config.first_player in (1, 2):
            first = self.config.first_player
        else:
            first = self.rng.choice([1, 2])
        self._first_player = first
        second = 3 - first
        steps.draw(self, self.players[first], 7)
        steps.draw(self, self.players[second], 6)
        for pid in (first, second):
            yield from self._maybe_mulligan(pid)

    def _maybe_mulligan(self, pid: int):
        player = self.players[pid]
        choice = yield Decision(pid, DecisionKind.MULLIGAN, "Mulligan your hand?", [True, False], 1, 1)
        if choice:
            n = len(player.hand)
            cards = player.hand.take_all()
            player.deck.shuffle_in(cards, self.rng)
            steps.draw(self, player, max(0, n - 1))
            self.log.add("mulligan", player=pid)

    # -------------------------------------------------------- main loop ----

    def _run(self):
        yield from self._setup()
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

        # Step 1: forge a key
        cost = player.get_key_forge_cost(self)
        if player.aember >= cost:
            if player.get_can_key_forge(self):
                player.aember -= cost
                player.keys += 1
                self.log.add("forge_key", player=pid, keys=player.keys, cost=cost, turn=self.turn_number)
                if player.keys >= 3:
                    self.result = {"winner": pid, "turns": self.turn_number, "reason": "3 keys"}
                    return True
            else:
                source = self._effect_source("CanKeyForge", pid)
                self.log.add("forge_skipped", player=pid, aember=player.aember, cost=cost, source=source)

        # Step 2: choose a house
        forced = player.get_house_selection(self)
        if forced is not None:
            house = forced
            self.log.add("house_forced", player=pid, house=house.value, source=self._effect_source("HouseSelection", pid))
        else:
            options = self.player_houses(pid)
            house = yield Decision(pid, DecisionKind.CHOOSE_HOUSE, "Choose your house", options, 1, 1)
        player.selected_house = house
        for c in player.play_area.all_cards():
            c.CanBeUsed = c.house == house
        self.log.add("choose_house", player=pid, house=house.value)

        # Step 3: optional archive pickup
        if len(player.archive) > 0:
            take = yield Decision(
                pid, DecisionKind.TAKE_ARCHIVE, "Take your archive into your hand?", [True, False], 1, 1
            )
            if take:
                for c in player.archive.take_all():
                    player.hand.add(c)
                self.log.add("take_archive", player=pid)

        # Step 4: main action loop
        while True:
            options = self._legal_actions(pid)
            choice = yield Decision(pid, DecisionKind.CHOOSE_ACTION, "Choose an action", options, 1, 1)
            if isinstance(choice, EndTurn):
                break
            yield from self._resolve_action(pid, choice)

        # Step 5: cleanup
        self._cleanup_turn(pid)
        # Step 6: draw
        self._draw_step(pid)
        return False

    def _cleanup_turn(self, pid: int):
        player = self.players[pid]
        for c in player.play_area.all_cards():
            c.Exhausted = False
            c.CanBeUsed = False
            c.fought_this_turn = False
        for p in self.players.values():
            for c in p.play_area.creatures:
                c.type_object.armor_used_this_turn = 0
        player.reset_turn_counters()
        self.active_effects.end_of_turn_tick()

    def _draw_step(self, pid: int):
        player = self.players[pid]
        limit = player.get_draw_up_to_limit(self)
        base = max(0, limit - len(player.hand))
        if base == 0:
            return
        n = base + player.get_card_draw_modifier(self)
        if player.chains > 0:
            n -= math.ceil(player.chains / 6)
            player.chains -= 1
        steps.draw(self, player, max(0, n))

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

    def why_not_playable(self, pid: int, card: Card) -> Optional[str]:
        """Why `card` in `pid`'s hand can't be played right now, or None if it can.
        Uses exactly the checks `_legal_actions` uses, so the two can't disagree."""
        player = self.players[pid]
        if card not in player.hand.cards():
            return "Not in hand"
        if pid != self.active_player_id or player.selected_house is None:
            return "Not your turn"
        if self.first_turn_limited(pid):
            return "First turn: you may play or discard only one card"
        house = player.selected_house
        allowance = player.get_non_logos_cards_playable(self)
        if not ((card.house == house) or (card.house != House.LOGOS and allowance > 0)):
            return f"Not of your active house ({house.value})"
        limit = player.get_card_played_limit(self)
        if limit is not None and player.hand_plays_this_turn >= limit:
            source = self._effect_source("CardPlayedLimit", pid)
            return f"Card play limit reached ({limit}{', ' + source if source else ''})"
        if card.type == CardType.CREATURE and not player.get_can_play_creatures(self):
            source = self._effect_source("CanPlayCreatures", pid)
            return "You cannot play creatures this turn" + (f" ({source})" if source else "")
        if card.type == CardType.ACTION and not player.get_can_play_actions(self):
            source = self._effect_source("CanPlayActions", pid)
            return "You cannot play actions this turn" + (f" ({source})" if source else "")
        if card.type == CardType.UPGRADE and not (self.players[1].play_area.creatures or self.players[2].play_area.creatures):
            return "No creature to attach it to"
        if not self._rule_of_six_ok(player, card.name):
            return "Rule of six: already played or used 6 times this turn"
        return None

    def _legal_actions(self, pid: int):
        player = self.players[pid]
        opponent = self.players[3 - pid]
        house = player.selected_house
        actions = []
        limited_now = self.first_turn_limited(pid)

        if not limited_now:
            for card in player.hand.cards():
                if self.why_not_playable(pid, card) is None:
                    actions.append(PlayCard(card))
                if card.house == house:
                    actions.append(DiscardCard(card))

        for card in player.play_area.creatures:
            if card.CanBeUsed and not card.Exhausted and self._rule_of_six_ok(player, card.name):
                actions.append(Reap(card))
                if opponent.play_area.creatures:
                    actions.append(Fight(card))
                if card.card_def.on_action is not None:
                    actions.append(UseAction(card))
            if not card.Exhausted and card.card_def.on_omni is not None and self._rule_of_six_ok(player, card.name):
                actions.append(UseOmni(card))

        for card in player.play_area.artifacts:
            if not card.Exhausted:
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
            self._discard_card(pid, action.card)
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

        house = player.selected_house
        if house is not None and card.house != house and card.house != House.LOGOS:
            if player.NonLogosCardsPlayable > 0:
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
            choice = yield from self.choose_cards(pid, f"Attach {card.name} to a creature", targets, 1, 1)
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
        )

        if card.type == CardType.CREATURE:
            player.play_area.add_creature(card, flank)
            card.Exhausted = True
            # Step 2 of the turn only flags cards already in play; anything
            # entering play later must be flagged too, or a card readied on
            # entry (Silvertooth) could never be used that turn.
            card.CanBeUsed = card.house == player.selected_house
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._play_resolution(card)
        elif card.type == CardType.ARTIFACT:
            player.play_area.add_artifact(card)
            card.Exhausted = True
            card.CanBeUsed = card.house == player.selected_house
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._play_resolution(card)
        elif card.type == CardType.UPGRADE:
            card.type_object.host = host
            host.type_object.upgrades.append(card)
            card.controller = host.controller
            if cdef.register_passive:
                cdef.register_passive(self, card)
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._run_play_trigger_check(card)
        else:  # Action
            if card.aember_on_play:
                player.aember += card.aember_on_play
            yield from self._play_resolution(card)
            if not self._card_is_somewhere(card):
                self.players[card.owner].discard.push(card)
        return True

    def _play_resolution(self, card: Card):
        """Resolve a played card's Play effect and its play-trigger check, in the
        tied order described in the plan (3.5)."""
        pre_existing = [
            t for t in self.active_effects.triggers_for("card_played") if t.controller == card.controller
        ]
        if pre_existing:
            order = yield from self.order_effects(
                card.controller, ["effect", "check"], f"{card.name}: choose what resolves first"
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
                event_player, triggers, f"Playing {card.name} triggered these: choose their order"
            )
        else:
            ordered = triggers
        for trig in ordered:
            yield from trig.handler(self, {"player": event_player, "card": card})

    def play_card_from_deck_top(self, player: Player, top_card: Card, ignore_house: bool = True):
        player.deck.remove(top_card)
        ok = yield from self._play_card(player.id, top_card, from_deck_top=True)
        if not ok:
            player.deck.put_on_top(top_card)

    # --------------------------------------------------- reap/fight/use ----

    def _use_action(self, pid: int, card: Card):
        player = self.players[pid]
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        self.log.add("use_action", player=pid, card=card.name, iid=card.instance_id)
        yield from card.card_def.on_action(self, card)

    def _use_omni(self, pid: int, card: Card):
        player = self.players[pid]
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        self.log.add("use_omni", player=pid, card=card.name, iid=card.instance_id)
        yield from card.card_def.on_omni(self, card)

    def _reap(self, pid: int, card: Card):
        player = self.players[pid]
        card.Exhausted = True
        player.used_this_turn[card.name] = player.used_this_turn.get(card.name, 0) + 1
        player.aember += 1
        self.log.add("reap", player=pid, card=card.name, iid=card.instance_id)
        cdef = card.card_def
        if cdef.on_reap is not None:
            yield from cdef.on_reap(self, card)
        for extra in list(card.extra_triggers.get("after_reap", [])):
            yield from extra(self, card)

    def _fight(self, pid: int, attacker: Card):
        player = self.players[pid]
        opponent = self.players[3 - pid]
        attacker.Exhausted = True
        player.used_this_turn[attacker.name] = player.used_this_turn.get(attacker.name, 0) + 1
        targets = list(opponent.play_area.creatures)
        if not targets:
            return
        choice = yield from self.choose_cards(pid, f"Choose a target for {attacker.name} to fight", targets, 1, 1)
        target = choice[0]
        self.log.add(
            "fight",
            attacker=attacker.name,
            attacker_iid=attacker.instance_id,
            target=target.name,
            target_iid=target.instance_id,
        )

        destroyed = yield from self.check_destroyed([attacker, target])
        if attacker in destroyed or target in destroyed:
            return

        skip_fight = target.Elusive and not target.fought_this_turn and not target.IgnoreElusive
        target.fought_this_turn = True
        if not skip_fight:
            if not attacker.Skirmish:
                steps.deal_damage(self, attacker, self.get_power(target))
            if not target.Skirmish:
                steps.deal_damage(self, target, self.get_power(attacker))
            destroyed = yield from self.check_destroyed([attacker, target])
            if attacker in destroyed:
                return

        cdef = attacker.card_def
        if cdef.on_fight is not None:
            yield from cdef.on_fight(self, attacker)
        for extra in list(attacker.extra_triggers.get("after_fight", [])):
            yield from extra(self, attacker)

    def use_creature_ability(self, card: Card):
        """Reap, fight, or use the Action of a friendly creature, whichever is
        possible (Dominator Bauble). Ignores house restrictions."""
        pid = card.controller
        player = self.players[pid]
        opponent = self.players[3 - pid]
        if card.Exhausted or not self._rule_of_six_ok(player, card.name):
            return
        options = ["reap"]
        if opponent.play_area.creatures:
            options.append("fight")
        if card.card_def.on_action is not None:
            options.append("action")
        if len(options) > 1:
            choice = yield from self.choose_cards(pid, f"Use {card.name}", options, 1, 1)
            kind = choice[0]
        else:
            kind = options[0]
        if kind == "reap":
            yield from self._reap(pid, card)
        elif kind == "fight":
            yield from self._fight(pid, card)
        elif kind == "action":
            yield from self._use_action(pid, card)

    # ------------------------------------------------------------ destroy ----

    def check_destroyed(self, cards: List[Card]):
        to_destroy = []
        for c in cards:
            if c.destroyed or not isinstance(c.type_object, CreatureType):
                continue
            if c not in self.players[c.controller].play_area.creatures:
                continue
            if c.type_object.damage >= self.get_power(c):
                to_destroy.append(c)
        destroyed = yield from self.destroy_cards(to_destroy)
        return destroyed

    def destroy_cards(self, cards: List[Card]):
        batch = []
        for c in cards:
            if c.destroyed:
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
        for c in batch:
            self._move_destroyed_card(c)
        return batch

    def _move_destroyed_card(self, card: Card):
        controller = self.players[card.controller]
        owner = self.players[card.owner]
        destination = card.destined_zone or "discard"
        if card in controller.play_area.creatures or card in controller.play_area.artifacts:
            controller.play_area.remove(card)
            self.leave_play(card)
        if destination == "hand":
            owner.hand.add(card)
        else:
            owner.discard.push(card)
        self.log.add("destroyed", card=card.name, iid=card.instance_id, destination=destination)

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
            card.aember_captured = 0
        card.reset_on_leave_play()

"""A simple, rules-aware opponent that plays like a reasonable human.

RandomBot picks End Turn as often as anything else and mulligans at random,
which makes for an opponent that doesn't feel like KeyForge at all. This
bot keeps things deliberately simple and readable:

- choose the house that lets it do the most this turn;
- play every card it sensibly can, use artifacts/actions, reap with ready
  creatures, fight only when the fight is favorable;
- end the turn only when nothing useful is left.

It only reads the redacted `PlayerView`, like any other Controller, and is
seeded so games stay reproducible.
"""

from __future__ import annotations

import random

from keyforge.actions import DiscardCard, EndTurn, Fight, PlayCard, Reap, UseAction, UseOmni
from keyforge.cards.card import Card
from keyforge.enums import Affects, CardType, DecisionIntent, DecisionKind, House

from .base import Controller

# Cards whose effect is usually bad to fire blindly.
_CAREFUL_PLAYS = {"Gateway to Dis", "Three Fates", "Pawn Sacrifice", "One Last Job"}
_NEVER_USE = {"Timetraveller", "Lifeward"}  # shuffles itself away / sacrifices itself


def _power(card: Card) -> int:
    return getattr(card.type_object, "base_power", 0)


def _remaining(card: Card) -> int:
    to = card.type_object
    return getattr(to, "base_power", 0) - getattr(to, "damage", 0)


def _keywords(card: Card) -> set:
    """Printed keywords plus anything granted by attached upgrades --
    mirrors `Game.get_keywords` without needing a `Game` reference. Misses
    keywords granted by another card's ModifierEffect (rare, and would need
    a Game to see) -- same documented gap as `_armor`/`_hazardous`/`_assault`
    below."""
    kw = set(card.keywords)
    for upg in getattr(card.type_object, "upgrades", []):
        kw |= set(upg.card_def.grants_keywords)
    return kw


def _armor(card: Card) -> int:
    """Mirrors `Game.get_armor` (own printed armor + upgrades), minus any
    cross-card ModifierEffect bonus, which needs a Game to see."""
    if card.armor_negated:
        return 0
    to = card.type_object
    bonus = sum(upg.card_def.armor_bonus for upg in getattr(to, "upgrades", []))
    return max(0, getattr(to, "base_armor", 0) + bonus)


def _armor_remaining(card: Card) -> int:
    return max(0, _armor(card) - getattr(card.type_object, "armor_used_this_turn", 0))


def _hazardous(card: Card) -> int:
    """Mirrors `Game.get_hazardous`: own printed value plus upgrades."""
    if not hasattr(card.type_object, "upgrades"):
        return 0
    return card.card_def.hazardous + sum(upg.card_def.hazardous for upg in card.type_object.upgrades)


def _assault(card: Card) -> int:
    """Mirrors `Game.get_assault`: own printed value plus upgrades."""
    if not hasattr(card.type_object, "upgrades"):
        return 0
    return card.card_def.assault + sum(upg.card_def.assault for upg in card.type_object.upgrades)


class HeuristicBot(Controller):
    def __init__(self, seed=None, bid_ceiling=4):
        self.rng = random.Random(seed)
        self.bid_ceiling = bid_ceiling  # chains this bot will offer at most, for a decisively-won deck

    # ------------------------------------------------------------ helpers ----

    def _house_score(self, view, house) -> float:
        # Omni abilities are usable no matter which house is active (that's
        # the point of Omni), so they add the same value to every house's
        # score and are deliberately left out here -- they'd never change
        # which house the argmax in `decide()` picks, only inflate every
        # score by the same constant.
        me = view.me()
        score = 0.0
        for c in me.hand or []:
            if c.house == house:
                score += 1.0
        for c in me.creatures:
            if c.house != house:
                continue
            # A creature that can't reap only gives up its Fight (already
            # house-independent via off-house-fight exceptions this bot
            # can't see anyway) for choosing this house, not the usual
            # reap value -- unless it also has an Action, which this house
            # choice does unlock.
            if c.card_def.cannot_reap and c.card_def.on_action is None:
                score += 0.5
            else:
                score += 1.5
            if c.card_def.on_action is not None:
                score += 0.5
        for c in me.artifacts:
            if c.house == house and c.card_def.on_action is not None:
                score += 0.75
        return score + self.rng.random() * 0.01

    def _is_good_play(self, view, card: Card) -> bool:
        me, opp = view.me(), view.opponent()
        name = card.name
        if name == "Gateway to Dis":
            return len(opp.creatures) >= len(me.creatures) + 2
        if name == "Three Fates":
            ranked = sorted(me.creatures + opp.creatures, key=_power, reverse=True)[:3]
            return sum(1 for c in ranked if c in opp.creatures) >= 2
        if name == "Pawn Sacrifice":
            return bool(me.creatures) and len(opp.creatures) >= 1
        if name == "One Last Job":
            shadows = [c for c in me.creatures if c.house == House.SHADOWS]
            return len(shadows) >= 2 and opp.aember >= len(shadows)
        if name == "Too Much to Protect":
            return opp.aember > 6
        if name in ("The Terror",):
            return True
        return True

    def _fight_value(self, attacker: Card, target: Card) -> float:
        """Positive when the fight is worth it. Walks the engine's own fight
        order (assault, then hazardous, then the main power exchange) so
        armor, assault, and hazardous all factor in, not just raw power."""
        if target.damage_prevented:
            return -1.0  # every point of this fight's own damage would be wasted

        kw_a, kw_t = _keywords(attacker), _keywords(target)
        attacker_hp = _remaining(attacker)
        target_hp = _remaining(target)
        target_armor = _armor_remaining(target)

        # Assault: the attacker's assault damage lands on the target first,
        # through its armor, before anything else -- can end the fight
        # outright with the attacker taking nothing back.
        assault = _assault(attacker)
        if assault:
            absorbed = min(target_armor, assault)
            target_armor -= absorbed
            target_hp -= assault - absorbed
        if target_hp <= 0:
            return 3.0 + _power(target) * 0.1 + target.aember_captured

        # Hazardous: the target's hazardous damage lands on the attacker
        # next, bypassing armor -- can kill the attacker before it ever
        # gets to deal its own damage.
        hazardous = _hazardous(target)
        if hazardous >= attacker_hp:
            return -2.0
        attacker_hp -= hazardous

        if "elusive" in kw_t and not target.fought_this_turn:
            # The main power trade is skipped entirely; assault/hazardous
            # above are the only damage this fight would ever deal.
            return -1.0 - hazardous * 0.3

        kills = _power(attacker) >= target_hp + target_armor
        counter = 0 if "skirmish" in kw_a else _power(target)
        survives = counter < attacker_hp
        if kills and survives:
            return 3.0 + _power(target) * 0.1 + target.aember_captured - hazardous * 0.5
        if kills and not survives:
            return (1.0 if _power(target) > _power(attacker) else -0.5) - hazardous * 0.3
        return -1.0 - hazardous * 0.3

    # ------------------------------------------------------------ decide ----

    def decide(self, view, decision):
        kind = decision.kind
        opts = list(decision.options)

        # Match-level decisions (Reversal/Adaptive) arrive between games,
        # when `view` is a MatchView with no board to read.
        if kind == DecisionKind.CHOOSE_FIRST_PLAYER:
            return "first"  # initiative is generally worth taking
        if kind == DecisionKind.BID_CHAINS:
            return self._decide_bid(view, opts)

        me, opp = view.me(), view.opponent()

        if kind == DecisionKind.MULLIGAN:
            hand = me.hand or []
            best = max((sum(1 for c in hand if c.house == h) for h in {c.house for c in hand}), default=0)
            return True if best < 2 and True in opts else False

        if kind == DecisionKind.CHOOSE_HOUSE:
            return max(opts, key=lambda h: self._house_score(view, h))

        if kind == DecisionKind.TAKE_ARCHIVE:
            return True

        if kind == DecisionKind.CHOOSE_FLANK:
            return "right" if "right" in opts else opts[0]

        if kind == DecisionKind.YES_NO:
            return True

        if kind == DecisionKind.ORDER_EFFECTS:
            return opts

        if kind == DecisionKind.CHOOSE_HOUSE_FOR_EFFECT:
            prompt = decision.prompt.lower()
            if "arise" in prompt:
                return max(opts, key=lambda h: sum(1 for c in me.discard if c.house == h and c.type == CardType.CREATURE))
            # Control the Weak: force the house they have the fewest creatures in.
            return min(opts, key=lambda h: sum(1 for c in opp.creatures if c.house == h))

        if kind == DecisionKind.CHOOSE_ACTION:
            return self._choose_action(view, opts)

        if kind == DecisionKind.CHOOSE_CARDS:
            return self._choose_cards(view, decision, opts)

        if kind == DecisionKind.CHOOSE_NUMBER:
            # Dance of Doom: prefer the power value that destroys the most
            # enemy creatures for the fewest friendly ones.
            def score(n):
                enemy = sum(1 for c in opp.creatures if _power(c) == n)
                friendly = sum(1 for c in me.creatures if _power(c) == n)
                return enemy - friendly
            return max(opts, key=score)

        if kind == DecisionKind.CHOOSE_MODE:
            return self._choose_mode(view, decision, opts)

        return self.rng.choice(opts)

    # -------------------------------------------------------- match play ----

    def _bid_ceiling(self, view) -> int:
        """Chains this bot is willing to offer for the deck that won both
        games, scaled down the more decisively it won (a deck that closed
        games out fast is worth less of a handicap to hand the opponent)."""
        games = [g for g in (getattr(view, "games", None) or []) if g.winner is not None]
        if not games:
            return self.bid_ceiling
        avg_turns = sum(g.turns for g in games) / len(games)
        scale = max(0.3, min(1.0, 1.4 - avg_turns / 20))
        return max(1, round(self.bid_ceiling * scale))

    def _decide_bid(self, view, opts):
        raises = sorted(o for o in opts if isinstance(o, int))
        if not raises or raises[0] > self._bid_ceiling(view):
            return "pass"
        return raises[0]

    def _choose_action(self, view, opts):
        me, opp = view.me(), view.opponent()
        # 1. Play cards: creatures and artifacts first (board presence), then actions.
        plays = [o for o in opts if isinstance(o, PlayCard) and self._is_good_play(view, o.card)]
        type_rank = {CardType.CREATURE: 0, CardType.ARTIFACT: 1, CardType.UPGRADE: 2, CardType.ACTION: 3}
        if plays:
            plays.sort(key=lambda o: (type_rank.get(o.card.type, 4), -o.card.aember_on_play))
            return plays[0]
        # 2. Use card abilities.
        uses = [
            o for o in opts
            if isinstance(o, (UseAction, UseOmni)) and o.card.name not in _NEVER_USE
        ]
        if uses:
            return uses[0]
        # 3. Favorable fights, then reaps.
        fights = [o for o in opts if isinstance(o, Fight)]
        best_fight, best_value = None, 0.0
        for f in fights:
            value = max((self._fight_value(f.card, t) for t in opp.creatures), default=-1.0)
            if value > best_value:
                best_fight, best_value = f, value
        if best_fight is not None:
            return best_fight
        reaps = [o for o in opts if isinstance(o, Reap)]
        if reaps:
            return reaps[0]
        # 4. Discard cards that couldn't be played anyway (thins the hand for the draw step).
        plays_possible = {id(o.card) for o in opts if isinstance(o, PlayCard)}
        discards = [o for o in opts if isinstance(o, DiscardCard) and id(o.card) not in plays_possible]
        if discards:
            return discards[0]
        return next(o for o in opts if isinstance(o, EndTurn))

    def _choose_cards(self, view, decision, opts):
        """Dispatches purely on `decision.intent`/`decision.source_card`/
        `decision.affects` (Agent Interface Plan, Milestone B) -- never on
        `decision.prompt`. The same `DecisionKind`+option-type pair means
        different things depending only on which card asked (Ozmo's target
        step is byte-identical whether the mode was "Heal 3" or "Stun"), and
        conditioning on prompt text is exactly the thing a learned agent
        can't do, so the oracle bot doesn't get to either."""
        cards = [o for o in opts if isinstance(o, Card)]
        if len(cards) != len(opts):
            return self._choose_non_card_options(view.opponent(), decision, opts)

        me = view.me()
        mine = set(id(c) for c in me.creatures + me.artifacts + (me.hand or []) + me.discard + (me.archive or []))
        intent = decision.intent
        if intent is None:
            raise ValueError(f"HeuristicBot: CHOOSE_CARDS decision has no intent tag ({decision.prompt!r})")
        handler_name = self._CARD_TARGET_HANDLERS.get(intent)
        if handler_name is None:
            raise ValueError(f"HeuristicBot: no CHOOSE_CARDS handler for intent {intent.name} ({decision.prompt!r})")
        return getattr(self, handler_name)(view, decision, cards, mine)

    def _choose_non_card_options(self, opp, decision, opts):
        """CHOOSE_CARDS decisions whose options aren't `Card`s at all: which
        discard pile to purge from (a player id), or which of a creature's
        abilities to use (mode strings, Dominator Bauble-style). Dispatched
        on option *shape*, which never needed prompt parsing."""
        if all(isinstance(o, int) and not isinstance(o, bool) for o in opts):
            return [opp.id] if opp.id in opts else [opts[0]]
        for preferred in ("fight", "reap", "action"):
            if preferred in opts:
                if preferred == "fight" and not opp.creatures:
                    continue
                return [preferred]
        return list(opts[: max(decision.min_n, 1)])

    def _take_n(self, decision, ranked):
        """Takes `decision.max_n` off an already-best-first-ranked list,
        clamped to what's available and to `min_n`."""
        n = max(decision.min_n, min(decision.max_n, len(ranked)))
        return list(ranked[:n])

    def _target_destroy(self, view, decision, cards, mine):
        # A forced self-destroy (bouncing_deathquark's friendly half, e.g.)
        # gives up the least; anything else takes the most dangerous target.
        reverse = decision.affects != Affects.FRIENDLY
        return self._take_n(decision, sorted(cards, key=_power, reverse=reverse))

    def _target_stun(self, view, decision, cards, mine):
        theirs = [c for c in cards if id(c) not in mine and not c.stunned]
        pool = theirs or [c for c in cards if not c.stunned] or cards
        return self._take_n(decision, sorted(pool, key=_power, reverse=True))

    def _target_sacrifice(self, view, decision, cards, mine):
        return self._take_n(decision, sorted(cards, key=_power))

    def _target_spare(self, view, decision, cards, mine):
        return [cards[0]]  # `choose_most_powerful` already narrowed this to an equal-power tie

    def _target_heal(self, view, decision, cards, mine):
        friendly = [c for c in cards if id(c) in mine]
        pool = friendly or cards
        return self._take_n(decision, sorted(pool, key=_remaining))

    def _target_ready(self, view, decision, cards, mine):
        # Anger, Ganger Chieftain, Gauntlet of Command, Sergeant Zakiel, One
        # Stood Against Many, Sanctum's ready-and-use: pick whichever
        # friendly creature makes the best fight against the current board.
        # Squawker/John Smyth/Commpod's plain readies (no attached fight)
        # fall back to the same ranking -- a creature worth a good fight is
        # also just a good creature to have ready.
        opp = view.opponent()
        if opp.creatures:
            ranked = sorted(
                cards, key=lambda c: max((self._fight_value(c, t) for t in opp.creatures), default=-1.0), reverse=True
            )
        else:
            ranked = sorted(cards, key=_power, reverse=True)
        return self._take_n(decision, ranked)

    def _target_capture(self, view, decision, cards, mine):
        # Captured Aember releases to the OPPONENT of whoever controls the
        # card when it leaves play (Game.leave_play) -- so capturing onto an
        # enemy creature pays off when it dies, and capturing onto your own
        # is a slow-motion gift to your opponent. Prefer the frailest enemy
        # (dies soonest, hands it back fastest) or the toughest friendly
        # (holds it captive longest) accordingly.
        if decision.affects == Affects.FRIENDLY:
            ranked = sorted(cards, key=_remaining, reverse=True)
        else:
            ranked = sorted(cards, key=_remaining)
        return self._take_n(decision, ranked)

    def _target_drain(self, view, decision, cards, mine):
        # Selwyn the Fence: drain the smaller stash, keep the bigger one
        # captured for later.
        ranked = sorted(cards, key=lambda c: c.aember_captured + c.aember_stored)
        return self._take_n(decision, ranked)

    def _target_discard(self, view, decision, cards, mine):
        house = view.me().selected_house
        ranked = sorted(cards, key=lambda c: (c.house == house, c.aember_on_play))
        return self._take_n(decision, ranked)

    def _target_archive(self, view, decision, cards, mine):
        if decision.affects == Affects.ENEMY:
            return self._take_n(decision, sorted(cards, key=_power, reverse=True))
        return self._target_discard(view, decision, cards, mine)

    def _target_return_to_hand(self, view, decision, cards, mine):
        if decision.affects == Affects.FRIENDLY:
            return self._take_n(decision, sorted(cards, key=_remaining))  # save the one closest to dying
        return self._take_n(decision, sorted(cards, key=_power, reverse=True))  # bounce their best

    def _target_shuffle_in(self, view, decision, cards, mine):
        me, opp = view.me(), view.opponent()
        live_ids = set(id(c) for c in me.creatures + opp.creatures)
        if all(id(c) in live_ids for c in cards):
            reverse = decision.affects != Affects.FRIENDLY  # pulling live creatures: keep mine, remove theirs
        else:
            reverse = True  # from a discard pile (World Tree): worth redrawing means worth keeping strong
        return self._take_n(decision, sorted(cards, key=_power, reverse=reverse))

    def _best_friendly_or_any(self, view, decision, cards, mine):
        friendly = [c for c in cards if id(c) in mine]
        pool = friendly or cards
        return [max(pool, key=_power)]

    _target_attach = _best_friendly_or_any
    _target_use_target = _best_friendly_or_any
    _target_modify = _best_friendly_or_any

    def _target_play(self, view, decision, cards, mine):
        return [max(cards, key=_power)]

    _target_copy = _target_play
    _target_swap = _target_play

    def _target_fight_target(self, view, decision, cards, mine):
        attacker = decision.source_card
        return [max(cards, key=lambda t: self._fight_value(attacker, t))]

    def _target_take_control(self, view, decision, cards, mine):
        return self._take_n(decision, sorted(cards, key=_power, reverse=True))

    def _target_reveal(self, view, decision, cards, mine):
        if decision.max_n > 1:
            return list(cards[: min(decision.max_n, len(cards))])
        # A single mandatory reveal that also archives the card (Incubation
        # Chamber, Zyzzix the Many): treat it like a discard.
        return self._target_discard(view, decision, cards, mine)[:1]

    _CARD_TARGET_HANDLERS = {
        DecisionIntent.DESTROY: "_target_destroy",
        DecisionIntent.DAMAGE: "_target_destroy",
        DecisionIntent.PURGE: "_target_destroy",
        DecisionIntent.REDIRECT: "_target_destroy",
        DecisionIntent.STUN: "_target_stun",
        DecisionIntent.EXHAUST: "_target_stun",
        DecisionIntent.SACRIFICE: "_target_sacrifice",
        DecisionIntent.SPARE: "_target_spare",
        DecisionIntent.HEAL: "_target_heal",
        DecisionIntent.READY: "_target_ready",
        DecisionIntent.CAPTURE: "_target_capture",
        DecisionIntent.DRAIN: "_target_drain",
        DecisionIntent.DISCARD: "_target_discard",
        DecisionIntent.ARCHIVE: "_target_archive",
        DecisionIntent.RETURN_TO_HAND: "_target_return_to_hand",
        DecisionIntent.SHUFFLE_IN: "_target_shuffle_in",
        DecisionIntent.ATTACH: "_target_attach",
        DecisionIntent.PLAY: "_target_play",
        DecisionIntent.FIGHT_TARGET: "_target_fight_target",
        DecisionIntent.TAKE_CONTROL: "_target_take_control",
        DecisionIntent.USE_TARGET: "_target_use_target",
        DecisionIntent.REVEAL: "_target_reveal",
        DecisionIntent.COPY: "_target_copy",
        DecisionIntent.SWAP: "_target_swap",
        DecisionIntent.MODIFY: "_target_modify",
    }

    def _choose_mode(self, view, decision, opts):
        me, opp = view.me(), view.opponent()
        name = decision.source_card.name if decision.source_card is not None else None
        if name == "Begone!":
            mode = "Destroy each Dis creature" if any(c.house == House.DIS for c in opp.creatures) else "Gain 1Æ"
        elif name == "Knowledge is Power":
            archived = len(me.archive or [])
            mode = "Gain 1Æ per archived card" if (not (me.hand or []) or archived >= 2) else "Archive a card"
        elif name == "Ozmo, Martianologist":
            mars_hurt = [c for c in me.creatures if c.house == House.MARS and _remaining(c) < _power(c)]
            mars_enemy = [c for c in opp.creatures if c.house == House.MARS]
            if mars_hurt:
                mode = "Heal 3"
            elif mars_enemy:
                mode = "Stun"
            else:
                mode = "Heal 3"  # neither is useful; healing a full-health ally is at least harmless
        elif name == "Squawker":
            mode = "Ready a Mars creature" if "Ready a Mars creature" in opts else opts[0]
        else:
            mode = opts[0]  # deterministic default beats an RNG coin flip here
        return mode

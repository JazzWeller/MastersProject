"""Agent Observation Plan, Part R (R0, R5): coverage-directed scenarios.

The 20,000-game fuzz reaches 404 of the reference engine's 413 suspension
points (tools/ref_coverage.py). Each point it misses gets a constructed
position here that reaches it, played in lockstep by the compiled engine and
the frozen reference (tools/diff_engines.py), or a written reason it can't
be reached:

- `Game._resume` (4 points): only `Game.copy()` of a native game at a
  boundary decision enters it; the fuzz never copies. Covered by
  tests/test_agent_interface_milestone_e.py's copy tests; compiled execution
  never uses it (a compiled copy keeps its frames).
- `tireless_crocag_register.handler`: Tireless Crocag destroys itself when
  the opponent's last creature is destroyed.
- `ozmo` (2 points): Ozmo's mode and target choices, with a Mars creature in
  play (unreachable before the 2026-10-05 rules fix).
- `epic_quest_omni` (2 points): Epic Quest's Omni after 7 Sanctum cards
  played in one turn.
"""

import unittest

from keyforge.actions import Fight, PlayCard, Reap, UseOmni
from keyforge.cards.card_data import CARD_DEFS
from keyforge.enums import CardType, DecisionKind, House
from tools import diff_engines as de


def _deck(name, required):
    """A 3-house deck holding every card in `required` (house -> names),
    padded with each house's first cards in the pool."""
    pods = {}
    for house, names in required.items():
        pool = [n for n, d in sorted(CARD_DEFS.items()) if d.house == house and n not in names]
        pods[house.value] = list(names) + pool[: 12 - len(names)]
    return {"name": name, "pods": pods}


def _play(spec, prefer, *, until, limit=400):
    """Plays `spec` with `prefer(game, decision) -> choice or None` (falling
    back to HeuristicBot) in lockstep, compiled against the reference, the
    whole state compared after every decision, until `until(game)`."""
    a, b = de.Engine("keyforge@compiled"), de.Engine("keyforge_ref")
    ls = de.Lockstep(a, b, spec)
    bots = de._bots("heuristic", spec["seed"])
    while not ls.ga.is_over and ls.n < limit and not until(ls.ga):
        d = ls.ga.pending_decision
        choice = prefer(ls.ga, d)
        if choice is None:
            choice = bots[d.player].decide(ls.ga.view_for(d.player), d)
        ls.step(choice, full=True)
    return ls.ga


def _logged(game, kind, **match):
    return any(e.kind == kind and all(e.data.get(k) == v for k, v in match.items()) for e in game.log.events)


def _first(options, test):
    return next((o for o in options if test(o)), None)


class TestCoverageScenarios(unittest.TestCase):
    def test_tireless_crocag_destroys_itself_when_the_enemy_has_no_creatures(self):
        spec = {"name": "crocag", "format": "game", "seed": 11, "max_turns": 6, "first_player": 1,
                "decks": [_deck("C1", {House.BROBNAR: ["Tireless Crocag"], House.LOGOS: [], House.DIS: []}),
                          _deck("C2", {House.LOGOS: ["Batdrone"], House.DIS: [], House.SHADOWS: []})],
                "setup_script": [("put_creature", 1, "Tireless Crocag", "left"), ("put_creature", 2, "Batdrone", "left")]}

        def prefer(game, d):
            if d.player != 1:
                return None
            if d.kind == DecisionKind.CHOOSE_ACTION:
                return _first(d.options, lambda o: isinstance(o, Fight) and o.card.name == "Tireless Crocag")
            if d.kind == DecisionKind.CHOOSE_CARDS and d.options and getattr(d.options[0], "name", None) == "Batdrone":
                return [d.options[0]]
            return None

        g = _play(spec, prefer, until=lambda g: _logged(g, "destroyed", card="Tireless Crocag"))
        self.assertTrue(_logged(g, "destroyed", card="Batdrone"))
        self.assertTrue(_logged(g, "destroyed", card="Tireless Crocag"))

    def test_ozmo_heals_a_mars_creature(self):
        spec = {"name": "ozmo", "format": "game", "seed": 12, "max_turns": 6, "first_player": 1,
                "decks": [_deck("O1", {House.LOGOS: ["Ozmo, Martianologist"], House.MARS: ["Zorg"], House.DIS: []}),
                          _deck("O2", {House.LOGOS: [], House.DIS: [], House.SHADOWS: []})],
                "setup_script": [("put_creature", 1, "Ozmo, Martianologist", "left"), ("put_creature", 1, "Zorg", "right"),
                                 ("damage", 1, "Zorg", 4)]}

        def prefer(game, d):
            if d.player != 1:
                return None
            if d.kind == DecisionKind.CHOOSE_HOUSE and House.LOGOS in d.options:
                return House.LOGOS
            if d.kind == DecisionKind.CHOOSE_ACTION:
                return _first(d.options, lambda o: isinstance(o, Reap) and o.card.name.startswith("Ozmo"))
            if d.kind == DecisionKind.CHOOSE_MODE and "Heal 3" in d.options:
                return "Heal 3"
            if d.kind == DecisionKind.CHOOSE_CARDS and d.source_card is not None and d.source_card.name.startswith("Ozmo"):
                return [_first(d.options, lambda o: o.name == "Zorg")]
            return None

        g = _play(spec, prefer, until=lambda g: _logged(g, "heal", card="Zorg"))
        self.assertTrue(_logged(g, "heal", card="Zorg", amount=3))

    def test_epic_quest_forges_after_seven_sanctum_cards(self):
        actions = ["Blinding Light", "Charge!", "Cleansing Wave", "Clear Mind", "Glorious Few", "Honorable Claim",
                   "Shield of Justice"]
        for n in actions:
            self.assertEqual((CARD_DEFS[n].house, CARD_DEFS[n].type), (House.SANCTUM, CardType.ACTION), n)
        spec = {"name": "epic", "format": "game", "seed": 13, "max_turns": 6, "first_player": 2,
                "decks": [_deck("E1", {House.SANCTUM: ["Epic Quest"] + actions, House.LOGOS: [], House.DIS: []}),
                          _deck("E2", {House.LOGOS: [], House.DIS: [], House.SHADOWS: []})],
                "setup_script": [("put_artifact", 1, "Epic Quest")] + [("hand_card", 1, n) for n in actions]}

        def prefer(game, d):
            if d.player != 1:
                return None
            if d.kind == DecisionKind.CHOOSE_HOUSE and House.SANCTUM in d.options:
                return House.SANCTUM
            if d.kind == DecisionKind.CHOOSE_ACTION:
                play = _first(d.options, lambda o: isinstance(o, PlayCard) and o.card.name in actions)
                if play is not None:
                    return play
                return _first(d.options, lambda o: isinstance(o, UseOmni) and o.card.name == "Epic Quest")
            return None

        g = _play(spec, prefer, until=lambda g: _logged(g, "forge_key", source="Epic Quest"))
        self.assertTrue(_logged(g, "forge_key", source="Epic Quest"))


if __name__ == "__main__":
    unittest.main()

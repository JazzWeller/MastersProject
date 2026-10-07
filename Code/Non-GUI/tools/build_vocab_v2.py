"""Builds (and extends, append-only) the v2 vocabularies in `agent/vocab/`
(Agent Observation Plan, Milestone O5).

    python -m tools.build_vocab_v2 [--fuzz 200] [--check]

Each vocabulary is harvested from the source of truth: the card data
(traits, keywords), the engine's source by AST (mode names, trigger
events, effect variables and kinds, destined zones, journal ops, the calls
card effects make), the engine's registries (log event kinds, cleanup
operations, effect ops) and the compiled resolution stack (routines,
sites, local roles, operations). `--fuzz N` also plays N games and adds
any value met at run time (a mode name built at run time, an event raised
dynamically). Existing entries keep their ids; new ones are appended in
sorted order. `--check` builds without writing and fails if anything
would be added (the test uses it).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Dict, Iterable, List, Set

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KEYFORGE = os.path.join(_ROOT, "keyforge")

ABILITY_KINDS = ("turn", "kernel", "effect", "play", "reap", "fight", "action", "omni", "destroyed",
                 "before_fight", "destroyed_fighting", "after_reap", "after_fight", "trigger")
ZONE_KINDS = ("hand", "deck", "discard", "archive", "purged", "battleline", "artifacts", "attached", "under",
              "limbo", "setup")
OPTION_FORMS = ("card", "end_turn", "trigger", "house", "bool", "value", "seq", "opaque", "stop")


def _sources(sub: str = "") -> Iterable[str]:
    base = os.path.join(_KEYFORGE, sub)
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("compiled", "__pycache__")]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _trees(sub: str = ""):
    for p in _sources(sub):
        with open(p, encoding="utf-8") as f:
            yield p, ast.parse(f.read())


def _const_strings(node) -> List[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _call_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _arg(call: ast.Call, index: int, keyword: str):
    if len(call.args) > index:
        return call.args[index]
    for k in call.keywords:
        if k.arg == keyword:
            return k.value
    return None


def harvest_static() -> Dict[str, Set[str]]:
    from agent import spec
    from keyforge import vm
    from keyforge.cards.card_data import CARD_DEFS
    from keyforge.effects import effect_object
    from keyforge.infoset import VERB_NAMES
    from keyforge.log import EVENT_SCHEMAS
    from keyforge.resolution import _role
    import keyforge.game  # noqa: F401 -- registers every card's cleanup operations

    out: Dict[str, Set[str]] = {n: set() for n in (
        "traits", "keywords", "modes", "events", "effect_variables", "effect_kinds", "effect_ops", "log_kinds",
        "ability_kinds", "cleanup_ops", "zone_kinds", "routines", "sites", "roles", "op_names", "journal_ops",
        "destined_zones", "verbs", "steps", "option_tags")}
    for cdef in CARD_DEFS.values():
        out["traits"].update(cdef.tags)
        out["keywords"].update(cdef.keywords)
        out["keywords"].update(cdef.grants_keywords)
    out["keywords"].update(spec.KEYWORDS)
    out["effect_variables"].update(spec.EFFECT_VARIABLES)
    out["effect_ops"].update(effect_object._OPS)
    out["log_kinds"].update(EVENT_SCHEMAS)
    out["ability_kinds"].update(ABILITY_KINDS)
    out["cleanup_ops"].update(effect_object._CLEANUP_OPERATIONS)
    out["zone_kinds"].update(ZONE_KINDS)
    out["verbs"].update(VERB_NAMES)
    out["verbs"].update(OPTION_FORMS)
    out["journal_ops"].update(("deal", "shuffle", "reveal", "search_reveal", "reveal_hand", "unreveal_hand",
                               "swap", "attach", "detach", "put_under", "take_under"))

    leave_ops, enter_ops = set(), set()
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name == "choose_mode":
                    a = _arg(node, 2, "modes")
                    out["modes"].update(_const_strings(a) if a is not None else [])
                elif name == "TriggerEffect":
                    a = _arg(node, 2, "event")
                    out["events"].update(_const_strings(a) if a is not None else [])
                elif name == "_fire_event":
                    a = _arg(node, 0, "event_name")
                    out["events"].update(_const_strings(a) if a is not None else [])
                elif name in ("InsteadEffect", "ModifierEffect"):
                    a = _arg(node, 2, "kind")
                    out["effect_kinds"].update(_const_strings(a) if a is not None else [])
                elif name in ("triggers_for", "insteads_for", "modifiers_for"):
                    a = _arg(node, 0, "event")
                    target = "events" if name == "triggers_for" else "effect_kinds"
                    out[target].update(_const_strings(a) if a is not None else [])
                elif name == "DurationEffect":
                    a = _arg(node, 4, "variable")
                    out["effect_variables"].update(_const_strings(a) if a is not None else [])
                    a = _arg(node, 5, "op")
                    out["effect_ops"].update(_const_strings(a) if a is not None else [])
                elif name in ("leave", "enter", "note", "reveal") and isinstance(node.func, ast.Attribute):
                    a = _arg(node, 2, "op") if name != "reveal" else _arg(node, 1, "op")
                    ops = _const_strings(a) if a is not None else []
                    (leave_ops if name == "leave" else enter_ops if name == "enter" else out["journal_ops"]).update(ops)
                elif name in ("get", "setdefault") and isinstance(node.func, ast.Attribute) \
                        and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "extra_triggers":
                    out["events"].update(_const_strings(node.args[0]) if node.args else [])
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == "destined_zone":
                        out["destined_zones"].update(_const_strings(node.value))
    # the zone methods' own ops (zones.py passes them as literals)
    for path, tree in _trees():
        if not path.endswith("zones.py"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in ("leave", "enter"):
                a = _arg(node, 2, "op")
                ops = _const_strings(a) if a is not None else []
                (leave_ops if _call_name(node) == "leave" else enter_ops).update(ops)
            if isinstance(node, ast.FunctionDef):
                for arg, default in zip(reversed(node.args.args), reversed(node.args.defaults)):
                    if arg.arg == "op":
                        enter_ops.update(_const_strings(default))
    enter_ops.add("deal")
    out["journal_ops"].update(leave_ops | enter_ops)
    out["journal_ops"].update(f"{a}/{b}" for a in leave_ops for b in enter_ops)
    # what card effects call: steps.* and game methods (engine signatures)
    for path, tree in _trees("effects"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in ("steps", "game", "generic"):
                out["steps"].add(f"{node.func.value.id}.{node.func.attr}")
    vm.load_compiled()
    for r in vm.ROUTINES.values():
        out["routines"].add(r.qualname)
        for pc in r.site_of_pc:
            out["sites"].add(f"{r.qualname}:{pc}")
        for slot in r.local_names:
            role = _role(slot)
            if role is not None:
                out["roles"].add(role)
        for pc, ops in r.ops_of_pc.items():
            for op in ops:
                out["op_names"].add(op_name(op))
    out["sites"].add("<native>")
    out["routines"].add("<native>")
    return out


from agent.vocab import op_name  # noqa: E402


def harvest_fuzz(n_games: int) -> Dict[str, Set[str]]:
    """Values met at run time, over `n_games` fuzz games of every pool."""
    from keyforge.decision import Decision
    from keyforge.enums import DecisionKind
    from keyforge.game import Game
    from tools.o1_acceptance import POOLS, bots_for, game_config

    from keyforge.infoset import VERB, _option_entry

    out: Dict[str, Set[str]] = {n: set() for n in ("modes", "events", "effect_variables", "effect_kinds",
                                                   "journal_ops", "destined_zones", "cleanup_ops", "log_kinds",
                                                   "option_tags", "effect_ops")}
    per_pool = max(1, n_games // len(POOLS))
    for pool in POOLS:
        for i in range(per_pool):
            config = game_config(pool, i)
            game = Game(config)
            bots = bots_for(i, config.seed)
            while not game.is_over:
                d = game.pending_decision
                if d.kind == DecisionKind.CHOOSE_MODE:
                    out["modes"].update(o for o in d.options if isinstance(o, str))
                for o in d.options:
                    try:
                        verb, _, payload = _option_entry(d, o, {})
                    except ValueError:
                        continue
                    if verb == VERB["trigger"] and isinstance(payload, str):
                        out["option_tags"].add(payload)
                    elif verb == VERB["mode"] and isinstance(payload, str):
                        out["modes"].add(payload)
                ae = game.active_effects
                out["events"].update(e.event for e in ae.trigger_effects)
                out["effect_variables"].update(e.variable for e in ae.duration_effects)
                out["effect_ops"].update(e.op for e in ae.duration_effects)
                out["effect_kinds"].update(e.kind for e in ae.instead_effects + ae.modifier_effects)
                out["cleanup_ops"].update(op for op, _ in game._end_of_turn_cleanups)
                for c in game._cards_by_id.values():
                    if c.destined_zone is not None:
                        out["destined_zones"].add(c.destined_zone)
                    out["events"].update(k for k, v in (c.extra_triggers or {}).items() if v)
                game.submit(bots[d.player].decide(game.view_for(d.player), d))
            out["journal_ops"].update(e[7] for e in game.journal.entries)
            out["log_kinds"].update(e.kind for e in game.log.events)
    return out


def build(fuzz: int = 0) -> Dict[str, List[str]]:
    from agent import vocab

    found = harvest_static()
    if fuzz:
        for k, v in harvest_fuzz(fuzz).items():
            found[k].update(v)
    result = {}
    for name in vocab.NAMES:
        existing = []
        p = vocab.path(name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                existing = json.load(f)["entries"]
        new = sorted(set(found.get(name, ())) - set(existing))
        result[name] = existing + new
    return result


def main(argv=None) -> int:
    from agent import vocab

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    built = build(args.fuzz)
    added = {}
    for name, entries in built.items():
        p = vocab.path(name)
        old = []
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                old = json.load(f)["entries"]
        if entries != old:
            added[name] = entries[len(old):]
            if not args.check:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"name": name, "version": 1, "entries": entries}, f, indent=1, ensure_ascii=False)
                    f.write("\n")
    for name, new in added.items():
        print(f"{name}: +{len(new)}" + (f" ({', '.join(new[:8])}{'...' if len(new) > 8 else ''})" if new else ""))
    if args.check and added:
        return 1
    print("vocabularies:", ", ".join(f"{n} {len(e)}" for n, e in built.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

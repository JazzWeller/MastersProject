"""Journal completeness (Agent Observation Plan O1): where the journal says
each card is must be where it is.

`journal_locations` folds the journal into instance id -> zone;
`actual_locations` reads the zones themselves. A card between zones (an
action being played) is in limbo in both.
"""

from __future__ import annotations

from typing import Dict, List

from keyforge.journal import LIMBO


def journal_locations(journal) -> Dict[int, tuple]:
    where: Dict[int, tuple] = {}
    for e in journal:
        iid, frm, to = e[2], e[5], e[6]
        if iid is None or frm == to:
            continue
        if frm != LIMBO and frm[0] != "setup" and where.get(iid, LIMBO) != frm:
            raise AssertionError(f"entry {e}: card {iid} left {frm} but the journal has it in {where.get(iid)}")
        where[iid] = to
    return where


def actual_locations(game) -> Dict[int, tuple]:
    where: Dict[int, tuple] = {}

    def put(card, zone):
        if card.instance_id in where:
            raise AssertionError(f"card {card.instance_id} is in {where[card.instance_id]} and {zone}")
        where[card.instance_id] = zone

    for pid, p in game.players.items():
        for kind in ("deck", "hand", "discard", "archive", "purged"):
            for c in getattr(p, kind).cards():
                put(c, (kind, pid))
        for c in p.play_area.creatures:
            put(c, ("battleline", pid))
        for c in p.play_area.artifacts:
            put(c, ("artifacts", pid))
    # every card, not just those in play: a host between zones (being
    # destroyed) still holds its upgrades
    for host in game._cards_by_id.values():
        for u in getattr(host.type_object, "upgrades", None) or ():
            put(u, ("attached", host.instance_id))
        for u in host.under_cards:
            put(u, ("under", host.instance_id))
    for iid in game._cards_by_id:
        where.setdefault(iid, LIMBO)
    return where


def mismatches(game) -> List[str]:
    j = journal_locations(game.journal)
    a = actual_locations(game)
    out = []
    for iid in sorted(set(j) | set(a)):
        if j.get(iid, LIMBO) != a.get(iid, LIMBO):
            out.append(f"card {iid} ({game._cards_by_id[iid].card_def.name}): journal {j.get(iid, LIMBO)}, actually {a.get(iid, LIMBO)}")
    return out

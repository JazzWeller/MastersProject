"""Deterministic, portable, per-event-keyed randomness.

See Code/AGENT_INTERFACE_PLAN.md, Milestone A ("determinism and identity").
Two separate problems live here:

- **Portability.** The stdlib only guarantees that `Random.random()`
  produces the same sequence for a given seed across CPython versions and
  implementations. `shuffle()`/`choice()`/`randrange()` are built on top of
  it, but nothing guarantees THEIR algorithms stay the same release to
  release -- only `random()` itself is a documented, stable contract. So
  every shuffle and pick the engine makes is hand-rolled on `random()`
  alone here, which is what lets a replay record stay valid across
  interpreters (PyPy, a future CPython) rather than depending on a
  shuffle algorithm nobody promised to keep.
- **Isolation.** A single shared RNG stream (the old `game.rng`) means any
  divergence in play -- an extra mulligan, an extra card-effect shuffle --
  shifts every later random draw for BOTH players, for the rest of the
  game. Keying each event by `(seed, player, event kind, a per-key
  counter)` keeps two games that have played identically so far dealt
  identically going forward, regardless of what else happened -- which is
  what makes paired-seed and duplicate evaluation (Milestone I) low
  variance.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, TypeVar

T = TypeVar("T")


def portable_shuffle(rng: random.Random, items: List) -> None:
    """In-place Fisher-Yates, built only on `rng.random()`."""
    for i in range(len(items) - 1, 0, -1):
        j = min(i, int(rng.random() * (i + 1)))
        items[i], items[j] = items[j], items[i]


def portable_choice(rng: random.Random, items: Sequence[T]) -> T:
    """Picks uniformly among `items`, built only on `rng.random()`."""
    if not items:
        raise IndexError("choice from an empty sequence")
    idx = min(len(items) - 1, int(rng.random() * len(items)))
    return items[idx]


def derive_rng(seed, player: Optional[int], kind: str, counter: int) -> random.Random:
    """A fresh, independent `random.Random` for one `(seed, player, kind,
    counter)` instance. Seeded from a plain `str` -- `random.Random` hashes
    a string seed via SHA-512 internally (a stable, documented algorithm,
    unlike `shuffle`/`choice` themselves), so this needs no hashing of its
    own and stays reproducible across interpreters."""
    return random.Random(f"{seed!r}|{player!r}|{kind}|{counter}")

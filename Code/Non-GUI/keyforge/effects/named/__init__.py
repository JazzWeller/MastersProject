"""Bespoke per-card effects that don't fit the generic factories, split by
house so no single file grows unbounded across the full 159-card pool.
Re-exported here so existing callers can keep writing `named.<fn>` regardless
of which house file actually defines it."""

from __future__ import annotations

from .dis import *  # noqa: F401,F403
from .logos import *  # noqa: F401,F403
from .shadows import *  # noqa: F401,F403
from . import dis, logos, shadows  # noqa: F401

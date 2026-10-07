"""The v1 feature encoding (Agent Training Plan, Milestone M1), frozen in
`agent/features_v1.py` (Agent Observation Plan O5, I7); the v2 encoding is
`agent/features_v2.py`. This module keeps every v1 import working."""

from .features_v1 import *  # noqa: F401,F403
from .features_v1 import (  # noqa: F401 -- names without a public alias that callers use
    Encoded,
    encode,
    stable_bucket,
    static_row,
    static_table,
    _VOCAB_IDS,
)

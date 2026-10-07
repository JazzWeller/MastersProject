"""Builds `agent/vocab/text_embedding.json`, the optional third form of
static card meaning (Agent Observation Plan, Milestone O5): a pretrained
sentence encoder's embedding of every card's text.

    python -m tools.build_text_embedding [--model all-MiniLM-L6-v2]

Needs `sentence-transformers` (WSL, `~/torchenv`) and downloads the model
on first use. Optional: `agent/static_v2.text_embedding()` returns None
and the network leaves the input out when the table doesn't exist.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.static_v2 import _EMBED_PATH
from keyforge.cards.card_data import CARD_DEFS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args(argv)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("needs sentence-transformers (pip install sentence-transformers)", file=sys.stderr)
        return 1
    model = SentenceTransformer(args.model)
    names = sorted(CARD_DEFS)
    texts = [f"{n}. {CARD_DEFS[n].text or ''}" for n in names]
    vectors = model.encode(texts, normalize_embeddings=True)
    out = {"model": args.model, "dim": int(vectors.shape[1]),
           "embeddings": {n: [round(float(x), 6) for x in v] for n, v in zip(names, vectors)}}
    with open(_EMBED_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"{len(names)} cards, dim {out['dim']} -> {_EMBED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

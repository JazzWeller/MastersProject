# Assets

`kf-bc-checkpoints.tar.gz` holds the two Tier 0b behaviour-cloning runs that Tier 2 self-play warm-starts from (`run.py tier2-v2` and `tier2-v1`):

- `runs/tier0b2-a7`: network v2. This is the Observation Plan's chosen configuration, A7.
- `runs/tier0b-a0`: network v1. It was trained on the same games as A7.

Each run includes its `bc_report.json`, `config.json` and checkpoint. Unpack it into the data root:

    tar xzf assets/kf-bc-checkpoints.tar.gz -C $KEYFORGE_DATA

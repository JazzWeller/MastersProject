# Hyperparameters of the main runs

| setting | m9-wt-1 | m9-fg-1 | m11-v1 | real-v1 |
|---|---|---|---|---|
| search.regime | within_turn | full_game | within_turn | within_turn |
| search.simulations | 100 | 100 | 100 | 100 |
| search.c_puct | 1.5 | 1.5 | 1.5 | 1.5 |
| search.dirichlet_alpha | 0.8 | 0.8 | 0.8 | 0.8 |
| search.dirichlet_eps | 0.25 | 0.25 | 0.25 | 0.25 |
| search.determinization | chance_exact | chance_exact | chance_exact | chance_exact |
| search.resample | all | all | all | all |
| search.leaf | student | student | student | student |
| search.rollout | heuristic | heuristic | heuristic | heuristic |
| search.temperature_high | 1.0 | 1.0 | 1.0 | 1.0 |
| search.temperature_low | 0.1 | 0.1 | 0.1 | 0.1 |
| search.temperature_moves | 15 | 15 | 15 | 15 |
| search.leaves_in_flight | 8 | 8 | 8 | 8 |
| search.virtual_loss | 1 | 1 | 1 | 1 |
| selfplay.games | 50000 | 50000 | 50000 | 50000 |
| selfplay.workers | 28 | 28 | 28 | 28 |
| selfplay.games_per_worker | 16 | 16 | 16 | 16 |
| selfplay.batch | 1024 | 1024 | 1024 | 1024 |
| selfplay.lr | 0.002 | 0.002 | 0.002 | 0.002 |
| selfplay.lr_min | 1e-05 | 1e-05 | 1e-05 | 1e-05 |
| selfplay.weight_decay | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| selfplay.grad_clip | 1.0 | 1.0 | 1.0 | 1.0 |
| selfplay.buffer_positions | 1000000 | 1000000 | 1000000 | 1000000 |
| selfplay.sample_reuse | 1 | 1 | 1 | 1 |
| selfplay.policy_reuse | 1 | 1 | 1 | 1 |
| selfplay.value_reuse | 1 | 1 | 1 | 1 |
| selfplay.playout_cap_full | 100 | 100 | 100 | 100 |
| selfplay.playout_cap_small | 25 | 25 | 25 | 25 |
| selfplay.playout_cap_full_fraction | 0.25 | 0.25 | 0.25 | 0.25 |
| selfplay.gate_every_steps | 1000 | 1000 | 1000 | 1000 |
| selfplay.gate_games | 400 | 400 | 400 | 400 |
| selfplay.gate_sprt | [0, 15] | [0, 15] | [0, 15] | [0, 15] |
| selfplay.gate_fallback_z | 1.0 | 1.0 | 1.0 | 1.0 |
| selfplay.precision | bf16 | bf16 | bf16 | bf16 |
| selfplay.resign_consecutive | 3 | 3 | 3 | 3 |
| selfplay.loss_weights | {"belief": 0.25, "distill": 0.0, "oracle": 0.25, "policy": 1.0, "value": 1.0} | {"belief": 0.25, "distill": 0.0, "oracle": 0.25, "policy": 1.0, "value": 1.0} | {"belief": 0.25, "distill": 0.0, "oracle": 0.25, "policy": 1.0, "value": 1.0} | {"belief": 0.25, "distill": 0.0, "oracle": 0.25, "policy": 1.0, "value": 1.0} |
| network.d_model | 128 | 128 | 128 | 128 |
| network.layers | 4 | 4 | 4 | 4 |
| network.heads | 4 | 4 | 4 | 4 |
| network.ff | 256 | 256 | 256 | 256 |
| network.card_embed | 64 | 64 | 64 | 64 |
| network.policy_head | pointer | pointer | pointer | pointer |
| network.multi_select | enumerate | enumerate | enumerate | enumerate |
| pool.deck_source | presets | presets | random | list |
| pool.deck_list |  |  |  | phase3 |
| pool.split |  |  |  | train |
| pool.decks | ["fignor", "igor"] | ["fignor", "igor"] | ["fignor", "igor"] | ["fignor", "igor"] |
| pool.format | archon | archon | archon | archon |
| pool.max_turns | 200 | 200 | 200 | 200 |
| dmc.epsilon_start | 0.3 | 0.3 | 0.3 | 0.3 |
| dmc.epsilon_end | 0.01 | 0.01 | 0.01 | 0.01 |
| dmc.epsilon_decay_games | 20000 | 20000 | 20000 | 20000 |
| seed | 1 | 1 | 0 | 0 |

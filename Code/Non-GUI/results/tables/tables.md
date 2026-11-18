## Table 1. Strength by deck set

Elo, HeuristicBot = 0, 100 simulations. Compare within a column: each column is its own ladder. Random decks: 32 held-out pairs, 256 games a match; otherwise 400.

| Agent | Trained by | Inputs | Fixed decks | Random decks | Real decks (test) | Real decks (train) |
|---|---|---|---|---|---|---|
| Imitation (compact) | Imitating HeuristicBot | Compact | 195 ± 12 | 1 ± 13 | 30 ± 10 | 29 ± 11 |
| Imitation (full) | Imitating HeuristicBot | Full | 156 ± 12 | -34 ± 13 | -16 ± 11 | -28 ± 11 |
| Specialist (compact) | Self-play, fixed decks | Compact | 665 ± 12 | 285 ± 13 | 317 ± 11 | 345 ± 11 |
| Specialist (full) | Self-play, fixed decks | Full | 641 ± 12 | 254 ± 13 | 310 ± 11 | 347 ± 11 |
| Generalist (compact) | Self-play, random decks | Compact | 580 ± 12 | 374 ± 13 | 386 ± 11 | 422 ± 11 |
| Generalist (full) | Self-play, random decks | Full | 545 ± 12 | 340 ± 13 | 372 ± 11 | 411 ± 11 |
| Real-deck agent | Self-play, real decks (train) | Compact | 596 ± 12 | 362 ± 13 | 404 ± 11 | 440 ± 11 |
| Real-deck agent (full) | Self-play, real decks (train) | Full | 560 ± 12 | 329 ± 13 | 386 ± 11 | 419 ± 11 |

## Table 2. Search at play time

Elo, HeuristicBot = 0, 400 games a match. Each column is one network in its own ladder.

| Search at play time | Real-deck agent, real decks (test) | Specialist (compact), fixed decks | Imitation (compact), fixed decks |
|---|---|---|---|
| None (network alone) | 262 ± 12 | 576 ± 23 | -20 ± 10 |
| 10 simulations | 344 ± 12 | 614 ± 23 | 21 ± 10 |
| 30 simulations | 377 ± 12 | 643 ± 23 | 63 ± 10 |
| 100 simulations | 412 ± 13 | 664 ± 23 | 184 ± 10 |
| 300 simulations | 431 ± 13 | 701 ± 23 | 323 ± 11 |
| 1,000 simulations | 459 ± 13 | 722 ± 23 | 405 ± 11 |
| 100 simulations, sees hidden cards | 420 ± 13 | 750 ± 23 | 219 ± 10 |

## Table 3. Design and training choices

Elo difference the change makes, measured within one ladder (several pairs: their mean). Fixed-deck play-time rows average M9's two training regimes; the training row its two seeds.

| Change | Compared with | Real decks (test) | Fixed decks |
|---|---|---|---|
| Within-turn search at play time | Full-game search | +48 ± 14 | +47 ± 8 |
| Belief-oracle leaf | Student-network leaf | -14 ± 14 | +7 ± 5 |
| Within-turn search during training | Full-game search during training | +8 ± 8 | -2 ± 5 |
| Full inputs (v2) | Compact inputs (v1), the three lineages | -13 ± 5 | -32 ± 4 |
| 4× bigger network (256×8) | 128×4, same data | -17 ± 11 | -12 ± 14 |
| + 20k self-play games | Real-deck agent | -15 ± 9 | -12 ± 9 |
| + 20k games, no gating | Real-deck agent | -5 ± 9 | -11 ± 9 |
| + 20k games, 3× training search | Real-deck agent | 0 ± 9 | +14 ± 9 |
| SAS-weighted outcome targets | The same three runs unweighted | -1 ± 5 | -1 ± 5 |

## Table 4. Deck strength (SAS) and the Adaptive format

Real test decks, 100 deck pairs per band; the agent is the Real-deck agent at 100 simulations. Pooled: 8.4 ± 0.6 Elo per SAS point (agent vs itself); skill over HeuristicBot 492 ± 21 Elo at equal decks, worth 49 SAS points.

| SAS gap | Mean gap | Higher deck's edge, agent vs itself (Elo) | vs HeuristicBot with the higher deck | vs HeuristicBot with the lower deck | Adaptive vs HeuristicBot, match wins | Adaptive mirror: reaches game 3 | Mean bid (chains) | Bid winner wins game 3 |
|---|---|---|---|---|---|---|---|---|
| 0-5 | 1.8 | +33 ± 33 | 96.6% | 93.9% | 99.8% | 56.0% | 4.23 | 46.4% |
| 5-10 | 7.1 | +49 ± 26 | 96.0% | 92.0% | 99.5% | 58.0% | 4.58 | 56.9% |
| 10-20 | 13.6 | +49 ± 26 | 98.0% | 86.5% | 99.8% | 63.5% | 5.9 | 49.6% |
| 20-60 | 28.9 | +295 ± 35 | 98.5% | 76.0% | 100.0% | 72.5% | 10.5 | 47.6% |

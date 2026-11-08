# What the deck embeddings tell us

Pool: the 1000 most-played training decks. All cards level 12, equal trophies.

## 1. Similar decks from embedding distance

'Matchup similarity' = correlation of two decks' cyclic matchup rows (they beat and lose to the same decks), from `attn_bc_inner` for every row so it is a fixed yardstick. Spearman over all 499,500 deck pairs. 'Within same overlap' = mean Spearman computed separately among pairs sharing exactly k cards (k = 0..6), i.e. what the embedding knows beyond counting shared cards. 'Top-10' = of each deck's 10 nearest neighbours by embedding cosine, how many are in its 10 most matchup-similar decks (mean over top-200 decks).

| embedding from | vs matchup similarity | vs matchup similarity (`bag_mlp` yardstick) | within same overlap | top-10 neighbours |
|---|---:|---:|---:|---:|
| (shared-card count) | 0.44 | 0.48 | – | 4.7 |
| `bag_bt` | 0.29 | 0.32 | -0.04 | 4.6 |
| `attn_bt` | 0.07 | 0.08 | 0.02 | 2.7 |
| `bag_bc_inner` | 0.73 | 0.76 | 0.67 | 5.6 |
| `bag_bc_dist` | 0.73 | 0.76 | 0.66 | 5.7 |
| `deepset_bc_dist` | 0.82 | 0.82 | 0.78 | 6.4 |
| `attn_bc_inner` | 0.87 | 0.81 | 0.86 | 7.3 |
| `attn_bc_dist` | 0.83 | 0.81 | 0.79 | 6.5 |
| `bag_mlp` | 0.73 | 0.78 | 0.67 | 5.7 |
| `deepset_mlp` | 0.79 | 0.80 | 0.73 | 5.6 |
| `attn_mlp` | 0.81 | 0.81 | 0.76 | 6.3 |

## 2. Counters read directly from the embeddings (blade-chest)

Blade-chest inner product: logit(A beats B) = s_A − s_B + blade_A·chest_B − blade_B·chest_A. So deck t's counters are the decks maximising the inner product of [blade, chest] with the query [chest_t, −blade_t]: a nearest-neighbour search in embedding space, no pairwise model evaluation needed.

| model | cyclic term vs Hodge cyclic part (Spearman) | counter rank agreement with `bag_mlp` (Spearman, mean over top-100 targets) |
|---|---:|---:|
| `bag_bc_inner` | 0.92 | 0.87 |
| `attn_bc_inner` | 0.82 | 0.85 |

Because the bag encoder is linear, logit(opponent beats t) is a *sum over the opponent's cards* of per-card scores, so the strongest possible counter to t (under `bag_bc_inner`) is just the 8 top-scoring cards. Built decks are then scored by the two other model families (`bag_mlp`, `attn_bc_inner`) as an independent check, and compared with the best counter that actually occurs in the pool. Treat built decks as suggestions: they can be card combinations nobody plays, where every model is extrapolating.

| target deck | built counter (top-8 cards) | P by bag_mlp / attn_bc_inner | best pool counter, P by bag_mlp | built deck played? |
|---|---|---:|---:|---:|
| Goblin Barrel: Knight, Princess, Ice Spirit, Goblin Gang, Inferno Tower, Rocket, The Log | Royal Hogs + Elixir Golem + Mortar + Graveyard: Royal Recruits, The Log, Earthquake, Barbarian Barrel | 0.71 / 0.78 | 0.81 | no |
| Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Cannon, Fireball, The Log | Giant + Golem + Royal Giant + Lava Hound + Mega Knight + Goblin Giant + Graveyard: Earthquake | 0.25 / 0.80 | 0.92 | no |
| Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison | Lava Hound: Goblins, P.E.K.K.A, Barbarians, Guards, Royal Recruits, Zappies, Goblin Cage | 0.74 / 0.91 | 0.80 | no |
| X-Bow: Knight, Archers, Skeletons, Ice Spirit, Tesla, Fireball, The Log | Giant + Golem + Royal Giant + Lava Hound + Goblin Giant + Elixir Golem: Royal Recruits, Earthquake | 0.27 / 0.91 | 0.91 | no |
| Golem + Baby Dragon: Lumberjack, Mega Minion, Night Witch, Lightning, Tornado, Barbarian Barrel | Prince + Ram Rider + Electro Giant: P.E.K.K.A, Inferno Dragon, Zappies, Inferno Tower, Barbarian Hut | 0.75 / 0.79 | 0.72 | no |

**Best deck against the whole meta** (play-weighted pool, built from per-card scores): Giant + Golem + Lava Hound + Battle Ram + Goblin Giant + Graveyard: Royal Recruits, Night Witch. Expected win rate vs the meta: 0.420 (bag_mlp) / 0.473 (attn_bc_inner); played in the data: no. Best deck that is actually played: Giant + Graveyard: Skeleton Army, Bowler, Night Witch, Bats, Arrows, Giant Snowball at 0.681.

Cards that help most / least against this meta (per-card score, logits): Lava Hound +0.28, Goblin Giant +0.19, Giant +0.15, Royal Recruits +0.15, Golem +0.14, Graveyard +0.14, Night Witch +0.13, Battle Ram +0.13 … Mirror -0.15, Fire Spirits -0.14, Firecracker -0.13, Wizard -0.12, Furnace -0.11

The unconstrained builds stack win conditions in a way no real deck does, and the two independent models disagree wildly on them: the models are only trustworthy near decks people actually play. A safer use is a *local* search: the best like-for-like single-card swap to a real deck. Even these lean on each card's overall strength, which can partly reflect who plays the card (player skill is not in the data), so read them as hypotheses to test, not advice.

### Best single-card swap against the meta, for the 5 most-played decks

Like-for-like swaps only (troop for troop, spell for spell, building for building, win condition for win condition), scored by expected win rate against the play-weighted pool. Ranked by the mean of `bag_mlp` and `attn_bc_inner`, so a swap only wins if both agree.

| deck | meta win rate now (mlp / attn) | best swap | after (mlp / attn) |
|---|---:|---|---:|
| Goblin Barrel: Knight, Princess, Ice Spirit, Goblin Gang, Inferno Tower, Rocket, The Log | 0.440 / 0.474 | Princess → **Cannon Cart** | 0.497 / 0.501 |
| Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Cannon, Fireball, The Log | 0.402 / 0.427 | Ice Golem → **Cannon Cart** | 0.504 / 0.515 |
| Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison | 0.551 / 0.498 | Magic Archer → **Flying Machine** | 0.587 / 0.521 |
| X-Bow: Knight, Archers, Skeletons, Ice Spirit, Tesla, Fireball, The Log | 0.464 / 0.482 | Skeletons → **Bowler** | 0.536 / 0.516 |
| Golem + Baby Dragon: Lumberjack, Mega Minion, Night Witch, Lightning, Tornado, Barbarian Barrel | 0.584 / 0.572 | Baby Dragon → **Lava Hound** | 0.660 / 0.625 |

## 3. How many 'rock-paper-scissors dimensions' are there?

An antisymmetric matrix splits into 2-D planes (real Schur form); each plane is one rock-paper-scissors cycle. Share of the cyclic matchup variance captured by the first k planes:

| model | 1 plane | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| `bag_mlp` | 36% | 55% | 68% | 84% | 97% |
| `attn_bc_inner` | 40% | 60% | 74% | 90% | 100% |
| `bag_bc_dist` | 41% | 62% | 75% | 90% | 100% |
| `attn_mlp` | 46% | 67% | 80% | 94% | 98% |

## 4. Archetypes from clustering the embeddings

KMeans (k = 14) on `attn_bc_inner` embeddings, weighted by plays. Name = dominant win condition + the two cards most over-represented in the cluster.

| archetype | share of pool plays | decks | win-con purity | best matchup (P) | worst matchup (P) |
|---|---:|---:|---:|---|---|
| Battle Ram (Bandit, Electro Wizard) | 12.6% | 112 | 59% | Giant (Bats, Zap) (0.60) | Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) (0.44) |
| Hog Rider (Skeletons, Ice Golem) | 11.5% | 57 | 81% | Prince + Goblin Barrel (Rascals, Princess) (0.59) | Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) (0.27) |
| Goblin Barrel (Inferno Tower, Princess) | 11.4% | 81 | 79% | Prince + Goblin Barrel (Rascals, Princess) (0.56) | Miner + Wall Breakers (Spear Goblins, Knight) (0.31) |
| Golem + Baby Dragon (Night Witch, Tornado) | 10.5% | 117 | 79% | X-Bow (Tesla, Skeletons) (0.73) | Battle Ram (Bandit, Electro Wizard) (0.44) |
| Giant (Bats, Zap) | 8.7% | 128 | 15% | X-Bow (Tesla, Skeletons) (0.59) | Golem + Baby Dragon (Night Witch, Tornado) (0.39) |
| Miner + Wall Breakers (Spear Goblins, Knight) | 7.3% | 72 | 46% | Goblin Barrel (Inferno Tower, Princess) (0.69) | Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) (0.31) |
| X-Bow (Tesla, Skeletons) | 6.3% | 26 | 99% | Goblin Barrel (Princess, The Log) (0.60) | Golem + Baby Dragon (Night Witch, Tornado) (0.27) |
| Royal Hogs (Goblin Hut, Archers) | 6.2% | 98 | 23% | X-Bow (Tesla, Skeletons) (0.55) | Golem + Baby Dragon (Night Witch, Tornado) (0.38) |
| Baby Dragon + Graveyard (Poison, Ice Wizard) | 5.7% | 51 | 78% | Hog Rider (Skeletons, Ice Golem) (0.69) | Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) (0.36) |
| Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) | 5.3% | 76 | 40% | Hog Rider (Skeletons, Ice Golem) (0.73) | Golem + Baby Dragon (Night Witch, Tornado) (0.44) |
| Goblin Barrel (Princess, The Log) | 4.6% | 39 | 52% | Prince + Goblin Barrel (Rascals, Princess) (0.62) | Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons) (0.31) |
| Balloon + Baby Dragon (Lumberjack, Freeze) | 4.0% | 56 | 41% | Goblin Barrel (Inferno Tower, Princess) (0.61) | Battle Ram (Bandit, Electro Wizard) (0.44) |
| Miner + Mega Knight + Skeleton Barrel (Inferno Dragon, Bats) | 3.1% | 40 | 74% | Hog Rider (Skeletons, Ice Golem) (0.59) | Golem + Baby Dragon (Night Witch, Tornado) (0.42) |
| Prince + Goblin Barrel (Rascals, Princess) | 2.9% | 47 | 51% | Royal Hogs (Goblin Hut, Archers) (0.61) | Miner + Wall Breakers (Spear Goblins, Knight) (0.37) |

### Counters grouped by archetype, for the 5 most-played decks

- Goblin Barrel: Knight, Princess, Ice Spirit, Goblin Gang, Inferno Tower, Rocket, The Log (Goblin Barrel (Inferno Tower, Princess)) is countered by:
  - **Miner + Wall Breakers (Spear Goblins, Knight)** avg P 0.69 (best: Skeleton Barrel + Mortar: Knight, Musketeer, Spear Goblins, Cannon Cart, Fireball, The Log, 0.79)
  - **Baby Dragon + Graveyard (Poison, Ice Wizard)** avg P 0.66 (best: Mortar + Graveyard: Knight, Spear Goblins, Hunter, Cannon Cart, Poison, Barbarian Barrel, 0.77)
  - **Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons)** avg P 0.63 (best: Balloon + Lava Hound + Miner: Guards, Inferno Dragon, Skeleton Dragons, Arrows, Zap, 0.74)
- Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Cannon, Fireball, The Log (Hog Rider (Skeletons, Ice Golem)) is countered by:
  - **Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons)** avg P 0.75 (best: Balloon + Lava Hound: Barbarians, Mega Minion, Skeleton Dragons, Tombstone, Arrows, Lightning, 0.85)
  - **Baby Dragon + Graveyard (Poison, Ice Wizard)** avg P 0.73 (best: Baby Dragon + Graveyard: Knight, Goblins, Bowler, Electro Wizard, Poison, Tornado, 0.82)
  - **Golem + Baby Dragon (Night Witch, Tornado)** avg P 0.69 (best: Balloon + Golem: Skeleton Army, Lumberjack, Night Witch, Electro Dragon, Fireball, Clone, 0.80)
- Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison (Battle Ram (Bandit, Electro Wizard)) is countered by:
  - **Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons)** avg P 0.59 (best: Balloon + Lava Hound: Barbarians, Mega Minion, Skeleton Dragons, Tombstone, Arrows, Lightning, 0.72)
  - **Goblin Barrel (Princess, The Log)** avg P 0.50 (best: Skeleton Barrel + Goblin Barrel: Knight, Skeletons, Dart Goblin, Bomb Tower, The Log, Royal Delivery, 0.61)
  - **Prince + Goblin Barrel (Rascals, Princess)** avg P 0.49 (best: Giant Skeleton + Skeleton Barrel: Barbarians, Hunter, Cannon Cart, Flying Machine, Clone, Heal Spirit, 0.68)
- X-Bow: Knight, Archers, Skeletons, Ice Spirit, Tesla, Fireball, The Log (X-Bow (Tesla, Skeletons)) is countered by:
  - **Golem + Baby Dragon (Night Witch, Tornado)** avg P 0.75 (best: Golem + Baby Dragon: Night Witch, Zappies, Flying Machine, Fisherman, Lightning, Barbarian Barrel, 0.85)
  - **Balloon + Lava Hound + Miner (Barbarians, Skeleton Dragons)** avg P 0.71 (best: Balloon + Lava Hound + Miner: Barbarians, Inferno Dragon, Skeleton Dragons, Arrows, Giant Snowball, 0.81)
  - **Giant (Bats, Zap)** avg P 0.59 (best: Giant + Graveyard: Skeleton Army, Bowler, Night Witch, Bats, Arrows, Giant Snowball, 0.91)
- Golem + Baby Dragon: Lumberjack, Mega Minion, Night Witch, Lightning, Tornado, Barbarian Barrel (Golem + Baby Dragon (Night Witch, Tornado)) is countered by:
  - **Battle Ram (Bandit, Electro Wizard)** avg P 0.59 (best: Battle Ram: P.E.K.K.A, Electro Wizard, Executioner, Bandit, Royal Ghost, Zap, Tornado, 0.72)
  - **Golem + Baby Dragon (Night Witch, Tornado)** avg P 0.53 (best: Golem + Baby Dragon: Night Witch, Zappies, Flying Machine, Fisherman, Lightning, Barbarian Barrel, 0.67)
  - **Baby Dragon + Graveyard (Poison, Ice Wizard)** avg P 0.48 (best: Baby Dragon + Graveyard: Knight, Ice Wizard, Barbarian Hut, Poison, Tornado, Barbarian Barrel, 0.57)

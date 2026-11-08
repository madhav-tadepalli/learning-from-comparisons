# Deck similarity and counters (`bag_mlp`, cross-checked with `attn_bc_inner`)

Pool: the 1000 most-played decks in train (80+ train plays each; together 19% of all deck appearances). All cards level 12, equal trophies.

## How similarity measures relate (all deck pairs in the pool)

| | profile sim (check model) | embedding cosine | shared cards |
|---|---:|---:|---:|
| profile sim (Spearman) | 0.90 | 0.72 | 0.48 |

Profile similarity = correlation of the two decks' cyclic matchup rows: high means they beat and lose to the same opponents (beyond overall strength).

| shared cards | pairs | mean profile sim |
|---:|---:|---:|
| 0 | 210,992 | -0.14 |
| 1 | 161,486 | -0.02 |
| 2 | 73,255 | +0.12 |
| 3 | 29,692 | +0.28 |
| 4 | 12,089 | +0.46 |
| 5 | 6,290 | +0.63 |
| 6 | 4,038 | +0.75 |
| 7 | 1,658 | +0.86 |

### Functional look-alikes: similar matchups, few shared cards (≤ 2)

- sim **0.86** (check model 0.77, 2 shared)
  - Goblin Giant: Minions, Musketeer, Mini P.E.K.K.A, Sparky, Rage, Zap, Heal Spirit
  - Goblin Giant: Dark Prince, Sparky, Mega Minion, Electro Wizard, Goblin Cage, Lightning, Giant Snowball
- sim **0.86** (check model 0.92, 1 shared)
  - Miner + Mega Knight + Wall Breakers: Musketeer, Mini P.E.K.K.A, Bats, Zap, Barbarian Barrel
  - Mega Knight + Royal Hogs: Electro Wizard, Bandit, Magic Archer, Goblin Cage, The Log, Heal Spirit
- sim **0.85** (check model 0.75, 2 shared)
  - Baby Dragon + Lava Hound: Lumberjack, Mega Minion, Flying Machine, Tombstone, Poison, Barbarian Barrel
  - Balloon + Lava Hound + Miner: Barbarians, Mega Minion, Skeleton Dragons, Fireball, Zap
- sim **0.85** (check model 0.71, 2 shared)
  - Giant: Mini P.E.K.K.A, Sparky, Electro Wizard, Fireball, Zap, Tornado, Heal Spirit
  - Goblin Giant: Dark Prince, Sparky, Mega Minion, Electro Wizard, Goblin Cage, Lightning, Giant Snowball
- sim **0.85** (check model 0.92, 2 shared)
  - Golem + Baby Dragon: Lumberjack, Mega Minion, Cannon Cart, Electro Dragon, Tornado, Barbarian Barrel
  - Giant + Prince + Miner: Mega Minion, Electro Dragon, Fireball, Zap, The Log
- sim **0.84** (check model 0.72, 2 shared)
  - Skeleton Barrel + Goblin Barrel: Knight, Skeletons, Dart Goblin, Electro Spirit, Bomb Tower, The Log
  - Skeleton Barrel + Mortar: Knight, Spear Goblins, Goblin Gang, Firecracker, Fireball, Barbarian Barrel
- sim **0.84** (check model 0.79, 2 shared)
  - Giant Skeleton + Royal Giant + Goblin Barrel: Skeleton Army, Hunter, Bats, Zappies, Firecracker
  - Giant Skeleton + Skeleton Barrel: Barbarians, Hunter, Cannon Cart, Flying Machine, Mother Witch, Clone
- sim **0.84** (check model 0.87, 2 shared)
  - Graveyard: P.E.K.K.A, Electro Wizard, Hunter, Fisherman, Goblin Cage, Poison, Giant Snowball
  - Miner: P.E.K.K.A, Royal Ghost, Zappies, Flying Machine, Goblin Hut, Zap, Poison

## Most similar decks and strongest counters, for the 5 most-played decks

P = model P(opponent beats target). C = cyclic part in logits (counter effect beyond the opponent's overall strength). Observed = raw win rate of that opponent vs target over all games (n ≥ 30, not level-adjusted).

### Goblin Barrel: Knight, Princess, Ice Spirit, Goblin Gang, Inferno Tower, Rocket, The Log
(61,411 plays, strength rank 814/1000)

Most similar (profile sim / shared cards):
- 0.98 / 7: Goblin Barrel: Knight, Skeletons, Princess, Ice Spirit, Inferno Tower, Rocket, The Log
- 0.98 / 7: Goblin Barrel: Knight, Princess, Ice Golem, Goblin Gang, Inferno Tower, Rocket, The Log
- 0.98 / 7: Goblin Barrel: Knight, Princess, Ice Spirit, Goblin Gang, Inferno Tower, Rocket, Barbarian Barrel
- 0.97 / 7: Goblin Barrel: Knight, Skeletons, Princess, Goblin Gang, Inferno Tower, Rocket, The Log
- 0.97 / 7: Goblin Barrel: Knight, Princess, Goblin Gang, Inferno Tower, Rocket, Mirror, The Log

Strongest counters (opponent → target):

| opponent | P | C | check-model P | observed (n) |
|---|---:|---:|---:|---:|
| Battle Ram: Electro Wizard, Bandit, Royal Ghost, Cannon Cart, Magic Archer, Mother Witch, Barbarian Barrel | 0.81 | +0.87 | 0.81 | – |
| Baby Dragon + Elixir Golem: Electro Dragon, Battle Healer, Barbarian Hut, Arrows, Tornado, Barbarian Barrel | 0.81 | +0.96 | 0.78 | – |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Mother Witch, Barbarian Hut, Fireball, Barbarian Barrel | 0.81 | +0.75 | 0.73 | – |
| Battle Ram: Electro Wizard, Bandit, Royal Ghost, Cannon Cart, Magic Archer, Electro Spirit, Barbarian Barrel | 0.81 | +0.72 | 0.79 | – |
| no win-con: Electro Wizard, Elite Barbarians, Bandit, Royal Ghost, Cannon Cart, Magic Archer, Barbarian Barrel, Giant Snowball | 0.80 | +0.73 | 0.79 | – |
| Battle Ram: Electro Wizard, Bandit, Royal Ghost, Cannon Cart, Magic Archer, Barbarian Barrel, Heal Spirit | 0.80 | +0.84 | 0.81 | – |

Best matchups for the target (target → opponent): no win-con (0.92); Giant (0.88); no win-con (0.88)

### Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Cannon, Fireball, The Log
(54,237 plays, strength rank 935/1000)

Most similar (profile sim / shared cards):
- 0.96 / 7: Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Cannon, Poison, The Log
- 0.95 / 7: Hog Rider: Skeletons, Musketeer, Ice Spirit, Ice Golem, Bomb Tower, Fireball, The Log
- 0.93 / 7: Hog Rider: Knight, Skeletons, Musketeer, Ice Spirit, Cannon, Fireball, The Log
- 0.93 / 7: Hog Rider: Knight, Skeletons, Musketeer, Ice Golem, Cannon, Fireball, The Log
- 0.92 / 7: Hog Rider: Skeletons, Musketeer, Ice Golem, Electro Spirit, Cannon, Fireball, The Log

Strongest counters (opponent → target):

| opponent | P | C | check-model P | observed (n) |
|---|---:|---:|---:|---:|
| Giant + Graveyard: Skeleton Army, Bowler, Night Witch, Bats, Arrows, Giant Snowball | 0.92 | +1.27 | 0.85 | – |
| Giant + Graveyard: Skeleton Army, Musketeer, Bowler, Night Witch, Arrows, Giant Snowball | 0.90 | +1.10 | 0.84 | – |
| Balloon + Lava Hound: Barbarians, Mega Minion, Skeleton Dragons, Tombstone, Arrows, Lightning | 0.85 | +0.74 | 0.78 | – |
| Balloon + Lava Hound: Minions, Guards, Mega Minion, Tombstone, Arrows, Lightning | 0.84 | +0.86 | 0.84 | – |
| Giant + Graveyard: Barbarians, Bowler, Mega Minion, Hunter, Arrows, Giant Snowball | 0.84 | +0.69 | 0.78 | – |
| Balloon + Lava Hound: Minions, Barbarians, Mega Minion, Tombstone, Arrows, Lightning | 0.84 | +0.75 | 0.79 | – |

Best matchups for the target (target → opponent): no win-con (0.92); no win-con (0.87); no win-con (0.81)

### Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison
(38,203 plays, strength rank 206/1000)

Most similar (profile sim / shared cards):
- 0.97 / 7: Battle Ram: P.E.K.K.A, Dark Prince, Electro Wizard, Bandit, Magic Archer, Zap, Poison
- 0.96 / 7: Battle Ram: P.E.K.K.A, Dark Prince, Electro Wizard, Royal Ghost, Magic Archer, Zap, Poison
- 0.95 / 7: Hog Rider: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison
- 0.95 / 7: Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Mother Witch, Zap, Poison
- 0.95 / 7: Battle Ram: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Fireball, Zap

Strongest counters (opponent → target):

| opponent | P | C | check-model P | observed (n) |
|---|---:|---:|---:|---:|
| Royal Hogs: Ice Spirit, Royal Recruits, Zappies, Flying Machine, Goblin Cage, Fireball, Barbarian Barrel | 0.80 | +1.36 | 0.83 | – |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Electro Spirit, Goblin Cage, Fireball, Barbarian Barrel | 0.78 | +1.15 | 0.85 | 0.82 (150) |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Electro Spirit, Goblin Cage, Poison, Barbarian Barrel | 0.78 | +1.03 | 0.86 | – |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Barbarian Hut, Fireball, Barbarian Barrel, Heal Spirit | 0.77 | +1.15 | 0.83 | – |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Goblin Cage, Fireball, Barbarian Barrel, Heal Spirit | 0.77 | +1.31 | 0.83 | – |
| Royal Hogs: Royal Recruits, Zappies, Flying Machine, Electro Spirit, Goblin Cage, Fireball, The Log | 0.76 | +1.07 | 0.86 | – |

Best matchups for the target (target → opponent): Royal Giant + Miner (0.93); no win-con (0.91); Baby Dragon + Royal Giant + Miner (0.88)

### X-Bow: Knight, Archers, Skeletons, Ice Spirit, Tesla, Fireball, The Log
(25,305 plays, strength rank 845/1000)

Most similar (profile sim / shared cards):
- 0.98 / 7: X-Bow: Knight, Archers, Skeletons, Ice Golem, Tesla, Fireball, The Log
- 0.97 / 7: X-Bow: Archers, Skeletons, Ice Spirit, Ice Golem, Tesla, Fireball, The Log
- 0.96 / 7: X-Bow: Knight, Archers, Skeletons, Fire Spirits, Tesla, Fireball, The Log
- 0.96 / 7: X-Bow: Knight, Archers, Skeletons, Ice Spirit, Tesla, Rage, The Log
- 0.96 / 7: X-Bow: Archers, Skeletons, Valkyrie, Ice Spirit, Tesla, Fireball, The Log

Strongest counters (opponent → target):

| opponent | P | C | check-model P | observed (n) |
|---|---:|---:|---:|---:|
| Giant + Graveyard: Skeleton Army, Bowler, Night Witch, Bats, Arrows, Giant Snowball | 0.91 | +1.36 | 0.87 | – |
| Giant + Graveyard: Skeleton Army, Musketeer, Bowler, Night Witch, Arrows, Giant Snowball | 0.91 | +1.39 | 0.88 | – |
| Giant + Graveyard: Barbarians, Bowler, Mega Minion, Hunter, Arrows, Giant Snowball | 0.86 | +1.03 | 0.84 | – |
| Giant Skeleton + Royal Giant: Hunter, Zappies, Fisherman, Electro Spirit, Earthquake, Barbarian Barrel | 0.85 | +1.08 | 0.68 | 0.77 (64) |
| Giant Skeleton + Royal Giant: Hunter, Zappies, Fisherman, Electro Spirit, The Log, Earthquake | 0.85 | +1.10 | 0.68 | – |
| Giant Skeleton + Royal Giant: Hunter, Zappies, Fisherman, Earthquake, Barbarian Barrel, Heal Spirit | 0.85 | +1.27 | 0.71 | – |

Best matchups for the target (target → opponent): no win-con (0.96); Goblin Barrel (0.85); no win-con (0.84)

### Golem + Baby Dragon: Lumberjack, Mega Minion, Night Witch, Lightning, Tornado, Barbarian Barrel
(17,774 plays, strength rank 155/1000)

Most similar (profile sim / shared cards):
- 0.97 / 7: Golem + Baby Dragon: Lumberjack, Mega Minion, Night Witch, Lightning, The Log, Tornado
- 0.94 / 7: Golem + Baby Dragon: Mega Minion, Night Witch, Cannon Cart, Lightning, Tornado, Barbarian Barrel
- 0.94 / 7: Golem + Baby Dragon: Dark Prince, Mega Minion, Night Witch, Lightning, Tornado, Barbarian Barrel
- 0.93 / 7: Golem + Baby Dragon: Lumberjack, Night Witch, Electro Dragon, Lightning, Tornado, Barbarian Barrel
- 0.92 / 6: Golem + Baby Dragon: Mega Minion, Night Witch, Cannon Cart, Fireball, Tornado, Barbarian Barrel

Strongest counters (opponent → target):

| opponent | P | C | check-model P | observed (n) |
|---|---:|---:|---:|---:|
| Battle Ram: P.E.K.K.A, Electro Wizard, Executioner, Bandit, Royal Ghost, Zap, Tornado | 0.72 | +0.94 | 0.66 | – |
| Giant Skeleton + Royal Giant: Hunter, Zappies, Fisherman, Electro Spirit, Earthquake, Barbarian Barrel | 0.69 | +0.68 | 0.70 | 0.72 (69) |
| Giant Skeleton + Royal Giant: Hunter, Zappies, Fisherman, Electro Spirit, The Log, Earthquake | 0.69 | +0.68 | 0.68 | – |
| Miner: P.E.K.K.A, Royal Ghost, Zappies, Flying Machine, Goblin Hut, Zap, Poison | 0.69 | +0.73 | 0.58 | – |
| Ram Rider: P.E.K.K.A, Dark Prince, Electro Wizard, Bandit, Magic Archer, Zap, Poison | 0.68 | +0.80 | 0.53 | – |
| Ram Rider: P.E.K.K.A, Electro Wizard, Bandit, Royal Ghost, Magic Archer, Zap, Poison | 0.68 | +0.80 | 0.56 | – |

Best matchups for the target (target → opponent): no win-con (0.94); Royal Giant + Miner (0.91); Royal Giant + Miner (0.88)

## Are the counters real? Held-out test games

12,578 test games have both decks in the pool. Binned by the model's cyclic counter score C(A,B) (computed at equal levels, from training data only). 'card_bt' already accounts for each card's strength, the real card levels and trophies, but has no matchup term.

| C(A,B) bin | games | A's observed win rate | card_bt predicted | full model predicted |
|---|---:|---:|---:|---:|
| [-inf, -0.4) | 2,236 | 0.322 | 0.496 | 0.335 |
| [-0.4, -0.2) | 1,768 | 0.442 | 0.498 | 0.422 |
| [-0.2, -0.1) | 1,211 | 0.496 | 0.501 | 0.463 |
| [-0.1, +0.0) | 1,118 | 0.493 | 0.497 | 0.481 |
| [+0.0, +0.1) | 1,103 | 0.534 | 0.500 | 0.514 |
| [+0.1, +0.2) | 1,169 | 0.541 | 0.501 | 0.537 |
| [+0.2, +0.4) | 1,776 | 0.587 | 0.505 | 0.581 |
| [+0.4, +inf) | 2,197 | 0.684 | 0.503 | 0.662 |

Correlation of C(A,B) with the card_bt residual (observed − predicted): +0.231.
Agreement between models on counter scores C (Spearman over all pairs): 0.86; on strengths s: 0.86.
Top-5 counter lists shared between the two models, averaged over the 100 most-played targets: 1.7 of 5.

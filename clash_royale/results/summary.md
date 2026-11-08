Test set: 234,559 matches. Δ = test log-loss minus card_bt's (negative = better), 95% paired bootstrap CI.

| run | params | val logloss | test logloss | test acc | test AUC | Δ vs card_bt [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| bag_mlp_s2 | 112,130 | 0.6415 | 0.6407 | 0.6259 | 0.6777 | -0.0267 [-0.0278, -0.0256] |
| bag_mlp_s1 | 112,130 | 0.6416 | 0.6406 | 0.6238 | 0.6769 | -0.0269 [-0.0279, -0.0257] |
| bag_mlp_wd1e-4 | 112,130 | 0.6417 | 0.6410 | 0.6235 | 0.6757 | -0.0265 [-0.0273, -0.0255] |
| bag_mlp | 112,130 | 0.6417 | 0.6410 | 0.6235 | 0.6757 | -0.0265 [-0.0273, -0.0255] |
| bag_bc_dist_k64 | 21,313 | 0.6422 | 0.6420 | 0.6251 | 0.6766 | -0.0254 [-0.0264, -0.0243] |
| attn_bc_inner | 115,265 | 0.6422 | 0.6417 | 0.6225 | 0.6743 | -0.0258 [-0.0267, -0.0247] |
| bag_mlp_d128 | 157,954 | 0.6422 | 0.6409 | 0.6244 | 0.6757 | -0.0266 [-0.0275, -0.0256] |
| cross | 171,586 | 0.6423 | 0.6413 | 0.6215 | 0.6739 | -0.0261 [-0.0269, -0.0252] |
| bag_bc_dist_k32 | 17,217 | 0.6425 | 0.6421 | 0.6254 | 0.6765 | -0.0254 [-0.0263, -0.0243] |
| gbdt | 2,293 | 0.6430 | 0.6419 | 0.6217 | 0.6732 | -0.0256 [-0.0263, -0.0248] |
| attn_bc_dist | 115,265 | 0.6431 | 0.6427 | 0.6210 | 0.6723 | -0.0248 [-0.0257, -0.0238] |
| attn_bc_inner_s1 | 115,265 | 0.6434 | 0.6424 | 0.6219 | 0.6725 | -0.0251 [-0.0259, -0.0241] |
| bag_bc_dist | 15,169 | 0.6441 | 0.6441 | 0.6224 | 0.6727 | -0.0234 [-0.0242, -0.0224] |
| bag_bc_dist_s1 | 15,169 | 0.6442 | 0.6441 | 0.6224 | 0.6727 | -0.0234 [-0.0243, -0.0225] |
| deepset_bc_dist | 179,841 | 0.6442 | 0.6438 | 0.6198 | 0.6709 | -0.0237 [-0.0246, -0.0226] |
| deepset_bc_dist_s1 | 179,841 | 0.6442 | 0.6433 | 0.6207 | 0.6720 | -0.0242 [-0.0252, -0.0231] |
| bag_bc_dist_s2 | 15,169 | 0.6442 | 0.6439 | 0.6226 | 0.6731 | -0.0236 [-0.0244, -0.0226] |
| deepset_bc_dist_s2 | 179,841 | 0.6444 | 0.6443 | 0.6189 | 0.6701 | -0.0232 [-0.0242, -0.0222] |
| attn_mlp | 212,226 | 0.6454 | 0.6445 | 0.6170 | 0.6682 | -0.0229 [-0.0239, -0.0219] |
| deepset_bc_inner | 179,841 | 0.6454 | 0.6446 | 0.6189 | 0.6696 | -0.0229 [-0.0238, -0.0218] |
| bag_bc_inner_k64 | 21,313 | 0.6462 | 0.6459 | 0.6217 | 0.6713 | -0.0216 [-0.0224, -0.0206] |
| bag_bc_inner_k32 | 17,217 | 0.6463 | 0.6459 | 0.6224 | 0.6712 | -0.0215 [-0.0224, -0.0206] |
| bag_bc_inner | 15,169 | 0.6463 | 0.6463 | 0.6212 | 0.6704 | -0.0212 [-0.0220, -0.0203] |
| bag_bc_inner_s1 | 15,169 | 0.6464 | 0.6463 | 0.6214 | 0.6702 | -0.0211 [-0.0220, -0.0202] |
| bag_bc_inner_s2 | 15,169 | 0.6466 | 0.6462 | 0.6215 | 0.6706 | -0.0213 [-0.0221, -0.0204] |
| deepset_mlp | 276,802 | 0.6471 | 0.6462 | 0.6146 | 0.6654 | -0.0212 [-0.0222, -0.0203] |
| bag_bc_inner_k8 | 14,145 | 0.6484 | 0.6484 | 0.6177 | 0.6661 | -0.0190 [-0.0198, -0.0182] |
| bag_bc_inner_k4 | 13,633 | 0.6523 | 0.6524 | 0.6108 | 0.6579 | -0.0151 [-0.0158, -0.0144] |
| bag_bc_inner_k2 | 13,377 | 0.6574 | 0.6574 | 0.6013 | 0.6463 | -0.0101 [-0.0106, -0.0095] |
| bag_bc_inner_k1 | 13,249 | 0.6618 | 0.6613 | 0.5933 | 0.6366 | -0.0062 [-0.0066, -0.0057] |
| deepset_bt | 177,793 | 0.6619 | 0.6613 | 0.5874 | 0.6311 | -0.0061 [-0.0066, -0.0056] |
| attn_bt | 113,217 | 0.6625 | 0.6618 | 0.5859 | 0.6293 | -0.0056 [-0.0061, -0.0051] |
| bag_mlp_decksonly | 112,130 | 0.6656 | 0.6645 | 0.5892 | 0.6270 | -0.0029 [-0.0041, -0.0018] |
| attn_bc_inner_decksonly | 115,265 | 0.6662 | 0.6650 | 0.5878 | 0.6253 | -0.0024 [-0.0036, -0.0012] |
| bag_bt | 13,121 | 0.6678 | 0.6675 | 0.5781 | 0.6185 | +0.0000 [+0.0000, +0.0000] |
| card_bt | 205 | 0.6678 | 0.6675 | 0.5783 | 0.6186 | +0.0000 [+0.0000, +0.0000] |
| card_bt_s2 | 205 | 0.6678 | 0.6675 | 0.5782 | 0.6186 | +0.0000 [-0.0000, +0.0000] |
| card_bt_s1 | 205 | 0.6678 | 0.6675 | 0.5784 | 0.6186 | +0.0000 [-0.0000, +0.0000] |
| bag_bc_dist_decksonly | 15,169 | 0.6693 | 0.6685 | 0.5824 | 0.6186 | +0.0011 [-0.0001, +0.0022] |
| gbdt_decksonly | 1,000 | 0.6696 | 0.6681 | 0.5837 | 0.6204 | +0.0007 [-0.0003, +0.0016] |
| attn_mlp_decksonly | 212,226 | 0.6697 | 0.6690 | 0.5804 | 0.6154 | +0.0016 [+0.0005, +0.0028] |
| deckid_bc_inner | 235,586 | 0.6711 | 0.6702 | 0.5704 | 0.6082 | +0.0027 [+0.0022, +0.0032] |
| bag_bc_inner_decksonly | 15,169 | 0.6712 | 0.6702 | 0.5841 | 0.6183 | +0.0027 [+0.0017, +0.0039] |
| deckid_bt | 234,562 | 0.6714 | 0.6705 | 0.5697 | 0.6072 | +0.0031 [+0.0026, +0.0036] |
| deepset_mlp_decksonly | 276,802 | 0.6715 | 0.6708 | 0.5762 | 0.6101 | +0.0033 [+0.0022, +0.0044] |
| trophy_level | 2 | 0.6755 | 0.6755 | 0.5642 | 0.6000 | +0.0080 [+0.0075, +0.0085] |
| card_bt_decksonly | 205 | 0.6911 | 0.6909 | 0.5201 | 0.5303 | +0.0234 [+0.0226, +0.0243] |
| trophy | 2 | 0.6925 | 0.6926 | 0.5118 | 0.5184 | +0.0251 [+0.0243, +0.0260] |
| constant | 0 | 0.6931 | 0.6931 | 0.5007 | 0.5000 | +0.0257 [+0.0249, +0.0266] |

## Seed variance (seed 0 = the round-1 run)

| model | seeds | test logloss mean ± sd | test acc mean ± sd |
|---|---:|---|---|
| bag_mlp | 3 | 0.6408 ± 0.0002 | 0.6244 ± 0.0013 |
| attn_bc_inner | 2 | 0.6420 ± 0.0005 | 0.6222 ± 0.0004 |
| bag_bc_dist | 3 | 0.6440 ± 0.0001 | 0.6225 ± 0.0001 |
| deepset_bc_dist | 3 | 0.6438 ± 0.0005 | 0.6198 ± 0.0009 |
| bag_bc_inner | 3 | 0.6463 ± 0.0001 | 0.6213 ± 0.0002 |
| card_bt | 3 | 0.6675 ± 0.0000 | 0.5783 ± 0.0001 |

## Card strengths (card_bt, logit units, centred; cards in >0.5% of decks)

Strongest: Electro Giant (+0.105), Skeleton Barrel (+0.102), Lava Hound (+0.100), Cannon Cart (+0.091), Bowler (+0.088), Mega Knight (+0.087), Night Witch (+0.084), Barbarian Barrel (+0.082)
Weakest: Executioner (-0.090), Mother Witch (-0.095), Baby Dragon (-0.099), Archers (-0.102), Wizard (-0.121), Minion Horde (-0.141), Mirror (-0.144), Elixir Golem (-0.157)

Trophy-level model: +1 total card level ⇒ logit +0.069; +100 trophies ⇒ logit +0.342
One full level on every card (+8 levels) ⇒ win prob 0.635 vs an otherwise equal opponent
Mean level slope per card (card_bt): +0.091 per level

## Strongest card-vs-card counters (bag blade-chest; logit added to the side holding the first card)

- P.E.K.K.A vs Electro Giant: +0.607
- Sparky vs Electro Giant: +0.586
- Electro Dragon vs Sparky: +0.423
- Graveyard vs Bomb Tower: +0.414
- Elite Barbarians vs Electro Giant: +0.413
- Inferno Dragon vs P.E.K.K.A: +0.409
- Giant vs X-Bow: +0.395
- Cannon vs Electro Giant: +0.394
- Inferno Tower vs Golem: +0.393
- Graveyard vs Cannon: +0.389
- Lava Hound vs Cannon: +0.381
- Bomb Tower vs Electro Giant: +0.370

## Deck matchup structure (top-100 most-played decks, all cards level 12, equal trophies)

Cyclic share = fraction of matchup-logit variance not explained by a single strength per deck (0% = pure ranking). Cyclic triangles = share of deck triples (top-60) forming a rock-paper-scissors loop.

| model | matchup logit std | cyclic share of variance | cyclic triangles |
|---|---:|---:|---:|
| card_bt | 0.210 | 0.0% | 0.0% |
| deepset_bt | 0.370 | 0.0% | 0.0% |
| bag_bc_inner | 0.440 | 85.2% | 21.3% |
| bag_bc_dist | 0.500 | 59.7% | 15.5% |
| deepset_bc_dist | 0.685 | 46.7% | 11.4% |
| attn_bc_inner | 0.605 | 64.1% | 16.4% |
| cross | 0.558 | 58.8% | 14.7% |
| bag_mlp | 0.674 | 50.2% | 12.0% |

Strongest rock-paper-scissors loops among the top-60 decks (`bag_mlp`; each beats the next):

1. weakest link p = 0.709  (P: 0.72, 0.71, 0.74)
   - **Giant + Prince + Miner**: Dark Prince, Mega Minion, Electro Wizard, Fireball, Zap  (8,480 plays)
   - **X-Bow**: Archers, Skeletons, Ice Spirit, Ice Golem, Tesla, Fireball, The Log  (6,205 plays)
   - **no win-con**: Barbarians, Skeleton Army, Bomber, Musketeer, Spear Goblins, Minion Horde, Goblin Hut, Barbarian Hut  (3,880 plays)
2. weakest link p = 0.675  (P: 0.69, 0.68, 0.85)
   - **X-Bow**: Knight, Skeletons, Ice Wizard, Tesla, Rocket, The Log, Tornado  (16,675 plays)
   - **Skeleton Barrel + Goblin Barrel**: Knight, Skeleton Army, Princess, Dart Goblin, Bomb Tower, The Log  (3,012 plays)
   - **Royal Hogs**: Royal Recruits, Zappies, Flying Machine, Electro Spirit, Goblin Cage, Fireball, Barbarian Barrel  (6,830 plays)
3. weakest link p = 0.675  (P: 0.72, 0.68, 0.79)
   - **X-Bow**: Knight, Archers, Skeletons, Ice Spirit, Tesla, Fireball, The Log  (25,305 plays)
   - **Skeleton Barrel + Goblin Barrel**: Knight, Skeleton Army, Princess, Dart Goblin, Bomb Tower, The Log  (3,012 plays)
   - **Royal Hogs**: Royal Recruits, Zappies, Flying Machine, Electro Spirit, Goblin Cage, Fireball, Barbarian Barrel  (6,830 plays)

Sanity check: 102 top-100 deck pairs have ≥100 head-to-head games. Correlation between the model's matchup probability and the raw head-to-head win rate: **0.76** (raw rates are not level/trophy adjusted)

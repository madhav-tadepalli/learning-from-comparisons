#!/usr/bin/env bash
# Full experiment queue. Each run writes results/runs/<name>.json (skips runs already done).
cd "$(dirname "$0")"
PY=${PY:-python}  # override with e.g. PY=.venv/bin/python
run() { [[ -f results/runs/$1.json ]] && return; $PY src/train.py "$@" 2>&1 | grep --line-buffered -E "DONE|Error|ep " >> results/train.log; }

mkdir -p results
# baselines
run constant constant
run trophy trophy
run trophy_level trophy_level
run card_bt card_bt
[[ -f results/runs/gbdt.json ]] || $PY src/train_gbdt.py gbdt >> results/train.log 2>&1

# encoder x pairing head grid
for enc in bag deepset attn; do
  for head in bt bc_inner bc_dist mlp; do
    run ${enc}_${head} enc:$enc:$head
  done
done
run cross cross

# classic blade-chest: one vector per deck id
run deckid_bt deckid:bt d=32
run deckid_bc_inner deckid:bc_inner d=32

# ablation: decks only (no card levels, no trophies)
run card_bt_decksonly card_bt levels=0 trophies=0
for m in bag_bc_inner:enc:bag:bc_inner attn_bc_inner:enc:attn:bc_inner attn_mlp:enc:attn:mlp deepset_mlp:enc:deepset:mlp cross:cross; do
  run ${m%%:*}_decksonly ${m#*:} levels=0 trophies=0
done
[[ -f results/runs/gbdt_decksonly.json ]] || $PY src/train_gbdt.py gbdt_decksonly levels=0 trophies=0 >> results/train.log 2>&1

# blade-chest dimension sweep (bag-of-cards encoder)
for k in 1 2 4 8 32 64; do run bag_bc_inner_k$k enc:bag:bc_inner k=$k; done
echo ALL DONE >> results/train.log

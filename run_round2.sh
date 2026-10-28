#!/usr/bin/env bash
# Round 2: seed-to-seed variance of the strongest models, decks-only for the best heads, light tuning.
# Training checkpoints every epoch, so re-running this script resumes an interrupted run.
cd "$(dirname "$0")"
PY=${PY:-python}  # override with e.g. PY=.venv/bin/python
run() { [[ -f results/runs/$1.json ]] && return; $PY src/train.py "$@" 2>&1 | grep --line-buffered -E "DONE|Error|ep |resumed" >> results/train.log; }

# seed variance for the headline models (seed 0 = the round-1 run)
for s in 1 2; do
  run card_bt_s$s card_bt seed=$s
  run bag_mlp_s$s enc:bag:mlp seed=$s
  run bag_bc_dist_s$s enc:bag:bc_dist seed=$s
  run bag_bc_inner_s$s enc:bag:bc_inner seed=$s
done

# decks only, best fast heads
run bag_mlp_decksonly enc:bag:mlp levels=0 trophies=0
run bag_bc_dist_decksonly enc:bag:bc_dist levels=0 trophies=0

# capacity / regularisation
run bag_mlp_d128 enc:bag:mlp d=128
run bag_mlp_wd1e-4 enc:bag:mlp wd=1e-4
run bag_bc_dist_k32 enc:bag:bc_dist k=32
run bag_bc_dist_k64 enc:bag:bc_dist k=64

# slower models: one more seed each
for s in 1 2; do run deepset_bc_dist_s$s enc:deepset:bc_dist seed=$s; done
run attn_bc_inner_s1 enc:attn:bc_inner seed=1
echo ROUND2 DONE >> results/train.log

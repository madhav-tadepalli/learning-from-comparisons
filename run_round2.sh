#!/usr/bin/env bash
# Round 2: tune the strongest models, measure seed-to-seed variance, give GBDT a longer budget.
cd "$(dirname "$0")"
PY=${PY:-python}  # override with e.g. PY=.venv/bin/python
run() { [[ -f results/runs/$1.json ]] && return; $PY src/train.py "$@" 2>&1 | grep --line-buffered -E "DONE|Error|ep " >> results/train.log; }

# seed variance for the headline models
for s in 1 2; do
  run card_bt_s$s card_bt seed=$s
  run bag_bc_dist_s$s enc:bag:bc_dist seed=$s
  run bag_mlp_s$s enc:bag:mlp seed=$s
done

# capacity / regularisation
run bag_mlp_d128 enc:bag:mlp d=128
run bag_mlp_wd1e-4 enc:bag:mlp wd=1e-4
run bag_bc_dist_k32 enc:bag:bc_dist k=32
run bag_bc_dist_k64 enc:bag:bc_dist k=64
run deepset_mlp_lr1e-3_wd1e-4 enc:deepset:mlp lr=1e-3 wd=1e-4
run attn_mlp_lr1e-3 enc:attn:mlp lr=1e-3

[[ -f results/runs/gbdt_3000.json ]] || $PY src/train_gbdt.py gbdt_3000 iters=3000 >> results/train.log 2>&1
echo ROUND2 DONE >> results/train.log

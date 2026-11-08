"""Gradient-boosted trees baseline (sklearn HistGradientBoosting) on hand-built features.

Features per side: level-weighted card presence (0 if absent, the card's level 1-13 if present),
total level, trophies. Trained on both orderings of each match and
antisymmetrised at prediction time: logit = (f(A,B) - f(B,A)) / 2.

Usage: python src/train_gbdt.py <run_name> [levels=1] [trophies=1] [iters=3000]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train import DATA, OUT, metrics
import torch

run = sys.argv[1]
o = {"levels": 1, "trophies": 1, "iters": 3000}
for kv in sys.argv[2:]:
    k, v = kv.split("="); o[k] = int(v)

df = pd.read_parquet(DATA)
C = len(pd.read_csv("data/processed/cards.csv"))


def side(s):
    cards = df[[f"card{i}_{s}" for i in range(8)]].to_numpy()
    lv = df[[f"level{i}_{s}" for i in range(8)]].to_numpy().astype(np.float32)
    x = np.zeros((len(df), C), np.float32)
    np.put_along_axis(x, cards, lv if o["levels"] else 1.0, 1)
    extra = []
    if o["levels"]: extra.append(lv.sum(1, keepdims=True))
    if o["trophies"]: extra.append(df[[f"trophies_{s}"]].to_numpy(np.float32))
    return np.hstack([x] + extra)


xa, xb = side("a"), side("b")
y = df.a_wins.to_numpy()
sp = df.split.to_numpy()
tr = sp == "train"

X = np.vstack([np.hstack([xa[tr], xb[tr]]), np.hstack([xb[tr], xa[tr]])])
Y = np.concatenate([y[tr], 1 - y[tr]])
t0 = time.time()
clf = HistGradientBoostingClassifier(max_iter=o["iters"], learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0,
                                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                     random_state=0, verbose=0)
clf.fit(X, Y)


def logit(mask):
    f = lambda a, b: clf.decision_function(np.hstack([a, b]))
    return (f(xa[mask], xb[mask]) - f(xb[mask], xa[mask])) / 2


res = {"run": run, "model": "gbdt", "opts": o, "params": int(clf.n_iter_), "train_sec": round(time.time() - t0, 1)}
for s in ("val", "test"):
    m = sp == s
    res[s] = metrics(torch.tensor(logit(m)), torch.tensor(y[m], dtype=torch.float64))
OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / f"{run}.json", "w"), indent=1)
np.save(OUT / f"{run}_test_logit.npy", logit(sp == "test"))
print(f"[{run}] DONE iters={clf.n_iter_} test {res['test']}")

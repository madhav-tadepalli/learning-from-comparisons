"""Which inputs give GBDT its edge? Fit GBDT on feature subsets (600K-match train subsample)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

df = pd.read_parquet("data/processed/s18_ladder_12292020.parquet")
C = len(pd.read_csv("data/processed/cards.csv"))
rng = np.random.default_rng(0)
tr = np.flatnonzero(df.split == "train"); tr = rng.choice(tr, 600_000, replace=False)
te = np.flatnonzero(df.split == "test")


def feats(s, what):
    cards = df[[f"card{i}_{s}" for i in range(8)]].to_numpy()
    lv = df[[f"level{i}_{s}" for i in range(8)]].to_numpy().astype(np.float32)
    cols = []
    if "cards" in what:
        x = np.zeros((len(df), C), np.float32); np.put_along_axis(x, cards, 1.0, 1); cols.append(x)
    if "cardlevels" in what:
        x = np.zeros((len(df), C), np.float32); np.put_along_axis(x, cards, lv, 1); cols.append(x)
    if "levelsum" in what: cols.append(lv.sum(1, keepdims=True))
    if "levelstats" in what: cols.append(np.stack([lv.min(1), lv.max(1), np.sort(lv, 1)[:, 3]], 1))
    if "trophies" in what: cols.append(df[[f"trophies_{s}"]].to_numpy(np.float32))
    return np.hstack(cols)


def fit(what):
    xa, xb = feats("a", what), feats("b", what)
    y = df.a_wins.to_numpy()
    X = np.vstack([np.hstack([xa[tr], xb[tr]]), np.hstack([xb[tr], xa[tr]])]); Y = np.concatenate([y[tr], 1 - y[tr]])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(X, Y)
    f = lambda a, b: clf.decision_function(np.hstack([a, b]))
    lo = (f(xa[te], xb[te]) - f(xb[te], xa[te])) / 2
    p = 1 / (1 + np.exp(-lo))
    print(f"{'+'.join(what):40s} logloss {log_loss(y[te], p):.4f} acc {((p > .5) == y[te]).mean():.4f} auc {roc_auc_score(y[te], p):.4f}", flush=True)


for what in [("levelsum",), ("trophies",), ("levelsum", "trophies"), ("levelsum", "levelstats", "trophies"),
             ("cards",), ("cards", "levelsum", "trophies"), ("cardlevels", "levelsum", "trophies")]:
    fit(what)

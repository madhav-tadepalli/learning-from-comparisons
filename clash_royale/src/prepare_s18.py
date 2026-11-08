"""Clean one S18 battle file into a compact match table.

Keeps only what the models need: both decks (card ids + levels) and both
players' starting trophies. Winner/loser columns are randomly assigned to
sides A/B so the label isn't leaked by column order.

Usage: python src/prepare_s18.py data/raw/s18/BattlesStaging_12292020_WL_tagged.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LADDER = 72000006
RAW = Path(sys.argv[1])
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

cards = pd.read_csv("data/raw/s18/CardMasterListSeason18_12082020.csv", encoding="utf-8-sig")
cards.columns = ["card_id", "name"]
cards = cards.sort_values("card_id").reset_index(drop=True)
cards["idx"] = np.arange(len(cards))
id2idx = dict(zip(cards.card_id, cards.idx))

sides = ["winner", "loser"]
card_cols = [f"{s}.card{i}.{f}" for s in sides for i in range(1, 9) for f in ("id", "level")]
use = ["battleTime", "gameMode.id", "winner.tag", "loser.tag",
       "winner.startingTrophies", "loser.startingTrophies"] + card_cols
df = pd.read_csv(RAW, usecols=use, engine="pyarrow")
log = {"raw rows": len(df)}

df = df[df["gameMode.id"] == LADDER]
log["normal ladder"] = len(df)
df = df.dropna()
log["no missing values"] = len(df)
df = df.drop_duplicates(["battleTime", "winner.tag", "loser.tag"])
log["deduplicated"] = len(df)

ids = {s: df[[f"{s}.card{i}.id" for i in range(1, 9)]].to_numpy(np.int64) for s in sides}
lvls = {s: df[[f"{s}.card{i}.level" for i in range(1, 9)]].to_numpy(np.int64) for s in sides}
ok = np.ones(len(df), bool)
for s in sides:
    ok &= np.isin(ids[s], cards.card_id.to_numpy()).all(1)
    ok &= (np.sort(ids[s], 1)[:, 1:] != np.sort(ids[s], 1)[:, :-1]).all(1)  # 8 distinct cards
    ok &= ((lvls[s] >= 1) & (lvls[s] <= 13)).all(1)
log["valid decks"] = int(ok.sum())

df = df[ok].reset_index(drop=True)
ids = {s: v[ok] for s, v in ids.items()}
lvls = {s: v[ok] for s, v in lvls.items()}

# sort each deck by card index (order in the API is meaningless), carry levels along
for s in sides:
    idx = np.vectorize(id2idx.get)(ids[s])
    o = np.argsort(idx, 1)
    ids[s] = np.take_along_axis(idx, o, 1)
    lvls[s] = np.take_along_axis(lvls[s], o, 1)

rng = np.random.default_rng(0)
a_won = rng.random(len(df)) < 0.5
pick = lambda w, l: np.where(a_won[:, None] if w.ndim == 2 else a_won, w, l)

out = pd.DataFrame({
    "time": pd.to_datetime(df.battleTime, utc=True),
    "tag_a": np.where(a_won, df["winner.tag"], df["loser.tag"]),
    "tag_b": np.where(a_won, df["loser.tag"], df["winner.tag"]),
    "trophies_a": pick(df["winner.startingTrophies"].to_numpy(), df["loser.startingTrophies"].to_numpy()).astype(np.int32),
    "trophies_b": pick(df["loser.startingTrophies"].to_numpy(), df["winner.startingTrophies"].to_numpy()).astype(np.int32),
    "a_wins": a_won.astype(np.int8),
})
for name, (w, l) in {"a": ("winner", "loser"), "b": ("loser", "winner")}.items():
    c = pick(ids[w], ids[l]); v = pick(lvls[w], lvls[l])
    for i in range(8):
        out[f"card{i}_{name}"] = c[:, i].astype(np.int16)
        out[f"level{i}_{name}"] = v[:, i].astype(np.int8)

# time-based split: last 10% of battles = test, previous 10% = val
out = out.sort_values("time").reset_index(drop=True)
n = len(out)
out["split"] = np.where(np.arange(n) < 0.8 * n, "train", np.where(np.arange(n) < 0.9 * n, "val", "test"))

out.to_parquet(OUT / "s18_ladder_12292020.parquet", index=False)
cards[["idx", "card_id", "name"]].to_csv(OUT / "cards.csv", index=False)

for k, v in log.items():
    print(f"{k:>20}: {v:,}")
print(out.split.value_counts().to_dict())
print("time range:", out.time.min(), "->", out.time.max())
print("A win rate:", out.a_wins.mean().round(4))

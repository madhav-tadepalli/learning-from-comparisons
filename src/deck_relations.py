"""Which decks are similar, and which decks counter which, from a trained matchup model.

Pool = the N most-played training decks. Matchups are evaluated with all cards at level 12 and equal
trophies, so they reflect the decks only. M = logit matrix, split into strength s (transitive) and
cyclic part C (rock-paper-scissors).

  similarity:  correlation of two decks' cyclic matchup rows (they beat / lose to the same decks),
               compared with embedding cosine and plain card overlap.
  counters:    P(opponent beats target), and C (how much more than its overall strength predicts).
  validation:  held-out test games — do model-identified counters win more than a transitive model
               (card_bt, with real levels and trophies) predicts?

Usage: python src/deck_relations.py [model=bag_mlp] [check=attn_bc_inner] [n=1000]
Writes results/deck_relations.md, results/deck_{pool,similar,counters}.csv, results/figures/deck_similarity.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import decks as K
import train as T

o = {"model": "bag_mlp", "check": "attn_bc_inner", "n": "1000"}
for kv in sys.argv[1:]:
    k, v = kv.split("="); o[k] = v
N = int(o["n"])
out = []
say = lambda s="": (out.append(s), print(s))
sig = lambda x: 1 / (1 + np.exp(-x))

df = pd.read_parquet(T.DATA)
D, n_tr, n_all = K.deck_pool(df, N)
labels = [K.label(d) for d in D]
say(f"# Deck similarity and counters (`{o['model']}`, cross-checked with `{o['check']}`)\n")
say(f"Pool: the {N} most-played decks in train ({n_tr[-1]}+ train plays each; together "
    f"{n_all.sum() / (2 * len(df)):.0%} of all deck appearances). All cards level 12, equal trophies.\n")

model, check = K.load_model(o["model"]), K.load_model(o["check"])
M, M2 = K.matchups(model, D, D), K.matchups(check, D, D)
s, C = K.split_cyclic(M)
s2, C2 = K.split_cyclic(M2)
P = sig(M)
iu = np.triu_indices(N, 1)

# ------------------------------------------------------------ similarity
prof = np.corrcoef(C)                       # matchup-profile similarity
prof2 = np.corrcoef(C2)
Z = K.embed(model, D); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
cos = Z @ Z.T
onehot = np.zeros((N, len(K.name))); np.put_along_axis(onehot, D, 1, 1)
overlap = onehot @ onehot.T                 # shared cards, 0..8

say("## How similarity measures relate (all deck pairs in the pool)\n")
say("| | profile sim (check model) | embedding cosine | shared cards |")
say("|---|---:|---:|---:|")
say(f"| profile sim (Spearman) | {spearmanr(prof[iu], prof2[iu])[0]:.2f} | {spearmanr(prof[iu], cos[iu])[0]:.2f} | "
    f"{spearmanr(prof[iu], overlap[iu])[0]:.2f} |")
say("\nProfile similarity = correlation of the two decks' cyclic matchup rows: high means they beat and lose to "
    "the same opponents (beyond overall strength).\n")
say("| shared cards | pairs | mean profile sim |")
say("|---:|---:|---:|")
for k in range(9):
    m = overlap[iu] == k
    if m.sum(): say(f"| {k} | {m.sum():,} | {prof[iu][m].mean():+.2f} |")

say("\n### Functional look-alikes: similar matchups, few shared cards (≤ 2)\n")
cand = np.where((overlap[iu] <= 2))[0]
top = cand[np.argsort(-prof[iu][cand])][:8]
for t in top:
    i, j = iu[0][t], iu[1][t]
    say(f"- sim **{prof[i, j]:.2f}** (check model {prof2[i, j]:.2f}, {int(overlap[i, j])} shared)\n"
        f"  - {labels[i]}\n  - {labels[j]}")

# ------------------------------------------------------------ examples for the most-played decks
show = list(range(5))
say("\n## Most similar decks and strongest counters, for the 5 most-played decks\n")
say("P = model P(opponent beats target). C = cyclic part in logits (counter effect beyond the opponent's overall "
    "strength). Observed = raw win rate of that opponent vs target over all games (n ≥ 30, not level-adjusted).\n")
ka, kb = K.deck_keys(df, "a"), K.deck_keys(df, "b")
pos = {tuple(d): i for i, d in enumerate(D)}
pa = np.array([pos.get(d, -1) for d in ka]); pb = np.array([pos.get(d, -1) for d in kb])
yv = df.a_wins.to_numpy()
both = (pa >= 0) & (pb >= 0) & (pa != pb)
h2h_w = np.zeros((N, N)); h2h_n = np.zeros((N, N))
np.add.at(h2h_w, (pa[both], pb[both]), yv[both]); np.add.at(h2h_w, (pb[both], pa[both]), 1 - yv[both])
np.add.at(h2h_n, (pa[both], pb[both]), 1); np.add.at(h2h_n, (pb[both], pa[both]), 1)
rank = (-s).argsort().argsort()
for t in show:
    say(f"### {labels[t]}\n({n_all[t]:,} plays, strength rank {rank[t] + 1}/{N})\n")
    sim = np.argsort(-prof[t]); sim = [j for j in sim if j != t][:5]
    say("Most similar (profile sim / shared cards):")
    for j in sim:
        say(f"- {prof[t, j]:.2f} / {int(overlap[t, j])}: {labels[j]}")
    say("\nStrongest counters (opponent → target):\n")
    say("| opponent | P | C | check-model P | observed (n) |")
    say("|---|---:|---:|---:|---:|")
    for j in np.argsort(-M[:, t])[:6]:
        ob = f"{h2h_w[j, t] / h2h_n[j, t]:.2f} ({int(h2h_n[j, t])})" if h2h_n[j, t] >= 30 else "–"
        say(f"| {labels[j]} | {P[j, t]:.2f} | {C[j, t]:+.2f} | {sig(M2[j, t]):.2f} | {ob} |")
    say("\nBest matchups for the target (target → opponent): " +
        "; ".join(f"{labels[j].split(':')[0]} ({P[t, j]:.2f})" for j in np.argsort(-M[t])[:3]) + "\n")

# ------------------------------------------------------------ validation on held-out games
say("## Are the counters real? Held-out test games\n")
te = (df.split.to_numpy() == "test") & both
lt = np.load(K.RUNS / f"{o['model']}_test_logit.npy"); lb = np.load(K.RUNS / "card_bt_test_logit.npy")
te_idx = np.flatnonzero(df.split.to_numpy() == "test")
m = both[te_idx]
a, b, y = pa[te_idx][m], pb[te_idx][m], yv[te_idx][m]
p_bt, p_mod = sig(lb[m]), sig(lt[m])
c = C[a, b]
say(f"{m.sum():,} test games have both decks in the pool. Binned by the model's cyclic counter score C(A,B) "
    "(computed at equal levels, from training data only). 'card_bt' already accounts for each card's strength, "
    "the real card levels and trophies, but has no matchup term.\n")
say("| C(A,B) bin | games | A's observed win rate | card_bt predicted | full model predicted |")
say("|---|---:|---:|---:|---:|")
edges = [-np.inf, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, np.inf]
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (c >= lo) & (c < hi)
    if k.sum() >= 50:
        say(f"| [{lo:+.1f}, {hi:+.1f}) | {k.sum():,} | {y[k].mean():.3f} | {p_bt[k].mean():.3f} | {p_mod[k].mean():.3f} |")
r = np.corrcoef(c, y - p_bt)[0, 1]
say(f"\nCorrelation of C(A,B) with the card_bt residual (observed − predicted): {r:+.3f}.")
say(f"Agreement between models on counter scores C (Spearman over all pairs): "
    f"{spearmanr(C[iu], C2[iu])[0]:.2f}; on strengths s: {spearmanr(s, s2)[0]:.2f}.")
top5 = lambda X, t: set(np.argsort(-X[:, t])[:5])
say(f"Top-5 counter lists shared between the two models, averaged over the 100 most-played targets: "
    f"{np.mean([len(top5(M, t) & top5(M2, t)) for t in range(100)]):.1f} of 5.")

# ------------------------------------------------------------ files + figure
pd.DataFrame({"deck": range(N), "label": labels, "cards": [", ".join(K.name[c] for c in d) for d in D],
              "plays_train": n_tr, "plays_all": n_all, "strength": s, "strength_rank": rank + 1}
             ).to_csv("results/deck_pool.csv", index=False)
rows = []
for t in range(N):
    for j in [j for j in np.argsort(-prof[t]) if j != t][:10]:
        rows.append({"deck": t, "similar": j, "profile_sim": prof[t, j], "embed_cos": cos[t, j],
                     "shared_cards": int(overlap[t, j]), "similar_label": labels[j]})
pd.DataFrame(rows).to_csv("results/deck_similar.csv", index=False)
rows = []
for t in range(N):
    for j in np.argsort(-M[:, t])[:10]:
        rows.append({"target": t, "counter": j, "p_counter_wins": P[j, t], "cyclic": C[j, t],
                     "check_p": sig(M2[j, t]), "h2h_games": int(h2h_n[j, t]),
                     "h2h_counter_winrate": h2h_w[j, t] / h2h_n[j, t] if h2h_n[j, t] else np.nan,
                     "counter_label": labels[j]})
pd.DataFrame(rows).to_csv("results/deck_counters.csv", index=False)

n_fig = 40
sub = prof[:n_fig, :n_fig]
order = leaves_list(linkage(C[:n_fig], "average", metric="correlation"))
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(sub[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
short = [l.split(":")[0] + " (" + ", ".join(x.strip() for x in l.split(":")[1].split(",")[:2]) + ")" for l in labels]
ax.set_xticks(range(n_fig)); ax.set_yticks(range(n_fig))
ax.set_xticklabels([short[i] for i in order], rotation=90, fontsize=7)
ax.set_yticklabels([short[i] for i in order], fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.04, label="matchup-profile similarity")
ax.set_title(f"Top-{n_fig} decks, clustered by who they beat and lose to ({o['model']})")
fig.tight_layout(); fig.savefig("results/figures/deck_similarity.png", dpi=150); plt.close(fig)

Path("results/deck_relations.md").write_text("\n".join(out) + "\n")

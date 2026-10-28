"""Look up any 8-card deck: similar decks, its archetype, what counters it, what it beats.

The deck doesn't need to appear in the data: the encoders build an embedding from its cards.
Comparisons are against the 1000 most-played decks, all at level 12 with equal trophies.
Needs results/archetypes.csv (run src/embedding_analysis.py once).

Usage: python src/deck_query.py "Hog Rider, Musketeer, Ice Golem, Ice Spirit, Skeletons, Cannon, Fireball, The Log"
"""
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import decks as K
import train as T

sig = lambda x: 1 / (1 + np.exp(-x))
lookup = {n.lower(): i for i, n in enumerate(K.name)}


def parse(text):
    idx = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        hit = lookup.get(part.lower()) or lookup.get(
            (difflib.get_close_matches(part.lower(), lookup, n=1, cutoff=0.6) or [None])[0])
        if hit is None: sys.exit(f"unknown card: {part!r}")
        idx.append(hit)
    if len(set(idx)) != 8: sys.exit(f"need 8 distinct cards, got {[K.name[i] for i in idx]}")
    return np.sort(idx)


q = parse(" ".join(sys.argv[1:]))
df = pd.read_parquet(T.DATA)
D, _, n_all = K.deck_pool(df, 1000)
arch = pd.read_csv("results/archetypes.csv").archetype.to_numpy()
w = n_all / n_all.sum()
mlp, bc = K.load_model("bag_mlp"), K.load_model("attn_bc_inner")
X = np.vstack([q[None], D])
P = (sig(K.matchups(mlp, X[:1], X)[0]) + sig(K.matchups(bc, X[:1], X)[0])) / 2   # P(query beats deck), 2-model mean
P = P[1:]
Z = K.embed(bc, X); Z -= K.embed(bc, D).mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
cos = Z[1:] @ Z[0]

print(f"\nDeck: {K.label(q)}")
hit = np.flatnonzero((D == q).all(1))
print(f"In the top-1000 pool: {'yes (#%d, %d plays)' % (hit[0] + 1, n_all[hit[0]]) if len(hit) else 'no'}")
print(f"Expected win rate vs the meta (play-weighted top-1000, equal levels): {P @ w:.3f}")

near = np.argsort(-cos)[:8]
votes = pd.Series(w[near], index=arch[near]).groupby(level=0).sum().sort_values(ascending=False)
print(f"Closest archetype: {votes.index[0]}")
print("\nMost similar decks (embedding cosine):")
for j in near[:5]:
    print(f"  {cos[j]:.2f}  {K.label(D[j])}")

a = pd.DataFrame({"arch": arch, "p": 1 - P, "w": w})
ga = a.groupby("arch").apply(lambda g: np.average(g.p, weights=g.w), include_groups=False).sort_values(ascending=False)
print("\nArchetypes that beat it (avg P they win):")
for n, v in ga.head(3).items():
    print(f"  {v:.2f}  {n}")
print("Archetypes it beats:")
for n, v in ga.tail(3)[::-1].items():
    print(f"  {1 - v:.2f}  {n}")
print("\nStrongest single counter decks:")
for j in np.argsort(P)[:5]:
    print(f"  {1 - P[j]:.2f}  {K.label(D[j])}")

ctype = K.cards.card_id.to_numpy() // 1_000_000
cand, swaps = [], []
for i in range(8):
    for c in range(len(K.name)):
        if c in q or ctype[c] != ctype[q[i]] or (c in K.wc_idx) != (q[i] in K.wc_idx): continue
        d = q.copy(); d[i] = c; cand.append(np.sort(d)); swaps.append((K.name[q[i]], K.name[c]))
cand = np.array(cand)
ev = lambda m: (sig(K.matchups(m, cand, D)) * w).sum(1)
e = (ev(mlp) + ev(bc)) / 2
print("\nBest like-for-like single-card swaps (meta win rate; hypotheses, not advice):")
for k in np.argsort(-e)[:3]:
    print(f"  {swaps[k][0]} -> {swaps[k][1]}: {e[k]:.3f}")

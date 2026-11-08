"""What can be read off the learned deck embeddings?

1. Similarity: is embedding distance a good proxy for "plays the same matchups"?
2. Counters: in blade-chest models the cyclic term is blade_A.chest_B - blade_B.chest_A, so the counters of a
   deck are a max-inner-product query in embedding space. With the linear bag encoder this even gives the
   best 8-card counter deck in closed form (per-card scores, take the top 8).
3. Low-dimensional cyclic structure: Schur planes of the cyclic matchup matrix (mElo-style "RPS map").
4. Archetypes: cluster the embeddings, then archetype-vs-archetype matchups and grouped counter lists.

Usage: python src/embedding_analysis.py [n=1000] [clusters=14]
Writes results/embedding_analysis.md, results/archetypes.csv, results/figures/{rps_plane,archetype_matchups}.png
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
import decks as K
import train as T

o = {"n": "1000", "clusters": "14"}
for kv in sys.argv[1:]:
    k, v = kv.split("="); o[k] = v
N, NC = int(o["n"]), int(o["clusters"])
out = []
say = lambda s="": (out.append(s), print(s))
sig = lambda x: 1 / (1 + np.exp(-x))
FIG = Path("results/figures")

df = pd.read_parquet(T.DATA)
D, n_tr, n_all = K.deck_pool(df, N)
labels = [K.label(d) for d in D]
w = n_all / n_all.sum()                     # meta weights = play share within the pool
iu = np.triu_indices(N, 1)
onehot = np.zeros((N, len(K.name))); np.put_along_axis(onehot, D, 1, 1)
overlap = onehot @ onehot.T

say("# What the deck embeddings tell us\n")
say(f"Pool: the {N} most-played training decks. All cards level 12, equal trophies.\n")


def unit(Z):
    Z = Z - Z.mean(0)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


# ------------------------------------------------------------ 1. similarity
runs = [r for r in ["bag_bt", "attn_bt", "bag_bc_inner", "bag_bc_dist", "deepset_bc_dist", "attn_bc_inner",
                    "attn_bc_dist", "bag_mlp", "deepset_mlp", "attn_mlp"] if (K.RUNS / f"{r}.pt").exists()]
models, Ms, Zs = {}, {}, {}
for r in runs:
    models[r] = K.load_model(r); Ms[r] = K.matchups(models[r], D, D); Zs[r] = K.embed(models[r], D)
ref = "attn_bc_inner"
_, Cref = K.split_cyclic(Ms[ref])
prof_ref = np.corrcoef(Cref)
prof_mlp = np.corrcoef(K.split_cyclic(Ms["bag_mlp"])[1])  # second yardstick, so no model is only graded by itself

say("## 1. Similar decks from embedding distance\n")
say(f"'Matchup similarity' = correlation of two decks' cyclic matchup rows (they beat and lose to the same "
    f"decks), from `{ref}` for every row so it is a fixed yardstick. Spearman over all {len(iu[0]):,} deck pairs. "
    "'Within same overlap' = mean Spearman computed separately among pairs sharing exactly k cards (k = 0..6), "
    "i.e. what the embedding knows beyond counting shared cards. 'Top-10' = of each deck's 10 nearest "
    "neighbours by embedding cosine, how many are in its 10 most matchup-similar decks (mean over top-200 decks).\n")
say("| embedding from | vs matchup similarity | vs matchup similarity (`bag_mlp` yardstick) | within same overlap | top-10 neighbours |")
say("|---|---:|---:|---:|---:|")
say(f"| (shared-card count) | {spearmanr(overlap[iu], prof_ref[iu])[0]:.2f} | {spearmanr(overlap[iu], prof_mlp[iu])[0]:.2f} | – | "
    f"{np.mean([len(set(np.argsort(-overlap[t], kind="stable")[1:11]) & set(np.argsort(-prof_ref[t])[1:11])) for t in range(200)]):.1f} |")
for r in runs:
    Zu = unit(Zs[r]); cos = Zu @ Zu.T
    within = np.mean([spearmanr(cos[iu][m], prof_ref[iu][m])[0]
                      for m in [overlap[iu] == k for k in range(7)] if m.sum() > 200])
    top = np.mean([len(set(np.argsort(-cos[t])[1:11]) & set(np.argsort(-prof_ref[t])[1:11])) for t in range(200)])
    say(f"| `{r}` | {spearmanr(cos[iu], prof_ref[iu])[0]:.2f} | {spearmanr(cos[iu], prof_mlp[iu])[0]:.2f} | {within:.2f} | {top:.1f} |")

# ------------------------------------------------------------ 2. counters straight from blade/chest vectors
say("\n## 2. Counters read directly from the embeddings (blade-chest)\n")
say("Blade-chest inner product: logit(A beats B) = s_A − s_B + blade_A·chest_B − blade_B·chest_A. "
    "So deck t's counters are the decks maximising the inner product of [blade, chest] with the query "
    "[chest_t, −blade_t]: a nearest-neighbour search in embedding space, no pairwise model evaluation needed.\n")
say("| model | cyclic term vs Hodge cyclic part (Spearman) | counter rank agreement with `bag_mlp` (Spearman, mean over top-100 targets) |")
say("|---|---:|---:|")
for r in [r for r in runs if "bc_inner" in r]:
    h = models[r].head
    with torch.no_grad():
        z = torch.tensor(Zs[r], device=K.dev)
        b, c, s = h.blade(z).cpu().numpy(), h.chest(z).cpu().numpy(), h.s(z).cpu().numpy()[:, 0]
    G = b @ c.T; Cbc = G - G.T
    assert np.allclose(Ms[r], s[:, None] - s[None] + Cbc, atol=1e-3)
    _, Cr = K.split_cyclic(Ms[r])
    agree = np.mean([spearmanr(Ms[r][:, t], Ms["bag_mlp"][:, t])[0] for t in range(100)])
    say(f"| `{r}` | {spearmanr(Cbc[iu], Cr[iu])[0]:.2f} | {agree:.2f} |")

# best counter deck in closed form (bag encoder is linear: z = sum of card tokens)
m = models["bag_bc_inner"]; h = m.head
with torch.no_grad():
    tok = m.enc.tok.emb.weight                                  # level 12 => level term is zero
    tb, tc, ts = h.blade(tok).cpu().numpy(), h.chest(tok).cpu().numpy(), h.s(tok).cpu().numpy()[:, 0]
zb = onehot @ tb; zc = onehot @ tc                              # blade / chest of every pool deck


def card_scores(t):
    """Contribution of each card to logit(opponent containing it beats deck t)."""
    return ts + tb @ zc[t] - tc @ zb[t]


say("\nBecause the bag encoder is linear, logit(opponent beats t) is a *sum over the opponent's cards* of "
    "per-card scores, so the strongest possible counter to t (under `bag_bc_inner`) is just the 8 top-scoring "
    "cards. Built decks are then scored by the two other model families (`bag_mlp`, `attn_bc_inner`) as an "
    "independent check, and compared with the best counter that actually occurs in the pool. Treat built decks "
    "as suggestions: they can be card combinations nobody plays, where every model is extrapolating.\n")
say("| target deck | built counter (top-8 cards) | P by bag_mlp / attn_bc_inner | best pool counter, P by bag_mlp | built deck played? |")
say("|---|---|---:|---:|---:|")
seen = set(map(tuple, D)) | set(K.deck_keys(df, "a")) | set(K.deck_keys(df, "b"))
for t in range(5):
    best = np.sort(np.argsort(-card_scores(t))[:8])
    B = best[None]
    p1 = sig(K.matchups(models["bag_mlp"], B, D[t:t + 1])[0, 0]); p2 = sig(K.matchups(models[ref], B, D[t:t + 1])[0, 0])
    j = np.argmax(Ms["bag_mlp"][:, t])
    say(f"| {labels[t]} | {K.label(best)} | {p1:.2f} / {p2:.2f} | {sig(Ms['bag_mlp'][j, t]):.2f} | "
        f"{'yes' if tuple(best) in seen else 'no'} |")

# best deck against the whole meta (expected logit vs play-weighted pool is again linear in cards)
meta_score = ts + tb @ (w @ zc) - tc @ (w @ zb)
best = np.sort(np.argsort(-meta_score)[:8])
ev = lambda X, r: (sig(K.matchups(models[r], X, D)) * w).sum(1)
pool_best = np.argmax(ev(D, "bag_mlp"))
say(f"\n**Best deck against the whole meta** (play-weighted pool, built from per-card scores): {K.label(best)}. "
    f"Expected win rate vs the meta: {ev(best[None], 'bag_mlp')[0]:.3f} (bag_mlp) / {ev(best[None], ref)[0]:.3f} "
    f"({ref}); played in the data: {'yes' if tuple(best) in seen else 'no'}. Best deck that is actually played: "
    f"{labels[pool_best]} at {ev(D[pool_best:pool_best + 1], 'bag_mlp')[0]:.3f}.\n")
cs = pd.Series(meta_score, index=K.name).sort_values()
say("Cards that help most / least against this meta (per-card score, logits): "
    + ", ".join(f"{n} {v:+.2f}" for n, v in cs[::-1][:8].items()) + " … "
    + ", ".join(f"{n} {v:+.2f}" for n, v in cs[:5].items()) + "\n")

say("The unconstrained builds stack win conditions in a way no real deck does, and the two independent models "
    "disagree wildly on them: the models are only trustworthy near decks people actually play. A safer use is "
    "a *local* search: the best like-for-like single-card swap to a real deck. Even these lean on each card's overall "
    "strength, which can partly reflect who plays the card (player skill is not in the data), so read them as "
    "hypotheses to test, not advice.\n")
say("### Best single-card swap against the meta, for the 5 most-played decks\n")
ctype = K.cards.card_id.to_numpy() // 1_000_000  # 26 troop, 27 building, 28 spell
say("Like-for-like swaps only (troop for troop, spell for spell, building for building, win condition for win "
    "condition), scored by expected win rate against the "
    "play-weighted pool. Ranked by the mean of `bag_mlp` and `attn_bc_inner`, so a swap only wins if both agree.\n")
say("| deck | meta win rate now (mlp / attn) | best swap | after (mlp / attn) |")
say("|---|---:|---|---:|")
for t in range(5):
    cand = []
    for i in range(8):
        for c in range(len(K.name)):
            if c in D[t] or ctype[c] != ctype[D[t][i]] or (c in K.wc_idx) != (D[t][i] in K.wc_idx): continue
            d = D[t].copy(); d[i] = c; cand.append(np.sort(d))
    cand = np.array(cand)
    e1, e2 = ev(cand, "bag_mlp"), ev(cand, ref)
    k = np.argmax(e1 + e2); new = cand[k]
    out_c = [K.name[c] for c in D[t] if c not in new][0]; in_c = [K.name[c] for c in new if c not in D[t]][0]
    b1, b2 = ev(D[t:t + 1], "bag_mlp")[0], ev(D[t:t + 1], ref)[0]
    say(f"| {labels[t]} | {b1:.3f} / {b2:.3f} | {out_c} → **{in_c}** | {e1[k]:.3f} / {e2[k]:.3f} |")
say("")

# ------------------------------------------------------------ 3. low-dimensional cyclic structure
say("## 3. How many 'rock-paper-scissors dimensions' are there?\n")
say("An antisymmetric matrix splits into 2-D planes (real Schur form); each plane is one rock-paper-scissors "
    "cycle. Share of the cyclic matchup variance captured by the first k planes:\n")
say("| model | 1 plane | 2 | 3 | 5 | 10 |")
say("|---|---:|---:|---:|---:|---:|")
for r in ["bag_mlp", ref, "bag_bc_dist", "attn_mlp"]:
    _, Cr = K.split_cyclic(Ms[r])
    sv = np.linalg.svd(Cr, compute_uv=False) ** 2   # singular values come in equal pairs, one pair per plane
    cum = np.cumsum(sv) / sv.sum()
    say(f"| `{r}` | " + " | ".join(f"{cum[2 * k - 1]:.0%}" for k in (1, 2, 3, 5, 10)) + " |")

_, Cb = K.split_cyclic(Ms["bag_mlp"])
U, S, _ = np.linalg.svd(Cb)
a, b = U[:, 0], U[:, 1]
sgn = np.sign(a @ Cb @ b)
P2 = np.c_[a, sgn * b] * np.sqrt(S[0])   # C_ij ≈ cross(p_i, p_j): i beats j when j lies counter-clockwise of i

# ------------------------------------------------------------ 4. archetypes from embedding clusters
say("\n## 4. Archetypes from clustering the embeddings\n")
Zu = unit(Zs[ref])
km = KMeans(NC, n_init=20, random_state=0).fit(Zu, sample_weight=w)
lab = km.labels_


def arch_name(k):
    mem = np.flatnonzero(lab == k)
    wc = Counter()
    for i in mem:
        wc[labels[i].split(":")[0]] += n_all[i]
    top = wc.most_common(1)[0][0]
    lift = (onehot[mem] * n_all[mem, None]).sum(0) / n_all[mem].sum() - (onehot * n_all[:, None]).sum(0) / n_all.sum()
    keys = [K.name[c] for c in np.argsort(-lift) if K.name[c] not in top][:2]
    return f"{top} ({', '.join(keys)})"


names = [arch_name(k) for k in range(NC)]
share = np.array([w[lab == k].sum() for k in range(NC)])
say(f"KMeans (k = {NC}) on `{ref}` embeddings, weighted by plays. Name = dominant win condition + the two "
    "cards most over-represented in the cluster.\n")
Pb = sig(Ms["bag_mlp"])
A = np.zeros((NC, NC))
for i in range(NC):
    for j in range(NC):
        wi, wj = w * (lab == i), w * (lab == j)
        A[i, j] = wi @ Pb @ wj / (wi.sum() * wj.sum())
order = np.argsort(-share)
say("| archetype | share of pool plays | decks | win-con purity | best matchup (P) | worst matchup (P) |")
say("|---|---:|---:|---:|---|---|")
for k in order:
    mem = np.flatnonzero(lab == k)
    wcs = Counter()
    for i in mem:
        wcs[labels[i].split(':')[0]] += n_all[i]
    pur = wcs.most_common(1)[0][1] / n_all[mem].sum()
    o_ = [j for j in np.argsort(-A[k]) if j != k]
    say(f"| {names[k]} | {share[k]:.1%} | {len(mem)} | {pur:.0%} | {names[o_[0]]} ({A[k, o_[0]]:.2f}) | "
        f"{names[o_[-1]]} ({A[k, o_[-1]]:.2f}) |")

say("\n### Counters grouped by archetype, for the 5 most-played decks\n")
for t in range(5):
    pa = [(Pb[lab == k, t] @ w[lab == k]) / w[lab == k].sum() for k in range(NC)]
    top = np.argsort(pa)[::-1][:3]
    bits = []
    for k in top:
        mem = np.flatnonzero(lab == k); j = mem[np.argmax(Pb[mem, t])]
        bits.append(f"**{names[k]}** avg P {pa[k]:.2f} (best: {labels[j]}, {Pb[j, t]:.2f})")
    say(f"- {labels[t]} ({names[lab[t]]}) is countered by:\n  - " + "\n  - ".join(bits))

pd.DataFrame({"deck": range(N), "label": labels, "archetype": [names[k] for k in lab], "plays_all": n_all,
              "rps_x": P2[:, 0], "rps_y": P2[:, 1]}).to_csv("results/archetypes.csv", index=False)

# ------------------------------------------------------------ figures
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(10, 9))
for k in order:
    mm = lab == k
    ax.scatter(P2[mm, 0], P2[mm, 1], s=np.sqrt(n_all[mm]) / 3, color=cmap(k % 20), alpha=.6, label=names[k])
for k in order:
    mm = lab == k; c = (P2[mm] * n_all[mm, None]).sum(0) / n_all[mm].sum()
    ax.annotate(names[k].split(" (")[0], c, fontsize=8, weight="bold", ha="center", color=cmap(k % 20),
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")])
ax.axhline(0, color="grey", lw=.5); ax.axvline(0, color="grey", lw=.5)
ax.set_title("First rock-paper-scissors plane of the cyclic matchups (bag_mlp)\n"
             "decks tend to beat decks lying counter-clockwise of them (within 180°);\n"
             "farther from the centre = stronger effect", fontsize=10)
ax.legend(fontsize=7, markerscale=.6, loc="center left", bbox_to_anchor=(1, .5))
fig.tight_layout(); fig.savefig(FIG / "rps_plane.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(A[np.ix_(order, order)], cmap="RdBu", vmin=0.3, vmax=0.7)
ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
ax.set_xticklabels([names[k] for k in order], rotation=90, fontsize=7)
ax.set_yticklabels([names[k] for k in order], fontsize=7)
for i, ki in enumerate(order):
    for j, kj in enumerate(order):
        ax.text(j, i, f"{A[ki, kj]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, fraction=0.04, label="P(row archetype beats column), bag_mlp, equal levels")
ax.set_title("Archetype matchups (embedding clusters)")
fig.tight_layout(); fig.savefig(FIG / "archetype_matchups.png", dpi=150); plt.close(fig)

Path("results/embedding_analysis.md").write_text("\n".join(out) + "\n")

"""Summarise all runs and probe what the best models learned. Writes results/summary.md + figures."""
import json
import sys
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T

RUNS, FIG = Path("results/runs"), Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)
out = []
say = lambda s="": (out.append(s), print(s))

df = pd.read_parquet(T.DATA)
cards = pd.read_csv("data/processed/cards.csv")
name = cards.name.to_numpy()
y_test = df.a_wins.to_numpy()[df.split.to_numpy() == "test"].astype(float)

# ------------------------------------------------------------ results table + paired bootstrap
rows = []
for f in sorted(RUNS.glob("*.json")):
    r = json.load(open(f))
    rows.append({"run": r["run"], "params": r["params"], "train_s": r["train_sec"],
                 **{f"val_{k}": v for k, v in r["val"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}})
res = pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)


def nll(run):
    p = 1 / (1 + np.exp(-np.load(RUNS / f"{run}_test_logit.npy").astype(float)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))


ref = nll("card_bt")


def ci(run, B=500):
    d = nll(run) - ref
    rng = np.random.default_rng(0)  # same resamples for every run
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


res[["d_vs_cardbt", "ci_lo", "ci_hi"]] = [ci(r) for r in res.run]
res.to_csv("results/all_runs.csv", index=False)
say(f"Test set: {len(y_test):,} matches. Δ = test log-loss minus card_bt's (negative = better), 95% paired bootstrap CI.\n")
say("| run | params | val logloss | test logloss | test acc | test AUC | Δ vs card_bt [95% CI] |")
say("|---|---:|---:|---:|---:|---:|---|")
for _, r in res.iterrows():
    say(f"| {r.run} | {r.params:,} | {r.val_logloss:.4f} | {r.test_logloss:.4f} | {r.test_acc:.4f} | {r.test_auc:.4f} | "
        f"{r.d_vs_cardbt:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] |")

seeded = res[res.run.str.contains(r"_s\d+$")].run.str.replace(r"_s\d+$", "", regex=True).unique()
if len(seeded):
    say("\n## Seed variance (seed 0 = the round-1 run)\n")
    say("| model | seeds | test logloss mean ± sd | test acc mean ± sd |")
    say("|---|---:|---|---|")
    for b in seeded:
        g = res[(res.run == b) | res.run.str.fullmatch(rf"{b}_s\d+")]
        say(f"| {b} | {len(g)} | {g.test_logloss.mean():.4f} ± {g.test_logloss.std():.4f} | "
            f"{g.test_acc.mean():.4f} ± {g.test_acc.std():.4f} |")

dev = T.device()


def load_model(run):
    r = json.load(open(RUNS / f"{run}.json"))
    m = T.build(r["model"], len(cards), 1, r["opts"]).to(dev)
    m.load_state_dict(torch.load(RUNS / f"{run}.pt", map_location=dev)); m.eval()
    return m, r


# ------------------------------------------------------------ card strengths & level effects
m, _ = load_model("card_bt")
w = m.w.weight.detach().cpu().numpy().ravel(); v = m.v.weight.detach().cpu().numpy().ravel()
use = pd.Series(np.concatenate([df[[f"card{i}_a" for i in range(8)]].to_numpy().ravel(),
                                df[[f"card{i}_b" for i in range(8)]].to_numpy().ravel()])).value_counts()
cs = pd.DataFrame({"card": name, "strength": w - w[use.index].mean(), "level_slope": v,
                   "usage_pct": (use.reindex(range(len(name))).fillna(0) / (2 * len(df)) * 100).round(2)})
cs = cs[cs.usage_pct > 0.5].sort_values("strength", ascending=False)
cs.to_csv("results/card_strengths.csv", index=False)
say("\n## Card strengths (card_bt, logit units, centred; cards in >0.5% of decks)\n")
say("Strongest: " + ", ".join(f"{r.card} ({r.strength:+.3f})" for r in cs.head(8).itertuples()))
say("Weakest: " + ", ".join(f"{r.card} ({r.strength:+.3f})" for r in cs.tail(8).itertuples()))
tl = json.load(open(RUNS / "trophy_level.json"))
mt, _ = load_model("trophy_level")
g, bt = mt.gamma.item(), mt.skill.beta.item()
say(f"\nTrophy-level model: +1 total card level ⇒ logit {g:+.3f}; +100 trophies ⇒ logit {bt:+.3f}")
say(f"One full level on every card (+8 levels) ⇒ win prob {1 / (1 + np.exp(-8 * g)):.3f} vs an otherwise equal opponent")
say(f"Mean level slope per card (card_bt): {v[cs.index].mean():+.3f} per level")

fig, ax = plt.subplots(figsize=(7, 9))
top = pd.concat([cs.head(15), cs.tail(15)])
ax.barh(top.card[::-1], top.strength[::-1], color=["#c0392b" if s < 0 else "#2e86c1" for s in top.strength[::-1]])
ax.set_xlabel("card strength (logit, centred)"); ax.set_title("Strongest / weakest cards (card BT model)")
fig.tight_layout(); fig.savefig(FIG / "card_strengths.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ card-vs-card counters (bag blade-chest)
if (RUNS / "bag_bc_inner.json").exists():
    m, _ = load_model("bag_bc_inner")
    E = m.enc.tok.emb.weight.detach().cpu()
    Wb, Wc = m.head.blade.weight.detach().cpu(), m.head.chest.weight.detach().cpu()
    S = Wb.T @ Wc - Wc.T @ Wb
    K = (E @ S @ E.T).numpy()  # K[i, j] = logit contribution of card i (mine) vs card j (theirs)
    common = cs.index.to_numpy()
    sub = K[np.ix_(common, common)]
    order = np.dstack(np.unravel_index(np.argsort(-sub.ravel()), sub.shape))[0]
    say("\n## Strongest card-vs-card counters (bag blade-chest; logit added to the side holding the first card)\n")
    seen = 0
    for i, j in order:
        if i == j: continue
        say(f"- {name[common[i]]} vs {name[common[j]]}: {sub[i, j]:+.3f}")
        seen += 1
        if seen == 12: break


# ------------------------------------------------------------ deck-level matchup structure
def deck_table(n_top):
    ka = df[[f"card{i}_a" for i in range(8)]].to_numpy(); kb = df[[f"card{i}_b" for i in range(8)]].to_numpy()
    keys = pd.Series([tuple(r) for r in np.vstack([ka, kb])]).value_counts()
    return [np.array(k) for k in keys.index[:n_top]], keys.to_numpy()[:n_top]


wc = pd.read_csv("data/raw/s18/Wincons.csv")
wc_idx = set(cards[cards.card_id.isin(wc.card_id)].idx)
card_use = use.reindex(range(len(name))).fillna(0).to_numpy()


def label(d, short=False):
    w = [name[c] for c in d if c in wc_idx] or ["no win-con"]
    if not short:
        return f"**{' + '.join(w)}**: " + ", ".join(name[c] for c in d if c not in wc_idx)
    rare = sorted([c for c in d if c not in wc_idx], key=lambda c: card_use[c])[:2]  # most distinctive
    return " + ".join(w) + " (" + ", ".join(name[c] for c in rare) + ")"


decks, counts = deck_table(100)
D = torch.tensor(np.stack(decks), dtype=torch.long, device=dev)
L = torch.zeros_like(D, dtype=torch.float32)  # all cards at level 12 (centred 0), equal trophies
n = len(decks)
ia, ib = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
ia, ib = ia.ravel().to(dev), ib.ravel().to(dev)
zero = torch.zeros(len(ia), device=dev)


def matchups(model):
    with torch.no_grad():
        return model({"cards_a": D[ia], "cards_b": D[ib], "level_a": L[ia], "level_b": L[ib], "troph_a": zero,
                      "troph_b": zero, "deck_a": zero.long(), "deck_b": zero.long()}).view(n, n).cpu().numpy()


def structure(logits, m=60):
    # Hodge-style split: transitive part s_i - s_j (least squares = row means), cyclic part = remainder
    s = logits.mean(1)
    cyc = logits - (s[:, None] - s[None, :])
    win = logits[:m, :m] > 0
    tri = [(i, j, k) for i in range(m) for j in range(i + 1, m) for k in range(j + 1, m)]
    ncyc = sum((win[i, j] and win[j, k] and win[k, i]) or (win[j, i] and win[k, j] and win[i, k]) for i, j, k in tri)
    return s, (cyc ** 2).sum() / (logits ** 2).sum(), ncyc / len(tri)


say("\n## Deck matchup structure (top-100 most-played decks, all cards level 12, equal trophies)\n")
say("Cyclic share = fraction of matchup-logit variance not explained by a single strength per deck (0% = pure ranking). "
    "Cyclic triangles = share of deck triples (top-60) forming a rock-paper-scissors loop.\n")
say("| model | matchup logit std | cyclic share of variance | cyclic triangles |")
say("|---|---:|---:|---:|")
for r in ["card_bt", "deepset_bt", "bag_bc_inner", "bag_bc_dist", "deepset_bc_dist", "attn_bc_inner", "cross", "bag_mlp"]:
    if not (RUNS / f"{r}.json").exists(): continue
    lg = matchups(load_model(r)[0]); _, fr, ct = structure(lg)
    say(f"| {r} | {lg.std():.3f} | {fr:.1%} | {ct:.1%} |")

best = res[res.run.str.contains("bc_|mlp|cross") & ~res.run.str.contains("decksonly|deckid|_k")].iloc[0].run
mb, rb = load_model(best)
logits = matchups(mb)
s, frac, _ = structure(logits)
P = 1 / (1 + np.exp(-logits))
say(f"\nStrongest rock-paper-scissors loops among the top-60 decks (`{best}`; each beats the next):\n")
cycles = []
for i, j, k in permutations(range(60), 3):
    if i < j and i < k:
        mrg = min(P[i, j], P[j, k], P[k, i]) - 0.5
        if mrg > 0: cycles.append((mrg, i, j, k))
cycles.sort(reverse=True)
shown = set()
for mrg, i, j, k in cycles:
    sig = frozenset(label(decks[x], True) for x in (i, j, k))
    if sig in shown: continue
    shown.add(sig)
    say(f"{len(shown)}. weakest link p = {0.5 + mrg:.3f}  (P: {P[i, j]:.2f}, {P[j, k]:.2f}, {P[k, i]:.2f})")
    for x in (i, j, k):
        say(f"   - {label(decks[x])}  ({counts[x]:,} plays)")
    if len(shown) == 3: break

# check against raw head-to-head data for the most-played pairs
key = lambda c: [tuple(r) for r in c]
da = key(df[[f"card{i}_a" for i in range(8)]].to_numpy()); db = key(df[[f"card{i}_b" for i in range(8)]].to_numpy())
pos = {tuple(d): i for i, d in enumerate(decks)}
pa = np.array([pos.get(d, -1) for d in da]); pb = np.array([pos.get(d, -1) for d in db])
mask = (pa >= 0) & (pb >= 0) & (pa != pb)
h2h = pd.DataFrame({"i": np.minimum(pa, pb)[mask], "j": np.maximum(pa, pb)[mask],
                    "i_won": np.where(pa[mask] < pb[mask], df.a_wins.to_numpy()[mask], 1 - df.a_wins.to_numpy()[mask])})
g = h2h.groupby(["i", "j"]).i_won.agg(["mean", "size"]).reset_index()
g = g[g["size"] >= 100]
if len(g) > 5:
    g["model"] = P[g.i, g.j]
    corr = np.corrcoef(g["mean"], g["model"])[0, 1]
    say(f"\nSanity check: {len(g)} top-100 deck pairs have ≥100 head-to-head games. Correlation between the model's "
        f"matchup probability and the raw head-to-head win rate: **{corr:.2f}** (raw rates are not level/trophy adjusted)")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(g["model"], g["mean"], s=np.sqrt(g["size"]) * 2, alpha=.6)
    ax.plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1)
    ax.set_xlabel("model P(deck i beats deck j)"); ax.set_ylabel("observed head-to-head win rate")
    ax.set_title(f"Top-deck matchups ({best})"); fig.tight_layout(); fig.savefig(FIG / "h2h_check.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 7))
o = np.argsort(-s)[:30]
im = ax.imshow(P[np.ix_(o, o)], cmap="RdBu", vmin=0.2, vmax=0.8)
ax.set_xticks([]); ax.set_yticks(range(len(o))); ax.set_yticklabels([label(decks[i], True) for i in o], fontsize=6)
ax.set_title("P(row deck beats column deck), top-30 decks sorted by strength", fontsize=9); fig.colorbar(im)
fig.tight_layout(); fig.savefig(FIG / "matchup_matrix.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ deck embedding map
if hasattr(mb, "embed"):
    decks2, cnt2 = deck_table(2000)
    D2 = torch.tensor(np.stack(decks2), dtype=torch.long, device=dev)
    with torch.no_grad():
        Z = mb.embed(D2, torch.zeros_like(D2, dtype=torch.float32)).cpu().numpy()
    Z = Z - Z.mean(0); U, Sv, Vt = np.linalg.svd(Z, full_matrices=False); XY = Z @ Vt[:2].T
    wcs = [next((name[c] for c in d if c in wc_idx), "none") for d in decks2]
    topwc = pd.Series(wcs).value_counts().index[:10]
    fig, ax = plt.subplots(figsize=(8, 7))
    for w_ in topwc:
        sel = np.array([x == w_ for x in wcs])
        ax.scatter(XY[sel, 0], XY[sel, 1], s=np.sqrt(cnt2[sel]) * 1.5, alpha=.6, label=w_)
    ax.legend(fontsize=7, markerscale=.6)
    ax.set_title(f"Deck embeddings, top-2000 decks, PCA ({best}); colour = first win condition", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "deck_embeddings.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ calibration of the best model
bestall = res.iloc[0].run
p = 1 / (1 + np.exp(-np.load(RUNS / f"{bestall}_test_logit.npy").astype(float)))
bins = np.clip((p * 20).astype(int), 0, 19)
cal = pd.DataFrame({"b": bins, "p": p, "y": y_test}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(cal.p, cal.y, "o-"); ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("predicted P(A wins)"); ax.set_ylabel("observed"); ax.set_title(f"Calibration, test set ({bestall})")
fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 7))
r2 = res[~res.run.str.contains("_k\\d|constant")].sort_values("test_logloss", ascending=False)
ax.barh(r2.run, r2.test_logloss, color=["#7f8c8d" if "decksonly" in x else "#2e86c1" for x in r2.run])
ax.set_xlim(r2.test_logloss.min() - 0.01, min(0.694, r2.test_logloss.max() + 0.002))
ax.set_xlabel("test log-loss (lower is better)"); ax.set_title("All models (grey = decks only, no levels/trophies)")
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=150); plt.close(fig)

open("results/summary.md", "w").write("\n".join(out) + "\n")

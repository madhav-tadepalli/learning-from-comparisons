"""Step 4: on the real sushi data, does the oiliness context effect survive once we model
differences between people?

Part A (paper's data: one choice per person)
  heterogeneity models with and without A_oo (and the other significant single effects):
    RC-oil / RC-all : normal random coefficients on oiliness / on all 6 features (diagonal)
    LC2/LC3-oil, LC3-all : latent classes
    SQ              : squared oiliness term (curvature control)
    DEMO            : observed heterogeneity, tastes depend on gender, age, east/west origin
  and the paper's whole-model test (36 A entries) on top of RC-all.

Part B (the rest of each ranking, which the paper discards)
  Exploded top-k rankings give k choices per person, so tastes are identified within person.
  Panel RC-all with and without A_oo, k = 1, 2, 3, 5.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, load_users, sushi  # noqa: E402
from models import Model, lrt  # noqa: E402

OUT = os.path.join(ROOT, "results", "real")
D = len(FEATURES)
K = OIL
PRICE = 4
ALL = list(range(D))


def nested(make, data, entries, restarts=2, seed=0, R=None):
    """Fit model without and with the given A entries; return (small, big, LRT, p)."""
    kw = {} if R is None else {"R": R}
    small = make(lcl=None, **kw).fit(data, restarts=restarts, seed=seed)
    init = {k: v.clone() for k, v in small.params.items()}
    init["a"] = torch.zeros(len(entries))
    big = make(lcl=entries, **kw).fit(data, restarts=restarts, seed=seed + 1, init=init)
    back = {k: v.clone() for k, v in big.params.items() if k != "a"}
    alt = make(lcl=None, **kw).fit(data, restarts=1, init=back)
    if alt.ll > small.ll:
        small = alt
    stat, p = lrt(big.ll, small.ll, len(entries))
    return small, big, stat, p


def demo_matrix():
    u = load_users()
    age = u["age"].astype(float)
    W = np.stack([u["gender"], (age - age.mean()) / age.std(), u["west15"]], 1).astype(float)
    return torch.tensor(W - W.mean(0))  # centred so theta stays the population-average taste


def fmt(x):
    return np.array2string(np.asarray(x), precision=3, suppress_small=True)


def main():
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    data = sushi(top_k=1)
    res, lines = {}, []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    oo = [(K, K)]
    specs = {
        "MNL": lambda lcl=None, **kw: Model(D, lcl=lcl),
        "SQ-oil": lambda lcl=None, **kw: Model(D, lcl=lcl, quad=[K]),
        "RC-oil": lambda lcl=None, **kw: Model(D, lcl=lcl, het="normal", het_idx=[K], **kw),
        "RC-all": lambda lcl=None, **kw: Model(D, lcl=lcl, het="normal", het_idx=ALL, **kw),
        "LC2-oil": lambda lcl=None, **kw: Model(D, lcl=lcl, het="classes", het_idx=[K], K=2),
        "LC3-oil": lambda lcl=None, **kw: Model(D, lcl=lcl, het="classes", het_idx=[K], K=3),
        "LC2-all": lambda lcl=None, **kw: Model(D, lcl=lcl, het="classes", het_idx=ALL, K=2),
        "LC3-all": lambda lcl=None, **kw: Model(D, lcl=lcl, het="classes", het_idx=ALL, K=3),
        "DEMO": lambda lcl=None, **kw: Model(D, lcl=lcl, demo=demo_matrix()),
        "DEMO+RC-all": lambda lcl=None, **kw: Model(D, lcl=lcl, demo=demo_matrix(), het="normal", het_idx=ALL, **kw),
    }

    log("== Part A: top-1 data (the paper's), test of A[oil,oil] inside each model ==")
    log(f"{'base model':12s} {'ll(base)':>10s} {'ll(+A_oo)':>10s} {'A_oo':>7s} {'LRT':>6s} {'p':>8s}   heterogeneity estimates")
    mnl_ll = None
    for name, make in specs.items():
        restarts = 6 if name.startswith("LC") else 2
        small, big, stat, p = nested(make, data, oo, restarts=restarts)
        if name == "MNL":
            mnl_ll = small.ll
        sm = small.summary()
        extra = ""
        if "sd" in sm:
            extra = f"sd={fmt(sm['sd'])}"
        if "pi" in sm:
            extra = f"pi={fmt(sm['pi'])} delta_oil={fmt(sm['delta'][:, small.het_idx.index(K)])}"
        if "q" in sm:
            extra = f"q_oil={fmt(sm['q'])}"
        if "B" in sm:
            extra = f"B[oil]=(female, age, west) {fmt(sm['B'][K])}"
        a = big.params["a"][0].item()
        log(f"{name:12s} {small.ll:10.1f} {big.ll:10.1f} {a:+7.3f} {stat:6.1f} {p:8.2g}   {extra}")
        res[name] = {"ll": small.ll, "ll_oo": big.ll, "a_oo": a, "lrt": stat, "p": p, "n_params": small.n_params(),
                     "summary": {k: np.asarray(v).tolist() for k, v in sm.items()}}
    log("\nFit vs MNL (higher ll = better; n_params):")
    for name in specs:
        r = res[name]
        aic = -2 * r["ll"] + 2 * r["n_params"]
        log(f"  {name:12s} ll {r['ll']:9.1f}  (+{r['ll'] - mnl_ll:6.1f} vs MNL)  params {r['n_params']:3d}  AIC {aic:8.1f}")

    log("\n== Other significant single effects in the paper's sense, with and without RC-all ==")
    for (p_, q_) in [(PRICE, PRICE), (K, PRICE)]:
        for name in ["MNL", "RC-all"]:
            _, big, stat, p = nested(specs[name], data, [(p_, q_)])
            log(f"  {FEATURES[q_]} on {FEATURES[p_]:12s} in {name:7s}: A={big.params['a'][0].item():+.3f}  LRT {stat:5.1f}  p={p:.2g}")

    log("\n== Paper's whole-model test (all 36 A entries) ==")
    for name in ["MNL", "RC-all", "DEMO+RC-all"]:
        small, big, stat, p = nested(specs[name], data, [(i, j) for i in range(D) for j in range(D)])
        log(f"  LCL vs {name:12s}: LRT {stat:6.1f} (df 36)  p={p:.2g}")
        res[f"full36_{name}"] = {"lrt": stat, "p": p}

    log("\n== Part B: exploded rankings (k choices per person), panel models ==")
    log(f"{'k':>2s} {'model':8s} {'A_oo':>7s} {'LRT':>6s} {'p':>8s}   sd estimates")
    for k in [1, 2, 3, 5]:
        dk = sushi(top_k=k)
        for name in ["MNL", "RC-all"]:
            small, big, stat, p = nested(specs[name], dk, oo, R=100 if k > 1 else None)
            sm = small.summary()
            log(f"{k:2d} {name:8s} {big.params['a'][0].item():+7.3f} {stat:6.1f} {p:8.2g}   "
                f"{fmt(sm['sd']) if 'sd' in sm else ''}")
            res[f"panel_k{k}_{name}"] = {"a_oo": big.params["a"][0].item(), "lrt": stat, "p": p,
                                         "sd": sm["sd"].tolist() if "sd" in sm else None}

    with open(os.path.join(OUT, "real_data.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "real_data.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""Step 3: simulation study on the REAL sushi choice sets (5000 people, the actual 10-item sets).

Ground truths (oiliness = feature k):
  mnl          no context effect, no heterogeneity                       (null control)
  lcl          true context effect A_kk = -0.266 (the paper's isolated estimate), homogeneous people
  both         true A_kk = -0.15 plus normal heterogeneity s = 0.75
  normal  s    continuous taste spread: theta_k,n = mu_k + s z_n            (type 1 heterogeneity)
  cls50   s    two equal segments, theta_k = mu_k +- s                       (type 2: discrete segments)
  cls20   s    a 20% minority segment, same variance s^2 (skewed)            (type 2b)

Every replicate is fitted with:
  MNL; full LCL (the paper's whole-model test); LCL with only A_kk (the paper's single-effect test);
  heterogeneity-aware models, with and without A_kk: normal random coefficient on k (RC),
  2 latent classes on k (LC2), and an MNL with a squared term in k (SQ, the cheapest control).

Output: results/sim/sim_raw.csv (one row per replicate). Summarise with src/summarise_sim.py.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, sushi  # noqa: E402
from models import Model, simulate_top1  # noqa: E402

OUT = os.path.join(ROOT, "results", "sim")
K = OIL
D = len(FEATURES)
S_GRID = [0.5, 0.75, 1.0, 1.25]

_DATA = None
_BASE = None


def _init():
    global _DATA, _BASE
    torch.set_num_threads(1)
    _DATA = sushi(top_k=1)
    mnl = Model(D).fit(_DATA)
    lcl1 = Model(D, lcl=[(K, K)]).fit(_DATA)
    _BASE = {"mu": mnl.params["theta"], "theta_lcl": lcl1.params["theta"], "a_lcl": lcl1.params["a"][0].item()}


def cells():
    out = [("mnl", 0.0), ("lcl", 0.0), ("both", 0.75)]
    for kind in ["normal", "cls50", "cls20"]:
        out += [(kind, s) for s in S_GRID]
    return out


def truth_kwargs(kind, s):
    mu = _BASE["mu"]
    if kind == "mnl":
        return dict(theta=mu)
    if kind == "lcl":
        A = torch.zeros(D, D)
        A[K, K] = _BASE["a_lcl"]
        return dict(theta=_BASE["theta_lcl"], A=A)
    if kind == "both":
        A = torch.zeros(D, D)
        A[K, K] = -0.15
        return dict(theta=mu, A=A, sd=torch.tensor([s]), het_idx=[K])
    if kind == "normal":
        return dict(theta=mu, sd=torch.tensor([s]), het_idx=[K])
    pi = 0.5 if kind == "cls50" else 0.2
    a, b = s * np.sqrt((1 - pi) / pi), -s * np.sqrt(pi / (1 - pi))
    return dict(theta=mu, het_idx=[K], classes=(torch.tensor([pi, 1 - pi]), torch.tensor([[a], [b]])))


KINDS = ["mnl", "lcl", "both", "normal", "cls50", "cls20"]


def task_seed(kind, s, rep):
    return int(np.random.SeedSequence([KINDS.index(kind), int(round(100 * s)), rep]).generate_state(1)[0])


def fit_nested(small, big, data, restarts, seed):
    """Fit small, then big starting from small's optimum (a = 0), then small again from big's
    optimum with a dropped; keep the best of each. Guarantees ll_big >= ll_small."""
    small.fit(data, restarts=restarts, seed=seed)
    init = {k: v.clone() for k, v in small.params.items()}
    init["a"] = torch.zeros(len(big.lcl))
    big.fit(data, restarts=restarts, seed=seed + 1, init=init)
    back = {k: v.clone() for k, v in big.params.items() if k != "a"}
    ll_small, p_small = small.ll, small.params
    small.fit(data, restarts=1, init=back)
    if ll_small > small.ll:
        small.ll, small.params = ll_small, p_small
    if small.ll > big.ll:  # cannot happen at a true optimum; refit big from small
        ll_big, p_big = big.ll, big.params
        init = {k: v.clone() for k, v in small.params.items()}
        init["a"] = torch.zeros(len(big.lcl))
        big.fit(data, restarts=1, init=init)
        if ll_big > big.ll:
            big.ll, big.params = ll_big, p_big


def run_one(task):
    kind, s, rep = task
    t0 = time.time()
    gen = torch.Generator().manual_seed(task_seed(kind, s, rep))
    data = _DATA.with_choice(simulate_top1(_DATA, gen=gen, **truth_kwargs(kind, s)))
    A1 = [(K, K)]
    fits = {
        "mnl": Model(D),
        "lcl_full": Model(D, lcl="full"),
        "lcl_kk": Model(D, lcl=A1),
        "rc": Model(D, het="normal", het_idx=[K]),
        "rc_kk": Model(D, lcl=A1, het="normal", het_idx=[K]),
        "lc2": Model(D, het="classes", het_idx=[K], K=2),
        "lc2_kk": Model(D, lcl=A1, het="classes", het_idx=[K], K=2),
        "sq": Model(D, quad=[K]),
        "sq_kk": Model(D, lcl=A1, quad=[K]),
    }
    for m in ("mnl", "lcl_full", "lcl_kk"):
        fits[m].fit(data)
    fit_nested(fits["rc"], fits["rc_kk"], data, restarts=2, seed=rep)
    fit_nested(fits["lc2"], fits["lc2_kk"], data, restarts=4, seed=rep)
    fit_nested(fits["sq"], fits["sq_kk"], data, restarts=1, seed=rep)
    row = {"kind": kind, "s": s, "rep": rep}
    for name, m in fits.items():
        row[f"ll_{name}"] = m.ll
        if m.lcl:
            row[f"a_{name}"] = m.A(m.params)[K, K].item()
        if m.het == "normal":
            row[f"sd_{name}"] = torch.exp(m.params["log_sd"])[0].item()
        if m.quad:
            row[f"q_{name}"] = m.params["q"][0].item()
    row["secs"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "sim_raw.csv")
    done = set()
    if os.path.exists(path):  # resumable
        with open(path) as f:
            done = {(r["kind"], float(r["s"]), int(r["rep"])) for r in csv.DictReader(f)}
    tasks = [(k, s, r) for k, s in cells() for r in range(args.reps) if (k, s, r) not in done]
    print(f"{len(tasks)} replicates to run ({len(done)} already done)", flush=True)
    writer, fh = None, None
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, row in enumerate(ex.map(run_one, tasks, chunksize=4)):
            if fh is None:
                new = not os.path.exists(path)
                fh = open(path, "a", newline="")
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()

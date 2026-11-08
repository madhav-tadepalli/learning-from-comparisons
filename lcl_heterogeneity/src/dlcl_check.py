"""Step 8: does the heterogeneity critique extend to the DLCL?

The DLCL (Tomlinson & Benson) is a mixture of d = 6 logit components. Component q has its own full
preference vector B_q and the context effect of feature q only. With all context entries set to zero
it is the paper's 6-component "mixed logit" (ML6), so DLCL against ML6 is a test of context effects
that already allows for 6 latent classes of people.

Tests (each an LRT, nested fits started from each other's optimum):
  dlcl36   : DLCL (36 entries) vs ML6                 (the paper's Table 10 test)
  dlcl_oo  : DLCL with only A[oil,oil] (acts in the oiliness component) vs ML6
  ml6_oo   : ML6 + an LCL self-effect A[oil,oil] shared by all classes vs ML6
  lcl36_ml6: ML6 + full shared LCL (36 entries) vs ML6

Modes
  real      : sushi data, top-k rankings for k = 1, 2, 3 (classes shared across a person's choices)
  sim       : simulated top-1 data on the real menus (same ground truths as simulate.py)

A tiny ridge (1e-3 on class offsets and DLCL entries) keeps tiny latent classes from running off to
infinite coefficients, which otherwise happens with one choice per person.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, sushi  # noqa: E402
from models import Model, simulate_top1  # noqa: E402

D = len(FEATURES)
K = OIL
RIDGE = 1e-3
OUT = os.path.join(ROOT, "results", "dlcl")
ALL36 = [(p, q) for p in range(D) for q in range(D)]


def ml6(**kw):
    return Model(D, het="classes", K=D, ridge=RIDGE, **kw)


def fit_family(data, restarts=4, seed=0):
    """Fit ML6 and the three bigger models, each started from ML6's optimum (plus random restarts)."""
    base = ml6().fit(data, restarts=restarts, seed=seed)
    out = {"ml6": base}
    specs = {"dlcl36": dict(dlcl="full"), "dlcl_oo": dict(dlcl=[(K, K)]),
             "ml6_oo": dict(lcl=[(K, K)]), "lcl36_ml6": dict(lcl="full")}
    for name, kw in specs.items():
        m = ml6(**kw)
        init = {k: v.clone() for k, v in base.params.items()}
        if "dlcl" in kw:
            init["da"] = torch.zeros(len(m.dlcl))
        if "lcl" in kw:
            init["a"] = torch.zeros(len(m.lcl))
        m.fit(data, restarts=max(1, restarts // 2), seed=seed + 7, init=init)
        out[name] = m
    return out


def tests(f):
    def lrt(big, df):
        stat = max(0.0, 2 * (f[big].ll - f["ml6"].ll))
        return stat, chi2.sf(stat, df)
    r = {}
    for name, df in [("dlcl36", 36), ("dlcl_oo", 1), ("ml6_oo", 1), ("lcl36_ml6", 36)]:
        r[f"lrt_{name}"], r[f"p_{name}"] = lrt(name, df)
    r["a_dlcl_oo"] = f["dlcl_oo"].A_dlcl(f["dlcl_oo"].params)[K, K].item()
    r["a_dlcl36_oo"] = f["dlcl36"].A_dlcl(f["dlcl36"].params)[K, K].item()
    r["a_ml6_oo"] = f["ml6_oo"].A(f["ml6_oo"].params)[K, K].item()
    r["nll_ml6"], r["nll_dlcl36"] = -f["ml6"].ll, -f["dlcl36"].ll
    return r


# ------------------------------------------------------------------ real data
def run_real():
    torch.set_num_threads(8)
    os.makedirs(OUT, exist_ok=True)
    lines, res = [], {}
    for k in [1, 2, 3]:
        f = fit_family(sushi(top_k=k), restarts=8 if k == 1 else 4)
        r = tests(f)
        res[f"k{k}"] = r
        line = (f"k={k}: NLL ML6 {r['nll_ml6']:.1f}, DLCL {r['nll_dlcl36']:.1f} | "
                f"DLCL vs ML6 LRT {r['lrt_dlcl36']:.1f} p={r['p_dlcl36']:.2g} | "
                f"DLCL A_oo only {r['a_dlcl_oo']:+.3f} LRT {r['lrt_dlcl_oo']:.1f} p={r['p_dlcl_oo']:.2g} | "
                f"ML6 + shared A_oo {r['a_ml6_oo']:+.3f} LRT {r['lrt_ml6_oo']:.1f} p={r['p_ml6_oo']:.2g} | "
                f"ML6 + LCL(36) LRT {r['lrt_lcl36_ml6']:.1f} p={r['p_lcl36_ml6']:.2g} | "
                f"A_oo inside full DLCL {r['a_dlcl36_oo']:+.3f}")
        print(line, flush=True)
        lines.append(line)
    with open(os.path.join(OUT, "dlcl_real.txt"), "w") as fh:
        fh.write("Paper (Adam): ML 9793, DLCL 9764 (EM 9709); LRT 59, p = 0.0089\n" + "\n".join(lines) + "\n")
    with open(os.path.join(OUT, "dlcl_real.json"), "w") as fh:
        json.dump(res, fh, indent=1)


# ------------------------------------------------------------------ simulation
_DATA, _BASE = None, None
CELLS = [("mnl", 0.0), ("lcl", 0.0), ("normal", 0.75), ("normal", 1.0), ("cls50", 1.0), ("cls20", 1.0)]
KINDS = ["mnl", "lcl", "normal", "cls50", "cls20"]


def _init():
    global _DATA, _BASE
    torch.set_num_threads(1)
    import simulate
    simulate._init()
    _DATA, _BASE = simulate._DATA, simulate._BASE


def _truth(kind, s):
    import simulate
    return simulate.truth_kwargs(kind, s)


def run_sim_one(task):
    kind, s, rep = task
    seed = int(np.random.SeedSequence([KINDS.index(kind), int(round(100 * s)), rep, 99]).generate_state(1)[0])
    gen = torch.Generator().manual_seed(seed)
    data = _DATA.with_choice(simulate_top1(_DATA, gen=gen, **_truth(kind, s)))
    r = tests(fit_family(data, restarts=3, seed=rep))
    r.update(kind=kind, s=s, rep=rep)
    return r


def run_sim(reps, workers, rep_start=0, cells=None, out_name="dlcl_sim_raw.csv"):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["kind"], float(r["s"]), int(r["rep"])) for r in csv.DictReader(fh)}
    use = CELLS if cells is None else [CELLS[i] for i in cells]
    tasks = [(k, s, r) for k, s in use for r in range(rep_start, rep_start + reps) if (k, s, r) not in done]
    print(f"{len(tasks)} replicates to run", flush=True)
    fh, w = None, None
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for i, row in enumerate(ex.map(run_sim_one, tasks)):
            if fh is None:
                new = not os.path.exists(path)
                fh = open(path, "a", newline="")
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
            w.writerow(row)
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    if fh:
        fh.close()
    summarise_sim(path)


def summarise_sim(path):
    """Summarise one CSV, or all dlcl_sim_raw*.csv files in OUT (local + Kaggle parts) when path is a dir."""
    import glob

    import pandas as pd
    files = (sorted(glob.glob(os.path.join(path, "**", "dlcl_sim_raw*.csv"), recursive=True))
             if os.path.isdir(path) else [path])
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["kind", "s", "rep"])
    df["order"] = [CELLS.index((k, s)) for k, s in zip(df.kind, df.s)]
    df = df.sort_values(["order", "rep"])
    lines = ["| truth | s | reps | DLCL vs ML6 (36 df) | DLCL A_oo only: mean, flagged | ML6 + shared A_oo: mean, flagged | ML6 + LCL(36) |",
             "|---|---|---|---|---|---|---|"]
    for (kind, s), g in df.groupby(["kind", "s"], sort=False):
        lines.append(f"| {kind} | {s:.2f} | {len(g)} | {np.mean(g.p_dlcl36 < 1e-3):.0%} | "
                     f"{g.a_dlcl_oo.mean():+.3f}, {np.mean(g.p_dlcl_oo < 1e-3):.0%} | "
                     f"{g.a_ml6_oo.mean():+.3f}, {np.mean(g.p_ml6_oo < 1e-3):.0%} | {np.mean(g.p_lcl36_ml6 < 1e-3):.0%} |")
    txt = "Flagged = p < 0.001.\n\n" + "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "dlcl_sim.md"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "sim", "summary"])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--cells", type=int, nargs="*", default=None, help="indices into CELLS")
    ap.add_argument("--out-name", default="dlcl_sim_raw.csv")
    a = ap.parse_args()
    if a.mode == "real":
        run_real()
    elif a.mode == "sim":
        run_sim(a.reps, a.workers, a.rep_start, a.cells, a.out_name)
    else:
        summarise_sim(OUT)

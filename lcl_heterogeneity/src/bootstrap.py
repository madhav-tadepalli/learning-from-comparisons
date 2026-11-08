"""Step 5: parametric bootstrap. If sushi choosers had NO context effects but differed in taste as
much as the sushi data itself says they do, how often would the paper's analysis report
  (a) an oiliness self-effect at least as negative as the paper's (A_oo <= -0.266, alone), and
  (b) LRT statistics at least as large as the paper's (single 23.1, whole-model 97.3)?

Null models are fitted to the real top-1 sushi data, then 5000 choices are re-simulated on the
real choice sets from each null, and the paper's MNL / LCL pipeline is rerun.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, sushi  # noqa: E402
from models import Model, simulate_top1  # noqa: E402

OUT = os.path.join(ROOT, "results", "bootstrap")
D = len(FEATURES)
K = OIL
ALL = list(range(D))
NULLS = ["MNL", "RC-oil", "RC-all", "LC2-oil", "LC3-all"]

_DATA, _NULL = None, None


def fit_nulls(data):
    out = {}
    m = Model(D).fit(data)
    out["MNL"] = dict(theta=m.params["theta"])
    m = Model(D, het="normal", het_idx=[K]).fit(data, restarts=2)
    out["RC-oil"] = dict(theta=m.params["theta"], sd=torch.exp(m.params["log_sd"]), het_idx=[K])
    m = Model(D, het="normal", het_idx=ALL).fit(data, restarts=2)
    out["RC-all"] = dict(theta=m.params["theta"], sd=torch.exp(m.params["log_sd"]), het_idx=ALL)
    m = Model(D, het="classes", het_idx=[K], K=2).fit(data, restarts=6)
    out["LC2-oil"] = dict(theta=m.params["theta"], het_idx=[K],
                          classes=(torch.softmax(m.params["pi_logit"], 0), m.params["delta"]))
    m = Model(D, het="classes", het_idx=ALL, K=3).fit(data, restarts=6)
    out["LC3-all"] = dict(theta=m.params["theta"], het_idx=ALL,
                          classes=(torch.softmax(m.params["pi_logit"], 0), m.params["delta"]))
    return out


def _init():
    global _DATA, _NULL
    torch.set_num_threads(1)
    _DATA = sushi(top_k=1)
    _NULL = fit_nulls(_DATA)


def run_one(task):
    null, b = task
    seed = int(np.random.SeedSequence([NULLS.index(null), b, 7]).generate_state(1)[0])
    gen = torch.Generator().manual_seed(seed)
    data = _DATA.with_choice(simulate_top1(_DATA, gen=gen, **_NULL[null]))
    mnl = Model(D).fit(data)
    kk = Model(D, lcl=[(K, K)]).fit(data)
    full = Model(D, lcl="full").fit(data)
    return {"null": null, "b": b, "a_kk": kk.params["a"][0].item(), "lrt_kk": 2 * (kk.ll - mnl.ll),
            "a_full_oo": full.A(full.params)[K, K].item(), "lrt_full": 2 * (full.ll - mnl.ll)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    torch.set_num_threads(4)
    nulls = fit_nulls(sushi(top_k=1))
    desc = {}
    for k, v in nulls.items():
        desc[k] = {kk: (vv.tolist() if torch.is_tensor(vv) else
                        [x.tolist() for x in vv] if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
    with open(os.path.join(OUT, "null_params.json"), "w") as f:
        json.dump(desc, f, indent=1)

    tasks = [(n, b) for n in NULLS for b in range(args.B)]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    with open(os.path.join(OUT, "bootstrap_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    obs = json.load(open(os.path.join(ROOT, "results", "reproduce", "sushi_lcl.json")))
    a_obs = next(r["A_alone"] for r in obs["single_effects"] if r["p"] == K and r["q"] == K)
    lrt_obs = next(r["LRT"] for r in obs["single_effects"] if r["p"] == K and r["q"] == K)
    full_obs = obs["lrt"]
    lines = [f"Observed (real sushi): A_oo alone {a_obs:+.3f}, single LRT {lrt_obs:.1f}, whole-model LRT {full_obs:.1f}",
             f"B = {args.B} simulated datasets per null; 'p_boot' = share of simulations at least as extreme.",
             "",
             "| null model (fitted to sushi) | mean A_oo | sd A_oo | P(A_oo <= obs) | P(single LRT >= obs) | "
             "P(single test p<0.001) | P(whole LRT >= obs) | P(whole test p<0.001) |",
             "|---|---|---|---|---|---|---|---|"]
    from scipy.stats import chi2
    for n in NULLS:
        g = [r for r in rows if r["null"] == n]
        a = np.array([r["a_kk"] for r in g])
        l1 = np.array([r["lrt_kk"] for r in g])
        l36 = np.array([r["lrt_full"] for r in g])
        lines.append(f"| {n} | {a.mean():+.3f} | {a.std():.3f} | {np.mean(a <= a_obs):.1%} | {np.mean(l1 >= lrt_obs):.1%} | "
                     f"{np.mean(chi2.sf(l1, 1) < 1e-3):.1%} | {np.mean(l36 >= full_obs):.1%} | "
                     f"{np.mean(chi2.sf(l36, 36) < 1e-3):.1%} |")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "bootstrap.md"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()

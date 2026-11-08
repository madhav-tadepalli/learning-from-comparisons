"""Step 6: is the panel (exploded-ranking) test trustworthy?

On the real sushi data, the oiliness self-effect vanishes once we use each person's top-2 or top-3
choices in a random-coefficient model. Before believing that, check on simulated rankings
(real sushi sets, parameters fitted to sushi) that the panel test
  * keeps its size when there is NO context effect but people differ (null: RC-all panel fit), and
  * still DETECTS a true context effect of the paper's size (A_oo = -0.266), with or without
    heterogeneity.
Rankings are simulated stage by stage, so a true LCL context effect is recomputed on the
remaining items at every stage (exactly the exploded-LCL likelihood).
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, ChoiceData, sushi  # noqa: E402
from models import Model, lrt  # noqa: E402

OUT = os.path.join(ROOT, "results", "panel_check")
D = len(FEATURES)
K = OIL
ALL = list(range(D))
TRUTHS = ["het_null", "het_plus_A", "A_only"]
KS = [1, 2, 3]

_SETS, _NULL = None, None


def explode(set_feats, order, top_k):
    """set_feats P x 10 x d, order P x 10 (ranking, positions into the set) -> exploded ChoiceData."""
    P, J, _ = set_feats.shape
    ranked = torch.gather(set_feats, 1, order[..., None].expand(-1, -1, set_feats.shape[2]))
    feats, valid, choice, person = [], [], [], []
    for t in range(top_k):
        v = torch.zeros(P, J, dtype=torch.bool)
        v[:, t:] = True
        feats.append(ranked * v[..., None])
        valid.append(v)
        choice.append(torch.full((P,), t, dtype=torch.long))
        person.append(torch.arange(P))
    return ChoiceData(torch.cat(feats), torch.cat(valid), torch.cat(choice), torch.cat(person))


@torch.no_grad()
def simulate_ranking(set_feats, theta, sd=None, a_oo=0.0, top_k=3, gen=None):
    P, J, d = set_feats.shape
    coef = theta[None, :].repeat(P, 1)
    if sd is not None:
        coef = coef + torch.randn(P, d, generator=gen) * sd
    remaining = torch.ones(P, J, dtype=torch.bool)
    order = []
    for _ in range(top_k):
        xbar = (set_feats * remaining[..., None]).sum(1) / remaining.sum(1, keepdim=True)
        c = coef.clone()
        c[:, K] += a_oo * xbar[:, K]
        u = (set_feats * c[:, None, :]).sum(-1).masked_fill(~remaining, -1e30)
        pick = torch.multinomial(torch.softmax(u, -1), 1, generator=gen).squeeze(1)
        order.append(pick)
        remaining[torch.arange(P), pick] = False
    rest = torch.argsort((~remaining).double() * -1 - torch.arange(J) * 1e-3, 1, descending=True)
    # positions not yet chosen, appended in any order after the chosen ones
    chosen = torch.stack(order, 1)
    tail = torch.stack([r[~torch.isin(r, ch)] for r, ch in zip(rest, chosen)])
    return torch.cat([chosen, tail], 1)


def _init():
    global _SETS, _NULL
    torch.set_num_threads(1)
    d1 = sushi(top_k=1)
    _SETS = d1.feats  # P x 10 x d (top-1 data has every item valid)
    d3 = sushi(top_k=3)
    m = Model(D, het="normal", het_idx=ALL, R=100).fit(d3, restarts=1)
    _NULL = {"theta": m.params["theta"], "sd": torch.exp(m.params["log_sd"])}


def fit_pair(data, het, R=100):
    if het:
        small = Model(D, het="normal", het_idx=ALL, R=R).fit(data)
        init = {k: v.clone() for k, v in small.params.items()}
        init["a"] = torch.zeros(1)
        big = Model(D, lcl=[(K, K)], het="normal", het_idx=ALL, R=R).fit(data, init=init)
    else:
        small = Model(D).fit(data)
        big = Model(D, lcl=[(K, K)]).fit(data)
    stat, p = lrt(big.ll, small.ll, 1)
    return big.params["a"][0].item(), stat, p


def run_one(task):
    truth, rep = task
    gen = torch.Generator().manual_seed(int(np.random.SeedSequence([TRUTHS.index(truth), rep, 11]).generate_state(1)[0]))
    th, sd = _NULL["theta"], _NULL["sd"]
    if truth == "het_null":
        order = simulate_ranking(_SETS, th, sd=sd, gen=gen)
    elif truth == "het_plus_A":
        order = simulate_ranking(_SETS, th, sd=sd, a_oo=-0.266, gen=gen)
    else:
        order = simulate_ranking(_SETS, th, a_oo=-0.266, gen=gen)
    row = {"truth": truth, "rep": rep}
    for k in KS:
        data = explode(_SETS, order, k)
        for het in (False, True):
            a, stat, p = fit_pair(data, het)
            tag = f"k{k}_{'rc' if het else 'mnl'}"
            row[f"a_{tag}"], row[f"lrt_{tag}"], row[f"p_{tag}"] = a, stat, p
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tasks = [(t, r) for t in TRUTHS for r in range(args.reps)]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(run_one, tasks)):
            rows.append(r)
            print(f"  {i+1}/{len(tasks)}", flush=True)
    with open(os.path.join(OUT, "panel_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["truth | k | model | mean A_oo | sd | % p<0.001 | % p<0.05", "---|---|---|---|---|---|---"]
    for t in TRUTHS:
        g = [r for r in rows if r["truth"] == t]
        for k in KS:
            for m in ("mnl", "rc"):
                a = np.array([r[f"a_k{k}_{m}"] for r in g])
                p = np.array([r[f"p_k{k}_{m}"] for r in g])
                lines.append(f"{t} | {k} | {'LCL' if m == 'mnl' else 'RC-all + A'} | {a.mean():+.3f} | {a.std():.3f} | "
                             f"{np.mean(p < 1e-3):.0%} | {np.mean(p < 0.05):.0%}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "panel_check.md"), "w") as f:
        f.write(f"{args.reps} simulated ranking datasets per truth, real sushi sets; null = RC-all panel fit (k=3).\n"
                "het_null: people differ, no context effect. het_plus_A: both, A_oo=-0.266. A_only: no heterogeneity, A_oo=-0.266.\n\n"
                + txt + "\n")


if __name__ == "__main__":
    main()

"""Step 7: is the oiliness context effect specific to the FIRST pick?

Panel random-coefficient model on exploded top-k rankings with two separate self-effects:
a_first (stage 1, full menu of 10) and a_later (stages 2..k, the remaining items).
Implemented by adding two regressors x_oil * xbar_oil * 1[stage] to the item features.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, ChoiceData, sushi  # noqa: E402
from models import Model, lrt  # noqa: E402

D = len(FEATURES)


def augment(d, k):
    P = d.P
    stage = torch.arange(k).repeat_interleave(P)          # sushi() stacks stage blocks of P rows
    inter = d.feats[..., OIL] * d.xbar[:, OIL:OIL + 1]
    first = inter * (stage == 0)[:, None]
    later = inter * (stage > 0)[:, None]
    f = torch.cat([d.feats, first[..., None], later[..., None]], -1)
    return ChoiceData(f, d.valid, d.choice, d.person, d.item)


def main():
    torch.set_num_threads(8)
    lines = []
    for k in [2, 3]:
        d = augment(sushi(top_k=k), k)
        for het in (False, True):
            kw = dict(het="normal", het_idx=list(range(D)), R=100) if het else {}
            m = Model(D + 2, **kw).fit(d, restarts=2 if het else 1)
            th = m.params["theta"]
            # tests: a_first = 0 and a_later = 0, each by dropping that regressor
            res = []
            for drop in (D, D + 1):
                keep = [i for i in range(D + 2) if i != drop]
                d2 = ChoiceData(d.feats[..., keep], d.valid, d.choice, d.person, d.item)
                m2 = Model(D + 1, **kw).fit(d2, restarts=2 if het else 1)
                res.append(lrt(m.ll, m2.ll, 1))
            line = (f"k={k} {'RC-all' if het else 'MNL   '}: a_first {th[D]:+.3f} (LRT {res[0][0]:5.1f}, p={res[0][1]:.2g})"
                    f"   a_later {th[D+1]:+.3f} (LRT {res[1][0]:5.1f}, p={res[1][1]:.2g})")
            print(line, flush=True)
            lines.append(line)
    os.makedirs(os.path.join(ROOT, "results", "real"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "real", "stage_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

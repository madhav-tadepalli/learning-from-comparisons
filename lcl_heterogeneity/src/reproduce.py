"""Step 1: reproduce the paper's sushi results (Tables 3, 5, 10) with exact MLE.

Paper (Adam, weight decay 0.001, 500 epochs): MNL NLL 9821, LCL 9773, LRT 97 (p=1.6e-7);
largest effects in sushi (Table 5): oiliness on oiliness A=-0.20 (alone -0.26, LRT 23, p=1.5e-6).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, ROOT, sushi  # noqa: E402
from models import Model, lrt  # noqa: E402

OUT = os.path.join(ROOT, "results", "reproduce")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = sushi(top_k=1)
    d = len(FEATURES)
    mnl = Model(d).fit(data)
    lcl = Model(d, lcl="full").fit(data)
    stat, pval = lrt(lcl.ll, mnl.ll, d * d)
    print(f"MNL NLL {-mnl.ll:.1f}   LCL NLL {-lcl.ll:.1f}   LRT {stat:.1f} (df {d*d}) p={pval:.2g}")
    print("paper:  MNL 9821 LCL 9773 LRT 97 p=1.6e-7")
    A = lcl.summary()["A"]

    rows = []
    for p_ in range(d):
        for q_ in range(d):
            one = Model(d, lcl=[(p_, q_)]).fit(data)
            s, pv = lrt(one.ll, mnl.ll, 1)
            rows.append({"effect": f"{FEATURES[q_]} on {FEATURES[p_]}", "p": p_, "q": q_, "A_full": float(A[p_, q_]),
                         "A_alone": float(one.params['a'][0]), "LRT": s, "pval": pv})
    rows.sort(key=lambda r: -abs(r["A_full"]))
    print("\nlargest |A| (q on p)        A_full  A_alone   LRT    p")
    for r in rows[:8]:
        print(f"{r['effect']:28s} {r['A_full']:+.3f}  {r['A_alone']:+.3f}  {r['LRT']:6.1f}  {r['pval']:.2g}")
    print("\nall single effects with p < 0.001:")
    for r in sorted(rows, key=lambda r: r["pval"]):
        if r["pval"] < 1e-3:
            print(f"  {r['effect']:28s} A_alone {r['A_alone']:+.3f}  LRT {r['LRT']:.1f}  p={r['pval']:.2g}")

    with open(os.path.join(OUT, "sushi_lcl.json"), "w") as f:
        json.dump({"mnl_nll": -mnl.ll, "lcl_nll": -lcl.ll, "lrt": stat, "pval": pval,
                   "theta_mnl": mnl.params["theta"].tolist(), "theta_lcl": lcl.params["theta"].tolist(),
                   "A": A.tolist(), "single_effects": rows}, f, indent=1)
    np.set_printoptions(precision=3, suppress=True)
    print("\nA (rows = influenced feature p, cols = exerting feature q):\n", A)
    print("theta (MNL):", mnl.params["theta"].numpy())


if __name__ == "__main__":
    main()

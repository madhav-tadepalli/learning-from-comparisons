"""Summarise results/sim/sim_raw.csv into results/sim/summary.md and results/sim/summary.csv."""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT  # noqa: E402

SIM = os.path.join(ROOT, "results", "sim")
ALPHA = 1e-3
PAPER_A = -0.266  # the paper's isolated oiliness self-effect

LABEL = {"mnl": "no effect (MNL)", "lcl": "true context effect", "both": "context + heterogeneity",
         "normal": "normal spread", "cls50": "2 segments 50/50", "cls20": "2 segments 20/80"}


def pval(df, big, small, k):
    return chi2.sf((2 * (df[big] - df[small])).clip(lower=0), k)


def summarise(raw):
    raw = raw.copy()
    raw["p_full"] = pval(raw, "ll_lcl_full", "ll_mnl", 36)
    raw["p_kk"] = pval(raw, "ll_lcl_kk", "ll_mnl", 1)
    raw["p_rc"] = pval(raw, "ll_rc_kk", "ll_rc", 1)
    raw["p_lc2"] = pval(raw, "ll_lc2_kk", "ll_lc2", 1)
    raw["p_sq"] = pval(raw, "ll_sq_kk", "ll_sq", 1)
    rows = []
    for (kind, s), g in raw.groupby(["kind", "s"], sort=False):
        rows.append({
            "truth": LABEL[kind], "kind": kind, "s": s, "reps": len(g),
            "A_oo LCL (mean)": g["a_lcl_kk"].mean(), "A_oo LCL (sd)": g["a_lcl_kk"].std(),
            "A_oo full LCL": g["a_lcl_full"].mean(),
            "paper 36-df test": (g["p_full"] < ALPHA).mean(),
            "paper single test": (g["p_kk"] < ALPHA).mean(),
            "as strong as paper": ((g["p_kk"] < ALPHA) & (g["a_lcl_kk"] <= PAPER_A)).mean(),
            "A_oo in RC": g["a_rc_kk"].mean(), "test in RC": (g["p_rc"] < ALPHA).mean(),
            "A_oo in LC2": g["a_lc2_kk"].mean(), "test in LC2": (g["p_lc2"] < ALPHA).mean(),
            "A_oo in SQ": g["a_sq_kk"].mean(), "test in SQ": (g["p_sq"] < ALPHA).mean(),
            "sd in RC": g["sd_rc"].mean(),
            "RC beats LCL_oo": (g["ll_rc"] > g["ll_lcl_kk"]).mean(),
        })
    return pd.DataFrame(rows), raw


def to_md(df):
    cols = ["truth", "s", "reps", "A_oo LCL (mean)", "A_oo LCL (sd)", "paper 36-df test", "paper single test",
            "as strong as paper", "A_oo in RC", "test in RC", "test in LC2", "test in SQ", "sd in RC", "RC beats LCL_oo"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("truth",):
                cells.append(str(v))
            elif c == "reps":
                cells.append(str(int(v)))
            elif c == "s":
                cells.append(f"{v:.2f}")
            elif "test" in c or "strong" in c or "beats" in c:
                cells.append(f"{100 * v:.0f}%")
            else:
                cells.append(f"{v:+.3f}" if ("A_oo" in c and "(sd)" not in c) else f"{v:.3f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    raw = pd.read_csv(os.path.join(SIM, "sim_raw.csv"))
    df, raw = summarise(raw)
    df.to_csv(os.path.join(SIM, "summary.csv"), index=False)
    md = to_md(df)
    with open(os.path.join(SIM, "summary.md"), "w") as f:
        f.write(f"Rejection rates at p < {ALPHA}. s = sd of the oiliness taste (0 for the first three rows; "
                "'context + heterogeneity' uses s = 0.75 and A_oo = -0.15).\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()

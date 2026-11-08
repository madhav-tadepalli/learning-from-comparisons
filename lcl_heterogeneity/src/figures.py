"""Figures for the report (static PNG + PDF, light background, sized for a LaTeX page)."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT  # noqa: E402
from summarise_sim import summarise  # noqa: E402

FIG = os.path.join(ROOT, "figures")
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
SERIES = {"normal": "#2a78d6", "cls50": "#eb6834", "cls20": "#1baf7a"}  # categorical slots 1-3
NAMES = {"normal": "normal spread", "cls50": "2 segments 50/50", "cls20": "2 segments 20/80"}
PAPER_A = -0.266

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def sushi_sd():
    p = os.path.join(ROOT, "results", "real", "real_data.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return r["RC-oil"]["summary"]["sd"][0]


def fig_spurious_effect(df):
    """Mean plain-LCL A_oo vs taste spread, per heterogeneity type."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    base = df[df.kind == "mnl"].iloc[0]
    for kind, col in SERIES.items():
        g = df[df.kind == kind].sort_values("s")
        s = np.r_[0.0, g.s.values]
        m = np.r_[base["A_oo LCL (mean)"], g["A_oo LCL (mean)"].values]
        sd = np.r_[base["A_oo LCL (sd)"], g["A_oo LCL (sd)"].values]
        ax.fill_between(s, m - sd, m + sd, color=col, alpha=0.10, lw=0)
        ax.plot(s, m, color=col, marker="o", ms=5, label=NAMES[kind])
    ax.axhline(PAPER_A, color=INK, ls="--", lw=1.2)
    ax.text(0.02, PAPER_A - 0.035, "paper's estimate (−0.27)", color=INK, fontsize=9, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    sd_hat = sushi_sd()
    if sd_hat is not None:
        ax.axvline(sd_hat, color=INK2, ls=":", lw=1.2)
        ax.text(sd_hat + 0.02, -0.40, f"taste spread estimated\nfrom real sushi data ({sd_hat:.2f})",
                color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("spread of the oiliness taste across people, s (sd of coefficient)")
    ax.set_ylabel("estimated self-effect A[oil, oil]")
    ax.set_title("No one has a context effect, yet LCL finds one", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "fig1_spurious_effect")


def fig_rejections(df):
    """Rejection rate of the single-effect test: paper's LCL vs heterogeneity-aware LCL."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    tests = [("paper single test", "paper's test (LCL vs MNL)"), ("test in RC", "same test inside a mixed logit")]
    for ax, (col, title) in zip(axes, tests):
        base = df[df.kind == "mnl"].iloc[0][col]
        for kind, c in SERIES.items():
            g = df[df.kind == kind].sort_values("s")
            ax.plot(np.r_[0, g.s], 100 * np.r_[base, g[col]], color=c, marker="o", ms=5, label=NAMES[kind])
        true = df[df.kind == "lcl"].iloc[0][col]
        ax.scatter([0], [100 * true], s=60, marker="D", color=INK, zorder=5)
        ax.annotate(f"true effect: {100*true:.0f}%", (0, 100 * true), xytext=(8, -2), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("taste spread s")
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("% of simulations flagged (p < 0.001)")
    axes[1].legend(loc="center right", fontsize=8.5)
    save(fig, "fig2_rejections")


def fig_bootstrap():
    p = os.path.join(ROOT, "results", "bootstrap", "bootstrap_raw.csv")
    if not os.path.exists(p):
        return
    b = pd.read_csv(p)
    # The three heterogeneity nulls (RC-oil, LC2-oil, RC-all) give near-identical histograms; show one.
    order = [("MNL", "people identical (MNL)", INK2), ("RC-oil", "people differ in oiliness taste (fitted, sd 0.45)", SERIES["normal"])]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(-0.36, 0.2, 29)
    for n, lab, c in order:
        a = b[b.null == n].a_kk.values
        if len(a):
            ax.hist(a, bins=bins, histtype="step", lw=2, color=c, label=f"{lab}: mean {a.mean():+.2f}")
    ax.axvline(PAPER_A, color=INK, ls="--", lw=1.2)
    ax.text(PAPER_A - 0.008, 30, "real sushi\nestimate\n(−0.27)", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 52)
    ax.set_xlabel("A[oil, oil] estimated by the paper's LCL on re-simulated data")
    ax.set_ylabel("simulated datasets (of 200)")
    ax.set_title("Heterogeneity explains about a third of the sushi effect", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig3_bootstrap")


def main():
    raw = pd.read_csv(os.path.join(ROOT, "results", "sim", "sim_raw.csv"))
    df, _ = summarise(raw)
    fig_spurious_effect(df)
    fig_rejections(df)
    fig_bootstrap()


if __name__ == "__main__":
    main()

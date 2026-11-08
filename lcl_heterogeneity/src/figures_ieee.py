"""Column-width versions of the figures for the two-column IEEE report (figures/ieee_*.pdf).

No in-figure titles (the captions carry them); 8 pt text so the figures stay legible at
\\columnwidth (about 3.5 in) or \\textwidth (about 7.2 in).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT  # noqa: E402
from figures import INK, INK2, GRID, NAMES, PAPER_A, SERIES, sushi_sd  # noqa: E402
from summarise_sim import summarise  # noqa: E402

FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "lines.linewidth": 1.6, "lines.markersize": 3.5,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=250)
    plt.close(fig)


def fig_spurious(df):
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    base = df[df.kind == "mnl"].iloc[0]
    for kind, col in SERIES.items():
        g = df[df.kind == kind].sort_values("s")
        s = np.r_[0.0, g.s.values]
        m = np.r_[base["A_oo LCL (mean)"], g["A_oo LCL (mean)"].values]
        sd = np.r_[base["A_oo LCL (sd)"], g["A_oo LCL (sd)"].values]
        ax.fill_between(s, m - sd, m + sd, color=col, alpha=0.10, lw=0)
        ax.plot(s, m, color=col, marker="o", label=NAMES[kind])
    ax.axhline(PAPER_A, color=INK, ls="--", lw=1)
    ax.text(0.02, PAPER_A - 0.025, "sushi estimate (−0.27)", color=INK, fontsize=7, va="top")
    ax.axhline(0, color=INK2, lw=0.7)
    sd_hat = sushi_sd()
    ax.axvline(sd_hat, color=INK2, ls=":", lw=1)
    ax.text(sd_hat + 0.02, 0.05, f"fitted on sushi ({sd_hat:.2f})", color=INK2, fontsize=7, va="top")
    ax.set_xlabel("taste spread $s$ (sd of the oiliness coefficient)")
    ax.set_ylabel(r"$A_{\mathrm{oil,oil}}$ estimated by LCL")
    ax.legend(loc="lower left")
    save(fig, "ieee_spurious")


def fig_rejections(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    tests = [("paper single test", "(a) LCL against MNL (the paper's test)"),
             ("test in RC", "(b) the same test inside a random-coefficient logit")]
    for ax, (col, title) in zip(axes, tests):
        base = df[df.kind == "mnl"].iloc[0][col]
        for kind, c in SERIES.items():
            g = df[df.kind == kind].sort_values("s")
            ax.plot(np.r_[0, g.s], 100 * np.r_[base, g[col]], color=c, marker="o", label=NAMES[kind])
        true = df[df.kind == "lcl"].iloc[0][col]
        ax.scatter([0], [100 * true], s=22, marker="D", color=INK, zorder=5)
        ax.annotate(f"true effect: {100*true:.0f}%", (0, 100 * true), xytext=(6, -1), textcoords="offset points",
                    fontsize=7, color=INK, va="center")
        ax.set_title(title, loc="left", fontsize=8, color=INK)
        ax.set_xlabel("taste spread $s$")
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("% flagged at $p<0.001$")
    axes[1].legend(loc="center right")
    save(fig, "ieee_rejections")


def fig_bootstrap():
    b = pd.read_csv(os.path.join(ROOT, "results", "bootstrap", "bootstrap_raw.csv"))
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    bins = np.linspace(-0.36, 0.2, 29)
    for n, lab, c in [("MNL", "MNL", INK2), ("RC-oil", "RC, sd 0.45", SERIES["normal"])]:
        a = b[b.null == n].a_kk.values
        ax.hist(a, bins=bins, histtype="step", lw=1.5, color=c, label=f"{lab} (mean {a.mean():+.2f})")
    ax.axvline(PAPER_A, color=INK, ls="--", lw=1)
    ax.text(PAPER_A - 0.006, 30, "sushi\n(−0.27)", ha="right", va="center", fontsize=7, color=INK)
    ax.set_ylim(0, 50)
    ax.set_xlabel(r"$A_{\mathrm{oil,oil}}$ estimated by LCL on simulated data")
    ax.set_ylabel("datasets (of 200)")
    ax.legend(loc="upper right")
    save(fig, "ieee_bootstrap")


def main():
    df, _ = summarise(pd.read_csv(os.path.join(ROOT, "results", "sim", "sim_raw.csv")))
    fig_spurious(df)
    fig_rejections(df)
    fig_bootstrap()


if __name__ == "__main__":
    main()

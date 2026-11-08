"""Step 2: check the second-order (small-heterogeneity) theory on the real sushi choice sets.

Theory. Let person n have theta_n = mu + eta_n e_k with E eta = 0, Var eta = s^2. In a set C,
with MNL probabilities p_j under mu, m_C = sum_j p_j x_jk and v_C = sum_j p_j (x_jk - m_C)^2:

    log P(i | C) = mu^T x_i + (s^2/2) (x_ik - m_C)^2 - log-normaliser + O(s^3)

i.e. to second order the aggregate behaves like an MNL with
    * an own-squared term     q_k = s^2 / 2                         (curvature)
    * a set-dependent slope   -s^2 m_C on feature k                  (looks like a negative self-effect)
and only the VARIANCE of tastes enters (Gaussian vs latent classes agree at this order).

Predictions for fits on large samples:
  (P1) MNL + squared term:  q_k ~ s^2/2.
  (P2) LCL + squared term:  A_kk ~ -s^2 * beta, beta = slope of m_C on xbar_C (set-mean of feature k);
       i.e. heterogeneity masquerades as a NEGATIVE self-effect, tied to the curvature: A_kk ~ -2 beta q_k.
  (P3) Plain LCL (the paper's model, no squared term): within a set, x^2 projects onto x with slope
       2 m_C under logit weights, which cancels the -s^2 m_C slope. What survives is third-order
       (skewness of the taste distribution, skew of x within sets), so A_kk is small and its sign
       depends on the TYPE of heterogeneity.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURES, OIL, ROOT, ChoiceData, sushi  # noqa: E402
from models import Model, simulate_top1  # noqa: E402

OUT = os.path.join(ROOT, "results", "theory")
REPS = 20  # 20 copies of the 5000 real choice sets -> 100,000 choices (near-population fits)


def tile(data, reps):
    P = data.N
    return ChoiceData(data.feats.repeat(reps, 1, 1), data.valid.repeat(reps, 1), data.choice.repeat(reps),
                      torch.arange(P * reps), data.item.repeat(reps, 1))


def taste_designs(s):
    """Three heterogeneity designs on oiliness, all with taste variance s^2."""
    yield "normal", dict(sd=torch.tensor([s]))
    yield "2 classes 50/50", dict(classes=(torch.tensor([0.5, 0.5]), torch.tensor([[-s], [s]])))
    pi = 0.2  # minority segment: skewed taste distribution with the same variance
    a = s * np.sqrt((1 - pi) / pi)
    b = -s * np.sqrt(pi / (1 - pi))
    yield "2 classes 20/80", dict(classes=(torch.tensor([pi, 1 - pi]), torch.tensor([[a], [b]])))


def main():
    os.makedirs(OUT, exist_ok=True)
    base = sushi(top_k=1)
    d = len(FEATURES)
    mu = Model(d).fit(base).params["theta"]
    big = tile(base, REPS)

    # beta: slope of the MNL-weighted mean m_C on the plain mean xbar_C (oiliness)
    with torch.no_grad():
        u = (base.feats * mu).sum(-1)
        w = torch.softmax(u, -1)
        m = (w * base.feats[..., OIL]).sum(1)
        xb = base.xbar[:, OIL]
        beta = float(((m - m.mean()) * (xb - xb.mean())).sum() / ((xb - xb.mean()) ** 2).sum())
    print(f"beta (slope of weighted mean m_C on set mean xbar_C, oiliness) = {beta:.3f}")

    # Oracle check of the expansion itself: fit an MNL with two extra regressors that are exactly
    # the predicted terms, x_k^2 and m_C * x_k (m_C computed from the true mu). Theory: coefficients
    # s^2/2 and -s^2, with mu unchanged.
    gen = torch.Generator().manual_seed(0)
    mbig = m.repeat(REPS)
    xk = big.feats[..., OIL]
    aug_feats = torch.cat([big.feats, (xk ** 2)[..., None], (mbig[:, None] * xk)[..., None]], -1)
    aug = ChoiceData(aug_feats, big.valid, big.choice, big.person, big.item)
    oracle = ["design | s | coef x^2 [pred s^2/2] | coef m_C*x [pred -s^2]"]
    print(oracle[0])
    for s in [0.25, 0.5, 1.0, 1.5]:
        for name, kw in taste_designs(s):
            ch = simulate_top1(big, mu, het_idx=[OIL], gen=gen, **kw)
            th = Model(d + 2).fit(aug.with_choice(ch)).params["theta"]
            line = f"{name:16s} | {s:.2f} | {th[d]:+.3f} [{s*s/2:+.3f}] | {th[d+1]:+.3f} [{-s*s:+.3f}]"
            print(line, flush=True)
            oracle.append(line)

    lines = ["design | s | q_oil (MNL+sq) [pred s^2/2] | A_oo (LCL+sq, A_oo only) [pred -beta s^2] "
             "| A_oo (plain LCL, A_oo only) | A_oo (plain full LCL)"]
    print(lines[0])
    for s in [0.5, 1.0, 1.5]:
        for name, kw in taste_designs(s):
            ch = simulate_top1(big, mu, het_idx=[OIL], gen=gen, **kw)
            dat = big.with_choice(ch)
            q = Model(d, quad=[OIL]).fit(dat).params["q"][0].item()
            a_sq = Model(d, lcl=[(OIL, OIL)], quad=[OIL]).fit(dat).params["a"][0].item()
            a_plain = Model(d, lcl=[(OIL, OIL)]).fit(dat).params["a"][0].item()
            a_full = Model(d, lcl="full").fit(dat).summary()["A"][OIL, OIL]
            line = (f"{name:16s} | {s:.1f} | {q:+.3f} [{s*s/2:+.3f}] | {a_sq:+.3f} [{-beta*s*s:+.3f}] "
                    f"| {a_plain:+.3f} | {a_full:+.3f}")
            print(line, flush=True)
            lines.append(line)
    with open(os.path.join(OUT, "theory_check.txt"), "w") as f:
        f.write(f"beta = {beta:.3f}; {REPS}x tiled real sushi sets (N = {big.N})\n\n"
                "Oracle second-order model:\n" + "\n".join(oracle) + "\n\nFits of misspecified models:\n"
                + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

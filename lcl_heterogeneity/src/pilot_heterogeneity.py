"""Pilot: does taste heterogeneity alone make LCL report context effects?

Data-generating process (no context effects at the individual level):
  - 100 items with d=6 standardised Gaussian features (like sushi: 100 items, 6 features)
  - each chooser sees a uniformly random set of 10 items (sushi's set-up, so no confounding)
  - chooser n has theta_n = mu + sigma * z_n * e_0   (heterogeneous taste for feature 0 only)
  - chooser picks one item from an MNL with theta_n (i.e. data = a mixed logit)

We fit MNL, full LCL (theta, A), and a constrained LCL with only A[0,0] free (the paper's
single-effect test), then run the paper's likelihood-ratio tests at p < 0.001.

Usage: python lcl/pilot_heterogeneity.py [--seeds 20] [--n 5000]
"""

import argparse

import numpy as np
import torch
from scipy.stats import chi2

torch.set_default_dtype(torch.float64)

D, N_ITEMS, SET_SIZE = 6, 100, 10
MU = torch.tensor([-0.3, 0.5, 0.2, 0.0, 0.4, -0.2])


def simulate(n, sigma, rng):
    X = rng.standard_normal((N_ITEMS, D))
    X = (X - X.mean(0)) / X.std(0)
    sets = np.stack([rng.choice(N_ITEMS, SET_SIZE, replace=False) for _ in range(n)])
    feats = torch.tensor(X[sets])                                   # n x k x d
    theta = MU.repeat(n, 1)
    theta[:, 0] += sigma * torch.tensor(rng.standard_normal(n))
    util = (feats * theta[:, None, :]).sum(-1)
    probs = torch.softmax(util, -1).numpy()
    choice = torch.tensor([rng.choice(SET_SIZE, p=p / p.sum()) for p in probs])
    return feats, choice


def nll(feats, choice, theta, A):
    xbar = feats.mean(1)                                           # n x d
    coef = theta[None, :] + (xbar @ A.T if A is not None else 0.0)  # n x d   (theta + A xbar)
    util = (feats * coef[:, None, :]).sum(-1)
    return -(torch.log_softmax(util, -1).gather(1, choice[:, None])).sum()


def fit(feats, choice, mode):
    theta = torch.zeros(D, requires_grad=True)
    params = [theta]
    if mode == "lcl":
        A = torch.zeros(D, D, requires_grad=True)
        params.append(A)
    elif mode == "a00":
        a = torch.zeros(1, requires_grad=True)
        params.append(a)
    opt = torch.optim.LBFGS(params, max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def get_A():
        if mode == "lcl":
            return A
        if mode == "a00":
            M = torch.zeros(D, D)
            M[0, 0] = 1.0
            return M * a
        return None

    def closure():
        opt.zero_grad()
        loss = nll(feats, choice, theta, get_A())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = nll(feats, choice, theta, get_A()).item()
        A_hat = get_A()
    return loss, (A_hat.detach().numpy() if A_hat is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    args = ap.parse_args()

    print(f"n={args.n} choosers, {args.seeds} seeds per sigma, sets of {SET_SIZE} from {N_ITEMS} items, d={D}")
    print("sigma | A00 full (mean±sd) | A00 alone (mean±sd) | full LRT sig | A00 LRT sig | A00 is largest |A|")
    for sigma in args.sigmas:
        a_full, a_alone, sig_full, sig_a00, a00_max = [], [], 0, 0, 0
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * s + int(10 * sigma))
            feats, choice = simulate(args.n, sigma, rng)
            l_mnl, _ = fit(feats, choice, "mnl")
            l_lcl, A = fit(feats, choice, "lcl")
            l_a00, A1 = fit(feats, choice, "a00")
            sig_full += chi2.sf(2 * (l_mnl - l_lcl), D * D) < 1e-3
            sig_a00 += chi2.sf(2 * (l_mnl - l_a00), 1) < 1e-3
            a00_max += np.unravel_index(np.abs(A).argmax(), A.shape) == (0, 0)
            a_full.append(A[0, 0])
            a_alone.append(A1[0, 0])
        k = args.seeds
        print(f"{sigma:5.1f} | {np.mean(a_full):+.3f} ± {np.std(a_full):.3f}      "
              f"| {np.mean(a_alone):+.3f} ± {np.std(a_alone):.3f}       "
              f"| {sig_full:2d}/{k}       | {sig_a00:2d}/{k}      | {a00_max:2d}/{k}", flush=True)


if __name__ == "__main__":
    main()

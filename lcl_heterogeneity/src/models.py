"""Discrete choice models with optional linear context effects and population heterogeneity.

One flexible model covers everything we need:

  utility of item i for person n in set C
      = (theta + A xbar_C + B w_n + eta_n)^T x_i  +  sum_k q_k x_ik^2

  - A        : LCL context-effect matrix (any subset of entries free)            [Tomlinson & Benson]
  - q        : own-squared terms (a plain MNL with curvature; no context)
  - B w_n    : observed heterogeneity (demographics w_n change the tastes)
  - eta_n    : unobserved heterogeneity, either
                 'normal'  : eta_n ~ N(0, diag(sd^2)) on chosen features (random-coefficient mixed logit)
                 'classes' : eta_n = delta_c with prob pi_c (latent-class / discrete mixed logit)
    eta_n is per PERSON, so with several choices per person (exploded rankings) it is a panel model.

All models are fitted by exact maximum likelihood with L-BFGS in float64 (no weight decay), so
likelihood-ratio tests are clean. Normal mixtures use fixed scrambled-Sobol draws (simulated ML).
"""

import math

import numpy as np
import torch
from scipy.stats import chi2

torch.set_default_dtype(torch.float64)


def logmeanexp(x, dim):
    return torch.logsumexp(x, dim) - math.log(x.shape[dim])


def normal_draws(P, R, k, seed=0):
    """P x R x k standard-normal draws: one scrambled Sobol sequence, randomly shifted per person."""
    if k == 0:
        return torch.zeros(P, R, 0)
    sob = torch.quasirandom.SobolEngine(k, scramble=True, seed=seed).draw(R).double()  # R x k
    g = torch.Generator().manual_seed(seed + 1)
    shift = torch.rand(P, 1, k, generator=g)
    u = (sob[None] + shift) % 1.0
    u = u.clamp(1e-10, 1 - 1e-10)
    return torch.distributions.Normal(0.0, 1.0).icdf(u)


class Model:
    def __init__(self, d, lcl=None, quad=(), het=None, het_idx=None, K=2, demo=None, demo_idx=None, R=200,
                 draw_seed=0, dlcl=None, ridge=0.0):
        """
        lcl     : None, 'full', or list of (p, q) entries of A to estimate
        quad    : features with an own-squared term
        het     : None, 'normal' or 'classes'
        het_idx : features whose coefficient varies across people (default all)
        K       : number of latent classes
        demo    : P x m tensor of person covariates (observed heterogeneity), or None
        demo_idx: features whose coefficient depends on demo (default all)
        dlcl    : DLCL context effects (Tomlinson & Benson's decomposed LCL): None, 'full' or a list of
                  (p, q) entries. Needs het='classes', K=d, het_idx=all. Component q has preference
                  vector B_q = theta + delta_q and the context effect of feature q only:
                  coefficient p in component q is B_qp + A[p, q] * xbar_q. With dlcl=None this is the
                  paper's d-component "mixed logit".
        ridge   : optional L2 penalty on class offsets and DLCL entries (fit objective only)
        """
        self.d = d
        if lcl == "full":
            lcl = [(p, q) for p in range(d) for q in range(d)]
        self.lcl = list(lcl) if lcl else []
        if dlcl == "full":
            dlcl = [(p, q) for p in range(d) for q in range(d)]
        self.dlcl = list(dlcl) if dlcl else []
        self.ridge = ridge
        if self.dlcl:
            assert het == "classes" and K == d and (het_idx is None or list(het_idx) == list(range(d)))
        self.quad = list(quad)
        self.het = het
        self.het_idx = list(range(d)) if het_idx is None else list(het_idx)
        self.K = K
        self.demo = demo
        self.demo_idx = list(range(d)) if demo_idx is None else list(demo_idx)
        self.R = R
        self.draw_seed = draw_seed
        self._draws = None

    # ---------------------------------------------------------------- parameters
    def init_params(self, rng=None, scale=0.5):
        g = torch.Generator().manual_seed(int(rng.integers(1 << 30)) if rng is not None else 0)
        p = {"theta": torch.zeros(self.d)}
        if self.lcl:
            p["a"] = torch.zeros(len(self.lcl))
        if self.quad:
            p["q"] = torch.zeros(len(self.quad))
        if self.demo is not None:
            p["B"] = torch.zeros(len(self.demo_idx), self.demo.shape[1])
        if self.het == "normal":
            p["log_sd"] = torch.full((len(self.het_idx),), math.log(scale)) + 0.1 * torch.randn(len(self.het_idx), generator=g)
        elif self.het == "classes":
            p["delta"] = scale * torch.randn(self.K, len(self.het_idx), generator=g)
            p["pi_logit"] = torch.zeros(self.K)
        if self.dlcl:
            p["da"] = torch.zeros(len(self.dlcl))
        return p

    def A_dlcl(self, p):
        """d x d matrix of DLCL entries: column q = context effect of feature q in component q."""
        M = torch.zeros(self.d, self.d)
        if self.dlcl:
            idx = torch.tensor(self.dlcl)
            M = M.index_put((idx[:, 0], idx[:, 1]), p["da"])
        return M

    def A(self, p):
        A = torch.zeros(self.d, self.d)
        if self.lcl:
            idx = torch.tensor(self.lcl)
            A = A.index_put((idx[:, 0], idx[:, 1]), p["a"])
        return A

    def n_params(self):
        n = self.d + len(self.lcl) + len(self.quad)
        if self.demo is not None:
            n += len(self.demo_idx) * self.demo.shape[1]
        if self.het == "normal":
            n += len(self.het_idx)
        elif self.het == "classes":
            n += self.K * len(self.het_idx) + self.K - 1 - len(self.het_idx)  # delta has 1 redundant location per feature
        return n + len(self.dlcl)

    # ---------------------------------------------------------------- likelihood
    def base_utility(self, p, data):
        coef = p["theta"][None, :].expand(data.N, -1)
        if self.lcl:
            coef = coef + data.xbar @ self.A(p).T
        if self.demo is not None:
            extra = torch.zeros(data.N, self.d)
            extra[:, self.demo_idx] = self.demo[data.person] @ p["B"].T
            coef = coef + extra
        u = (data.feats * coef[:, None, :]).sum(-1)
        if self.quad:
            u = u + (data.feats[..., self.quad] ** 2 * p["q"]).sum(-1)
        return u  # N x J

    def eta(self, p, P):
        """Per-person taste shifts: P x R x len(het_idx) (normal) or K x len(het_idx) (classes)."""
        if self.het == "normal":
            if self._draws is None or self._draws.shape[0] != P:
                self._draws = normal_draws(P, self.R, len(self.het_idx), self.draw_seed)
            return self._draws * torch.exp(p["log_sd"])
        return p["delta"]

    def log_probs(self, p, data):
        """log P(chosen) for every observation, per draw/class: N x R (R=1 without heterogeneity)."""
        u = self.base_utility(p, data)
        neg = torch.finfo(u.dtype).min / 4
        if self.het is None:
            u = u.masked_fill(~data.valid, neg)
            return torch.log_softmax(u, -1).gather(1, data.choice[:, None])
        xf = data.feats[..., self.het_idx]                         # N x J x k
        e = self.eta(p, data.P)
        if self.het == "normal":
            extra = torch.einsum("njk,nrk->nrj", xf, e[data.person])  # N x R x J
        else:
            extra = torch.einsum("njk,ck->ncj", xf, e)                 # N x K x J
            if self.dlcl:
                # component c gets coefficient increment A_dlcl[p, c] * xbar_c on feature p
                coefD = self.A_dlcl(p).T[None, :, :] * data.xbar[:, :, None]   # N x K(=d) x d
                extra = extra + torch.einsum("njp,ncp->ncj", data.feats, coefD)
        u = (u[:, None, :] + extra).masked_fill(~data.valid[:, None, :], neg)
        lp = torch.log_softmax(u, -1)
        idx = data.choice[:, None, None].expand(-1, lp.shape[1], 1)
        return lp.gather(2, idx).squeeze(-1)

    def loglik(self, p, data):
        lp = self.log_probs(p, data)
        if self.het is None:
            return lp.sum()
        S = torch.zeros(data.P, lp.shape[1]).index_add(0, data.person, lp)  # per-person sums
        if self.het == "normal":
            return logmeanexp(S, 1).sum()
        return torch.logsumexp(S + torch.log_softmax(p["pi_logit"], 0), 1).sum()

    # ---------------------------------------------------------------- fitting
    def fit(self, data, restarts=1, seed=0, max_iter=2000, init=None, verbose=False):
        rng = np.random.default_rng(seed)
        best = None
        r = 0
        extra = 5  # if every planned restart fails, try up to 5 more random starts
        while r < restarts or (best is None and r < restarts + extra):
            p = init_copy(init) if (init is not None and r == 0) else self.init_params(rng)
            r += 1
            for v in p.values():
                v.requires_grad_(True)
            opt = torch.optim.LBFGS(list(p.values()), max_iter=max_iter, tolerance_grad=1e-7,
                                    tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = -self.loglik(p, data)
                if self.ridge:
                    for k in ("delta", "da"):
                        if k in p:
                            loss = loss + self.ridge * (p[k] ** 2).sum()
                loss.backward()
                return loss

            try:
                opt.step(closure)
                with torch.no_grad():
                    ll = self.loglik(p, data).item()
            except RuntimeError as err:  # rare line-search failures on bad restarts
                if verbose:
                    print("restart failed:", err)
                continue
            if verbose:
                print(f"  restart {r}: ll={ll:.3f}")
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, {k: v.detach().clone() for k, v in p.items()})
        if best is None:  # every start failed: report -inf so callers can fall back
            self.ll, self.params = -np.inf, None
            return self
        self.ll, self.params = best
        return self

    def summary(self):
        p = self.params
        out = {"ll": self.ll, "theta": p["theta"].numpy()}
        if self.lcl:
            out["A"] = self.A(p).numpy()
        if self.quad:
            out["q"] = p["q"].numpy()
        if self.het == "normal":
            out["sd"] = torch.exp(p["log_sd"]).numpy()
        if self.het == "classes":
            out["pi"] = torch.softmax(p["pi_logit"], 0).numpy()
            out["delta"] = p["delta"].numpy()
        if self.dlcl:
            out["A_dlcl"] = self.A_dlcl(p).numpy()
        if self.demo is not None:
            out["B"] = p["B"].numpy()
        return out


def init_copy(p):
    return {k: v.detach().clone() for k, v in p.items()}


def lrt(ll_big, ll_small, df):
    stat = 2 * (ll_big - ll_small)
    return stat, chi2.sf(max(stat, 0.0), df)


# -------------------------------------------------------------------- simulation
@torch.no_grad()
def simulate_top1(data, theta, A=None, q=None, quad=(), sd=None, het_idx=None, classes=None, gen=None):
    """Draw one choice per observation (top-1 data, one observation per person).

    sd      : per-feature sds of a normal random coefficient on het_idx
    classes : (pi, delta) latent classes, delta K x len(het_idx)
    """
    N, J, d = data.feats.shape
    coef = theta[None, :].expand(N, -1).clone()
    if A is not None:
        coef += data.xbar @ A.T
    if sd is not None:
        z = torch.randn(N, len(het_idx), generator=gen)
        coef[:, het_idx] += z * sd
    if classes is not None:
        pi, delta = classes
        c = torch.multinomial(pi, N, replacement=True, generator=gen)
        coef[:, het_idx] += delta[c]
    u = (data.feats * coef[:, None, :]).sum(-1)
    if q is not None:
        u += (data.feats[..., list(quad)] ** 2 * q).sum(-1)
    u = u.masked_fill(~data.valid, -1e30)
    return torch.multinomial(torch.softmax(u, -1), 1, generator=gen).squeeze(1)

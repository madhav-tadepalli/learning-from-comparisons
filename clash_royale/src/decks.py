"""Shared helpers for deck-level analysis: deck pool, labels, trained-model loading, matchup matrices."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train as T

RUNS = Path("results/runs")
dev = T.device()

cards = pd.read_csv("data/processed/cards.csv")
name = cards.name.to_numpy()
wc = pd.read_csv("data/raw/s18/Wincons.csv")
wc_idx = set(cards[cards.card_id.isin(wc.card_id)].idx)


def load_model(run):
    r = json.load(open(RUNS / f"{run}.json"))
    m = T.build(r["model"], len(cards), 1, r["opts"]).to(dev)
    m.load_state_dict(torch.load(RUNS / f"{run}.pt", map_location=dev)); m.eval()
    return m


def deck_keys(df, side):
    return [tuple(r) for r in df[[f"card{i}_{side}" for i in range(8)]].to_numpy()]


def deck_pool(df, n):
    """The n most-played decks in the training split, with play counts (train) and overall."""
    tr = df.split.to_numpy() == "train"
    ka, kb = deck_keys(df, "a"), deck_keys(df, "b")
    cnt_tr = pd.Series([k for k, t in zip(ka, tr) if t] + [k for k, t in zip(kb, tr) if t]).value_counts()
    cnt_all = pd.Series(ka + kb).value_counts()
    keys = list(cnt_tr.index[:n])
    return np.array(keys), cnt_tr.to_numpy()[:n], cnt_all.reindex(keys).to_numpy()


def label(d):
    w = [name[c] for c in d if c in wc_idx] or ["no win-con"]
    return " + ".join(w) + ": " + ", ".join(name[c] for c in d if c not in wc_idx)


@torch.no_grad()
def embed(model, D, level=12):
    D = torch.as_tensor(D, dtype=torch.long, device=dev)
    return model.embed(D, torch.full(D.shape, level - 12.0, device=dev)).cpu().numpy()


@torch.no_grad()
def matchups(model, A, B, level=12, bs=1 << 18):
    """logit(row deck beats column deck), all cards at `level`, equal trophies. Antisymmetric."""
    A = torch.as_tensor(A, dtype=torch.long, device=dev); B = torch.as_tensor(B, dtype=torch.long, device=dev)
    ia, ib = torch.meshgrid(torch.arange(len(A), device=dev), torch.arange(len(B), device=dev), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    out = []
    if hasattr(model, "head"):  # siamese: embed each deck once, then only the pairing head runs per pair
        za, zb = (model.embed(X, torch.full(X.shape, level - 12.0, device=dev)) for X in (A, B))
        for s in range(0, len(ia), bs):
            out.append(model.head(za[ia[s:s + bs]], zb[ib[s:s + bs]]).cpu())
        return torch.cat(out).view(len(A), len(B)).numpy().astype(np.float64)
    for s in range(0, len(ia), bs):
        a, b = A[ia[s:s + bs]], B[ib[s:s + bs]]
        lv = torch.full(a.shape, level - 12.0, device=dev); z = torch.zeros(len(a), device=dev)
        out.append(model({"cards_a": a, "cards_b": b, "level_a": lv, "level_b": lv, "troph_a": z, "troph_b": z,
                          "deck_a": z.long(), "deck_b": z.long()}).cpu())
    return torch.cat(out).view(len(A), len(B)).numpy().astype(np.float64)


def split_cyclic(M):
    """Hodge-style split of an antisymmetric matchup matrix: M = (s_i - s_j) + C."""
    s = M.mean(1)
    return s, M - (s[:, None] - s[None, :])

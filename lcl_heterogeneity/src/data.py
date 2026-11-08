"""Sushi data, preprocessed exactly as in Tomlinson & Benson (feature-context-effects/datasets.py).

- sushi3b.5000.10.order: each respondent ranks a set of 10 sushi drawn from 100.
- The paper uses only the top-ranked item, as one choice from the set of 10.
- Features (sushi3.idata columns 2,3,5,6,7,8): is maki, is seafood, oiliness, popularity,
  price, availability. The first two are flipped to 1 = maki / seafood.
- Standardised with the mean and sd over all item occurrences in all choice sets (as in
  load_standardized).

Note on orientation: the raw "oiliness" column is 0 = heavy/oily, 4 = light, and the paper uses
it unflipped. A self-effect A_pp does not depend on the sign of feature p, so the headline
"oiliness on oiliness" number is orientation-free.
"""

import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "sushi3-2016")

FEATURES = ["is_maki", "is_seafood", "oiliness", "popularity", "price", "availability"]
OIL = 2
CONTINUOUS = [2, 3, 4, 5]  # binary features cannot carry a separate squared term


def load_items():
    raw = np.loadtxt(os.path.join(RAW, "sushi3.idata"), usecols=(2, 3, 5, 6, 7, 8))
    raw[:, 0] = 1 - raw[:, 0]
    raw[:, 1] = 1 - raw[:, 1]
    return raw  # 100 x 6, unstandardised


def load_rankings():
    return np.loadtxt(os.path.join(RAW, "sushi3b.5000.10.order"), skiprows=1, usecols=range(2, 12), dtype=int)


def load_users():
    """Columns: gender (1 = female), age group 0-5, east/west at age 15 (1 = west)."""
    u = np.loadtxt(os.path.join(RAW, "sushi3.udata"), dtype=int)
    return {"gender": u[:, 1], "age": u[:, 2], "west15": u[:, 6], "west_now": u[:, 9]}


def standardiser(items, rankings):
    occ = items[rankings.reshape(-1)]
    return occ.mean(0), occ.std(0, ddof=1)  # torch .std default is unbiased


class ChoiceData:
    """A batch of choice observations, possibly several per person (exploded rankings).

    feats  : N x J x d   (padded)
    valid  : N x J bool
    choice : N           (index into J)
    person : N           (0..P-1)
    item   : N x J       (item ids, -1 for padding)
    """

    def __init__(self, feats, valid, choice, person, item=None):
        self.feats, self.valid, self.choice, self.person, self.item = feats, valid, choice, person, item
        self.P = int(person.max()) + 1
        n_valid = valid.sum(1, keepdim=True)
        self.xbar = (feats * valid[..., None]).sum(1) / n_valid  # N x d, mean features of the set

    @property
    def N(self):
        return self.feats.shape[0]

    def subset_people(self, people):
        people = torch.as_tensor(people)
        remap = torch.full((self.P,), -1, dtype=torch.long)
        remap[people] = torch.arange(len(people))
        keep = remap[self.person] >= 0
        return ChoiceData(self.feats[keep], self.valid[keep], self.choice[keep], remap[self.person[keep]],
                          None if self.item is None else self.item[keep])

    def with_choice(self, choice):
        return ChoiceData(self.feats, self.valid, choice, self.person, self.item)


def sushi(top_k=1, item_feats=None):
    """Exploded top-k rankings. top_k=1 is the paper's dataset (one choice from 10 per person).

    Stage t: choose the rank-t item from the items ranked t..9.
    item_feats: optional 100 x d standardised item features (default: the paper's).
    """
    items, ranks = load_items(), load_rankings()
    if item_feats is None:
        mu, sd = standardiser(items, ranks)
        item_feats = (items - mu) / sd
    X = torch.tensor(item_feats)
    P, J = ranks.shape
    feats, valid, choice, person, item = [], [], [], [], []
    for t in range(top_k):
        ids = torch.tensor(ranks)                      # P x 10
        v = torch.zeros(P, J, dtype=torch.bool)
        v[:, t:] = True
        feats.append(X[ids] * v[..., None])
        valid.append(v)
        choice.append(torch.full((P,), t, dtype=torch.long))
        person.append(torch.arange(P))
        item.append(torch.where(v, ids, torch.full_like(ids, -1)))
    return ChoiceData(torch.cat(feats), torch.cat(valid), torch.cat(choice), torch.cat(person), torch.cat(item))


def sushi_standardised_items():
    items, ranks = load_items(), load_rankings()
    mu, sd = standardiser(items, ranks)
    return (items - mu) / sd

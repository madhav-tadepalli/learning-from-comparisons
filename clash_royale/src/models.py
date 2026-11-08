"""Matchup models: P(A beats B) = sigmoid(logit(A, B)), with logit(A, B) = -logit(B, A).

A batch is a dict of tensors:
    cards_a, cards_b   (N, 8) long   card index 0..C-1
    level_a, level_b   (N, 8) float  card level, centred (level - 12)
    troph_a, troph_b   (N,)   float  starting trophies / 100
    deck_a, deck_b     (N,)   long   deck-vocabulary id (0 = unseen deck), only for deck-id models
"""
import torch
import torch.nn as nn


class Skill(nn.Module):
    """Linear player-skill term from the trophy gap, shared by every model."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, b):
        return self.beta * (b["troph_a"] - b["troph_b"])


# ---------------------------------------------------------------- baselines

class Constant(nn.Module):
    def forward(self, b):
        return torch.zeros_like(b["troph_a"])


class TrophyLevel(nn.Module):
    """Logistic regression on trophy gap (+ optionally total card-level gap)."""

    def __init__(self, use_levels=True):
        super().__init__()
        self.skill = Skill()
        self.use_levels = use_levels
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, b):
        out = self.skill(b)
        if self.use_levels:
            out = out + self.gamma * (b["level_a"].sum(1) - b["level_b"].sum(1))
        return out


class CardBT(nn.Module):
    """Bradley-Terry with deck strength = sum of card strengths (+ per-card level slope)."""

    def __init__(self, n_cards, use_levels=True, use_trophies=True):
        super().__init__()
        self.w = nn.Embedding(n_cards, 1)
        self.v = nn.Embedding(n_cards, 1)
        nn.init.zeros_(self.w.weight); nn.init.zeros_(self.v.weight)
        self.use_levels, self.use_trophies = use_levels, use_trophies
        self.skill = Skill()

    def strength(self, cards, level):
        s = self.w(cards).squeeze(-1)
        if self.use_levels:
            s = s + self.v(cards).squeeze(-1) * level
        return s.sum(1)

    def forward(self, b):
        out = self.strength(b["cards_a"], b["level_a"]) - self.strength(b["cards_b"], b["level_b"])
        return out + self.skill(b) if self.use_trophies else out


# ---------------------------------------------------------------- deck encoders

class CardTokens(nn.Module):
    """Card embedding plus a card-specific direction scaled by the card's level."""

    def __init__(self, n_cards, d, use_levels=True):
        super().__init__()
        self.emb = nn.Embedding(n_cards, d)
        self.lvl = nn.Embedding(n_cards, d)
        nn.init.normal_(self.emb.weight, std=0.1); nn.init.zeros_(self.lvl.weight)
        self.use_levels = use_levels

    def forward(self, cards, level):
        t = self.emb(cards)
        if self.use_levels:
            t = t + self.lvl(cards) * level.unsqueeze(-1)
        return t


class BagEncoder(nn.Module):
    """Deck embedding = sum of card tokens (linear, no card-card interactions)."""

    def __init__(self, n_cards, d, use_levels=True):
        super().__init__()
        self.tok = CardTokens(n_cards, d, use_levels)

    def forward(self, cards, level):
        return self.tok(cards, level).sum(1)


class DeepSetEncoder(nn.Module):
    """rho(sum phi(card)): permutation invariant, nonlinear synergies."""

    def __init__(self, n_cards, d, h=256, use_levels=True):
        super().__init__()
        self.tok = CardTokens(n_cards, d, use_levels)
        self.phi = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, cards, level):
        return self.rho(self.phi(self.tok(cards, level)).sum(1))


class AttnEncoder(nn.Module):
    """Self-attention over the deck's 8 cards, then mean-pool (models card-card synergy)."""

    def __init__(self, n_cards, d, layers=2, heads=4, use_levels=True, dropout=0.1):
        super().__init__()
        self.tok = CardTokens(n_cards, d, use_levels)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, cards, level):
        return self.norm(self.enc(self.tok(cards, level)).mean(1))


class DeckIdEncoder(nn.Module):
    """Classic blade-chest setting: one free vector per frequent deck (id 0 = unseen)."""

    def __init__(self, n_decks, d):
        super().__init__()
        self.emb = nn.Embedding(n_decks, d)
        nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, deck_ids):
        return self.emb(deck_ids)


# ---------------------------------------------------------------- pairing heads (all antisymmetric)

class BTHead(nn.Module):
    """Transitive: s(A) - s(B)."""

    def __init__(self, d):
        super().__init__()
        self.s = nn.Linear(d, 1, bias=False)

    def forward(self, za, zb):
        return (self.s(za) - self.s(zb)).squeeze(-1)


class BladeChestInner(nn.Module):
    """Chen & Joachims (2016) inner-product blade-chest: s(A)-s(B) + blade_A.chest_B - blade_B.chest_A."""

    def __init__(self, d, k):
        super().__init__()
        self.s = nn.Linear(d, 1, bias=False)
        self.blade = nn.Linear(d, k, bias=False)
        self.chest = nn.Linear(d, k, bias=False)

    def forward(self, za, zb):
        bt = (self.s(za) - self.s(zb)).squeeze(-1)
        ba, ca, bb, cb = self.blade(za), self.chest(za), self.blade(zb), self.chest(zb)
        return bt + (ba * cb).sum(-1) - (bb * ca).sum(-1)


class BladeChestDist(nn.Module):
    """Blade-chest distance: A wins when its blade is close to B's chest."""

    def __init__(self, d, k):
        super().__init__()
        self.s = nn.Linear(d, 1, bias=False)
        self.blade = nn.Linear(d, k, bias=False)
        self.chest = nn.Linear(d, k, bias=False)

    def forward(self, za, zb):
        bt = (self.s(za) - self.s(zb)).squeeze(-1)
        ba, ca, bb, cb = self.blade(za), self.chest(za), self.blade(zb), self.chest(zb)
        return bt + ((bb - ca) ** 2).sum(-1) - ((ba - cb) ** 2).sum(-1)


class MLPHead(nn.Module):
    """Unrestricted matchup function, antisymmetrised: g(A,B) - g(B,A)."""

    def __init__(self, d, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, za, zb):
        return (self.g(torch.cat([za, zb], -1)) - self.g(torch.cat([zb, za], -1))).squeeze(-1)


ENCODERS = {"bag": BagEncoder, "deepset": DeepSetEncoder, "attn": AttnEncoder}
HEADS = {"bt": BTHead, "bc_inner": BladeChestInner, "bc_dist": BladeChestDist, "mlp": MLPHead}


class EncoderHead(nn.Module):
    """Siamese: the same encoder embeds both decks, a head pairs them."""

    def __init__(self, n_cards, encoder="bag", head="bt", d=64, k=16, use_levels=True, use_trophies=True):
        super().__init__()
        self.enc = ENCODERS[encoder](n_cards, d, use_levels=use_levels)
        self.head = HEADS[head](d) if head in ("bt", "mlp") else HEADS[head](d, k)
        self.use_trophies = use_trophies
        self.skill = Skill()

    def embed(self, cards, level):
        return self.enc(cards, level)

    def forward(self, b):
        out = self.head(self.embed(b["cards_a"], b["level_a"]), self.embed(b["cards_b"], b["level_b"]))
        return out + self.skill(b) if self.use_trophies else out


class DeckIdModel(nn.Module):
    """Deck-id blade-chest / BT, with level-sum and trophy terms (decks never seen in train share id 0)."""

    def __init__(self, n_decks, head="bc_inner", d=32, k=16):
        super().__init__()
        self.enc = DeckIdEncoder(n_decks, d)
        self.head = HEADS[head](d) if head in ("bt", "mlp") else HEADS[head](d, k)
        self.base = TrophyLevel(use_levels=True)

    def forward(self, b):
        return self.head(self.enc(b["deck_a"]), self.enc(b["deck_b"])) + self.base(b)


class CrossAttention(nn.Module):
    """Joint transformer over all 16 cards (side embedding marks own/opponent), so attention can
    learn card-vs-card counters directly. Antisymmetrised by scoring both orderings."""

    def __init__(self, n_cards, d=64, layers=3, heads=4, use_levels=True, use_trophies=True, dropout=0.1):
        super().__init__()
        self.tok = CardTokens(n_cards, d, use_levels)
        self.side = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.use_trophies = use_trophies
        self.skill = Skill()

    def score(self, ca, la, cb, lb):
        x = torch.cat([self.tok(ca, la) + self.side.weight[0], self.tok(cb, lb) + self.side.weight[1]], 1)
        h = self.enc(x)
        return self.out(torch.cat([self.norm(h[:, :8].mean(1)), self.norm(h[:, 8:].mean(1))], -1)).squeeze(-1)

    def forward(self, b):
        ca, la, cb, lb = b["cards_a"], b["level_a"], b["cards_b"], b["level_b"]
        out = self.score(ca, la, cb, lb) - self.score(cb, lb, ca, la)
        return out + self.skill(b) if self.use_trophies else out

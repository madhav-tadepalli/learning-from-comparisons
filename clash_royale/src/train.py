"""Train one matchup model and save its metrics, weights and test predictions.

Usage: python src/train.py <run_name> <model> [key=value ...]
  model: constant | trophy | trophy_level | card_bt | enc:<encoder>:<head> | cross | deckid:<head>
  options: d, k, lr, wd, bs, epochs, patience, levels (0/1), trophies (0/1), seed, min_count
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import models as M

DATA = Path("data/processed/s18_ladder_12292020.parquet")
OUT = Path("results/runs")


def device():
    if os.environ.get("DEVICE"): return os.environ["DEVICE"]  # e.g. DEVICE=cpu to leave the GPU to a training run
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load(device, min_count=20):
    df = pd.read_parquet(DATA)
    n_cards = len(pd.read_csv("data/processed/cards.csv"))
    t = lambda a, dt: torch.tensor(np.ascontiguousarray(a), dtype=dt)
    ca = df[[f"card{i}_a" for i in range(8)]].to_numpy(); cb = df[[f"card{i}_b" for i in range(8)]].to_numpy()

    # deck vocabulary from training decks with >= min_count appearances; id 0 = unseen/rare
    key = lambda c: [" ".join(map(str, r)) for r in c]
    ka, kb = key(ca), key(cb)
    tr = (df.split == "train").to_numpy()
    counts = pd.Series(np.concatenate([np.array(ka)[tr], np.array(kb)[tr]])).value_counts()
    vocab = {k: i + 1 for i, k in enumerate(counts[counts >= min_count].index)}

    data = {
        "cards_a": t(ca, torch.long), "cards_b": t(cb, torch.long),
        "level_a": t(df[[f"level{i}_a" for i in range(8)]].to_numpy() - 12.0, torch.float32),
        "level_b": t(df[[f"level{i}_b" for i in range(8)]].to_numpy() - 12.0, torch.float32),
        "troph_a": t(df.trophies_a.to_numpy() / 100.0, torch.float32),
        "troph_b": t(df.trophies_b.to_numpy() / 100.0, torch.float32),
        "deck_a": t([vocab.get(k, 0) for k in ka], torch.long),
        "deck_b": t([vocab.get(k, 0) for k in kb], torch.long),
        "y": t(df.a_wins.to_numpy(), torch.float32),
    }
    split = {s: torch.tensor(np.flatnonzero(df.split.to_numpy() == s)) for s in ("train", "val", "test")}
    data = {k: v.to(device) for k, v in data.items()}
    return data, {k: v.to(device) for k, v in split.items()}, n_cards, len(vocab) + 1


def build(name, n_cards, n_decks, o):
    lv, tr = bool(o["levels"]), bool(o["trophies"])
    if name == "constant": return M.Constant()
    if name == "trophy": return M.TrophyLevel(use_levels=False)
    if name == "trophy_level": return M.TrophyLevel(use_levels=True)
    if name == "card_bt": return M.CardBT(n_cards, lv, tr)
    if name.startswith("enc:"):
        _, enc, head = name.split(":")
        return M.EncoderHead(n_cards, enc, head, d=o["d"], k=o["k"], use_levels=lv, use_trophies=tr)
    if name == "cross": return M.CrossAttention(n_cards, d=o["d"], use_levels=lv, use_trophies=tr)
    if name.startswith("deckid:"): return M.DeckIdModel(n_decks, name.split(":")[1], d=o["d"], k=o["k"])
    raise ValueError(name)


@torch.no_grad()
def predict(model, data, idx, bs=65536):
    model.eval()
    out = [model({k: v[idx[i:i + bs]] for k, v in data.items()}) for i in range(0, len(idx), bs)]
    return torch.cat(out)


def metrics(logit, y):
    p = torch.sigmoid(logit.cpu().double()).numpy(); y = y.cpu().double().numpy()
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {"logloss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
            "acc": float(((p > 0.5) == y).mean()), "auc": float(roc_auc_score(y, p)),
            "brier": float(((p - y) ** 2).mean())}


def main():
    run, name = sys.argv[1], sys.argv[2]
    o = dict(d=64, k=16, lr=2e-3, wd=1e-5, bs=8192, epochs=40, patience=3, levels=1, trophies=1, seed=0, min_count=20)
    for kv in sys.argv[3:]:
        k, v = kv.split("="); o[k] = type(o[k])(float(v)) if isinstance(o[k], (int, float)) else v
    torch.manual_seed(o["seed"]); np.random.seed(o["seed"])
    dev = device()

    data, split, n_cards, n_decks = load(dev, o["min_count"])
    model = build(name, n_cards, n_decks, o).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    y = data["y"]
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    ckpt = OUT / f"{run}.ckpt"  # per-epoch checkpoint so a killed run can resume; deleted when the run finishes
    OUT.mkdir(parents=True, exist_ok=True)

    if n_params > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["wd"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)
        tr = split["train"]
        start = 0
        if ckpt.exists():
            c = torch.load(ckpt, map_location=dev, weights_only=False)
            model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
            best, best_state, bad, hist, start = c["best"], c["best_state"], c["bad"], c["hist"], c["epoch"] + 1
            torch.set_rng_state(c["rng"].cpu()); t0 -= c["elapsed"]
            print(f"[{run}] resumed from checkpoint after epoch {c['epoch']}", flush=True)
        stopped = bad >= o["patience"]
        for ep in range(start, o["epochs"]):
            if stopped: break
            model.train(); perm = tr[torch.randperm(len(tr), device=dev)]; tot = 0.0
            for i in range(0, len(perm), o["bs"]):
                idx = perm[i:i + o["bs"]]
                loss = F.binary_cross_entropy_with_logits(model({k: v[idx] for k, v in data.items()}), y[idx])
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
            val = metrics(predict(model, data, split["val"]), y[split["val"]])["logloss"]
            sched.step(val)
            hist.append({"epoch": ep, "train": tot / len(perm), "val": val, "sec": round(time.time() - t0, 1)})
            print(f"[{run}] ep {ep:2d} train {tot / len(perm):.5f} val {val:.5f} ({time.time() - t0:.0f}s)", flush=True)
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                stopped = bad >= o["patience"]
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                        "best": best, "best_state": best_state, "bad": bad, "hist": hist, "rng": torch.get_rng_state(),
                        "elapsed": time.time() - t0}, tmp)
            tmp.replace(ckpt)  # atomic, so a kill mid-save never leaves a corrupt checkpoint
        model.load_state_dict(best_state)

    res = {"run": run, "model": name, "opts": o, "params": n_params, "train_sec": round(time.time() - t0, 1), "history": hist}
    for s in ("val", "test"):
        res[s] = metrics(predict(model, data, split[s]), y[split[s]])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f"{run}.pt")
    np.save(OUT / f"{run}_test_logit.npy", predict(model, data, split["test"]).cpu().numpy())
    json.dump(res, open(OUT / f"{run}.json", "w"), indent=1)  # written last: the queue treats the json as "done"
    ckpt.unlink(missing_ok=True)
    print(f"[{run}] DONE params={n_params:,} test {res['test']}", flush=True)


if __name__ == "__main__":
    main()

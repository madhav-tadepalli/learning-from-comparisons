# Clash Royale deck matchups

Course project for DA5453 (IIT Madras). The goal: given two Clash Royale decks, their card levels and the players' trophies, predict who wins. Along the way, learn an embedding for each deck and model matchups that can be intransitive (rock-paper-scissors), for example with blade-chest.

**Results and discussion: [REPORT.md](REPORT.md).** Deck similarity, counters and archetypes: REPORT §6, [results/deck_relations.md](results/deck_relations.md) and [results/embedding_analysis.md](results/embedding_analysis.md).

Headline: on 235K held-out matches, a sum-of-card-embeddings encoder with an MLP matchup head gets **62.3%** accuracy (log-loss 0.641). A per-card strength model gets 57.8%. The gain comes from the matchup head, not from a fancier encoder.

## Setup (Windows / Linux / macOS)

This folder lives inside the course `Project/` folder, which keeps one shared environment at `Project/.venv` (also used by `../lcl_heterogeneity`). To create it from scratch:

```bash
cd ..                                # the Project/ folder
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r clash_royale/requirements.txt     # NVIDIA GPU: install torch from pytorch.org first
```

Run every command from this `clash_royale/` folder, because the paths are relative. The code uses CUDA, then Apple MPS, then CPU, whichever is available (`train.device()`).

## Data

The data isn't in the repo (1.4 GB). Fetch the one S18 battle file we use, plus the card lists. This doesn't need a Kaggle login:

```bash
python src/download_data.py                  # → data/raw/s18/
python src/prepare_s18.py data/raw/s18/BattlesStaging_12292020_WL_tagged.csv
                                             # → data/processed/s18_ladder_12292020.parquet + cards.csv
```

The source is [Clash Royale S18 Ladder Datasets](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset). The full dataset is 22 GB across 10 battle files; we only use `BattlesStaging_12292020_WL_tagged.csv`.

`prepare_s18.py` does the cleaning:
- keeps normal ladder games only
- drops rows with missing values and duplicate battles
- requires valid decks: 8 distinct known cards, levels 1–13
- sorts the cards in each deck
- randomly assigns winner/loser to sides A/B, so column order doesn't leak the label
- splits by time into 80/10/10 train/val/test

Result: 2,345,592 matches.

## Running things

```bash
python src/train.py <run_name> <model> [key=value ...]   # one model → results/runs/<run_name>.{json,pt} + test logits
python src/train_gbdt.py gbdt                            # gradient-boosted trees baseline
bash run_all.sh                                          # full experiment queue (skips runs that already exist)
python src/analyze.py                                    # results table, figures → results/
python src/deck_relations.py                             # deck similarity + counters → results/deck_*.csv, deck_relations.md
python src/embedding_analysis.py                         # what the embeddings encode: similarity, counters, archetypes, RPS planes
python src/deck_query.py "Hog Rider, Musketeer, Ice Golem, Ice Spirit, Skeletons, Cannon, Fireball, The Log"
                                                         # any 8-card deck → similar decks, archetype, counters, swaps
```

Set `DEVICE=cpu` (or `cuda`, `mps`) to override the device choice. For example, run the analysis on CPU while a training run uses the GPU.

Models for `train.py`:

| `<model>` | what it is |
|---|---|
| `constant`, `trophy`, `trophy_level` | baselines (side bias / trophy gap / + total card level) |
| `card_bt` | Bradley–Terry: deck strength = sum of per-card strengths (+ level slopes) |
| `enc:<encoder>:<head>` | Siamese deck encoder + pairing head. Encoders: `bag` (sum of card embeddings), `deepset`, `attn` (self-attention). Heads: `bt`, `bc_inner` / `bc_dist` (blade-chest), `mlp` (antisymmetrised MLP) |
| `cross` | cross-attention transformer over all 16 cards |
| `deckid:<head>` | classic blade-chest with one vector per frequent deck |

Options (defaults): `d=64 k=16 lr=2e-3 wd=1e-5 bs=8192 epochs=40 patience=3 levels=1 trophies=1 seed=0`. Use `levels=0 trophies=0` for decks-only. All models are antisymmetric: P(A beats B) = 1 − P(B beats A).

Example: `python src/train.py bag_mlp_s1 enc:bag:mlp seed=1`

**Trained weights for every run are already in `results/runs/`**, so `analyze.py` and `deck_relations.py` work without retraining once the data is prepared. On a 2025 MacBook (M5) most runs take 1–30 min; `cross` takes about 3 h. Training writes a checkpoint every epoch (`results/runs/<run>.ckpt`). Re-running the same command, or the queue script, resumes from it, and the checkpoint is deleted when the run finishes.

### Running on Kaggle instead of your machine

`kaggle/push.py` sends any of the commands above to a private Kaggle notebook. The notebook attaches the S18 dataset, rebuilds the cleaned table, runs the commands and keeps `results/` as its output. Nothing is downloaded locally. You need the Kaggle CLI, signed in (`pip install kaggle`, then an API token).

```bash
python kaggle/push.py "python src/train_gbdt.py gbdt"                        # CPU notebook
python kaggle/push.py "python src/train.py bag_mlp_s3 enc:bag:mlp seed=3" --gpu
kaggle kernels status <your-username>/clash-royale-runs
kaggle kernels output <your-username>/clash-royale-runs -p kaggle/out        # then copy kaggle/out/results/... into results/
```

The log prints a `fingerprint` line (row count + hash) for the rebuilt table. It should match `2345592 b08be8b53b96f44437c865ecdc88a9fa`, the locally prepared data.

## Layout

```
src/
  download_data.py   fetch the S18 files
  prepare_s18.py     clean → parquet
  models.py          all model classes (encoders, heads, cross-attention, deck-id)
  train.py           training loop, data loading, metrics, device choice
  train_gbdt.py      HistGradientBoosting baseline
  diag_gbdt.py       which feature groups the GBDT relies on
  analyze.py         results table with bootstrap CIs, card strengths, matchup structure, figures
  decks.py           helpers: deck pool, labels, matchup matrices, Hodge split
  deck_relations.py  similar decks, counters, held-out validation
  embedding_analysis.py  embedding similarity, blade-chest counters, archetype clusters, RPS planes
  deck_query.py      look up any deck
kaggle/push.py       run commands on a Kaggle notebook
run_all.sh           experiment queue that produced the results
run_round2.sh        round 2: extra seeds, decks-only for the best heads, light tuning
results/             metrics (all_runs.csv, summary.md), figures/, runs/ (weights + test logits)
REPORT.md            write-up
```

## Open next steps

See REPORT.md, "Next steps":
- **Generalisation:** test on a later day.
- **A level-aware matchup head:** let the level gap change the matchup, not just shift strength.
- **Deck recommendation:** needs constraints that keep it near real decks (see REPORT §6.4).

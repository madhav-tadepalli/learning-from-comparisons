# Context effects or different people? A check of LCL on sushi

DA5453 project (Advik Kabra, Madhav Tadepalli). Paper: Tomlinson & Benson, *Learning Interpretable
Feature Context Effects in Discrete Choice* (KDD 2021).

This folder is separate from the Clash Royale work in the parent directory. It uses the parent's
Python environment (`../.venv`: torch, numpy, scipy, pandas, matplotlib).

**Question.** In the sushi data the paper finds that people avoid oily sushi more when the menu is
oilier (oiliness self-effect A = −0.20, −0.27 alone, p = 1.6e-6). Because sushi menus are random,
the paper calls this a candidate causal effect. Random menus rule out confounding, but they do not
rule out *people simply differing in how much they like oily food*. Can that alone produce the effect?

Results and discussion: [REPORT.md](REPORT.md). LaTeX report draft: [report/main.tex](report/main.tex) → `report/main.pdf`
(build with `cd report && tectonic -X compile main.tex`; figures come from `figures/`).

## Layout

| path | what |
|---|---|
| `src/data.py` | sushi loading, preprocessed exactly as the authors' code (features, flips, standardisation) |
| `src/models.py` | MNL / LCL / squared terms / demographics / normal random coefficients / latent classes, panel-capable, exact MLE with L-BFGS |
| `src/reproduce.py` | reproduces the paper's sushi numbers (Tables 3, 5, 10) |
| `src/theory_check.py` | checks the second-order theory on the real sushi sets |
| `src/simulate.py` → `src/summarise_sim.py` | simulation study: 2 kinds of heterogeneity (normal spread; discrete segments 50/50 and 20/80) plus controls |
| `src/real_data.py` | heterogeneity-aware tests on the real data, including the full rankings |
| `src/bootstrap.py` | parametric bootstrap from heterogeneity models fitted to sushi |
| `src/panel_check.py` | checks that the ranking-based (panel) test has the right size and power |
| `src/stage_check.py` | first-pick vs later-pick self-effect in the panel model |
| `src/figures.py` | figures in `figures/` |
| `src/pilot_heterogeneity.py` | first pilot (Gaussian synthetic items), kept for the record |
| `results/` | all outputs (text, CSV, JSON) |

## Reproduce

Run from this folder:

```bash
mkdir -p data/raw && cd data/raw && curl -sSLO https://www.kamishima.net/asset/sushi3-2016.zip && unzip sushi3-2016.zip && cd ../..
../.venv/bin/python -I src/reproduce.py        # seconds
../.venv/bin/python -I src/theory_check.py     # ~1 min
../.venv/bin/python -I src/real_data.py        # ~10-15 min
../.venv/bin/python -I src/simulate.py         # ~2-3 h on 8 cores (resumable)
../.venv/bin/python -I src/summarise_sim.py
../.venv/bin/python -I src/bootstrap.py        # ~15 min
../.venv/bin/python -I src/panel_check.py      # ~1 h
../.venv/bin/python -I src/stage_check.py      # ~10 min
../.venv/bin/python -I src/figures.py
```

The sushi data may not be redistributed (Kamishima's licence), so it is not included; cite
Kamishima (KDD 2003) when using it.

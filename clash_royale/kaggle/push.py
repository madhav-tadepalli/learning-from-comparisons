"""Run repo commands on a Kaggle notebook instead of the laptop.

Bundles src/*.py into one self-contained Kaggle script that attaches the S18 dataset,
rebuilds the processed parquet from the 12/29 file, runs the given commands, and leaves
results/runs/ as the notebook output. Needs the Kaggle CLI, signed in.

Usage:  python kaggle/push.py "python src/train_gbdt.py gbdt_lvfix iters=3000" [--gpu]
Fetch:  kaggle kernels output <user>/clash-royale-runs -p kaggle/out
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG = "clash-royale-runs"
DATASET = "bwandowando/clash-royale-season-18-dec-0320-dataset"

cmds = [a for a in sys.argv[1:] if a != "--gpu"]
gpu = "--gpu" in sys.argv
user = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
user = next(l.split(":", 1)[1].strip() for l in user.splitlines() if "username" in l)

src = {p.name: p.read_text() for p in sorted((ROOT / "src").glob("*.py"))}
script = f'''import glob, hashlib, os, shutil, subprocess, sys
from pathlib import Path
SRC = {src!r}
CMDS = {cmds!r}
W = Path("/kaggle/working"); os.chdir(W)
(W / "src").mkdir(exist_ok=True)
for n, t in SRC.items(): (W / "src" / n).write_text(t)
raw = W / "data/raw/s18"; raw.mkdir(parents=True, exist_ok=True)
find = lambda pat: glob.glob("/kaggle/input/**/" + pat, recursive=True)[0]
for pat in ["CardMasterListSeason18_12082020.csv", "Wincons.csv"]:
    (raw / pat).unlink(missing_ok=True); (raw / pat).symlink_to(find(pat))
battles = find("BattlesStaging_12292020_WL_tagged.csv")
run = lambda c: subprocess.run(c, shell=True, check=True)
run(f"python src/prepare_s18.py {{battles}}")
import pandas as pd
df = pd.read_parquet("data/processed/s18_ladder_12292020.parquet")
print("fingerprint", len(df), hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest(), flush=True)
for c in CMDS: run(c)
shutil.rmtree(W / "data"); shutil.rmtree(W / "src")
'''
k = ROOT / "kaggle" / "build"
k.mkdir(parents=True, exist_ok=True)
(k / "run.py").write_text(script)
json.dump({"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "run.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": False,
           "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []},
          open(k / "kernel-metadata.json", "w"), indent=1)
subprocess.run(["kaggle", "kernels", "push", "-p", str(k)], check=True)
print(f"status: kaggle kernels status {user}/{SLUG}")

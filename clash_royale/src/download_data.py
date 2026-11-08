"""Download the S18 files this project uses (one ~1.4 GB battle file + two small card lists) into data/raw/s18.

Uses kagglehub (no Kaggle login needed for this public dataset). Only the named files are fetched,
not the full 22 GB dataset.

Usage: python src/download_data.py
"""
import shutil
from pathlib import Path

import kagglehub

DATASET = "bwandowando/clash-royale-season-18-dec-0320-dataset"
FILES = ["CardMasterListSeason18_12082020.csv", "Wincons.csv", "BattlesStaging_12292020_WL_tagged.csv"]
OUT = Path("data/raw/s18")
OUT.mkdir(parents=True, exist_ok=True)

for f in FILES:
    if (OUT / f).exists():
        print(f"have {f}")
        continue
    print(f"downloading {f} ...")
    shutil.copy(kagglehub.dataset_download(DATASET, path=f), OUT / f)
print("done:", sorted(p.name for p in OUT.iterdir()))

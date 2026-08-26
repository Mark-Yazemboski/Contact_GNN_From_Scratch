"""
check_wind_labels.py

Run this on a dataset folder BEFORE launching any training on it.

Verifies that the wind vector actually reached disk in BOTH places the pipeline
reads it from:
    raw[1]              <- get_gns_features / get_clean_positions read this
    raw[3]['wind']      <- analysis scripts read this

Usage:
    python check_wind_labels.py data/mojoco_paper_replica_20_wind
    python check_wind_labels.py data/mojoco_paper_replica          # expects zeros
"""

import os
import sys
import numpy as np
import torch

script_dir   = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")
expect_wind = "wind" in os.path.basename(folder)

files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")],
               key=lambda s: int(s[:-3]))
print(f"{folder}: {len(files)} trajectories   (expecting wind: {expect_wind})")

mags, n_zero_slot1, n_zero_params, n_mismatch = [], 0, 0, 0
for fn in files:
    raw = torch.load(os.path.join(folder, fn), weights_only=False)
    w1 = np.asarray(raw[1], dtype=float).reshape(3)
    wp = np.asarray(raw[3].get("wind", np.zeros(3)), dtype=float).reshape(3)
    if np.linalg.norm(w1) < 1e-9:
        n_zero_slot1 += 1
    if np.linalg.norm(wp) < 1e-9:
        n_zero_params += 1
    if np.linalg.norm(w1 - wp) > 1e-5:
        n_mismatch += 1
    mags.append(float(np.linalg.norm(w1)))

mags = np.array(mags)
print(f"  |wind| from raw[1]: median {np.median(mags):.3f}  "
      f"mean {mags.mean():.3f}  range [{mags.min():.3f}, {mags.max():.3f}] m/s")
print(f"  zero in raw[1]: {n_zero_slot1}   zero in params['wind']: {n_zero_params}"
      f"   slot/params mismatch: {n_mismatch}")

ok = True
if expect_wind and n_zero_slot1 > 0:
    print(f"  FAIL: {n_zero_slot1} trajectories have zero wind in raw[1] "
          f"-- training would see NO wind information."); ok = False
if expect_wind and n_zero_params > 0:
    print(f"  FAIL: {n_zero_params} trajectories have zero params['wind'] "
          f"-- analysis scripts would bin everything at wind=0."); ok = False
if n_mismatch:
    print(f"  FAIL: {n_mismatch} trajectories disagree between raw[1] and "
          f"params['wind']."); ok = False
if not expect_wind and n_zero_slot1 < len(files):
    print("  NOTE: this folder has non-zero wind but 'wind' is not in its name.")

print("  PASS" if ok else "  DO NOT TRAIN ON THIS FOLDER")
sys.exit(0 if ok else 1)
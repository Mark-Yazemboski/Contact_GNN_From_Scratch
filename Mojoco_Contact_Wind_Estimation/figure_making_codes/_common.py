"""
_common.py  —  small, boring helpers shared by the figure scripts.

Nothing here decides which runs go in a figure. That decision lives in each
figure script, written out as plain dictionaries, so you can read it and change
it without coming back here.

What this file gives you:

    load(csv)                 read the CSV, add three convenience columns
    select(df, criteria)      keep rows whose settings match; print what matched
    summarize(df, by, metric) mean / std / n / run names, grouped
    report(...)               print a table of what feeds each point
    save(fig, name)           write SVG + PNG to figure_assets/
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_DEFAULT = "Master_Data_CSV_9_14_26.csv"
OUTDIR = "figure_assets"

# poster palette
INK   = "#1a1a1a"
GRAY  = "#8a8a8a"
ORANGE = "#d95f02"
BLUE  = "#0072b2"
GREEN = "#009e73"
RED   = "#c0392b"
PURPLE = "#7b3294"

# ground truth constants
MU_TRUE = 0.198
K_TRUE = 0.0285

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#4a4a4a", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "figure.dpi": 110,
})


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def wind_from_dataset(name):
    """'mojoco_paper_replica_20_wind' -> 20.0
       'tosses_processed'             -> nan   (real data, no wind axis)"""
    match = re.search(r"replica_(\d+)_wind", str(name))
    if match:
        return float(match.group(1))
    return np.nan


def load(csv=CSV_DEFAULT):
    """Read the master CSV and add three columns we use everywhere:

        dataset_name  the folder name, without the path
        group         run_name with the trailing _1 / _2 / _3 stripped
        wind_max      the wind level parsed out of the dataset name
    """
    df = pd.read_csv(csv)

    dataset_name = df["settings.dataset"].astype(str).str.split("/").str[-1]
    group = df["run_name"].str.replace(r"_\d+$", "", regex=True)
    wind_max = dataset_name.apply(wind_from_dataset)

    extra = pd.DataFrame({"dataset_name": dataset_name,
                          "group": group,
                          "wind_max": wind_max}, index=df.index)
    return pd.concat([df, extra], axis=1)


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------

def select(df, criteria, label="", verbose=True):
    """Keep the rows whose settings match every key/value pair in `criteria`.

    criteria is a plain dict of column name -> required value, e.g.

        {"settings.use_wind": False, "settings.w_fluid_anchor": 0.0}

    Prints what it matched so you can check it against what you expect.
    """
    mask = pd.Series(True, index=df.index)
    for column, required in criteria.items():
        if column not in df.columns:
            print(f"    !! column not in CSV, ignored: {column}")
            continue
        mask = mask & (df[column] == required)

    chosen = df[mask].copy()

    if verbose:
        shown = ", ".join(f"{k.replace('settings.', '')}={v}"
                          for k, v in criteria.items())
        print(f"\n  [{label}]" if label else "")
        print(f"    criteria : {shown}")
        print(f"    matched  : {len(chosen)} runs, "
              f"{chosen['group'].nunique()} groups")
    return chosen


def summarize(df, by, metric):
    """Group and compute mean / std / n, and keep the run names that went in.

    Returns a DataFrame with one row per value of `by`, sorted by `by`.
    """
    rows = []
    for key, sub in df.groupby(by):
        values = sub[metric].to_numpy(dtype=float)
        rows.append({
            by: key,
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": sorted(sub["run_name"].tolist()),
            "groups": sorted(sub["group"].unique().tolist()),
        })
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def report(table, by, metric_label="value", unit=""):
    """Print one line per point: what it is, how many runs, and WHICH runs."""
    if table.empty:
        print("    (nothing)")
        return
    for _, row in table.iterrows():
        names = ", ".join(row["runs"])
        print(f"    {by}={row[by]:<6}  n={row['n']}  "
              f"{metric_label}={row['mean']:.4f} ± {row['std']:.4f}{unit}")
        print(f"        runs: {names}")
        if len(row["groups"]) > 1:
            print(f"        !! MORE THAN ONE GROUP AT THIS POINT: "
                  f"{row['groups']} — check your criteria")


def leftovers(df, used_indices, only_where=None):
    """Print rows that no arm claimed, so nothing goes missing silently."""
    rest = df.drop(index=used_indices)
    if only_where is not None:
        rest = rest[only_where.reindex(rest.index).fillna(False)]
    if rest.empty:
        print("\n  every candidate run was claimed by an arm")
        return
    print(f"\n  NOT used by any arm ({len(rest)} runs):")
    for group, sub in rest.groupby("group"):
        cfg = sub.iloc[0]
        print(f"    {group:<24} n={len(sub):<3} "
              f"use_wind={cfg['settings.use_wind']} "
              f"baseline={cfg['settings.use_drag_baseline']} "
              f"learn_k={cfg['settings.learn_k']} "
              f"anchor={cfg['settings.w_fluid_anchor']}")


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------

def save(fig, stem):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("svg", "png"):
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path, transparent=True, bbox_inches="tight",
                    pad_inches=0.05, dpi=220)
    plt.close(fig)
    print(f"\n  wrote {OUTDIR}/{stem}.svg  (+.png)")

"""
fig1_wind_ablation.py   —  rollout error vs wind, for each wind arm.

EVERYTHING THAT DECIDES WHICH RUNS APPEAR IS IN THE `ARMS` LIST BELOW.
Edit it there. Nothing is hidden in _common.py.

Each arm is a dict:

    label     what goes in the legend
    color     line colour
    marker    point marker
    criteria  column -> required value. A run is in this arm only if EVERY
              pair matches. These are settings columns from the CSV.

When you run it, it prints the criteria for each arm, how many runs matched,
and the exact run names behind every plotted point. Read that before you trust
the picture.

    python fig1_wind_ablation.py [path/to/master.csv]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from _common import (load, select, summarize, report, leftovers, save,
                     CSV_DEFAULT, INK, GRAY, ORANGE, RED, PURPLE)


# What we plot on the y axis.
METRIC = "metrics.center_error"
Y_LABEL = "rollout centre error  (block widths)"


# ======================================================================
# THE ARMS.  This is the part you edit.
# ======================================================================

ARMS = [
    {
        "label": "wind feature ON",
        "color": ORANGE,
        "marker": "o",
        "criteria": {
            "settings.use_wind":           True,
            "settings.use_drag_baseline":  False,
            "settings.learn_k":            True,
            "settings.w_fluid_anchor":     0.2,
            "settings.multistep":          4,
            "settings.w_fric_dir":         6.0,
        },
    },
    {
        "label": "wind feature OFF",
        "color": RED,
        "marker": "s",
        "criteria": {
            "settings.use_wind":           False,
            "settings.use_drag_baseline":  False,
            "settings.learn_k":            False,
            "settings.w_fluid_anchor":     0.0,
            "settings.multistep":          4,
            "settings.w_fric_dir":         6.0,
        },
    },
    # {
    #     "label": "inferred from velocity history",
    #     "color": PURPLE,
    #     "marker": "^",
    #     "criteria": {
    #         "settings.use_wind":           False,
    #         "settings.use_drag_baseline":  False,
    #         "settings.learn_k":            True,
    #         "settings.w_fluid_anchor":     0.2,
    #         "settings.multistep":          4,
    #         "settings.w_fric_dir":         6.0,
    #     },
    # },
]


# ======================================================================

def main(csv=CSV_DEFAULT):
    data = load(csv)

    # Only the simulated wind datasets have a wind axis. Real tosses are
    # excluded here; they get their own figure.
    is_wind_dataset = data["wind_max"].notna()
    candidates = data[is_wind_dataset]
    print(f"\n{len(candidates)} runs on simulated wind datasets "
          f"({candidates['group'].nunique()} groups)")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    claimed = []

    for arm in ARMS:
        runs = select(candidates, arm["criteria"], label=arm["label"])
        if runs.empty:
            print("    nothing matched — this arm is skipped")
            continue

        claimed.extend(runs.index.tolist())

        table = summarize(runs, "wind_max", METRIC)
        report(table, "wind_max", metric_label="centre error")

        ax.errorbar(table["wind_max"], table["mean"], yerr=table["std"],
                    marker=arm["marker"], ms=9, lw=2.2, capsize=5,
                    color=arm["color"], label=arm["label"], zorder=5)

    leftovers(candidates, claimed)

    ax.set_yscale("log")
    ax.set_xlabel("wind range in the dataset  (m/s)")
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Wind ablation", fontweight="bold", loc="left")
    ax.set_xticks(sorted(candidates["wind_max"].unique()))
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "mean ± s.d. over 3 runs",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=GRAY)

    save(fig, "fig1_wind_ablation")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT)

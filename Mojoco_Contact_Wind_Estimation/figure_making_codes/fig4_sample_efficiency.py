"""
fig4_sample_efficiency.py   —  rollout error vs number of training trajectories.

EVERYTHING THAT DECIDES WHICH RUNS APPEAR IS IN THE `ARMS` LIST BELOW.
Adding the Allen-style baseline later is one more dict in that list.

A sweep curve must be ONE dataset, so DATASET is pinned at the top. Pooling
tosses_processed with the simulated wind sets would average unlike things.

The script also reports total_optimizer_steps per point, because a sweep at a
fixed epoch cap does NOT give every point the same amount of training: steps
per epoch scale with n_train, so the small-data points run short. Read that
column before you read the curve.

    python fig4_sample_efficiency.py [path/to/master.csv]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from _common import (load, select, summarize, report, save, CSV_DEFAULT,
                     INK, GRAY, ORANGE, BLUE, RED)

METRIC = "metrics.center_error"
STEPS = "metrics.total_optimizer_steps"
DATASET = "tosses_processed"        # Allen et al.'s real cube tosses


# ======================================================================
# THE ARMS.  This is the part you edit.
# ======================================================================

ARMS = [
    {
        "label": "force architecture + multistep  (ours)",
        "color": ORANGE,
        "marker": "o",
        "criteria": {
            "settings.use_drag_baseline":  False,
            "settings.learn_k":            True,
            "settings.w_fluid_anchor":     0.2,
            "settings.multistep":          4,
            "settings.w_fric_dir":         6.0,
        },
    },
    # When the Allen-style baseline runs exist, uncomment and set the criteria
    # that identify them. Nothing else needs to change.
    #
    # {
    #     "label": "acceleration GNS, single-step  (Allen et al. style)",
    #     "color": GRAY,
    #     "marker": "s",
    #     "criteria": {
    #         "settings.multistep":      1,
    #         "settings.w_fric_dir":     0.0,
    #         "settings.w_fluid_anchor": 0.0,
    #     },
    # },
]


# ======================================================================

def main(csv=CSV_DEFAULT):
    data = load(csv)
    on_dataset = data[data["dataset_name"] == DATASET]
    print(f"\n{len(on_dataset)} runs on dataset '{DATASET}' "
          f"({on_dataset['group'].nunique()} groups)")
    if on_dataset.empty:
        print("  nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for arm in ARMS:
        runs = select(on_dataset, arm["criteria"], label=arm["label"])
        if runs.empty:
            print("    nothing matched — this arm is skipped")
            continue

        table = summarize(runs, "settings.n_train", METRIC)
        report(table, "settings.n_train", metric_label="centre error")

        # how much training each point actually got
        steps = summarize(runs, "settings.n_train", STEPS)
        print("    optimizer steps per point:")
        for _, row in steps.iterrows():
            print(f"        n_train={int(row['settings.n_train']):>4}  "
                  f"{row['mean']:>10,.0f} steps  (spread ±{row['std']:,.0f})")
        budget = steps["mean"].max()
        short = steps[steps["mean"] < 0.7 * budget]
        for _, row in short.iterrows():
            print(f"    !! n_train={int(row['settings.n_train'])} got "
                  f"{row['mean']/budget:.0%} of the largest point's training — "
                  f"it is step-limited, say so in the caption")

        ax.errorbar(table["settings.n_train"], table["mean"], yerr=table["std"],
                    marker=arm["marker"], ms=9, lw=2.2, capsize=5,
                    color=arm["color"], label=arm["label"], zorder=5)

        # mark points that did not get the full step budget
        # for _, row in short.iterrows():
        #     n = row["settings.n_train"]
        #     y = float(table.loc[table["settings.n_train"] == n, "mean"].iloc[0])
            # ax.annotate(f"{row['mean']/budget:.0%} of\nstep budget",
            #             xy=(n, y), xytext=(34, -6), textcoords="offset points",
            #             ha="left", va="center", fontsize=8.5, color=RED,
            #             arrowprops=dict(arrowstyle="-", lw=1.0, color=RED))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    levels = sorted(on_dataset["settings.n_train"].dropna().unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(v)) for v in levels])
    ax.set_xlabel("training trajectories")
    ax.set_ylabel("rollout centre error  (block widths)")
    ax.set_title("Sample efficiency on the real cube tosses",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.text(0.70, 0.015,
            "mean ± s.d. over 3 runs, each on a different training subset",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=GRAY)

    save(fig, "fig4_sample_efficiency")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT)

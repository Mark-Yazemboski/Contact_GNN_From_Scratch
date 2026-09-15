"""
fig4_sample_efficiency.py   —  error vs number of training trajectories,
for the three metrics Allen et al. report: position, rotation, penetration.

EVERYTHING THAT DECIDES WHICH RUNS APPEAR IS IN THE `ARMS` LIST.
WHAT EACH PANEL PLOTS IS IN THE `PANELS` LIST.

Writes three separate files, one per metric:
    fig4_position.svg                 position error,  % of cube width
    fig4_rotation.svg                 rotation error,  degrees
    fig4_penetration.svg              penetration,     % of cube width

Set COMBINED = True to also get all three in one row.

Y axes are LINEAR and start at zero, with plain numbers rather than
scientific notation, so the size of a gap between arms reads directly off the
axis instead of having to be decoded from a log scale.

A sweep curve must be ONE dataset, so DATASET is pinned below.

    python fig4_sample_efficiency.py [path/to/master.csv]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from _common import (load, select, summarize, report, save, CSV_DEFAULT,
                     INK, GRAY, ORANGE, BLUE, RED)

DATASET = "tosses_processed"        # Allen et al.'s real cube tosses
STEPS = "metrics.total_optimizer_steps"
MARK_STEP_LIMITED = False           # annotate points that ran short of budget
COMBINED = False                    # also write the three-in-a-row version


# ----------------------------------------------------------------------
# !! CHECK THIS BEFORE COMPARING TO THE PAPER !!
#
# Allen et al. report position error and penetration as a percentage of the
# object WIDTH w:   (1/w)||c_hat - c||,  w = 0.1048 m for your cube.
# WIDTH_SCALE just converts your stored fraction into a percentage.
#
# If evaluate_metrics.py divides by BLOCK_HALF_WIDTH (0.0524) instead of the
# full width, every position and penetration number here is 2x too large
# relative to the paper and WIDTH_SCALE should be 50.0, not 100.0. Grep
# evaluate_metrics.py for what it divides by before you put a comparison on a
# poster - it is the difference between beating them by a little and by a lot.
# ----------------------------------------------------------------------
WIDTH_SCALE = 100.0


# ======================================================================
# THE ARMS.  Which runs appear.
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
    {
        "label": "acceleration GNS, single-step  (Allen et al. style)",
        "color": GRAY,
        "marker": "s",
        "criteria": {
            # NOTE: keys must carry the "settings." prefix or select() ignores
            # them and the arm silently matches far too much. "use_wind" alone
            # is not a column; "settings.use_wind" is.
            "settings.multistep":   1,
            "settings.use_wind":    False,
        },
    },
]


# ======================================================================
# THE PANELS.  What each subplot shows.
# ======================================================================

PANELS = [
    {
        "metric": "metrics.center_error",
        "scale":  WIDTH_SCALE,
        "title":  "Positional error",
        "ylabel": "position error  (% of cube width)",
        "log":    False,
    },
    {
        "metric": "metrics.angle_error_deg",
        "scale":  1.0,
        "title":  "Rotational error",
        "ylabel": "rotation error  (degrees)",
        "log":    False,
    },
    {
        "metric": "metrics.floor_penetration",
        "scale":  WIDTH_SCALE,
        "title":  "Penetration",
        "ylabel": "penetration  (% of cube width)",
        "log":    False,
    },
]


# ======================================================================

def draw_panel(ax, on_dataset, panel, show_legend=False):
    """Draw one metric against n_train for every arm."""
    print(f"\n{'=' * 66}\n{panel['title']}   [{panel['metric']}]\n{'=' * 66}")

    for arm in ARMS:
        runs = select(on_dataset, arm["criteria"], label=arm["label"])
        if runs.empty:
            print("    nothing matched — this arm is skipped")
            continue

        table = summarize(runs, "settings.n_train", panel["metric"])
        report(table, "settings.n_train", metric_label=panel["title"].lower())

        ax.errorbar(table["settings.n_train"],
                    table["mean"] * panel["scale"],
                    yerr=table["std"] * panel["scale"],
                    marker=arm["marker"], ms=8, lw=2.2, capsize=4,
                    color=arm["color"], label=arm["label"], zorder=5)

        if MARK_STEP_LIMITED and STEPS in runs.columns:
            steps = summarize(runs, "settings.n_train", STEPS)
            budget = steps["mean"].max()
            for _, row in steps[steps["mean"] < 0.7 * budget].iterrows():
                n = row["settings.n_train"]
                y = float(table.loc[table["settings.n_train"] == n,
                                    "mean"].iloc[0]) * panel["scale"]
                ax.annotate(f"{row['mean'] / budget:.0%} of budget",
                            xy=(n, y), xytext=(28, -4),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=8, color=RED,
                            arrowprops=dict(arrowstyle="-", lw=0.9, color=RED))

    ax.set_xscale("log", base=2)
    if panel["log"]:
        ax.set_yscale("log")
    else:
        # Linear, anchored at zero, plain numbers. A gap on this axis is the
        # gap - no mental arithmetic, no 2x10^1 to decode.
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    levels = sorted(on_dataset["settings.n_train"].dropna().unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(v)) for v in levels])
    ax.set_xlabel("training trajectories")
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"], fontweight="bold", loc="left")
    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=9)


def report_step_budget(on_dataset):
    """Print how much training each sweep point actually got, once."""
    if STEPS not in on_dataset.columns:
        return
    print(f"\n{'=' * 66}\noptimizer steps per point\n{'=' * 66}")
    for arm in ARMS:
        runs = select(on_dataset, arm["criteria"], label=arm["label"],
                      verbose=False)
        if runs.empty or runs[STEPS].isna().all():
            continue
        steps = summarize(runs, "settings.n_train", STEPS)
        budget = steps["mean"].max()
        print(f"\n  [{arm['label']}]")
        for _, row in steps.iterrows():
            frac = row["mean"] / budget if budget else float("nan")
            flag = "   <-- STEP-LIMITED, say so in the caption" if frac < 0.7 else ""
            print(f"    n_train={int(row['settings.n_train']):>4}  "
                  f"{row['mean']:>10,.0f} steps  ({frac:>4.0%} of the largest "
                  f"point){flag}")


def main(csv=CSV_DEFAULT):
    data = load(csv)
    on_dataset = data[data["dataset_name"] == DATASET]
    print(f"\n{len(on_dataset)} runs on dataset '{DATASET}' "
          f"({on_dataset['group'].nunique()} groups)")
    if on_dataset.empty:
        print("  nothing to plot")
        return

    # ---- one file per metric ----
    for panel, stem in zip(PANELS, ("fig4_position", "fig4_rotation",
                                    "fig4_penetration")):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        draw_panel(ax, on_dataset, panel, show_legend=True)
        ax.text(0.99, 0.015, "mean ± s.d. over 3 runs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=GRAY)
        fig.tight_layout()
        save(fig, stem)

    if COMBINED:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
        for i, (ax, panel) in enumerate(zip(axes, PANELS)):
            draw_panel(ax, on_dataset, panel, show_legend=(i == 0))
        fig.tight_layout()
        save(fig, "fig4_sample_efficiency")

    report_step_budget(on_dataset)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT)

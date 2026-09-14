"""
fig2_learn_k.py   —  is the drag coefficient k/m identifiable?

EVERYTHING THAT DECIDES WHICH RUNS APPEAR IS IN THE `SERIES` LIST BELOW.

FRAMING, because it matters for what you can claim: k is INITIALISED at the
calibrated value (k_init ~ 0.0285). A final value near 0.0285 is RETENTION, not
recovery from scratch. The result that holds is the contrast - with flow in the
data the coefficient stays put, without flow it collapses toward zero. The
figure draws the initialisation so that is visible instead of hidden.

Runs with learn_k = False are excluded: k never moves there, so they would draw
a flat line at the calibrated value and imply a recovery that did not happen.

    python fig2_learn_k.py [path/to/master.csv]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from _common import (load, select, summarize, report, save, CSV_DEFAULT,
                     INK, GRAY, BLUE, RED, PURPLE, K_TRUE)

VALUE = "metrics.recovered_k_over_m"
INIT = "metrics.k_init"

REAL_X = -6.0          # x position for the real-toss dataset, which has no wind


# ======================================================================
# THE SERIES.  This is the part you edit.
# ======================================================================

SERIES = [
    {
        "label": "learned $k/m$",
        "color": BLUE,
        "marker": "o",
        "criteria": {
            "settings.use_wind":           True,
            "settings.use_drag_baseline":  False,
            "settings.learn_k":            True,
            "settings.w_fluid_anchor":     0.2,
            "settings.n_train":            256,
        },
    },
]


# ======================================================================

def main(csv=CSV_DEFAULT):
    data = load(csv)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    # calibrated truth
    ax.axhline(K_TRUE, color=INK, lw=1.5, ls=(0, (6, 4)), zorder=2, label="true $k/m$")
    # ax.text(20, K_TRUE * 1.10, f"calibrated  $k/m$ = {K_TRUE}",
    #         ha="right", va="bottom", fontsize=10, color=INK)

    # where each point started, and an arrow to where it ended
            # ax.scatter(start["x"], start["mean"], marker="-", s=430, lw=2.0,
            #            color=INK, zorder=4,
            #            label="true $k/m$" if spec is SERIES[0] else None)

    all_x = set()

    for spec in SERIES:
        runs = select(data, spec["criteria"], label=spec["label"])
        if runs.empty:
            print("    nothing matched — this series is skipped")
            continue

        runs = runs.copy()
        runs["x"] = runs["wind_max"]

        final = summarize(runs, "x", VALUE)
        start = summarize(runs, "x", INIT)
        report(final, "x", metric_label="k/m learned")
        all_x.update(final["x"].tolist())

        
        for _, row in final.iterrows():
            s0 = start.loc[start["x"] == row["x"], "mean"]
            # if not s0.empty:
            #     ax.annotate("", xy=(row["x"], row["mean"]),
            #                 xytext=(row["x"], float(s0.iloc[0])),
            #                 arrowprops=dict(arrowstyle="-|>", lw=1.3,
            #                                 color=spec["color"], alpha=0.5),
            #                 zorder=3)

        # solid marker where wind is present in the data, hollow where it is not
        has_wind = final["x"] > 0
        for mask, face in ((has_wind, spec["color"]), (~has_wind, "white")):
            part = final[mask]
            if part.empty:
                continue
            ax.errorbar(part["x"], part["mean"], yerr=part["std"],
                        marker=spec["marker"], ms=11, lw=0, elinewidth=1.8,
                        capsize=5, color=spec["color"], mfc=face, mew=2.0,
                        zorder=6,
                        label=spec["label"] if mask is has_wind else None)

    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(-3e-5, 0.060)
    ticks = sorted(all_x)
    ax.set_xticks(ticks)
    # ax.set_xticklabels(["real\ntosses" if t < 0 else f"{t:.0f}" for t in ticks])
    ax.set_xlabel("wind range in the dataset  (m/s)")
    ax.set_ylabel("quadratic drag coefficient  $k/m$")
    ax.set_title("The drag coefficient is identifiable only when the data has wind",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08),
              fontsize=10)
    ax.text(0.99, 0.015,
            "mean ± s.d. over 3 runs  ·  hollow = no wind in the data  ",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=GRAY)

    save(fig, "fig2_learn_k")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT)

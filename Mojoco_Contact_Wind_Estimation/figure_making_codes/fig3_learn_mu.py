"""
fig3_learn_mu.py   —  the friction coefficient, on the same wind axis as
fig1 and fig2.

EVERYTHING THAT DECIDES WHICH RUNS APPEAR IS IN THE `SERIES` LIST BELOW.
Each entry names both the runs (criteria) and WHICH COLUMN to plot (metric),
so the two different mu quantities are separate rows you can comment out.

The two quantities, because they disagree:

  metrics.recovered_mu       the LEARNABLE PARAMETER inside the friction loss.
                             Starts at 0.30 and settles near 0.157 in almost
                             every run, well under the true 0.198. The fact
                             that it lands on the same wrong value regardless
                             of the data says the loss geometry is pinning it,
                             not the data identifying it.

  metrics.diag_mu_implied    the coefficient IMPLIED BY THE PREDICTED FORCES,
                             ||f_t|| / f_n at contact. What the model's forces
                             actually do. Tracks truth much more closely.

Report the implied value as the headline and the parameter as the caveat.
Reporting only the flattering one invites the question you don't want.

    python fig3_learn_mu.py [path/to/master.csv]
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from _common import (load, select, summarize, report, save, CSV_DEFAULT,
                     INK, GRAY, ORANGE, GREEN, RED, MU_TRUE)

REAL_X = -6.0

WIND_ON = {
    "settings.use_wind":           True,
    "settings.use_drag_baseline":  False,
    "settings.learn_mu":           True,
    "settings.w_fluid_anchor":     0.2,
    "settings.n_train":            256,
}

WIND_BLIND = {
    "settings.use_wind":           False,
    "settings.use_drag_baseline":  False,
    "settings.learn_mu":           True,
    "settings.w_fluid_anchor":     0.0,
    "settings.n_train":            256,
}


# ======================================================================
# THE SERIES.  This is the part you edit.
# ======================================================================

SERIES = [
    {
        "label": r"implied by the forces,  wind given",
        "metric": "metrics.diag_mu_implied",
        "criteria": WIND_ON,
        "color": GREEN, "marker": "o", "linestyle": "-",
    },
    {
        "label": r"implied by the forces,  wind-blind",
        "metric": "metrics.diag_mu_implied",
        "criteria": WIND_BLIND,
        "color": RED, "marker": "o", "linestyle": "-",
    },
    {
        "label": r"loss parameter,  wind given",
        "metric": "metrics.recovered_mu",
        "criteria": WIND_ON,
        "color": GRAY, "marker": "s", "linestyle": "--",
    },
    {
        "label": r"loss parameter,  wind-blind",
        "metric": "metrics.recovered_mu",
        "criteria": WIND_BLIND,
        "color": "#c7a9a9", "marker": "s", "linestyle": "--",
    },
]


# ======================================================================

def main(csv=CSV_DEFAULT):
    data = load(csv)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.axhline(MU_TRUE, color=INK, lw=1.5, ls=(0, (6, 4)), zorder=2)
    ax.text(-6.0, MU_TRUE + 0.003, f"true  $\\mu$ = {MU_TRUE}",
            ha="left", va="bottom", fontsize=10.5, color=INK)

    all_x, init_values = set(), []

    for spec in SERIES:
        runs = select(data, spec["criteria"], label=spec["label"])
        if runs.empty:
            print("    nothing matched — this series is skipped")
            continue

        runs = runs.copy()
        runs["x"] = runs["wind_max"].fillna(REAL_X)

        table = summarize(runs, "x", spec["metric"])
        report(table, "x", metric_label=spec["metric"].split(".")[-1])
        all_x.update(table["x"].tolist())
        init_values.extend(runs["metrics.mu_init"].dropna().tolist())

        ax.errorbar(table["x"], table["mean"], yerr=table["std"],
                    marker=spec["marker"], ms=9, lw=2.0, capsize=5,
                    color=spec["color"], linestyle=spec["linestyle"],
                    label=spec["label"], zorder=5)

    if init_values:
        init = float(np.mean(init_values))
        ax.axhline(init, color="#d5d5d5", lw=1.2, ls=(0, (2, 3)), zorder=1)
        ax.text(max(all_x), init - 0.01, f"initialised at {init:.2f}",
                ha="right", va="bottom", fontsize=9.5, color="#9a9a9a")

    ticks = sorted(all_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["real\ntosses" if t < 0 else f"{t:.0f}" for t in ticks])
    ax.set_xlabel("wind range in the dataset  (m/s)")
    ax.set_ylabel(r"friction coefficient  $\mu$")
    ax.set_title("Friction: the forces get close, the loss parameter does not",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="center left",
              bbox_to_anchor=(0.02, 0.60))
    ax.text(0.99, 0.015, "mean ± s.d. over 3 runs  ·  drag baseline off",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=GRAY)

    save(fig, "fig3_learn_mu")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT)

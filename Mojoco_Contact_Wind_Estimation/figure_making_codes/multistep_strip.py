"""
multistep_strip.py

The multistep-unrolling panel, drawn as a sliding window rather than a row of
identical boxes.

Each unroll step consumes a window of W = h + 1 positions (h finite-difference
velocities need h+1 frames) and emits one new frame. The window slides right,
so it fills progressively with the model's OWN output - by the last step almost
nothing in the input is ground truth. That staircase is the argument for
multistep training, and a chain of identical GNS boxes hides it completely.

    python multistep_strip.py                 # M = 4, h = 3
    python multistep_strip.py -M 8            # eight unroll steps
    python multistep_strip.py --compact       # first + last row only, thinner band
    python multistep_strip.py -M 4 -o my_name

Cell styling is the whole colour scheme: filled gray = ground truth,
orange outline = a frame the model produced.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


# ======================================================================
# CONFIG
# ======================================================================

OUTDIR = "figure_assets"
ALSO_PNG = True

OLD = "#8a8a8a"          # inherited / ground truth
INK = "#1a1a1a"
NEW = "#d95f02"          # this work / model-generated
TRUE_FC = "#dcdcdc"      # ground-truth cell fill
PRED_FC = "#fdf1e7"      # predicted cell fill

CW, CH = 2.30, 0.78      # cell width, height
GAPX, GAPY = 1.05, 1.00  # gap between columns, between rows

FS = {"title": 16, "head": 12, "body": 11, "small": 10, "tiny": 8.5}

UNROLL_SYM = "M"         # NOT K - the processor already uses L x K


# ======================================================================
# primitives
# ======================================================================

def rbox(ax, x, y, w, h, ec, fc="none", lw=1.5, ls="-", r=0.16, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, size="body", color=INK, ha="center", va="center",
        weight=None, style=None, z=8):
    ax.text(x, y, s, fontsize=FS[size], color=color, ha=ha, va=va,
            weight=weight, style=style, zorder=z)


def arc_arrow(ax, p0, p1, color, rad=-0.45, lw=1.6, ms=11, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


# ======================================================================

def build(M=4, h=3, compact=False, name=None):
    W = h + 1                       # frames consumed per step
    n_cols = W + M                  # offsets -(W-1) ... +M
    offsets = list(range(-(W - 1), M + 1))

    rows = [1, M] if (compact and M > 2) else list(range(1, M + 1))
    gap_row = compact and M > 2

    pitch_x = CW + GAPX
    pitch_y = CH + GAPY

    def col_x(o):                   # left edge of the cell for offset o
        return 1.7 + (o + W - 1) * pitch_x

    def row_y(i):                   # bottom edge of row i (0 = topmost drawn)
        return top_row_y - i * pitch_y

    n_drawn = len(rows) + (1 if gap_row else 0)
    head_h, foot_h = 6.6, 4.3
    top_row_y = foot_h + (n_drawn - 1) * pitch_y
    W_units = col_x(M) + CW + 7.4
    H_units = top_row_y + CH + head_h

    fig = plt.figure(figsize=(W_units * 0.30, H_units * 0.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_units); ax.set_ylim(0, H_units)
    ax.set_aspect("equal"); ax.axis("off")

    # ---------------- header ----------------
    txt(ax, W_units / 2, H_units - 1.1,
        f"Multistep unrolling  (${UNROLL_SYM} = {M}$)", "title", INK,
        weight="bold")
    txt(ax, W_units / 2, H_units - 2.6,
        "each step runs on the previous step's output", "small", OLD)

    y_hdr = top_row_y + CH + 2.25
    for o in offsets:
        lab = "$t$" if o == 0 else (f"$t{o}$" if o < 0 else f"$t+{o}$")
        txt(ax, col_x(o) + CW / 2, y_hdr, lab, "small",
            INK if o <= 0 else NEW)

    # the given / rolled-out divider
    x_div = col_x(1) - GAPX / 2
    ax.plot([x_div, x_div], [foot_h - 1.5, y_hdr + 0.60], color="#c4c4c4",
            lw=1.2, ls=(0, (5, 4)), zorder=1)
    txt(ax, x_div - 0.35, y_hdr + 1.15, "given", "small", OLD, ha="right")
    txt(ax, x_div + 0.35, y_hdr + 1.15, "rolled out", "small", NEW, ha="left")

    # ---------------- rows ----------------
    drawn = 0
    for m in rows:
        if gap_row and drawn == 1:                       # vertical ellipsis row
            yv = row_y(drawn) + CH / 2
            for k in range(3):
                ax.add_patch(Circle((W_units / 2 - 2.0, yv + 0.45 - k * 0.45),
                                    0.075, fc="#bdbdbd", ec="none", zorder=5))
            drawn += 1

        y = row_y(drawn)
        win = list(range(m - W, m))                      # input offsets
        n_pred = sum(1 for o in win if o >= 1)

        for o in offsets:
            pred = o >= 1
            inside = o in win
            is_out = (o == m)
            if not inside and not is_out:
                rbox(ax, col_x(o), y, CW, CH, "#e8e8e8", fc="none", lw=0.9,
                     ls=(0, (2, 2.5)), z=1)
                continue
            rbox(ax, col_x(o), y, CW, CH,
                 NEW if pred else OLD,
                 fc=PRED_FC if pred else TRUE_FC,
                 lw=2.0 if is_out else 1.5, z=3)
            if is_out:
                txt(ax, col_x(o) + CW / 2, y + CH / 2,
                    rf"$\hat{{\mathbf{{x}}}}_{{t+{o}}}$", "small", NEW, z=9)

        # window bracket
        xa, xb = col_x(win[0]) - 0.20, col_x(win[-1]) + CW + 0.20
        rbox(ax, xa, y - 0.22, xb - xa, CH + 0.44, INK, fc="none", lw=1.3,
             ls=(0, (4, 3)), r=0.25, z=4)

        # the GNS hop, arcing over into the new frame
        arc_arrow(ax, (xb - 0.1, y + CH + 0.24),
                  (col_x(m) + CW / 2, y + CH + 0.24), NEW)
        txt(ax, (xb + col_x(m) + CW / 2) / 2, y + CH + 1.02, "GNS", "tiny",
            NEW, weight="bold")

        txt(ax, 1.15, y + CH / 2, f"${UNROLL_SYM.lower()}={m}$", "small", INK,
            ha="right")
        txt(ax, col_x(M) + CW + 0.8, y + CH / 2,
            f"{n_pred} of {W} predicted", "small",
            NEW if n_pred else OLD, ha="left")
        drawn += 1

    # ---------------- loss bracket ----------------
    bl, br = col_x(1) - 0.1, col_x(M) + CW + 0.1
    yb = foot_h - 1.05
    ax.plot([bl, br], [yb, yb], color=NEW, lw=1.8, zorder=5)
    for xx in (bl, br):
        ax.plot([xx, xx], [yb, yb + 0.42], color=NEW, lw=1.8, zorder=5)
    txt(ax, (bl + br) / 2, yb - 0.75,
        f"loss at every step — gradients flow through all ${UNROLL_SYM}$ steps "
        "and through the integrator", "small", NEW)

    # ---------------- legend ----------------
    lx, ly = 1.7, 1.05
    rbox(ax, lx, ly, 1.15, 0.55, OLD, fc=TRUE_FC, lw=1.4, r=0.12)
    txt(ax, lx + 1.45, ly + 0.28, "ground truth", "small", INK, ha="left")
    lx2 = lx + 5.4
    rbox(ax, lx2, ly, 1.15, 0.55, NEW, fc=PRED_FC, lw=1.4, r=0.12)
    txt(ax, lx2 + 1.45, ly + 0.28, "model's own prediction", "small", INK,
        ha="left")
    txt(ax, W_units - 0.8, ly + 0.28,
        f"window $W=h+1={W}$ frames", "small", OLD, ha="right")

    os.makedirs(OUTDIR, exist_ok=True)
    stem = name or ("multistep_strip_compact" if compact else "multistep_strip")
    p = os.path.join(OUTDIR, stem + ".svg")
    fig.savefig(p, transparent=True, bbox_inches="tight", pad_inches=0.06)
    if ALSO_PNG:
        fig.savefig(os.path.join(OUTDIR, stem + ".png"), transparent=True,
                    dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {p}   ({W_units:.1f} x {H_units:.1f} units, "
          f"aspect {W_units / H_units:.1f}:1)")


def _win_icon(ax, x, y, n_pred, W, cw=0.46, ch=0.52, g=0.11):
    """Tiny 4-cell window icon; the first n_pred cells are model-generated."""
    for k in range(W):
        pred = k >= (W - n_pred)
        rbox(ax, x + k * (cw + g), y, cw, ch,
             NEW if pred else OLD, fc=PRED_FC if pred else TRUE_FC,
             lw=1.0, r=0.07, z=3)
    return x + W * cw + (W - 1) * g


def build_compare(M=4, h=3, name=None):
    """Single-step vs multistep, with the gradient path drawn. This is the
    panel that actually says what multistep IS - the window version shows a
    rollout, which is what inference looks like too."""
    W = h + 1
    GW, GH = 2.5, 1.45          # GNS box
    XW = 1.75                   # x-hat cell
    A = 0.75                    # arrow run
    win_w = W * 0.46 + (W - 1) * 0.11
    pitch = win_w + A + GW + A + XW + 0.85

    x0 = 7.4
    yB = 6.6                    # multistep row baseline
    yA = 14.3                   # single-step row baseline
    W_units = x0 + M * pitch + 4.2
    H_units = yA + GH + 5.0

    fig = plt.figure(figsize=(W_units * 0.255, H_units * 0.255))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_units); ax.set_ylim(0, H_units)
    ax.set_aspect("equal"); ax.axis("off")

    

    def step(x, y, m, color, n_pred, show_loss):
        xe = _win_icon(ax, x, y + GH / 2 - 0.26, n_pred, W)
        arw = FancyArrowPatch((xe + 0.12, y + GH / 2), (xe + A, y + GH / 2),
                              arrowstyle="-|>", mutation_scale=11, lw=1.4,
                              color=OLD, zorder=5, shrinkA=0, shrinkB=0)
        ax.add_patch(arw)
        gx = xe + A + 0.15
        rbox(ax, gx, y, GW, GH, color, fc="white", lw=1.9, r=0.22, z=4)
        txt(ax, gx + GW / 2, y + GH / 2 + 0.22, "GNS", "head", color,
            weight="bold")
        txt(ax, gx + GW / 2, y + GH / 2 - 0.38, r"$\theta$", "small", color)
        ax.add_patch(FancyArrowPatch((gx + GW + 0.1, y + GH / 2),
                                     (gx + GW + A + 0.05, y + GH / 2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.4, color=OLD, zorder=5,
                                     shrinkA=0, shrinkB=0))
        xx = gx + GW + A + 0.2
        rbox(ax, xx, y + GH / 2 - 0.42, XW, 0.84, NEW, fc=PRED_FC, lw=1.6,
             r=0.14, z=4)
        txt(ax, xx + XW / 2, y + GH / 2,
            rf"$\hat{{\mathbf{{x}}}}_{{t+{m}}}$", "small", NEW, z=9)
        if show_loss:
            ly_ = y - 1.85
            rbox(ax, xx - 0.15, ly_, XW + 0.3, 0.86, NEW, fc="white", lw=1.5,
                 r=0.14, z=4)
            txt(ax, xx + XW / 2, ly_ + 0.43, rf"$\mathcal{{L}}_{{{m}}}$",
                "small", NEW, z=9)
            ax.add_patch(FancyArrowPatch((xx + XW / 2, y + GH / 2 - 0.46),
                                         (xx + XW / 2, ly_ + 0.92),
                                         arrowstyle="-|>", mutation_scale=9,
                                         lw=1.1, color=OLD, zorder=5,
                                         shrinkA=0, shrinkB=0))
        return gx, xx

    # ---------------- A: single-step ----------------
    txt(ax, x0 - 0.7, yA + GH / 2 + 0.45, "single-step", "head", OLD,
        ha="right", weight="bold")
    txt(ax, x0 - 0.7, yA + GH / 2 - 0.45, "Allen et al.", "small", OLD,
        ha="right")
    gxA, xxA = step(x0, yA, 1, OLD, 0, True)
    ax.add_patch(FancyArrowPatch((xxA - 0.25, yA - 1.42),
                                 (gxA + GW / 2, yA - 0.12),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.5,
                                 color="#c0392b", zorder=6, ls=(0, (5, 2.5)),
                                 connectionstyle="arc3,rad=0.35",
                                 shrinkA=0, shrinkB=0))
    txt(ax, x0 + pitch + 0.6, yA + GH / 2 + 0.45,
        "gradient reaches one GNS call and stops", "small", "#c0392b",
        ha="left")
    txt(ax, x0 + pitch + 0.6, yA + GH / 2 - 0.55,
        "every training input is clean ground truth", "small", OLD, ha="left")

    # ---------------- B: multistep ----------------
    txt(ax, x0 - 0.7, yB + GH / 2 + 0.45, "multistep", "head", NEW,
        ha="right", weight="bold")
    txt(ax, x0 - 0.7, yB + GH / 2 - 0.45, f"ours, ${UNROLL_SYM}={M}$", "small",
        NEW, ha="right")

    gxs, prev_xx = [], None
    for m in range(1, M + 1):
        gx, xx = step(x0 + (m - 1) * pitch, yB, m, NEW, min(m - 1, W), True)
        if prev_xx is not None:
            ax.add_patch(FancyArrowPatch(
                (prev_xx + XW + 0.12, yB + GH / 2),
                (x0 + (m - 1) * pitch - 0.08, yB + GH / 2),
                arrowstyle="-|>", mutation_scale=11, lw=1.3, color=NEW,
                ls=(0, (3, 2.2)), zorder=5, shrinkA=0, shrinkB=0))
        gxs.append(gx + GW / 2)
        prev_xx = xx

    # shared weights
    ty = yB + GH + 1.55
    ax.plot([gxs[0], gxs[-1]], [ty, ty], color=NEW, lw=1.3, zorder=4)
    for gx in gxs:
        ax.plot([gx, gx], [ty, yB + GH + 0.1], color=NEW, lw=1.1,
                ls=(0, (2.5, 2.5)), zorder=4)
    txt(ax, (gxs[0] + gxs[-1]) / 2, ty + 0.62,
        rf"one set of weights $\theta$, applied ${UNROLL_SYM}$ times",
        "small", NEW)

    # gradient lane
    gy = yB - 3.3
    ax.add_patch(FancyArrowPatch((gxs[-1] + 3, gy), (gxs[0] - 0.9, gy),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.7,
                                 color="#c0392b", zorder=6, ls=(0, (10, 2.5)),
                                 shrinkA=0, shrinkB=0))
    for gx in gxs:
        ax.plot([gx+3.05, gx+3.05], [gy, yB - 1.9], color="#c0392b", lw=1.1,
                ls=(0, (2.5, 2.5)), zorder=5)
    txt(ax, (gxs[0] + gxs[-1]) / 2, gy - 0.85,
        r"$\partial\mathcal{L}/\partial\theta$ accumulates through every "
        "step — and through the Newton–Euler integrator", "small", "#c0392b")

    # legend
    _win_icon(ax, 3.4, 1.15, 0, W)
    txt(ax, 3.4 + win_w + 0.45, 1.41, "ground-truth frames", "small", INK,
        ha="left")
    _win_icon(ax, 12.0, 1.15, W, W)
    txt(ax, 12.0 + win_w + 0.45, 1.41, "model-generated frames", "small", INK,
        ha="left")

    os.makedirs(OUTDIR, exist_ok=True)
    stem = name or "multistep_compare"
    pth = os.path.join(OUTDIR, stem + ".svg")
    fig.savefig(pth, transparent=True, bbox_inches="tight", pad_inches=0.06)
    if ALSO_PNG:
        fig.savefig(os.path.join(OUTDIR, stem + ".png"), transparent=True,
                    dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {pth}   (aspect {W_units / H_units:.1f}:1)")


def main():
    # ap = argparse.ArgumentParser(description="Multistep sliding-window strip.")
    # ap.add_argument("-M", type=int, default=4, help="unroll steps (default 4)")
    # ap.add_argument("-H", "--history", type=int, default=3,
    #                 help="h, velocity history length (default 3) -> window h+1")
    # ap.add_argument("--compact", action="store_true",
    #                 help="first and last row only, with an ellipsis between")
    # ap.add_argument("--style", choices=("window", "compare"), default="window",
    #                 help="window = sliding-window rollout; "
    #                      "compare = single-step vs multistep with the gradient path")
    # ap.add_argument("-o", "--out", default=None)
    # a = ap.parse_args()
    # if a.style == "compare":
    build_compare()
    # else:
    # build()


if __name__ == "__main__":
    main()

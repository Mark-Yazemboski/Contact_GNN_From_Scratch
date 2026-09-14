"""
make_architecture_figure.py

Full mock-up of the poster architecture figure, as one editable SVG.

Layout:
    row 1   state -> graph | features | encode-process-decode | rigid-body layer
    row 2   supervision: what the loss sees, and what it never sees
    row 3   multistep unrolling strip

Colour rule: GRAY = inherited from Allen et al. (2022).  ORANGE = this work
(contact side).  BLUE = this work (fluid side).  GREEN = the wall-distance
feature, whose clamp you changed.

Needs make_cube_assets.py next to it - the two 3D panels are drawn by the same
code that makes the standalone assets, so they can never drift apart.

    python make_architecture_figure.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_making_codes.make_cube_assets import (make_mesh_cube, make_contact_cube, C, OUT_DIR)
import os


# ======================================================================
# CONFIG
# ======================================================================

FIGSIZE = (19.0, 10.6)       # build at poster scale, then place as one object
NEW = "#d95f02"              # "this work"
OLD = "#8a8a8a"              # "inherited"
CONTACT = "#d95f02"
FLUID = "#0072b2"
FEAT = "#009e73"
INK = "#1a1a1a"

FS = {"block": 16, "head": 13, "body": 11.5, "small": 10, "tiny": 9}


# ======================================================================
# primitives
# ======================================================================

def box(ax, x, y, w, h, ec, lw=1.8, ls="-", fc="none", r=0.010, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls, zorder=z,
                                mutation_aspect=FIGSIZE[0] / FIGSIZE[1]))


def txt(ax, x, y, s, size="body", color=INK, ha="left", va="center", weight=None):
    ax.text(x, y, s, fontsize=FS[size], color=color, ha=ha, va=va,
            weight=weight, zorder=5)


def arw(ax, p0, p1, color=INK, lw=2.0, ls="-", ms=16, z=4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, ls=ls, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def inset3d(fig, rect):
    a = fig.add_axes(rect, projection="3d")
    a.patch.set_alpha(0.0)
    return a


# ======================================================================
# the figure
# ======================================================================

def build():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    TOP_Y, TOP_H = 0.415, 0.545          # main row
    yc = TOP_Y + TOP_H / 2

    # ------------------------------------------------------------------
    # BLOCK 1 - state -> graph
    # ------------------------------------------------------------------
    x1, w1 = 0.015, 0.185
    box(ax, x1, TOP_Y, w1, TOP_H, OLD, lw=1.6)
    txt(ax, x1 + w1 / 2, TOP_Y + TOP_H + 0.022, "1.  State " + r"$\rightarrow$" + " graph",
        "block", INK, ha="center", weight="bold")
    txt(ax, x1 + w1 / 2, TOP_Y + TOP_H - 0.030,
        r"COM position + quaternion $\rightarrow$ mesh", "small", OLD, ha="center")
    a = inset3d(fig, [x1 + 0.004, TOP_Y + 0.045, w1 - 0.008, TOP_H - 0.100])
    make_mesh_cube()
    txt(ax, x1 + w1 / 2, TOP_Y + 0.028, r"$N=8$ nodes,  $E=24$ directed edges",
        "small", OLD, ha="center")

    arw(ax, (x1 + w1 + 0.004, yc), (0.222, yc))

    # ------------------------------------------------------------------
    # BLOCK 2 - features
    # ------------------------------------------------------------------
    x2, w2 = 0.226, 0.175
    txt(ax, x2 + w2 / 2, TOP_Y + TOP_H + 0.022, "2.  Features", "block", INK,
        ha="center", weight="bold")

    nh = 0.305
    ny = TOP_Y + TOP_H - nh
    box(ax, x2, ny, w2, nh, NEW, lw=2.0)
    txt(ax, x2 + 0.010, ny + nh - 0.030, "node", "head", INK, weight="bold")
    rows_n = [(r"$\mathbf{v}^{\rm FD}_i$  velocity history", "3h = 9", OLD),
              (r"$\mathbf{u}=\mathbf{w}\Delta t-\mathbf{v}_{\rm curr}$", "3", NEW),
              (r"$\|\mathbf{u}\|$", "1", NEW),
              (r"$b_i$  wall distance", "1", FEAT)]
    yy = ny + nh - 0.072
    for s, n, c in rows_n:
        txt(ax, x2 + 0.012, yy, s, "body", c)
        txt(ax, x2 + w2 - 0.012, yy, n, "small", c, ha="right")
        yy -= 0.045
    ax.plot([x2 + 0.012, x2 + w2 - 0.012], [yy + 0.020, yy + 0.020],
            color="#cccccc", lw=1.0, zorder=3)
    txt(ax, x2 + 0.012, yy - 0.010, "14 channels", "small", INK, weight="bold")

    eh = 0.195
    ey = TOP_Y + 0.010
    box(ax, x2, ey, w2, eh, OLD, lw=1.6)
    txt(ax, x2 + 0.010, ey + eh - 0.030, "edge", "head", INK, weight="bold")
    txt(ax, x2 + 0.012, ey + eh - 0.078,
        r"$\mathbf{d}_{ij},\ \|\mathbf{d}_{ij}\|,\ \mathbf{d}^U_{ij},\ \|\mathbf{d}^U_{ij}\|$",
        "body", OLD)
    txt(ax, x2 + 0.012, ey + eh - 0.125, "8 channels  —  unchanged", "small", OLD)

    arw(ax, (x2 + w2 + 0.004, yc), (0.423, yc))

    # ------------------------------------------------------------------
    # BLOCK 3 - encode / process / decode
    # ------------------------------------------------------------------
    x3, w3 = 0.427, 0.235
    txt(ax, x3 + w3 / 2, TOP_Y + TOP_H + 0.022, "3.  Encode – Process – Decode",
        "block", INK, ha="center", weight="bold")

    eb_y, eb_h, eb_w = TOP_Y + TOP_H - 0.115, 0.075, 0.105
    box(ax, x3, eb_y, eb_w, eb_h, OLD, lw=1.6, fc="#f4f4f4")
    txt(ax, x3 + eb_w / 2, eb_y + eb_h / 2, "encoder", "head", OLD, ha="center")
    box(ax, x3 + 0.125, eb_y, eb_w + 0.005, eb_h, OLD, lw=1.6, fc="#f4f4f4")
    txt(ax, x3 + 0.125 + eb_w / 2, eb_y + eb_h / 2 + 0.014, "processor", "head",
        OLD, ha="center")
    txt(ax, x3 + 0.125 + eb_w / 2, eb_y + eb_h / 2 - 0.020,
        r"$L\times K$ message passing", "tiny", OLD, ha="center")
    arw(ax, (x3 + eb_w, eb_y + eb_h / 2), (x3 + 0.123, eb_y + eb_h / 2), OLD, lw=1.6, ms=12)

    dy, dh = TOP_Y + 0.025, 0.245
    box(ax, x3, dy, w3, dh, NEW, lw=2.0)
    txt(ax, x3 + 0.010, dy + dh - 0.028, "decoder  —  two heads", "head", INK,
        weight="bold")
    arw(ax, (x3 + w3 / 2, eb_y - 0.004), (x3 + w3 / 2, dy + dh + 0.004), OLD, lw=1.6, ms=12)

    hw = (w3 - 0.032) / 2
    box(ax, x3 + 0.010, dy + 0.016, hw, dh - 0.075, CONTACT, lw=1.7)
    txt(ax, x3 + 0.010 + hw / 2, dy + dh - 0.085, "contact", "head", CONTACT, ha="center",
        weight="bold")
    txt(ax, x3 + 0.010 + hw / 2, dy + dh - 0.125, r"$f_i$  per node", "small",
        CONTACT, ha="center")
    txt(ax, x3 + 0.010 + hw / 2, dy + dh - 0.160, "tangential + softplus normal",
        "tiny", CONTACT, ha="center")
    txt(ax, x3 + 0.010 + hw / 2, dy + dh - 0.190, r"$\times$ contact gate", "tiny",
        CONTACT, ha="center")

    box(ax, x3 + 0.022 + hw, dy + 0.016, hw, dh - 0.075, FLUID, lw=1.7)
    txt(ax, x3 + 0.022 + 1.5 * hw, dy + dh - 0.085, "fluid", "head", FLUID, ha="center",
        weight="bold")
    txt(ax, x3 + 0.022 + 1.5 * hw, dy + dh - 0.125,
        r"$\mathbf{F}_{\rm fluid},\ \boldsymbol{\tau}_{\rm fluid}$", "small",
        FLUID, ha="center")
    txt(ax, x3 + 0.022 + 1.5 * hw, dy + dh - 0.160, "one per body", "tiny",
        FLUID, ha="center")
    txt(ax, x3 + 0.022 + 1.5 * hw, dy + dh - 0.190, "from pooled latents", "tiny",
        FLUID, ha="center")

    arw(ax, (x3 + w3 + 0.004, yc), (0.684, yc))

    # ------------------------------------------------------------------
    # BLOCK 4 - rigid-body layer
    # ------------------------------------------------------------------
    x4, w4 = 0.688, 0.298
    box(ax, x4, TOP_Y, w4, TOP_H, INK, lw=2.0, ls=(0, (5, 4)))
    txt(ax, x4 + w4 / 2, TOP_Y + TOP_H + 0.022, "4.  Rigid-body layer", "block",
        INK, ha="center", weight="bold")
    txt(ax, x4 + w4 / 2, TOP_Y + TOP_H - 0.030,
        r"exact — fixed: $m$, $I$, $\mathbf{g}$, integrator", "small", INK,
        ha="center")

    a2 = inset3d(fig, [x4 + 0.006, TOP_Y + 0.115, w4 - 0.012, TOP_H - 0.160])
    make_contact_cube()

    txt(ax, x4 + 0.014, TOP_Y + 0.088,
        r"$\mathbf{a}_{\rm com}=\sum_i f_i+\mathbf{g}\Delta t^2+\mathbf{F}_{\rm fluid}$",
        "body", INK)
    txt(ax, x4 + 0.014, TOP_Y + 0.043,
        r"$\boldsymbol{\alpha}=\sum_i \mathbf{r}_i\times f_i\,/\,(I/m)+\boldsymbol{\tau}_{\rm fluid}$",
        "body", INK)

    # recovered constants - a result, not a caveat
    rw, rh = 0.205, 0.072
    rx = x4 + w4 - rw
    ry = TOP_Y - rh - 0.020
    box(ax, rx, ry, rw, rh, NEW, lw=2.0, fc="#fdf3ec")
    txt(ax, rx + rw / 2, ry + rh - 0.024, "recovered from data — never given",
        "small", NEW, ha="center", weight="bold")
    txt(ax, rx + rw / 2, ry + 0.022,
        r"$\mu = 0.1999 \pm 0.0052$   ·   $k/m = 0.02799 \pm 0.00091$",
        "small", INK, ha="center")

    # ------------------------------------------------------------------
    # supervision row
    # ------------------------------------------------------------------
    ly, lh, lw_ = 0.243, 0.100, 0.150
    lx = 0.427
    box(ax, lx, ly, lw_, lh, INK, lw=1.8, fc="#f7f7f7")
    txt(ax, lx + lw_ / 2, ly + lh - 0.028, "loss", "head", INK, ha="center",
        weight="bold")
    txt(ax, lx + lw_ / 2, ly + lh - 0.060, "acceleration MSE", "small", OLD, ha="center")
    txt(ax, lx + lw_ / 2, ly + lh - 0.085, "+ physics residuals", "small", NEW, ha="center")

    # from the integrator output
    arw(ax, (x4 + 0.02, TOP_Y - 0.004), (lx + lw_ + 0.004, ly + lh * 0.62), OLD,
        lw=1.7, ms=13, style="-|>")
    # from the force heads
    arw(ax, (x3 + w3 * 0.5, dy - 0.004), (lx + lw_ * 0.5, ly + lh + 0.004), NEW,
        lw=1.7, ms=13)

    # MuJoCo labels -> evaluation only
    ex, ew = 0.175, 0.180
    box(ax, ex, ly, ew, lh, OLD, lw=1.6, ls=(0, (4, 3)))
    txt(ax, ex + ew / 2, ly + lh - 0.030, "MuJoCo wrench labels", "head", OLD,
        ha="center")
    txt(ax, ex + ew / 2, ly + lh - 0.065, "evaluation only", "small", OLD, ha="center")
    arw(ax, (ex + ew + 0.004, ly + lh / 2), (lx - 0.004, ly + lh / 2), OLD,
        lw=1.6, ls=(0, (3, 2.5)), ms=13)
    # the bar: these never reach the loss
    xb = (ex + ew + lx) / 2
    ax.plot([xb - 0.009, xb + 0.009], [ly + lh / 2 - 0.021, ly + lh / 2 + 0.021],
            color="#c0392b", lw=2.6, zorder=6)
    ax.plot([xb - 0.009, xb + 0.009], [ly + lh / 2 + 0.021, ly + lh / 2 - 0.021],
            color="#c0392b", lw=2.6, zorder=6)
    txt(ax, xb, ly - 0.020, "no measured force enters training", "small", "#c0392b",
        ha="center")

    # ------------------------------------------------------------------
    # multistep strip
    # ------------------------------------------------------------------
    sy, sh = 0.104, 0.076
    sx0, K = 0.105, 4
    gw, gap = 0.074, 0.148
    txt(ax, 0.015, sy + sh + 0.052, "Multistep unrolling  (K = 4)", "block", INK,
        weight="bold")
    txt(ax, 0.015, sy + sh + 0.020,
        "trained on its own predictions, not only on clean inputs", "small", OLD)

    txt(ax, sx0 - 0.048, sy + sh / 2, r"$\mathbf{x}_t$", "body", INK, ha="center")
    txt(ax, sx0 - 0.020, sy - 0.014, "ground truth + noise", "tiny", INK,
        ha="center", va="top")

    for k in range(K):
        bx = sx0 + k * gap
        box(ax, bx, sy, gw, sh, OLD, lw=1.6, fc="#f4f4f4")
        txt(ax, bx + gw / 2, sy + sh / 2, "GNS", "head", OLD, ha="center")
        solid = (k == 0)
        arw(ax, (bx - 0.030, sy + sh / 2), (bx - 0.004, sy + sh / 2), INK if solid else OLD,
            lw=2.0 if solid else 1.5, ls="-" if solid else (0, (3, 2.5)), ms=13)
        out_x = bx + gw + 0.004
        txt(ax, out_x + 0.020, sy + sh / 2 + 0.020,
            rf"$\hat{{\mathbf{{x}}}}_{{t+{k+1}}}$", "small", INK, ha="center")

    txt(ax, sx0 + gap - 0.017, sy - 0.014, "model's own prediction", "tiny", OLD,
        ha="center", va="top")

    # loss bracket under every step
    bl, br_ = sx0 + gw + 0.004, sx0 + (K - 1) * gap + gw + 0.040
    ax.plot([bl, br_], [sy - 0.044, sy - 0.044], color=NEW, lw=1.8, zorder=4)
    for xx in (bl, br_):
        ax.plot([xx, xx], [sy - 0.044, sy - 0.034], color=NEW, lw=1.8, zorder=4)
    txt(ax, (bl + br_) / 2, sy - 0.060, "loss at every step — gradients flow through all K",
        "small", NEW, ha="center", va="top")

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------
    lgx, lgy = 0.700, 0.075
    for i, (c, s) in enumerate([(OLD, "inherited from Allen et al. (2022)"),
                                (NEW, "this work")]):
        ax.plot([lgx, lgx + 0.022], [lgy - i * 0.032, lgy - i * 0.032], color=c, lw=4)
        txt(ax, lgx + 0.030, lgy - i * 0.032, s, "small", INK)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"architecture_figure.{ext}"),
                    transparent=True, dpi=190)
        print(f"  wrote {OUT_DIR}/architecture_figure.{ext}")
    plt.close(fig)


def dot(ax, x, y, r=0.0075, color="#5a5a5a", z=6):
    """Circle that stays circular despite the non-square figure."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y), width=2 * r * FIGSIZE[1] / FIGSIZE[0],
                         height=2 * r, ec="none", fc=color, zorder=z))

def draw_mlp(ax, x, y, w, h, dims, tag, color=OLD, r=0.0040):
    """Small not-to-scale MLP sketch: one column of dots per layer, fully
    connected, with the true channel count printed under each column and a
    vertical ellipsis marking the columns that are truncated."""
    shown = [4, 5, 5][:len(dims)]
    xs = [x + w * (i + 0.5) / len(dims) for i in range(len(dims))]
    cols = []
    for xi, n in zip(xs, shown):
        ys = [y + h * (j + 1) / (n + 1) for j in range(n)]
        cols.append(ys)
    for a_, b_ in zip(cols[:-1], cols[1:]):
        i = cols.index(a_)
        for ya in a_:
            for yb in b_:
                ax.plot([xs[i], xs[i + 1]], [ya, yb], color="#d8d8d8",
                        lw=0.5, zorder=4)
    for xi, ys in zip(xs, cols):
        for yj in ys:
            dot(ax, xi, yj, r=r, color=color)
        for m in range(3):
            dot(ax, xi, y - 0.010 - m * 0.008, r=0.0018, color="#b0b0b0")
    for xi, d in zip(xs, dims):
        ax.text(xi, y - 0.030, str(d), fontsize=FS["small"], color=INK,
                ha="center", va="top", zorder=5)
    if tag:
        ax.text(x + w / 2, y + h + 0.009, tag, fontsize=FS["small"], color=color,
                ha="center", va="bottom", zorder=5)

if __name__ == "__main__":
    build()

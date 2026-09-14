"""
mlp_diagrams.py

Standalone SVG generator for the network-internals diagrams: plain MLPs, the
encoder pair, and the two-head decoder.

    python mlp_diagrams.py mlp 14 128 128 --title "node encoder"
    python mlp_diagrams.py mlp 8 128 128 -t "edge encoder" -o edge_encoder
    python mlp_diagrams.py encoder          # node + edge, side by side
    python mlp_diagrams.py decoder          # the two-head decoder
    python mlp_diagrams.py all

Everything comes out transparent, tightly cropped, and vector. Colours match
the architecture figure: gray = inherited, orange = contact side, blue = fluid.

Layer sizes are printed as text, so the dot columns are decorative and never
have to be to scale - `SHOWN` controls how many dots stand in for a column.

The decoder panel is the one worth reading carefully. The processor hands back
ONE latent per node. The contact head is applied to each of those independently
(shared weights, N times) and produces a per-node force. The fluid head is
applied ONCE, to the mean of the node latents, and produces a single force and
torque for the whole body. That asymmetry is the thing the picture exists to
show.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch


# ======================================================================
# CONFIG
# ======================================================================

OUTDIR = "figure_assets"
ALSO_PNG = True

OLD = "#8a8a8a"       # inherited from Allen et al.
INK = "#1a1a1a"
CONTACT = "#d95f02"
FLUID = "#0072b2"
WIRE = "#d8d8d8"
FILL = "#f8f8f8"

SHOWN = [4, 5, 5, 5, 5]     # dots drawn per column, purely cosmetic
DOT_R = 0.34                # in layout units
COL_GAP = 4.6               # horizontal spacing between layers
ROW_GAP = 1.25              # vertical spacing between dots

FS = {"title": 15, "head": 13, "body": 11, "small": 10, "tiny": 9}


# ======================================================================
# primitives  (everything in equal-aspect data units, so circles are circles)
# ======================================================================

def new_canvas(w_units, h_units, unit_in=0.22):
    fig = plt.figure(figsize=(w_units * unit_in, h_units * unit_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_units); ax.set_ylim(0, h_units)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def dot(ax, x, y, color=OLD, r=DOT_R, z=6):
    ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=0.6, zorder=z))


def rbox(ax, x, y, w, h, ec, fc="none", lw=1.6, ls="-", r=0.35, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, size="body", color=INK, ha="center", va="center", weight=None):
    ax.text(x, y, s, fontsize=FS[size], color=color, ha=ha, va=va,
            weight=weight, zorder=8)


def arw(ax, p0, p1, color=INK, lw=1.8, ls="-", ms=13, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, ls=ls, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def vdots(ax, x, y_top, color="#b0b0b0", n=3, gap=0.62, r=0.10):
    for m in range(n):
        ax.add_patch(Circle((x, y_top - m * gap), r, fc=color, ec="none", zorder=6))


def layer_stack(ax, x, y_mid, n_dots, color, row_gap=ROW_GAP):
    """One column of dots, vertically centred on y_mid. Returns the y list."""
    ys = [y_mid + (n_dots - 1) / 2 * row_gap - i * row_gap for i in range(n_dots)]
    for yy in ys:
        dot(ax, x, yy, color=color)
    return ys


def wire(ax, x0, ys0, x1, ys1):
    for a in ys0:
        for b in ys1:
            ax.plot([x0, x1], [a, b], color=WIRE, lw=0.5, zorder=3)


def mlp_block(ax, x0, y_mid, dims, color=OLD, dim_color=INK, show_dims=True,
              ellipsis=True, shown=None, row_gap=ROW_GAP, col_gap=COL_GAP,
              dim_off=3.2, ell_gap=0.62, r=DOT_R):
    """Draw a fully-connected sketch for `dims`. Returns (x_last, extent)."""
    shown = shown or SHOWN
    cols, xs = [], []
    for i, d in enumerate(dims):
        x = x0 + i * col_gap
        xs.append(x)
        n = shown[min(i, len(shown) - 1)]
        ys = [y_mid + (n - 1) / 2 * row_gap - j * row_gap for j in range(n)]
        for yy in ys:
            dot(ax, x, yy, color=color, r=r)
        cols.append(ys)
    for i in range(len(dims) - 1):
        wire(ax, xs[i], cols[i], xs[i + 1], cols[i + 1])
    y_bot = min(min(c) for c in cols)
    for x in xs:
        if ellipsis:
            vdots(ax, x, y_bot - 0.85, gap=ell_gap)
    if show_dims:
        for x, d in zip(xs, dims):
            txt(ax, x, y_bot - dim_off, str(d), "small", dim_color)
    return xs[-1], (y_bot - dim_off - 0.7, max(max(c) for c in cols) + r)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name + ".svg")
    fig.savefig(p, transparent=True, bbox_inches="tight", pad_inches=0.06)
    if ALSO_PNG:
        fig.savefig(os.path.join(OUTDIR, name + ".png"), transparent=True,
                    dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {p}")


# ======================================================================
# 1. plain MLP
# ======================================================================

def make_mlp(dims, title=None, subtitle=None, name="mlp", color=OLD):
    w = (len(dims) - 1) * COL_GAP + 6
    h = 16
    fig, ax = new_canvas(w, h)
    y_mid = h * 0.56
    mlp_block(ax, 3.0, y_mid, dims, color=color)
    if title:
        txt(ax, w / 2, h - 1.6, title, "title", INK, weight="bold")
    if subtitle:
        txt(ax, w / 2, h - 3.4, subtitle, "small", OLD)
    save(fig, name)


def make_encoder():
    """Node and edge encoders side by side - the real shapes from force_gns.py."""
    w, h = 26, 17
    fig, ax = new_canvas(w, h)
    txt(ax, w / 2, h - 1.5, "Encoder", "title", INK, weight="bold")
    txt(ax, w / 2, h - 3.3, r"Linear $\rightarrow$ act $\rightarrow$ Linear "
        r"$\rightarrow$ LayerNorm   (weights not shared)", "small", OLD)
    for x0, dims, tag in ((1.6, [14, 128, 128], "node features"),
                          (14.6, [8, 128, 128], "edge features")):
        txt(ax, x0 + COL_GAP, h - 5.4, tag, "head", OLD)
        mlp_block(ax, x0, 7.4, dims)
    save(fig, "diagram_encoder")


# ======================================================================
# 2. the two-head decoder
# ======================================================================

def make_decoder(N=8):
    w, h = 31, 31
    fig, ax = new_canvas(w, h)

    SM = dict(shown=[4, 4, 4], row_gap=1.05, col_gap=4.2, dim_off=2.9,
              ell_gap=0.52, r=0.30)

    # txt(ax, w / 2, h - 1.3, "Decoder — two heads", "title", INK, weight="bold")
    # txt(ax, w / 2, h - 3.1,
    #     "one latent per node in;  a force per NODE and one wrench per BODY out",
    #     "small", OLD)

    # ---- shared input: N node latents -------------------------------
    x_in = 2.4
    ys_in = layer_stack(ax, x_in, 13.6, N, OLD, row_gap=1.35)
    txt(ax, x_in, max(ys_in) + 1.9, "processor", "small", OLD)
    txt(ax, x_in, max(ys_in) + 0.9, "output", "small", OLD)
    txt(ax, x_in, min(ys_in) - 1.4, "128 each", "tiny", OLD)

    # ==================================================================
    # CONTACT HEAD - per node, shared weights
    # ==================================================================
    cx, cw, ch = 10.5, 12.5, 11.6
    cy = 16.0
    for k in (2, 1):                       # stacked cards = applied N times
        rbox(ax, cx + 0.45 * k, cy - 0.45 * k, cw, ch, CONTACT, fc="white",
             lw=1.0, z=1)
    rbox(ax, cx, cy, cw, ch, CONTACT, fc="white", lw=1.8, z=2)
    txt(ax, cx + cw / 2, cy + ch - 1.5, "contact head", "head", CONTACT,
        weight="bold")
    txt(ax, cx + cw / 2, cy + ch - 3.0,
        r"same MLP applied to every node, $\times N$", "small", CONTACT)
    mlp_block(ax, cx + 2, cy + 5.8, [128, 128, 4], color=CONTACT,
              dim_color=CONTACT, **SM)

    for yy in (max(ys_in), ys_in[len(ys_in) // 2], min(ys_in)):
        arw(ax, (x_in + 0.9, yy), (cx - 0.5, cy + ch * 0.5), CONTACT,
            lw=0.9, ls=(0, (2.5, 2.5)), ms=9)

    # ax_ = cx + cw + 1.0
    # arw(ax, (cx + cw + 0.3, cy + ch / 2), (ax_ + 1.2, cy + ch / 2), CONTACT, lw=1.8)
    # txt(ax, ax_ + 8.4, cy + ch - 2.0,
    #     r"$[\ \mathbf{t}_{\rm raw}\,(3)\ \ |\ \ n_{\rm raw}\,(1)\ ]$",
    #     "small", CONTACT)
    # txt(ax, ax_ + 8.4, cy + ch - 4.0,
    #     "project onto wall plane,  softplus on $n$", "tiny", CONTACT)
    # txt(ax, ax_ + 8.4, cy + ch - 5.6, r"$\times$ contact gate", "tiny", CONTACT)
    # txt(ax, ax_ + 8.4, cy + 2.2, r"$f_i$  —  one force per node", "head",
    #     CONTACT, weight="bold")

    # ==================================================================
    # FLUID HEAD - one per body, from the mean of the node latents
    # ==================================================================
    px, py = 6.9, 6.2
    rbox(ax, px - 1.5, py - 1.5, 3.0, 3.0, FLUID, fc="white", lw=1.7, r=1.5)
    txt(ax, px, py, r"$\frac{1}{N}\sum$", "body", FLUID)
    txt(ax, px, py - 2.6, "mean over", "tiny", FLUID)
    txt(ax, px, py - 3.7, "all nodes", "tiny", FLUID)
    for yy in (max(ys_in), ys_in[len(ys_in) // 2], min(ys_in)):
        arw(ax, (x_in + 0.9, yy), (px - 1.7, py + 0.9), FLUID, lw=0.9,
            ls=(0, (2.5, 2.5)), ms=9)

    fx, fw, fh = 11, 12.5, 11.0
    fy = py - 4.6
    rbox(ax, fx, fy, fw, fh, FLUID, fc="white", lw=1.8)
    txt(ax, fx + fw / 2, fy + fh - 1.5, "fluid head", "head", FLUID, weight="bold")
    txt(ax, fx + fw / 2, fy + fh - 3.0, "applied once, to the pooled latent",
        "small", FLUID)
    mlp_block(ax, fx + 2.0, fy + 5.4, [128, 128, 6], color=FLUID,
              dim_color=FLUID, **SM)
    arw(ax, (px + 1.7, py), (fx - 0.4, py), FLUID, lw=1.8)

    # gx = fx + fw + 1.0
    # arw(ax, (fx + fw + 0.3, fy + fh / 2), (gx + 1.2, fy + fh / 2), FLUID, lw=1.8)
    # txt(ax, gx + 8.0, fy + fh - 3.0, r"$\mathbf{F}_{\rm fluid}\,(3)$   and   "
    #     r"$\boldsymbol{\tau}_{\rm fluid}\,(3)$", "head", FLUID, weight="bold")
    # txt(ax, gx + 8.0, fy + fh - 5.2, "one wrench for the whole body", "small", FLUID)

    # ax.plot([x_in + 6.5, w - 1.0], [14.3, 14.3], color="#e2e2e2", lw=1.0,
    #         ls=(0, (5, 4)), zorder=0)
    save(fig, "diagram_decoder")


# ======================================================================

def main():

    # m = sub.add_parser("mlp", help="a plain fully-connected sketch")
    # m.add_argument("dims", nargs="+", type=int, help="layer sizes, e.g. 14 128 128")
    # m.add_argument("-t", "--title")
    # m.add_argument("--subtitle")
    # m.add_argument("-o", "--out", default=None)
    # m.add_argument("-c", "--color", default=OLD)

    # sub.add_parser("encoder", help="node + edge encoders")
    # d = sub.add_parser("decoder", help="the two-head decoder")
    # d.add_argument("-N", type=int, default=8, help="nodes per graph (default 8)")
    # sub.add_parser("all", help="encoder and decoder")
    make_encoder()
    make_decoder()
    # if a.cmd == "mlp":
    #     name = a.out or "mlp_" + "_".join(str(d) for d in a.dims)
    #     make_mlp(a.dims, title=a.title, subtitle=a.subtitle, name=name,
    #              color=a.color)
    # elif a.cmd == "encoder":
    #     make_encoder()
    # elif a.cmd == "decoder":
    #     make_decoder(N=a.N)
    # else:
    #     make_encoder(); make_decoder()


if __name__ == "__main__":
    main()

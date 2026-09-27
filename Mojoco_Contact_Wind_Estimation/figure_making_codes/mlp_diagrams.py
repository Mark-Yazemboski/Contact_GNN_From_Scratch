"""
mlp_diagrams.py

SVG generator for the network-internals diagrams.

Each piece of the decoder is drawn by its own function, so you can get the
whole assembly in one image OR each part on its own and arrange them yourself:

    python mlp_diagrams.py all         # everything below
    python mlp_diagrams.py decoder     # whole decoder assembly
    python mlp_diagrams.py contact     # contact head alone
    python mlp_diagrams.py fluid       # fluid head alone (with the pooling)
    python mlp_diagrams.py latents     # the processor-output node column alone
    python mlp_diagrams.py encoder     # node + edge encoders
    python mlp_diagrams.py mlp 14 128 128 -t "node encoder"

The heads are drawn as a narrowing TRAPEZOID - the standard shorthand for a
decoder, many channels in and few out. No layer sizes, no dot columns: the
point of the panel is that a decoder sits there, not what its widths are.

Everything is transparent, tightly cropped, and vector. Colours match the
architecture figure: gray = inherited, orange = contact, blue = fluid.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon


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

SHOWN = [4, 5, 5, 5, 5]     # dots per column in the MLP sketches (cosmetic)
DOT_R = 0.34
COL_GAP = 4.6
ROW_GAP = 1.25

# decoder trapezoid: tall edge on the left, short edge on the right
TRAP_W = 3.6
TRAP_H_IN = 3.6
TRAP_H_OUT = 1.4

FS = {"title": 15, "head": 13, "body": 11, "small": 10, "tiny": 9}


# ======================================================================
# primitives  (equal-aspect data units, so circles stay circular)
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


def trapezoid(ax, x, y_mid, color, w=TRAP_W, h_in=TRAP_H_IN, h_out=TRAP_H_OUT,
              label=None, fc="white", lw=1.8, z=4):
    """Decoder shorthand: wide edge in on the left, narrow edge out on the
    right. Returns (x_out, y_mid) so you can hang an arrow off the tip."""
    pts = [(x, y_mid - h_in / 2), (x, y_mid + h_in / 2),
           (x + w, y_mid + h_out / 2), (x + w, y_mid - h_out / 2)]
    ax.add_patch(Polygon(pts, closed=True, ec=color, fc=fc, lw=lw, zorder=z,
                         joinstyle="round"))
    if label:
        txt(ax, x + w * 0.42, y_mid, label, "tiny", color, weight="bold")
    return x + w, y_mid


def layer_stack(ax, x, y_mid, n_dots, color, row_gap=ROW_GAP, r=DOT_R):
    ys = [y_mid + (n_dots - 1) / 2 * row_gap - i * row_gap for i in range(n_dots)]
    for yy in ys:
        dot(ax, x, yy, color=color, r=r)
    return ys


def wire(ax, x0, ys0, x1, ys1):
    for a in ys0:
        for b in ys1:
            ax.plot([x0, x1], [a, b], color=WIRE, lw=0.5, zorder=3)


def vdots(ax, x, y_top, color="#b0b0b0", n=3, gap=0.62, r=0.10):
    for m in range(n):
        ax.add_patch(Circle((x, y_top - m * gap), r, fc=color, ec="none", zorder=6))


def mlp_block(ax, x0, y_mid, dims, color=OLD, dim_color=INK, show_dims=True,
              ellipsis=True, shown=None, row_gap=ROW_GAP, col_gap=COL_GAP,
              dim_off=3.2, ell_gap=0.62, r=DOT_R):
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
# THE PIECES.  Each draws into an axes you give it, so the same code makes
# the combined figure and the standalone ones.
# ======================================================================

def draw_node_latents(ax, x, y_mid, N=8, label=True):
    """The processor's output: one latent per node. Returns the y list."""
    ys = layer_stack(ax, x, y_mid, N, OLD, row_gap=1.35)
    if label:
        txt(ax, x, max(ys) + 1.9, "processor", "small", OLD)
        txt(ax, x, max(ys) + 0.9, "output", "small", OLD)
        txt(ax, x, min(ys) - 1.4, "128 per node", "tiny", OLD)
    return ys


def draw_contact_head(ax, x, y, w=9.0, h=7.4, cards=True):
    """Contact head: stacked cards = the same decoder run once per node."""
    if cards:
        for k in (2, 1):
            rbox(ax, x + 0.42 * k, y - 0.42 * k, w, h, CONTACT, fc="white",
                 lw=1.0, z=1)
    rbox(ax, x, y, w, h, CONTACT, fc="white", lw=1.8, z=2)
    txt(ax, x + w / 2, y + h - 1.3, "contact head", "head", CONTACT,
        weight="bold")
    txt(ax, x + w / 2, y + h - 2.6, r"one per node,  $\times N$", "tiny",
        CONTACT)
    tx = x + (w - TRAP_W) / 2
    return trapezoid(ax, tx, y + 2.4, CONTACT)


def draw_fluid_head(ax, x, y, w=9.0, h=7.4):
    """Fluid head: one decoder, run once on the pooled latent."""
    rbox(ax, x, y, w, h, FLUID, fc="white", lw=1.8, z=2)
    txt(ax, x + w / 2, y + h - 1.3, "fluid head", "head", FLUID, weight="bold")
    txt(ax, x + w / 2, y + h - 2.6, "one per body", "tiny", FLUID)
    tx = x + (w - TRAP_W) / 2
    return trapezoid(ax, tx, y + 2.4, FLUID)


def draw_pool(ax, x, y, r=1.5):
    """The mean-over-nodes symbol that feeds the fluid head."""
    rbox(ax, x - r, y - r, 2 * r, 2 * r, FLUID, fc="white", lw=1.7, r=r)
    txt(ax, x, y, r"$\frac{1}{N}\sum$", "body", FLUID)
    txt(ax, x, y - 2.5, "mean over", "tiny", FLUID)
    txt(ax, x, y - 3.5, "all nodes", "tiny", FLUID)
    return x + r, y


def fan(ax, x0, ys, target, color, n=3):
    """A few dashed arrows from the latent column to a destination."""
    picks = (max(ys), ys[len(ys) // 2], min(ys))[:n]
    for yy in picks:
        arw(ax, (x0 + 0.9, yy), target, color, lw=0.9, ls=(0, (2.5, 2.5)), ms=9)


# ======================================================================
# FIGURES
# ======================================================================

def make_decoder(N=8):
    """Everything assembled: latents -> contact head (per node)
                                    -> pool -> fluid head (per body)."""
    w, h = 22, 22
    fig, ax = new_canvas(w, h)
    cw, ch = 9.0, 7.4

    x_in = 2.2
    ys_in = draw_node_latents(ax, x_in, 12.4, N)

    cx, cy = 10.2, 13.2
    fan(ax, x_in, ys_in, (cx - 0.5, cy + ch * 0.5), CONTACT)
    draw_contact_head(ax, cx, cy, cw, ch)

    px, py = 6.4, 4.9
    fan(ax, x_in, ys_in, (px - 1.7, py + 0.8), FLUID)
    pout = draw_pool(ax, px, py)

    fx, fy = 10.2, 1.6
    arw(ax, (pout[0] + 0.3, py), (fx - 0.4, py), FLUID, lw=1.8)
    draw_fluid_head(ax, fx, fy, cw, ch)

    save(fig, "diagram_decoder")


def make_contact_head_alone():
    w, h = 11.0, 9.4
    fig, ax = new_canvas(w, h)
    draw_contact_head(ax, 0.8, 1.4, 9.0, 7.4)
    save(fig, "diagram_contact_head")


def make_fluid_head_alone(with_pool=True):
    w, h = (16.0 if with_pool else 10.6), 9.4
    fig, ax = new_canvas(w, h)
    x = 0.8
    ymid = 1.0 + 7.4 / 2
    if with_pool:
        pout = draw_pool(ax, 2.4, ymid)
        x = 5.6
        arw(ax, (pout[0] + 0.3, ymid), (x - 0.4, ymid), FLUID, lw=1.8)
    draw_fluid_head(ax, x, 1.0, 9.0, 7.4)
    save(fig, "diagram_fluid_head")


def make_latents_alone(N=8):
    w, h = 7.0, 17.0
    fig, ax = new_canvas(w, h)
    draw_node_latents(ax, w / 2, h / 2 - 0.6, N)
    save(fig, "diagram_processor_output")


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


def make_mlp(dims, title=None, subtitle=None, name="mlp", color=OLD):
    w = (len(dims) - 1) * COL_GAP + 6
    h = 16
    fig, ax = new_canvas(w, h)
    mlp_block(ax, 3.0, h * 0.56, dims, color=color)
    if title:
        txt(ax, w / 2, h - 1.6, title, "title", INK, weight="bold")
    if subtitle:
        txt(ax, w / 2, h - 3.4, subtitle, "small", OLD)
    save(fig, name)


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Network diagrams as SVG.")
    sub = ap.add_subparsers(dest="cmd")

    m = sub.add_parser("mlp", help="a plain fully-connected sketch")
    m.add_argument("dims", nargs="+", type=int)
    m.add_argument("-t", "--title")
    m.add_argument("--subtitle")
    m.add_argument("-o", "--out", default=None)
    m.add_argument("-c", "--color", default=OLD)

    sub.add_parser("encoder", help="node + edge encoders")
    sub.add_parser("contact", help="contact head alone")
    sub.add_parser("latents", help="processor-output node column alone")
    f = sub.add_parser("fluid", help="fluid head alone")
    f.add_argument("--no-pool", action="store_true",
                   help="leave out the mean-over-nodes symbol")
    d = sub.add_parser("decoder", help="whole decoder assembly")
    d.add_argument("-N", type=int, default=8)
    sub.add_parser("all", help="every figure in this file")

    a = ap.parse_args()
    cmd = a.cmd or "all"

    if cmd == "mlp":
        name = a.out or "mlp_" + "_".join(str(d) for d in a.dims)
        make_mlp(a.dims, title=a.title, subtitle=a.subtitle, name=name,
                 color=a.color)
    elif cmd == "encoder":
        make_encoder()
    elif cmd == "contact":
        make_contact_head_alone()
    elif cmd == "fluid":
        make_fluid_head_alone(with_pool=not a.no_pool)
    elif cmd == "latents":
        make_latents_alone()
    elif cmd == "decoder":
        make_decoder(N=a.N)
    else:
        make_encoder()
        make_decoder()
        make_contact_head_alone()
        make_fluid_head_alone()
        make_latents_alone()


if __name__ == "__main__":
    main()

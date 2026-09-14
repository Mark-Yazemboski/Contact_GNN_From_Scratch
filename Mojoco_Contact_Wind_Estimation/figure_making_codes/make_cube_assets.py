"""
make_cube_assets.py

Generates the two cube assets for the architecture figure, as transparent-
background SVG ready to drop into Inkscape / Illustrator / PowerPoint.

  asset_mesh_cube.svg      block 1 - state -> graph: the 8 corner nodes and the
                           knn edges, drawn from YOUR actual mesh/knn code so
                           the node and edge counts are provably right.

  asset_contact_cube.svg   block 4 - the rigid-body layer: cube at impact with
                           one corner in contact, normal + friction on the
                           gated node, F_fluid and tau_fluid at the COM, gravity,
                           wind streamlines, and the next-state ghost.

Run from the directory containing generate_node_states.py (it imports your real
mesh + knn functions; falls back to an inline copy if the import fails).

    python make_cube_assets.py

Everything you'd want to change lives in the CONFIG block below.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# ======================================================================
# CONFIG
# ======================================================================

OUT_DIR = "figure_assets"
ALSO_PNG = True          # PNG previews alongside the SVGs, for quick checking

NODES_PER_EDGE = 2       # -> 8 corner nodes
NEAREST_NEIGHBORS = 3    # your force-pipeline default
BLOCK_HALF_WIDTH = 0.0524

# Gray = inherited from Allen et al.  Accent = yours.
# Both chosen to stay distinguishable in grayscale and for red-green CVD.
C = {
    "node":      "#3d3d3d",   # mesh nodes           (inherited)
    "edge":      "#9a9a9a",   # mesh edges           (inherited)
    "cube":      "#5a5a5a",   # cube wireframe
    "floor":     "#3798f8",
    "contact":   "#d95f02",   # contact forces       (yours) - orange
    "fluid":     "#0072b2",   # fluid force/torque   (yours) - blue
    "wind":      "#56b4e9",   # wind streamlines
    "gravity":   "#8a8a8a",   # exact, not learned
    "ghost":     "#b4b4b4",   # next-state cube
    "text":      "#1a1a1a",
    "floor_edge":"#7a7a7a",   # section-cut ground line + hatching
    "feature":   "#009e73",   # b, the wall-distance node feature - green
    "vel":       "#3d3d3d",   # v^FD - inherited feature, so gray
}


# Mesh-cube annotations - flip any of these off if the poster panel gets busy.
MESH_ANNOT = {"b": True, "v": True, "w": True, "x": True, "d": True}
MESH_LIFT = 2.65
MESH_VIEW = (13, -73)     # (elev, azim) for the mesh panel          # how far the cube floats above the cut ground
 

LABEL_SIZE = 13          # bump in the editor; 24pt+ at FINAL PRINT SIZE
LW_CUBE = 1.8
LW_ARROW = 2.6

# Arrow lengths are NOT to scale - contact peaks run ~30 N against a fluid
# force under 1.5 N, so a literal scaling would make the fluid arrows invisible.
# Say so in the caption. These are in units of half-width.
ARROW = {
    "normal":   1.75,
    "friction": 1.35,
    "fluid":    1.55,
    "gravity":  1.15,
    "torque_r": 0.85,   # radius of the curved torque arc
}

SHOW_SYMBOLS = True      # f_i, F_fluid, tau_fluid, g labels on the arrows


# ======================================================================
# Geometry - imported from your code where possible
# ======================================================================

try:
    from generate_node_states import mesh_cube_surface, knn_adjacency
    _SOURCE = "generate_node_states.py"
except Exception:
    _SOURCE = "inline fallback copy"

    def mesh_cube_surface(side_length, nodes_per_edge):
        L = side_length / 2.0
        lin = np.linspace(-L, L, nodes_per_edge)
        nodes = []
        for x in [-L, L]:
            for y in lin:
                for z in lin:
                    nodes.append([x, y, z])
        for y in [-L, L]:
            for x in lin:
                for z in lin:
                    nodes.append([x, y, z])
        for z in [-L, L]:
            for x in lin:
                for y in lin:
                    nodes.append([x, y, z])
        return np.unique(np.array(nodes), axis=0)

    def knn_adjacency(nodes, k=4):
        N = nodes.shape[0]
        diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        edge_list = []
        for i in range(N):
            knn_idx = np.argsort(dist[i])[1:k + 1]
            for j in knn_idx:
                edge_list.append([i, j])
        return np.array(edge_list).T


def rot_xyz(rx, ry, rz):
    """Degrees -> rotation matrix, applied Z then Y then X."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


# ======================================================================
# 3D arrow that keeps a real arrowhead under orthographic projection
# ======================================================================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs3d)


def arrow(ax, start, vec, color, lw=LW_ARROW, ls="-", mutation=18):
    s = np.asarray(start, float)
    e = s + np.asarray(vec, float)
    ax.add_artist(Arrow3D([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                          mutation_scale=mutation, lw=lw, arrowstyle="-|>",
                          color=color, linestyle=ls, zorder=10))
    return e


def label(ax, pos, text, color, size=LABEL_SIZE, ha="left", va="bottom"):
    if SHOW_SYMBOLS:
        ax.text(pos[0], pos[1], pos[2], text, color=color, fontsize=size,
                ha=ha, va=va, zorder=20)


def clean_axes(ax, pts=None, pad=1.2, xlim=None, ylim=None, zlim=None):
    """Strip everything; either fit an equal-aspect cube around `pts`, or use
    the explicit ranges given. box_aspect is set from the ranges so a wide
    scene fills the canvas instead of floating in a square."""
    ax.set_axis_off()
    try:
        ax.set_proj_type("ortho")          # technical-drawing look, no perspective
    except Exception:
        pass
    if xlim is None:
        pts = np.asarray(pts, float)
        c = pts.mean(axis=0)
        r = np.abs(pts - c).max() * pad
        xlim = (c[0] - r, c[0] + r)
        ylim = (c[1] - r, c[1] + r)
        zlim = (c[2] - r, c[2] + r)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except Exception:
        pass


def save(fig, name):
    for a in fig.axes:
        a.set_position([0.0, 0.0, 1.0, 1.0])
    os.makedirs(OUT_DIR, exist_ok=True)
    svg = os.path.join(OUT_DIR, name + ".svg")
    fig.savefig(svg, transparent=True, bbox_inches="tight", pad_inches=0.02)
    if ALSO_PNG:
        fig.savefig(os.path.join(OUT_DIR, name + ".png"), transparent=True,
                    dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {svg}")


# ======================================================================
# ASSET 1 - mesh cube: nodes + knn edges
# ======================================================================


def hatched_floor(ax, x0, x1, y_front, z=0.0, n_slash=26, depth=0.42,
                  line_color=None, hatch_color=None):
    """Ground drawn as a section cut: solid top line with 45-degree slashes
    hanging below it, in the vertical plane y = y_front. Slashes are clipped
    at the left edge, which is what bounded section hatching actually looks
    like."""
    line_color = line_color or C["floor_edge"]
    hatch_color = hatch_color or C["floor_edge"]
    ax.plot([x0, x1], [y_front, y_front], [z, z],
            color=line_color, lw=1.9, zorder=2)
    for c in np.linspace(x0, x1, n_slash):
        t = min(depth, c - x0)
        if t < 0.05:
            continue
        ax.plot([c, c - t], [y_front, y_front], [z, z - t],
                color=hatch_color, lw=1.0, alpha=0.85, zorder=2)
 
 
def make_mesh_cube():
    rest = mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE) / BLOCK_HALF_WIDTH
    ei = knn_adjacency(rest, k=NEAREST_NEIGHBORS)
 
    # knn_adjacency returns DIRECTED edges (i->j and j->i both appear).
    # Draw each undirected segment once.
    undirected = sorted({tuple(sorted((int(a), int(b)))) for a, b in zip(ei[0], ei[1])})
 
    print(f"  nodes N = {len(rest)}   directed edges E = {ei.shape[1]}   "
          f"unique segments drawn = {len(undirected)}")
 
    # Mild tilt + lift clear of the ground, so b is visibly a per-node quantity
    # rather than one number for the whole body.
    R = rot_xyz(9, 15, 0)
    nodes = (R @ rest.T).T + np.array([0.0, 0.0, MESH_LIFT])
 
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
 
    for i, j in undirected:
        ax.plot(*zip(nodes[i], nodes[j]), color=C["edge"], lw=LW_CUBE, zorder=3)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=90, c=C["node"],
               depthshade=False, edgecolors="white", linewidths=1.1, zorder=5)
 
    # ---- b: wall distance, on the lowest node ----
    i_b = int(np.argmin(nodes[:, 2]))
    screen_level_floor(ax, nodes[i_b], elev=MESH_VIEW[0], azim=MESH_VIEW[1])
    if MESH_ANNOT["b"]:
        p = nodes[i_b]
        ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], 0.0], color=C["feature"],
                lw=1.5, ls=(0, (3, 2.5)), zorder=6)
        ax.scatter(*p, s=150, c=C["feature"], depthshade=False,
                   edgecolors="white", linewidths=1.2, zorder=7)
        for zz in (p[2], 0.0):   # end ticks
            ax.plot([p[0] - 0.16, p[0] + 0.16], [p[1], p[1]], [zz, zz],
                    color=C["feature"], lw=1.4, zorder=6)
        label(ax, [p[0] + 0.22, p[1], p[2] * 0.5], r"$b_i$", C["feature"],
              ha="left", va="center")
 
    # ---- v: finite-difference velocity, on a different node ----
    if MESH_ANNOT["v"]:
        i_v = int(np.argmax(nodes @ np.array([0.9, -0.2, 0.5])))
        p = nodes[i_v]
        v_hat = np.array([0.86, 0.0, -0.51]); v_hat /= np.linalg.norm(v_hat)
        e = arrow(ax, p, v_hat * 1.35, C["vel"])
        label(ax, e + np.array([0.12, 0.0, -0.10]), r"$\mathbf{v}_i^{\rm FD}$",
              C["vel"], ha="left", va="top")
 
    # ---- w: wind, streamlines in from the left ----
    if MESH_ANNOT["w"]:
        for z0 in (MESH_LIFT - 0.55, MESH_LIFT + 0.15, MESH_LIFT + 0.85):
            xs = np.linspace(-2.45, -1.70, 50)
            zs = z0 + 0.05 * np.sin(7.0 * xs)
            ax.plot(xs, np.zeros_like(xs), zs, color=C["wind"], lw=1.6,
                    alpha=0.95, zorder=4)
            arrow(ax, np.array([xs[-1], 0.0, zs[-1]]),
                  np.array([0.24, 0.0, 0.0]), C["wind"], lw=1.6, mutation=12)
        label(ax, [-2.45, 0.0, MESH_LIFT + 1.25], r"$\mathbf{w}$", C["wind"],
              ha="left", va="bottom")
 
    # ---- graph notation ----
    if MESH_ANNOT["x"]:
        i0 = int(np.argmax(nodes @ np.array([-0.5, 0.3, 1.0])))
        label(ax, nodes[i0] + np.array([-0.10, 0.0, 0.16]), r"$\mathbf{x}_i$",
              C["text"], ha="right", va="bottom")
    if MESH_ANNOT["d"]:
        mids = [((nodes[a] + nodes[b]) / 2, a, b) for a, b in undirected]
        mid, a, b = max(mids, key=lambda m: m[0][2] - 0.35 * abs(m[0][0]))
        label(ax, mid + np.array([0.0, 0.0, 0.26]), r"$\mathbf{d}_{ij}$",
              C["edge"], ha="center", va="bottom")
 
    ax.view_init(*MESH_VIEW)
    clean_axes(ax, xlim=(-2.9, 3.4), ylim=(-1.5, 1.5), zlim=(-0.55, 4.75))
    save(fig, "asset_mesh_cube")
 
 
 
def screen_level_floor(ax, through, elev, azim, left=3.4, right=3.4,
                       n_slash=30, depth=0.40, line_color=None,
                       hatch_color=None):
    """Section-cut ground that draws PERFECTLY HORIZONTAL on screen, whatever
    the 3D camera is doing.
 
    The camera's right vector r = (sin azim, -cos azim, 0) is horizontal in
    world space and perpendicular to the view direction, so a line along r
    projects to an exactly horizontal screen line. Running the ground along r
    through `through` (the point directly under the annotated node) also makes
    the b drop-line land on it by construction. A vertical world drop of d
    projects to d*cos(elev) on screen, so the 45-degree slashes divide by
    cos(elev) to come out at a true 45 degrees in the image."""
    line_color = line_color or C["floor_edge"]
    hatch_color = hatch_color or C["floor_edge"]
    a = np.radians(azim)
    r = np.array([np.sin(a), -np.cos(a), 0.0])
    r /= np.linalg.norm(r)
    zdrop = 1.0 / max(np.cos(np.radians(elev)), 1e-6)
    P = np.asarray(through, float) * np.array([1.0, 1.0, 0.0])
 
    A, B = P - r * left, P + r * right
    ax.plot(*zip(A, B), color=line_color, lw=1.9, zorder=2)
    for t in np.linspace(0.0, left + right, n_slash):
        Q = A + r * t
        s_ = min(depth, t)                       # clip the hatch at the left end
        if s_ < 0.05:
            continue
        E = Q - r * s_ - np.array([0.0, 0.0, s_ * zdrop])
        ax.plot(*zip(Q, E), color=hatch_color, lw=1.0, alpha=0.85, zorder=2)
 
 
def make_mesh_cube():
    rest = mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE) / BLOCK_HALF_WIDTH
    ei = knn_adjacency(rest, k=NEAREST_NEIGHBORS)
 
    # knn_adjacency returns DIRECTED edges (i->j and j->i both appear).
    # Draw each undirected segment once.
    undirected = sorted({tuple(sorted((int(a), int(b)))) for a, b in zip(ei[0], ei[1])})
 
    print(f"  nodes N = {len(rest)}   directed edges E = {ei.shape[1]}   "
          f"unique segments drawn = {len(undirected)}")
 
    # Mild tilt + lift clear of the ground, so b is visibly a per-node quantity
    # rather than one number for the whole body.
    R = rot_xyz(9, 15, 0)
    nodes = (R @ rest.T).T + np.array([0.0, 0.0, MESH_LIFT])
 
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
 
    for i, j in undirected:
        ax.plot(*zip(nodes[i], nodes[j]), color=C["edge"], lw=LW_CUBE, zorder=3)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=90, c=C["node"],
               depthshade=False, edgecolors="white", linewidths=1.1, zorder=5)
 
    # ---- b: wall distance, on the lowest node ----
    i_b = int(np.argmin(nodes[:, 2]))
    screen_level_floor(ax, nodes[i_b], elev=MESH_VIEW[0], azim=MESH_VIEW[1])
    if MESH_ANNOT["b"]:
        p = nodes[i_b]
        ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], 0.0], color=C["feature"],
                lw=1.5, ls=(0, (3, 2.5)), zorder=6)
        ax.scatter(*p, s=150, c=C["feature"], depthshade=False,
                   edgecolors="white", linewidths=1.2, zorder=7)
        for zz in (p[2], 0.0):   # end ticks
            ax.plot([p[0] - 0.16, p[0] + 0.16], [p[1], p[1]], [zz, zz],
                    color=C["feature"], lw=1.4, zorder=6)
        label(ax, [p[0] + 0.22, p[1], p[2] * 0.5], r"$b_i$", C["feature"],
              ha="left", va="center")
 
    # ---- v: finite-difference velocity, on a different node ----
    if MESH_ANNOT["v"]:
        i_v = int(np.argmax(nodes @ np.array([0.9, -0.2, 0.5])))
        p = nodes[i_v]
        v_hat = np.array([0.86, 0.0, -0.51]); v_hat /= np.linalg.norm(v_hat)
        e = arrow(ax, p, v_hat * 1.35, C["vel"])
        label(ax, e + np.array([0.12, 0.0, -0.10]), r"$\mathbf{v}_i^{\rm FD}$",
              C["vel"], ha="left", va="top")
 
    # ---- w: wind, streamlines in from the left ----
    if MESH_ANNOT["w"]:
        for z0 in (MESH_LIFT - 0.55, MESH_LIFT + 0.15, MESH_LIFT + 0.85):
            xs = np.linspace(-2.45, -1.70, 50)
            zs = z0 + 0.05 * np.sin(7.0 * xs)
            ax.plot(xs, np.zeros_like(xs), zs, color=C["wind"], lw=1.6,
                    alpha=0.95, zorder=4)
            arrow(ax, np.array([xs[-1], 0.0, zs[-1]]),
                  np.array([0.24, 0.0, 0.0]), C["wind"], lw=1.6, mutation=12)
        label(ax, [-2.45, 0.0, MESH_LIFT + 1.0], r"$\mathbf{wind}$", C["wind"],
              ha="left", va="bottom")
 
    # ---- graph notation ----
    if MESH_ANNOT["x"]:
        i0 = int(np.argmax(nodes @ np.array([-0.5, 0.3, 1.0])))
        label(ax, nodes[i0] + np.array([-0.10, 0.0, 0.16]), r"$\mathbf{x}_i$",
              C["text"], ha="right", va="bottom")
    if MESH_ANNOT["d"]:
        mids = [((nodes[a] + nodes[b]) / 2, a, b) for a, b in undirected]
        mid, a, b = max(mids, key=lambda m: m[0][2] - 0.35 * abs(m[0][0]))
        label(ax, mid + np.array([0.1, 0.0, 0.10]), r"$\mathbf{d}_{ij}$",
              C["edge"], ha="center", va="bottom")
 
    ax.view_init(*MESH_VIEW)
    clean_axes(ax, xlim=(-2.9, 3.4), ylim=(-1.5, 1.5), zlim=(-0.55, 4.75))
    save(fig, "asset_mesh_cube")
 
 


# ======================================================================
# ASSET 2 - contact-state cube with the force decomposition
# ======================================================================

def make_contact_cube():
    rest = mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE) / BLOCK_HALF_WIDTH
    ei = knn_adjacency(rest, k=NEAREST_NEIGHBORS)
    undirected = sorted({tuple(sorted((int(a), int(b)))) for a, b in zip(ei[0], ei[1])})

    # Tilt so exactly one corner is clearly the lowest, then drop it onto z = 0.
    R = rot_xyz(40, 30, 0)
    com = np.array([0.0, 0.0, 0.0])
    nodes = (R @ rest.T).T + com
    k_contact = int(np.argmin(nodes[:, 2]))
    lift = -nodes[k_contact, 2]
    nodes[:, 2] += lift
    com = com + np.array([0.0, 0.0, lift])
    p_c = nodes[k_contact]

    # Next-state cube: pushed well clear to the right so the two read as two
    # states, not one overlapping mess. Small extra rotation - the rotation is
    # the point, a pure translation would undersell the torque term.
    R2 = rot_xyz(9, 6, 4) @ R
    com2 = com + np.array([3.15, 0.0, 0.35])
    ghost = (R2 @ rest.T).T + com2

    fig = plt.figure(figsize=(8.6, 4.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # ---- floor ----
    gx, gy = np.meshgrid(np.linspace(-2.3, 2.3, 2), np.linspace(-2.2, 2.2, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), color=C["floor"],
                    alpha=.3, shade=False, zorder=0)

    # floor_z = -0.15
    # gx, gy = np.meshgrid(np.linspace(-2.3, 4.6, 2), np.linspace(-1.8, 1.8, 2))
    # ax.plot_surface(
    #     gx, gy,
    #     np.full_like(gx, floor_z),
    #     color=C["floor"],
    #     alpha=1,
    #     shade=False
    # )

    # ---- next-state cube ----
    # for i, j in undirected:
    #     ax.plot(*zip(ghost[i], ghost[j]), color=C["ghost"], lw=1.4,
    #             ls=(0, (4, 3)), zorder=1)
    # label(ax, com2 + np.array([0.0, 0.0, 1.85]),
    #       r"$t+1$", C["ghost"], size=LABEL_SIZE, ha="center")

    # ---- cube at time t ----
    for i, j in undirected:
        ax.plot(*zip(nodes[i], nodes[j]), color=C["cube"], lw=LW_CUBE, zorder=3)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=55, c=C["node"],
               depthshade=False, edgecolors="white", linewidths=1.0, zorder=4)

    # Gated contact node highlighted - only this one carries force.
    ax.scatter(*p_c, s=170, c=C["contact"], depthshade=False,
               edgecolors="white", linewidths=1.3, zorder=3)

    # ---- contact forces on the gated node, aimed into the open floor area ----
    n_hat = np.array([0.0, 0.0, .8])
    t_hat = np.array([-.8, 0.0, 0.0])
    e = arrow(ax, p_c, n_hat * ARROW["normal"], C["contact"])
    label(ax, e + np.array([.7, 0.0, -1.35]), r"$f_i^{\,n}$", C["contact"],
          ha="right", va="bottom")
    e = arrow(ax, p_c, t_hat * ARROW["friction"], C["contact"])
    label(ax, e + np.array([.85, 0.0, -0.095]), r"$f_i^{\,t}$", C["contact"],
          ha="right", va="top")

    # ---- COM ----
    ax.scatter(*com, s=60, c=C["cube"], depthshade=False, marker="o",
               edgecolors="white", linewidths=1.0, zorder=9)

    # fluid force at the COM, pointing downwind into clear space
    u = np.array([.8, 0, 0])
    e = arrow(ax, com, u * ARROW["fluid"], C["fluid"])
    label(ax, e + np.array([-0.70, 0.0, 0.12]), r"$\mathbf{F}_{\rm fluid}$",
          C["fluid"], ha="left", va="bottom")

    # gravity at the COM - exact, not learned
    e = arrow(ax, com, np.array([0, 0, -.8]) * ARROW["gravity"], C["gravity"])
    label(ax, e + np.array([-0.16, 0.0, 0.42]), r"$m\mathbf{g}$", C["gravity"],
          ha="right", va="center")

    # curved torque arrow, parked ABOVE the cube so it owns empty space
    c_t = com 
    e1, e2 = np.array([.5, 0.0, 0.0]), np.array([0.0, 0.0, .5])
    th = np.linspace(-2.45, -4.71, 60)
    r_t = ARROW["torque_r"]
    arc = c_t + r_t * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=C["fluid"],
            lw=LW_ARROW - 0.4, zorder=10)
    arrow(ax, arc[-1]+np.array([0, 0.0, 0.0]), np.array([0.26, 0.0, 0.0]) , C["fluid"],
          lw=LW_ARROW - 0.4, mutation=14)
    label(ax, c_t + np.array([-.3, 0.0, .80]),
          r"$\boldsymbol{\tau}_{\rm fluid}$", C["fluid"], ha="left", va="center")

    # ---- wind streamlines entering from -x ----
    for z0 in (0.75, 1.45, 2.15):
        xs = np.linspace(-2.80, -1.75, 60)
        zs = z0 + 0.06 * np.sin(6.0 * xs)
        ax.plot(xs, np.zeros_like(xs), zs, color=C["wind"], lw=1.6,
                alpha=0.95, zorder=2)
        arrow(ax, np.array([xs[-1], 0.0, zs[-1]]), np.array([0.26, 0.0, 0.0]),
              C["wind"], lw=1.6, mutation=12)
    label(ax, np.array([-2.60, 0.0, 2.42]), "wind", C["wind"], ha="left")

    # # ---- transition arrow between the two states ----
    # arrow(ax, np.array([1.55, 0.0, 0.32]), np.array([0.95, 0.0, 0.0]),
    #       C["text"], lw=2.0, mutation=16)

    ax.view_init(elev=14, azim=-80)
    clean_axes(ax, xlim=(-2.6, 4.7), ylim=(-1.9, 1.9), zlim=(-0.25, 3.15))
    save(fig, "asset_contact_cube")

def make_t_plus_1_cube():
    rest = mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE) / BLOCK_HALF_WIDTH
    ei = knn_adjacency(rest, k=NEAREST_NEIGHBORS)
    undirected = sorted({tuple(sorted((int(a), int(b)))) for a, b in zip(ei[0], ei[1])})

    # Tilt so exactly one corner is clearly the lowest, then drop it onto z = 0.
    R = rot_xyz(30, 40, 0)
    com = np.array([0.3, 0.0, 0.0])
    nodes = (R @ rest.T).T + com
    k_contact = int(np.argmin(nodes[:, 2]))
    lift = -nodes[k_contact, 2]
    nodes[:, 2] += lift
    com = com + np.array([0.0, 0.0, lift])
    p_c = nodes[k_contact]

    # Next-state cube: pushed well clear to the right so the two read as two
    # states, not one overlapping mess. Small extra rotation - the rotation is
    # the point, a pure translation would undersell the torque term.
    R2 = rot_xyz(9, 6, 4) @ R
    com2 = com + np.array([3.15, 0.0, 0.35])
    ghost = (R2 @ rest.T).T + com2

    fig = plt.figure(figsize=(8.6, 4.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # ---- floor ----
    gx, gy = np.meshgrid(np.linspace(-2.3, 2.3, 2), np.linspace(-2.2, 2.2, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), color=C["floor"],
                    alpha=.3, shade=False, zorder=0)

    
    # ---- cube at time t ----
    for i, j in undirected:
        ax.plot(*zip(nodes[i], nodes[j]), color=C["cube"], lw=LW_CUBE, zorder=3)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=55, c=C["node"],
                depthshade=False, edgecolors="white", linewidths=1.0, zorder=4)


    ax.view_init(elev=14, azim=-80)
    clean_axes(ax, xlim=(-2.6, 4.7), ylim=(-1.9, 1.9), zlim=(-0.25, 3.15))
    save(fig, "t_plus_1_cube")
        
if __name__ == "__main__":
    print(f"geometry source: {_SOURCE}")
    make_mesh_cube()

    make_contact_cube()
    make_t_plus_1_cube()
    print("\nBoth assets are transparent-background SVG with separate paths - "
          "recolour, restyle and move the arrows in Inkscape/Illustrator.")

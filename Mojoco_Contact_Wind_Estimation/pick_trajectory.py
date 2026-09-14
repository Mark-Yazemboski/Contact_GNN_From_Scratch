"""
pick_trajectory.py

Ranks test trajectories so you can choose a filmstrip subject by a stated rule
instead of by eye.

Everything here is computed from GROUND TRUTH ONLY - no model is loaded. That
is deliberate: if you rank by how well the model did and then show the winner,
you have cherry-picked, and the first person to ask how you chose it will say
so. Rank on the physics, show the result, and put the rule in the caption.

Two rankings, because the wind figure and the force-decomposition figure want
different trajectories:

  --mode wind     lateral drift while airborne, perpendicular to the initial
                  horizontal velocity, in block widths. This is the deflection
                  wind causes, separated from how hard the cube was thrown.

  --mode force    contact richness: number of distinct contact episodes, how
                  long the cube slides, and how much the orientation changes
                  during contact. A trajectory that lands flat and stops is
                  useless for showing friction.

USAGE
    python pick_trajectory.py --data data/mojoco_paper_replica_20_wind \\
                             --range 454-568 --mode wind --top 10
"""

import argparse
import os

import numpy as np
import torch

from train_force_gns import build_force_dataset
from generate_node_states import mesh_cube_surface, BLOCK_HALF_WIDTH

BLOCK_W = BLOCK_HALF_WIDTH * 2.0
CONTACT_EPS = 0.004          # metres; a node this close to z=0 counts as touching


# ----------------------------------------------------------------------

def node_heights(com, R, rest):
    """Lowest node height at every frame. com (T,3), R (T,3,3), rest (N,3)."""
    world = com[:, None, :] + torch.einsum("tij,nj->tni", R, rest)
    return world[..., 2].min(dim=1).values.numpy(), world.numpy()


def contact_episodes(min_z, eps=CONTACT_EPS):
    """(n_episodes, first_contact_frame, last_contact_frame) from the height trace."""
    on = min_z < eps
    if not on.any():
        return 0, -1, -1
    edges = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    n = len(edges) + (1 if on[0] else 0)
    return int(n), int(np.argmax(on)), int(len(on) - 1 - np.argmax(on[::-1]))


def metrics_for(traj, rest):
    com, R = traj["com"], traj["R"]
    T = com.shape[0]
    min_z, world = node_heights(com, R, rest)
    n_ep, t_first, t_last = contact_episodes(min_z)
    c = com.numpy()

    # horizontal launch direction, from the first few frames
    v0 = c[min(3, T - 1), :2] - c[0, :2]
    speed0 = float(np.linalg.norm(v0))
    if speed0 < 1e-9:
        along = np.array([1.0, 0.0])
    else:
        along = v0 / speed0
    perp = np.array([-along[1], along[0]])

    # t_first == -1 means it never touches (airborne throughout);
    # t_first == 0 means it starts in contact (no airborne phase at all).
    air_end = T if t_first < 0 else t_first
    rel = c[:air_end, :2] - c[0, :2]
    lateral = float(np.abs(rel @ perp).max()) / BLOCK_W if air_end > 1 else 0.0
    forward = float(np.abs(rel @ along).max()) / BLOCK_W if air_end > 1 else 0.0

    # how far it travels horizontally AFTER first touchdown - the sliding phase
    slide = (float(np.linalg.norm(c[-1, :2] - c[t_first, :2])) / BLOCK_W
             if t_first >= 0 else 0.0)

    # orientation change during contact, in degrees
    if t_first >= 0 and t_last > t_first:
        dR = R[t_last] @ R[t_first].transpose(-1, -2)
        cos = float(torch.clamp((torch.diagonal(dR).sum() - 1) / 2, -1, 1))
        spin = float(np.degrees(np.arccos(cos)))
    else:
        spin = 0.0

    wind = traj.get("wind", torch.zeros(3))
    return dict(T=T, n_episodes=n_ep, t_contact=t_first, t_last=t_last,
                airborne=(T if t_first < 0 else int(t_first)),
                lateral=lateral, forward=forward, slide=slide, spin=spin,
                wind_speed=float(torch.norm(wind)),
                drift_ratio=lateral / max(forward, 1e-6))


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="trajectory folder")
    ap.add_argument("--range", default="454-568",
                    help="START-STOP, e.g. the test split 454-568")
    ap.add_argument("--mode", choices=("wind", "force"), default="wind")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--nodes-per-edge", type=int, default=2)
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--unscale", action="store_true")
    a = ap.parse_args()

    start, stop = (int(v) for v in a.range.split("-"))
    idx = list(range(start, stop))
    rest = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, a.nodes_per_edge),
                        dtype=torch.float32)

    trajs, _ = build_force_dataset(idx, a.data, weights_only=a.weights_only,
                                   unscale_data=a.unscale, verbose_every=0)

    rows = []
    for n, tr in zip(idx, trajs):
        try:
            m = metrics_for(tr, rest)
            m["traj"] = n
            rows.append(m)
        except Exception as exc:
            print(f"  traj {n} skipped: {type(exc).__name__}: {exc}")

    if not rows:
        print("nothing to rank")
        return

    if a.mode == "wind":
        # deflection perpendicular to launch, and prefer a long flight so the
        # curvature has room to show
        for m in rows:
            m["score"] = m["lateral"] * np.sqrt(max(m["airborne"], 1))
        cols = [("traj", "traj", "{:>5d}"), ("score", "score", "{:8.2f}"),
                ("lateral", "lateral", "{:8.2f}"), ("forward", "fwd", "{:7.2f}"),
                ("drift_ratio", "lat/fwd", "{:8.3f}"),
                ("airborne", "airb", "{:>5d}"), ("wind_speed", "wind", "{:7.2f}")]
        header = ("ranked by lateral drift while airborne "
                  "(block widths, perpendicular to launch)")
    else:
        # contact richness: several impacts, real sliding, real rotation
        for m in rows:
            m["score"] = (m["n_episodes"] * 1.0 + m["slide"] * 2.0
                          + m["spin"] / 45.0)
        cols = [("traj", "traj", "{:>5d}"), ("score", "score", "{:8.2f}"),
                ("n_episodes", "bounces", "{:>8d}"), ("slide", "slide", "{:7.2f}"),
                ("spin", "spin deg", "{:9.1f}"),
                ("t_contact", "t_cont", "{:>7d}"), ("T", "T", "{:>5d}")]
        header = "ranked by contact richness (bounces, sliding distance, rotation)"

    rows.sort(key=lambda m: -m["score"])
    print(f"\n{header}\n")
    print("  " + "  ".join(f"{lab:>8}" for _, lab, _ in cols))
    for m in rows[:a.top]:
        print("  " + "  ".join(fmt.format(m[key]) for key, _, fmt in cols))

    best = [m["traj"] for m in rows[:3]]
    print(f"\n  look at these three before deciding: {best}")
    print("  then put the rule in the caption, e.g. "
          f"\"{'largest lateral drift' if a.mode == 'wind' else 'most contact-rich'} "
          "trajectory in the held-out test set\".\n")


if __name__ == "__main__":
    main()

"""
compare_models_gif.py

Roll out TWO models on the SAME test trajectories and save:

    <idx>_feature.gif        model A alone (prediction vs truth)
    <idx>_no_feature.gif     model B alone (prediction vs truth)
    <idx>_overlay.gif        truth + BOTH predictions in one view, with the
                             wind arrow and the assist ratio in the title
                             (this is the one to put on a slide)

Also prints per-trajectory metrics for both models side by side, plus the wind
magnitude and assist ratio so you know which regime each demo is showing.

Pick trajectory indices from wind_error_<dataset>_per_traj.csv -- it has
wind_mag and kappa_assist per index, so you can choose a strong tailwind
(kappa > 1, cube never stops) or a headwind (kappa < 0) on purpose.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import wall
from train_gnn_multi_step import GNSModel
from generate_node_states import mesh_cube_surface, BLOCK_HALF_WIDTH
from evaluate_metrics import compute_metrics, compute_phase_boundaries, BLOCK_WIDTH
from display_results import (rollout_trajectory_feedback_shape_match, animate_cube)

# ======================================================================
# CONFIG
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER  = os.path.join(script_dir, "data/mojoco_paper_replica_20_wind")
TRAJ_INDICES = [454, 500, 550]          # which test trajectories to render
OUT_DIR      = os.path.join(script_dir, "gif_comparisons")
WEIGHTS_ONLY = False
UNSCALE      = False

MODEL_A = dict(tag="feature",
               label="wind feature ON",
               model_path=os.path.join(script_dir, "models/CHANGE_ME_ON/256_train_gns_model_best_model.pt"),
               norms_path=os.path.join(script_dir, "models/CHANGE_ME_ON/256_train_gns_model_norms.pt"),
               use_wind=True, color="tab:blue")
MODEL_B = dict(tag="no_feature",
               label="wind feature OFF",
               model_path=os.path.join(script_dir, "models/CHANGE_ME_OFF/256_train_gns_model_best_model.pt"),
               norms_path=os.path.join(script_dir, "models/CHANGE_ME_OFF/256_train_gns_model_norms.pt"),
               use_wind=False, color="tab:red")

# physics (for the assist ratio shown in the overlay title)
DT      = 1.0 / 148.0
K_OVER_M = 0.02847        # measured by wind_error_analysis.py calibration (1/m)
MU      = 1.9 / 9.615
GRAV    = 9.615

H, NODES_PER_EDGE, K_NN = 3, 2, 3
LATENT, L_MP, K_REP = 128, 5, 1
INTERVAL = 60             # ms per frame in the saved gif
MAKE_SINGLE_GIFS = True
MAKE_OVERLAY_GIF = True

# ======================================================================
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))
nodes_body = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE),
                          dtype=torch.float32)


def load_model(entry):
    norms = torch.load(entry["norms_path"], weights_only=False)
    model = GNSModel(norms["x_mean"].shape[0], norms["e_mean"].shape[0],
                     latent_dim=LATENT, L=L_MP, K=K_REP)
    sd = torch.load(entry["model_path"], map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, norms


def rollout(entry, model, norms, idx):
    return rollout_trajectory_feedback_shape_match(
        DATA_FOLDER, model, Floor, throw_number=idx,
        nodes_per_edge=NODES_PER_EDGE, nearest_neighbors=K_NN,
        rest_positions=nodes_body,
        accel_std=norms["acc_std"], accel_mean=norms["acc_mean"],
        x_mean=norms["x_mean"], x_std=norms["x_std"],
        e_mean=norms["e_mean"], e_std=norms["e_std"],
        do_shape_match=True, shape_alpha=1.0, return_edge_info=True,
        weights_only_load=WEIGHTS_ONLY, unscale_trajectory_data=UNSCALE,
        h=H, use_wind=entry["use_wind"])


def trajectory_wind(idx):
    raw = torch.load(os.path.join(DATA_FOLDER, f"{idx}.pt"), weights_only=WEIGHTS_ONLY)
    w = np.asarray(raw[1], dtype=float).reshape(3)
    if np.allclose(w, 0) and len(raw) > 3 and isinstance(raw[3], dict):
        w = np.asarray(raw[3].get("wind", np.zeros(3)), dtype=float).reshape(3)
    return w


def assist_ratio(wind, true_pos, tc):
    """kappa at contact entry: drag along the slide / friction."""
    cm = true_pos.mean(dim=1).cpu().numpy()
    if tc < 2 or tc >= len(cm):
        return float("nan")
    v = (cm[tc - 1] - cm[tc - 2]) / DT
    s = v[:2]
    n = np.linalg.norm(s)
    if n < 1e-6:
        return float("nan")
    s_hat = s / n
    u = (wind - v)[:2]
    return K_OVER_M * np.linalg.norm(u) * float(np.dot(u, s_hat)) / (MU * GRAV)


# ----------------------------------------------------------------------
# Overlay animation: truth + both predictions in one 3D view
# ----------------------------------------------------------------------
def animate_overlay(true_pos, preds, labels, colors, edge_index, wind,
                    save_path, title="", interval=INTERVAL):
    true_pos = true_pos.cpu().numpy()
    preds = [p.cpu().numpy() for p in preds]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    E = edge_index.shape[1]

    allp = np.concatenate([true_pos] + preds, axis=0)
    lo, hi = allp.reshape(-1, 3).min(0), allp.reshape(-1, 3).max(0)
    pad = 0.1 * max(hi - lo)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(min(0.0, lo[2]), hi[2] + pad)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

    bodies, handles = [], []
    for series, lab, col, lw, alpha in (
            [(true_pos, "ground truth", "0.35", 2.4, 1.0)]
            + [(p, l, c, 1.8, 0.95) for p, l, c in zip(preds, labels, colors)]):
        lc = Line3DCollection([[(0, 0, 0), (0, 0, 0)]] * E,
                              colors=col, linewidths=lw, alpha=alpha)
        ax.add_collection3d(lc)
        bodies.append((series, lc))
        handles.append(plt.Line2D([0], [0], color=col, lw=lw, label=lab))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    # wind arrow (fixed, drawn at the corner of the domain)
    wmag = float(np.linalg.norm(wind))
    if wmag > 1e-6:
        span = float(max(hi[:2] - lo[:2]))
        base = np.array([lo[0], lo[1], max(0.0, lo[2])])
        d = np.array([wind[0], wind[1], 0.0]) / wmag * 0.25 * span
        ax.quiver(base[0], base[1], base[2] + 0.02, d[0], d[1], 0.0,
                  color="tab:green", linewidth=2.5, arrow_length_ratio=0.25)
        ax.text(base[0] + d[0], base[1] + d[1], base[2] + 0.02,
                f"  wind {wmag:.1f} m/s", color="tab:green", fontsize=9)

    ttl = ax.set_title(title, fontsize=11)

    def update(t):
        for series, lc in bodies:
            p = series[min(t, len(series) - 1)]
            lc.set_segments([[p[edge_index[0, e]], p[edge_index[1, e]]] for e in range(E)])
        ttl.set_text(f"{title}    frame {t}")
        return [lc for _, lc in bodies]

    n_frames = min([len(true_pos)] + [len(p) for p in preds])
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval, blit=False)
    ani.save(save_path, writer="pillow", fps=max(1, 1000 // interval))
    plt.close(fig)
    print(f"    saved {save_path}")


# ======================================================================
mA, nA = load_model(MODEL_A)
mB, nB = load_model(MODEL_B)

print(f"{'idx':>6} {'|wind|':>7} {'kappa':>7} "
      f"{MODEL_A['tag'] + ' center':>18} {MODEL_B['tag'] + ' center':>18} {'angle A/B':>16}")
print("-" * 84)

for idx in TRAJ_INDICES:
    wind = trajectory_wind(idx)
    predA, true, edgeA = rollout(MODEL_A, mA, nA, idx)
    predB, _, edgeB = rollout(MODEL_B, mB, nB, idx)

    tc, ts = compute_phase_boundaries(true)
    kappa = assist_ratio(wind, true, int(tc))
    mA_m = compute_metrics(predA, true, nodes_body)
    mB_m = compute_metrics(predB, true, nodes_body)

    regime = ("headwind" if kappa < -0.05 else
              "tailwind, never stops" if kappa > 1.0 else
              "tailwind" if kappa > 0.05 else "crosswind")
    print(f"{idx:>6} {np.linalg.norm(wind):>7.2f} {kappa:>7.2f} "
          f"{mA_m['center_error']:>18.4f} {mB_m['center_error']:>18.4f} "
          f"{mA_m['angle_error_deg']:>7.2f}/{mB_m['angle_error_deg']:<8.2f}")

    if MAKE_SINGLE_GIFS:
        animate_cube(predA, true, edge_info=edgeA, interval=INTERVAL,
                     save_path=os.path.join(OUT_DIR, f"{idx}_{MODEL_A['tag']}.gif"))
        animate_cube(predB, true, edge_info=edgeB, interval=INTERVAL,
                     save_path=os.path.join(OUT_DIR, f"{idx}_{MODEL_B['tag']}.gif"))

    if MAKE_OVERLAY_GIF:
        title = (f"traj {idx}   |wind| {np.linalg.norm(wind):.1f} m/s   "
                 f"kappa {kappa:.2f} ({regime})")
        animate_overlay(true, [predA, predB],
                        [MODEL_A["label"], MODEL_B["label"]],
                        [MODEL_A["color"], MODEL_B["color"]],
                        edgeA["edge_index"], wind,
                        os.path.join(OUT_DIR, f"{idx}_overlay.gif"),
                        title=title)

print(f"\nGIFs written to {OUT_DIR}/")

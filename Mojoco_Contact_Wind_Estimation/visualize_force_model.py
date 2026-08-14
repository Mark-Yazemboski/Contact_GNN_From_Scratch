"""
visualize_force_model.py

NEW FILE - does not modify any existing code.

Same job as animate_cube() in display_results.py (red predicted cube, blue
ground-truth cube, wireframe edges, 3D animation, optional GIF), but for the
force model - and it additionally draws the forces the model is predicting:

  * TWO arrows per node in contact:
      - NORMAL    (green)  along the wall normal, >= 0 by construction
      - TANGENTIAL(orange) in the floor plane - this is the friction force
    Nodes the contact gate has switched off draw nothing, so you can watch
    corners turn on and off as the cube tumbles.
  * ONE arrow at the COM (magenta) for the fluid force (learned residual plus
    the analytic drag baseline if that was enabled). No torque arrow.

ARROW SCALE IS PHYSICAL, NOT PER-FRAME NORMALIZED. A force equal to the cube's
weight (m g) is drawn MG_ARROW_WIDTHS block-widths long, so arrow length means
the same thing in every frame and every trajectory. The HUD prints the summed
normal force as a multiple of m g: when the cube comes to rest that number
should sit near 1.0, which is a free sanity check on the model that costs
nothing to look at.

Rollout only - the cube follows the model's own predictions and the forces are
whatever it predicts as it drifts.

USAGE: set the CONFIG block and run.  python visualize_force_model.py
"""

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import wall
from force_gns import ForceGNSModel, BLOCK_WIDTH
from train_force_gns import build_force_dataset, rollout_force_batched
from generate_node_states import mesh_cube_surface, knn_adjacency, BLOCK_HALF_WIDTH

# ======================================================================
# CONFIG
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")
MODEL_FOLDER = os.path.join(script_dir, "models/force_stage1")
MODEL_PREFIX = None          # None -> auto-detect the single "*_norms.pt" in MODEL_FOLDER
TRAJECTORY = 460             # which trajectory to animate
WEIGHTS_ONLY = False
UNSCALE = False

SAVE_PATH = os.path.join(MODEL_FOLDER, f"force_rollout_{TRAJECTORY}.gif")
SHOW = True                  # also open the interactive window
INTERVAL = 50                # ms per frame (GIF fps = 1000 // INTERVAL)

# --- arrow appearance ---
MG_ARROW_WIDTHS = 1.0        # a force of m*g draws this many block-widths long
MIN_ARROW_FRAC = 0.01        # skip arrows below this fraction of m*g (declutter)
DRAW_FLOOR = True
DRAW_GROUND_TRUTH = True

C_NORMAL, C_TANGENT, C_FLUID = "tab:green", "tab:orange", "magenta"

# ======================================================================
# Load model + config
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

if MODEL_PREFIX is None:
    cands = [f[:-len("_norms.pt")] for f in os.listdir(MODEL_FOLDER)
             if f.endswith("_norms.pt")]
    assert len(cands) == 1, f"set MODEL_PREFIX explicitly, found: {cands}"
    MODEL_PREFIX = cands[0]

norms = torch.load(os.path.join(MODEL_FOLDER, MODEL_PREFIX + "_norms.pt"),
                   weights_only=False)
cfg = norms["force_cfg"]
h, dt, mass, g = cfg["h"], cfg["dt"], cfg["mass"], cfg["gravity"]
MG = mass * g
print(f"Model: {MODEL_PREFIX}   dt={dt:.6f}s  m={mass}kg  g={g}  "
      f"drag_baseline={cfg['use_drag_baseline']}  use_wind={cfg['use_wind']}")

rest_nodes = torch.tensor(
    mesh_cube_surface(BLOCK_HALF_WIDTH * 2, cfg["nodes_per_edge"]), dtype=torch.float32)
edge_index = torch.tensor(
    knn_adjacency(rest_nodes.numpy(), k=cfg["nearest_neighbors"]), dtype=torch.long)

model = ForceGNSModel(norms["x_mean"].shape[0], norms["e_mean"].shape[0],
                      latent_dim=cfg["latent_dim"], L=cfg["L"], K=cfg["K"])
sd = torch.load(os.path.join(MODEL_FOLDER, MODEL_PREFIX + "_best_model.pt"),
                map_location=device, weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in sd):
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
model.load_state_dict(sd)
model.to(device).eval()

# ======================================================================
# Roll out the one trajectory, keeping the forces
# ======================================================================
trajs, _ = build_force_dataset([TRAJECTORY], DATA_FOLDER,
                               weights_only=WEIGHTS_ONLY, unscale_data=UNSCALE,
                               verbose_every=0)
trajs[0]["edge_index"] = edge_index
g_step = torch.tensor([0.0, 0.0, -g]) * dt * dt

center, angle, forces, (per_traj, pred_all, true_all, lengths) = rollout_force_batched(
    model, trajs, Floor, h, rest_nodes,
    norms["x_mean"], norms["x_std"], norms["e_mean"], norms["e_std"],
    cfg["scale_vec"], cfg["ang_scale_vec"], g_step, dt, device,
    use_wind=cfg["use_wind"], use_drag_baseline=cfg["use_drag_baseline"],
    k_over_m=cfg["k_over_m"], contact_d0=cfg["contact_d0"],
    contact_tau=cfg["contact_tau"], mass=mass,
    return_forces=True, return_per_traj=True)

L = lengths[0] - h                              # real (unpadded) frames in this view
pred = pred_all[0, :L]                          # (L, N, 3)
true = true_all[0, :L]
m0 = per_traj[0]
t_contact, t_settle = int(m0["t_contact"]), int(m0["t_settle"])
print(f"traj {TRAJECTORY}: center {m0['center_error']:.4f} widths | "
      f"angle {m0['angle_error_deg']:.2f} deg | contact@{t_contact} settle@{t_settle}")

# Force arrays are per PREDICTION STEP. Step i is computed at the state
# pred[h + i] and produces pred[h + 1 + i]. So frame f has forces iff
# f >= h, and its force index is (f - h).
f_norm = forces["node_normal"][0]               # (n_steps, N, 3) Newtons
f_tang = forces["node_tangent"][0]
f_fluid = forces["F_fluid"][0]                  # (n_steps, 3) Newtons
n_steps = f_norm.shape[0]
assert n_steps >= L - h - 1, "force/frame alignment mismatch"

# ======================================================================
# Figure setup (mirrors animate_cube)
# ======================================================================
ARROW_SCALE = MG_ARROW_WIDTHS * BLOCK_WIDTH / MG      # meters of arrow per Newton
MIN_F = MIN_ARROW_FRAC * MG

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ei = edge_index.numpy()
all_pos = torch.cat([pred, true], dim=0) if DRAW_GROUND_TRUTH else pred
pad = 0.5 * BLOCK_WIDTH
ax.set_xlim(float(all_pos[:, :, 0].min()) - pad, float(all_pos[:, :, 0].max()) + pad)
ax.set_ylim(float(all_pos[:, :, 1].min()) - pad, float(all_pos[:, :, 1].max()) + pad)
ax.set_zlim(min(0.0, float(all_pos[:, :, 2].min())) - 0.2 * pad,
            float(all_pos[:, :, 2].max()) + pad)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

if DRAW_FLOOR:
    span = float(max(all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
                     all_pos[:, :, 1].max() - all_pos[:, :, 1].min())) + 2 * pad
    centre = (float(all_pos[:, :, 0].mean()), float(all_pos[:, :, 1].mean()), 0.0)
    wall.wall(center_position=centre, size=(span, span),
              normal=(0, 0, 1)).show(ax, color="gray", alpha=0.15)

pred_scatter = ax.scatter([], [], [], c='r', s=25, label='Pred')
gt_scatter = (ax.scatter([], [], [], c='b', s=25, alpha=0.4, label='GT')
              if DRAW_GROUND_TRUTH else None)
pred_edge_lines = [ax.plot([], [], [], c='r', alpha=0.35, linewidth=1.0)[0]
                   for _ in range(ei.shape[1])]
gt_edge_lines = ([ax.plot([], [], [], c='b', alpha=0.2, linewidth=1.0)[0]
                  for _ in range(ei.shape[1])] if DRAW_GROUND_TRUTH else [])

# legend proxies for the arrow colors
ax.plot([], [], [], c=C_NORMAL, lw=2, label='contact normal')
ax.plot([], [], [], c=C_TANGENT, lw=2, label='contact tangential')
ax.plot([], [], [], c=C_FLUID, lw=2, label='fluid @ COM')
ax.legend(loc='upper left', fontsize=8)

hud = fig.text(0.015, 0.015, "", fontsize=9, family='monospace', va='bottom')
ax.set_title(f"traj {TRAJECTORY} - red=pred, blue=GT | "
             f"arrow: {MG_ARROW_WIDTHS:g} width = m g = {MG:.2f} N")

quivers = []            # mutable holder so update() can clear last frame's arrows


def _phase(frame):
    if frame < t_contact:
        return "airborne"
    return "contact" if frame < t_settle else "settled"


def update(frame):
    p = pred[frame].numpy()
    pred_scatter._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
    for i in range(ei.shape[1]):
        s, d = int(ei[0, i]), int(ei[1, i])
        pred_edge_lines[i].set_data([p[s, 0], p[d, 0]], [p[s, 1], p[d, 1]])
        pred_edge_lines[i].set_3d_properties([p[s, 2], p[d, 2]])

    if DRAW_GROUND_TRUTH:
        t = true[frame].numpy()
        gt_scatter._offsets3d = (t[:, 0], t[:, 1], t[:, 2])
        for i in range(ei.shape[1]):
            s, d = int(ei[0, i]), int(ei[1, i])
            gt_edge_lines[i].set_data([t[s, 0], t[d, 0]], [t[s, 1], t[d, 1]])
            gt_edge_lines[i].set_3d_properties([t[s, 2], t[d, 2]])

    # ---- arrows: clear previous frame, redraw ----
    for q in quivers:
        q.remove()
    quivers.clear()

    k = frame - h                       # force index for this frame's state
    if 0 <= k < n_steps:
        fn = f_norm[k].numpy()
        ft = f_tang[k].numpy()
        ff = f_fluid[k].numpy()

        for vecs, color in ((fn, C_NORMAL), (ft, C_TANGENT)):
            mags = np.linalg.norm(vecs, axis=1)
            sel = mags > MIN_F
            if sel.any():
                quivers.append(ax.quiver(
                    p[sel, 0], p[sel, 1], p[sel, 2],
                    vecs[sel, 0] * ARROW_SCALE,
                    vecs[sel, 1] * ARROW_SCALE,
                    vecs[sel, 2] * ARROW_SCALE,
                    color=color, linewidth=1.8, arrow_length_ratio=0.25))

        com = p.mean(axis=0)
        if np.linalg.norm(ff) > MIN_F:
            quivers.append(ax.quiver(
                com[0], com[1], com[2],
                ff[0] * ARROW_SCALE, ff[1] * ARROW_SCALE, ff[2] * ARROW_SCALE,
                color=C_FLUID, linewidth=2.2, arrow_length_ratio=0.25))

        n_active = int((np.linalg.norm(fn, axis=1) > MIN_F).sum())
        hud.set_text(
            f"frame {frame:3d}/{L-1}  [{_phase(frame)}]\n"
            f"sum |normal| = {np.linalg.norm(fn.sum(axis=0)) / MG:5.2f} m g   "
            f"({n_active} node{'s' if n_active != 1 else ''} in contact)\n"
            f"sum |tangent| = {np.linalg.norm(ft.sum(axis=0)) / MG:5.2f} m g   "
            f"|fluid| = {np.linalg.norm(ff) / MG:5.2f} m g")
    else:
        hud.set_text(f"frame {frame:3d}/{L-1}  [seeded from ground truth]")

    artists = [pred_scatter] + pred_edge_lines + quivers + [hud]
    if gt_scatter is not None:
        artists += [gt_scatter] + gt_edge_lines
    return tuple(artists)


ani = animation.FuncAnimation(fig, update, frames=L, interval=INTERVAL, blit=False)
try:
    ax.set_aspect('equal')
except Exception:
    pass                                     # older matplotlib 3d has no equal aspect

if SAVE_PATH is not None:
    print(f"Saving animation to {SAVE_PATH} ...")
    ani.save(SAVE_PATH, writer='pillow', fps=max(1, 1000 // INTERVAL))
    print("Saved successfully.")

if SHOW:
    plt.show()

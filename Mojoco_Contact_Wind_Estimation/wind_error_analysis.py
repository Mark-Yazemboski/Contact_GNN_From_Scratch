"""
wind_error_analysis.py

Per-trajectory error vs wind analysis for the contact+wind datasets.

The organizing variable is NOT |wind| -- it is the ASSIST RATIO:

    kappa_assist = (drag force component along the slide direction)
                   / (friction force)
                 = (k/m) * |u_rel| * (u_rel . s_hat) / (mu * g)

    where u_rel = w_wind - v_cube  (relative flow at contact entry)
          s_hat = unit slide direction at contact entry
          k     = 0.5 * rho * Cd * A   (CALIBRATED FROM THE DATA, see below)

  kappa_assist  <  0   wind opposes the slide  -> stops sooner
  kappa_assist ~= 0    crosswind / no wind     -> nominal
  kappa_assist ->  1   drag cancels friction   -> marginal, ill-conditioned
  kappa_assist  >  1   drag exceeds friction   -> cube sleds indefinitely

Drag calibration: during the airborne phase the only horizontal force is drag,
so  a_horiz = (k/m) |u_rel| u_rel_horiz.  We least-squares fit k/m over all
airborne frames of the test set -- no guessing at MuJoCo's ellipsoid Cd.

Supports SEVERAL models at once (e.g. wind-feature ON vs OFF, or single-step vs
multistep). With exactly two, it also plots the paired per-trajectory delta
against the assist ratio -- i.e. "does the feature help where physics says it
should?", which pooled means cannot answer.

Outputs: printed binned tables, <prefix>_per_traj.csv, <prefix>_scatter.png
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wall
from train_gnn_multi_step import GNSModel
from generate_node_states import mesh_cube_surface, BLOCK_HALF_WIDTH
from evaluate_metrics import compute_metrics, compute_phase_boundaries, BLOCK_WIDTH
from display_results import rollout_trajectory_feedback_shape_match

# ======================================================================
# CONFIG
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(script_dir, "data/mojoco_paper_replica_20_wind")
TEST_INDICES = range(454, 568)
WEIGHTS_ONLY = False        # MuJoCo-generated data
UNSCALE      = False

# One entry per model. use_wind must match how that model was trained.
MODELS = [
    dict(label="feature ON",
         model_path=os.path.join(script_dir, "models/CHANGE_ME_ON/256_train_gns_model_best_model.pt"),
         norms_path=os.path.join(script_dir, "models/CHANGE_ME_ON/256_train_gns_model_norms.pt"),
         use_wind=True),
    dict(label="feature OFF",
         model_path=os.path.join(script_dir, "models/CHANGE_ME_OFF/256_train_gns_model_best_model.pt"),
         norms_path=os.path.join(script_dir, "models/CHANGE_ME_OFF/256_train_gns_model_norms.pt"),
         use_wind=False),
]

DT   = 1.0 / 148.0          # replica record rate
MU   = 1.9 / 9.615          # the tuned replica friction (matches the generator)
GRAV = 9.615
MASS = 0.37

H, NODES_PER_EDGE, K_NN = 3, 2, 3
LATENT, L_MP, K_REP = 128, 5, 1
MIN_CONTACT_FRAMES = 8
OUT_PREFIX = os.path.join(script_dir, "wind_error_" + os.path.basename(DATA_FOLDER))

# ======================================================================
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


def load_raw(idx):
    raw = torch.load(os.path.join(DATA_FOLDER, f"{idx}.pt"), weights_only=WEIGHTS_ONLY)
    states = raw[0].float()
    wind = np.zeros(3)
    if len(raw) > 1:
        try:
            wind = np.asarray(raw[1], dtype=float).reshape(3)
        except Exception:
            pass
    if np.allclose(wind, 0) and len(raw) > 3 and isinstance(raw[3], dict):
        wind = np.asarray(raw[3].get("wind", np.zeros(3)), dtype=float).reshape(3)
    return states, wind


def com_kinematics(states):
    """COM position/velocity/acceleration in SI units."""
    cm = states[:, :3].numpy().astype(np.float64)
    v = (cm[1:] - cm[:-1]) / DT                                   # v[t] ~ t+1/2
    a = (cm[2:] - 2 * cm[1:-1] + cm[:-2]) / DT ** 2               # a[t] -> frame t+1
    return cm, v, a


# ======================================================================
# PASS 1 - calibrate drag from airborne frames (a_horiz = (k/m)|u|u)
# ======================================================================
print("Calibrating drag coefficient from airborne frames...")
num = den = 0.0
n_frames = 0
for idx in TEST_INDICES:
    try:
        states, wind = load_raw(idx)
    except FileNotFoundError:
        continue
    if np.linalg.norm(wind) < 1e-6:
        continue
    T = states.shape[0]
    corners = []
    for t in range(T):
        from generate_node_states import quat_to_rotmat
        R = quat_to_rotmat(states[t, 3:7])
        corners.append((R @ nodes_body.T).T + states[t, :3])
    tc, _ = compute_phase_boundaries(torch.stack(corners))
    cm, v, a = com_kinematics(states)
    for t in range(1, min(int(tc) - 1, len(a))):
        u = wind - v[t]                       # relative flow
        p = np.linalg.norm(u) * u             # |u| u
        num += float(np.dot(a[t][:2], p[:2]))
        den += float(np.dot(p[:2], p[:2]))
        n_frames += 1
k_over_m = num / den if den > 0 else 0.0
print(f"  fitted k/m = {k_over_m:.5f} 1/m   from {n_frames} airborne frames")
print(f"  -> implied Cd*A = {2 * k_over_m * MASS / 1.225:.5f} m^2 "
      f"(sphere of r=0.0524 has A = {np.pi * 0.0524**2:.5f} m^2)")
print(f"  -> terminal/blow-away wind speed = "
      f"{np.sqrt(MU * GRAV / max(k_over_m, 1e-9)):.2f} m/s")


def assist_ratio(wind, v_slide_vec):
    """kappa along the slide direction at contact entry."""
    s = v_slide_vec[:2]
    sn = np.linalg.norm(s)
    if sn < 1e-6:
        return np.nan, np.nan, np.nan, np.nan
    s_hat = s / sn
    u = (wind - v_slide_vec)[:2]              # relative flow, horizontal
    drag_along = k_over_m * np.linalg.norm(u) * float(np.dot(u, s_hat))
    kappa = drag_along / (MU * GRAV)          # signed: >0 assists the slide
    w_along = float(np.dot(wind[:2], s_hat))  # simple along-slide wind component
    cos_th = w_along / (np.linalg.norm(wind[:2]) + 1e-9)
    return kappa, w_along, cos_th, sn


# ======================================================================
# PASS 2 - rollouts + geometry
# ======================================================================
loaded = [(m["label"], m["use_wind"]) + load_model(m) for m in MODELS]
rows = []

for n_done, idx in enumerate(TEST_INDICES, 1):
    try:
        states, wind = load_raw(idx)
    except FileNotFoundError:
        continue
    T = states.shape[0]
    cm, v, a = com_kinematics(states)

    r = dict(idx=idx, wind_mag=float(np.linalg.norm(wind)))

    got_phase = False
    for label, use_wind, model, norms in loaded:
        pred, true, _ = rollout_trajectory_feedback_shape_match(
            DATA_FOLDER, model, Floor, throw_number=idx,
            nodes_per_edge=NODES_PER_EDGE, nearest_neighbors=K_NN,
            rest_positions=nodes_body,
            accel_std=norms["acc_std"], accel_mean=norms["acc_mean"],
            x_mean=norms["x_mean"], x_std=norms["x_std"],
            e_mean=norms["e_mean"], e_std=norms["e_std"],
            do_shape_match=True, shape_alpha=1.0, return_edge_info=True,
            weights_only_load=WEIGHTS_ONLY, unscale_trajectory_data=UNSCALE,
            h=H, use_wind=use_wind)
        m = compute_metrics(pred, true, nodes_body)
        tc, ts = int(m["t_contact"]), int(m["t_settle"])

        if not got_phase:
            got_phase = True
            if tc < 1 or tc >= T - 2:
                break
            v_entry = v[max(0, tc - 1)]
            kappa, w_along, cos_th, slide_speed = assist_ratio(wind, v_entry)
            hi = min(ts, T)
            path_m = float(np.linalg.norm(np.diff(cm[tc:hi, :2], axis=0), axis=1).sum()) \
                if hi > tc + 1 else np.nan
            r.update(t_contact=tc, t_settle=ts, kappa_assist=kappa,
                     w_along=w_along, cos_theta=cos_th, slide_speed=slide_speed,
                     path_m=path_m, path_widths=path_m / BLOCK_WIDTH,
                     settles=int(ts < T - 2))

        ce = m["center_error_t"].numpy()
        seg = ce[tc:ts]
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) \
            if len(seg) >= MIN_CONTACT_FRAMES else np.nan
        r[f"center[{label}]"] = float(m["center_error"])
        r[f"contact[{label}]"] = float(m["center_error_contact"])
        r[f"angle[{label}]"] = float(m["angle_error_deg"])
        r[f"slope[{label}]"] = slope
        pw = r.get("path_widths", np.nan)
        r[f"norm_contact[{label}]"] = (float(m["center_error_contact"]) / pw
                                       if pw and np.isfinite(pw) and pw > 0.1 else np.nan)

    if "t_contact" in r:
        rows.append(r)
    if n_done % 25 == 0:
        print(f"  {n_done}/{len(TEST_INDICES)}", flush=True)

labels = [m["label"] for m in MODELS]

# ======================================================================
# Binned tables
# ======================================================================
def binned(key, edges, value_key):
    print(f"\n{'bin ' + key:>22} {'n':>4} " + "".join(f"{l:>16}" for l in labels))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi]
        if not sel:
            continue
        line = f"{f'[{lo:g},{hi:g})':>22} {len(sel):>4} "
        for l in labels:
            vals = [r[f"{value_key}[{l}]"] for r in sel
                    if np.isfinite(r.get(f"{value_key}[{l}]", np.nan))]
            line += f"{np.mean(vals):>16.4f}" if vals else f"{'-':>16}"
        print(line)

print("\n" + "=" * 78)
print(f"BINNED RESULTS  ({len(rows)} trajectories, {os.path.basename(DATA_FOLDER)})")
print("=" * 78)
kappa_edges = [-2, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1.0, 1.5, 3, 10]
wind_edges = [0, 2, 4, 6, 8, 10, 14, 20, 30]
print("\n--- mean center error by |wind| ---")
binned("wind_mag", wind_edges, "center")
print("\n--- mean center error by ASSIST RATIO (kappa along slide) ---")
binned("kappa_assist", kappa_edges, "center")
print("\n--- PATH-NORMALIZED contact drift by assist ratio ---")
binned("kappa_assist", kappa_edges, "norm_contact")

settle_frac = np.mean([r["settles"] for r in rows])
print(f"\nfraction of trajectories that settle: {settle_frac:.2f}")
if len(labels) == 2:
    a, b = labels
    d = [(r["kappa_assist"], r[f"center[{a}]"] - r[f"center[{b}]"]) for r in rows
         if np.isfinite(r.get("kappa_assist", np.nan))]
    dd = np.array([x[1] for x in d])
    print(f"\npaired delta ({a} - {b}): mean {dd.mean():+.4f} "
          f"+- {dd.std(ddof=1)/np.sqrt(len(dd)):.4f} SEM (n={len(dd)})")
    print("  (negative = first model better)")

# ======================================================================
# Scatter plots
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
K = np.array([r.get("kappa_assist", np.nan) for r in rows])
Wm = np.array([r["wind_mag"] for r in rows])

for l in labels:
    C = np.array([r.get(f"center[{l}]", np.nan) for r in rows])
    axes[0, 0].scatter(Wm, C, s=16, alpha=0.55, label=l)
    axes[0, 1].scatter(K, C, s=16, alpha=0.55, label=l)
    N = np.array([r.get(f"norm_contact[{l}]", np.nan) for r in rows])
    axes[0, 2].scatter(K, N, s=16, alpha=0.55, label=l)
    S = np.array([r.get(f"slope[{l}]", np.nan) for r in rows])
    axes[1, 0].scatter(K, S, s=16, alpha=0.55, label=l)

axes[0, 0].set_xlabel("|wind| (m/s)"); axes[0, 0].set_ylabel("center error (/width)")
axes[0, 0].set_title("error vs wind magnitude")
axes[0, 1].set_xlabel("assist ratio  kappa"); axes[0, 1].set_ylabel("center error (/width)")
axes[0, 1].set_title("error vs assist ratio (kappa=1: drag cancels friction)")
axes[0, 2].set_xlabel("assist ratio  kappa"); axes[0, 2].set_ylabel("contact err / widths travelled")
axes[0, 2].set_title("PATH-NORMALIZED drift vs assist ratio")
axes[1, 0].set_xlabel("assist ratio  kappa"); axes[1, 0].set_ylabel("contact drift slope")
axes[1, 0].set_title("drift slope vs assist ratio")

if len(labels) == 2:
    a, b = labels
    D = np.array([r.get(f"center[{a}]", np.nan) - r.get(f"center[{b}]", np.nan) for r in rows])
    axes[1, 1].scatter(K, D, s=18, alpha=0.6, color="C3")
    axes[1, 1].axhline(0, color="k", lw=1)
    axes[1, 1].set_xlabel("assist ratio  kappa")
    axes[1, 1].set_ylabel(f"{a} - {b}  (center)")
    axes[1, 1].set_title("feature benefit vs regime (negative = feature helps)")

axes[1, 2].hist(K[np.isfinite(K)], bins=30, alpha=0.8)
axes[1, 2].axvline(1.0, color="r", ls="--", label="kappa=1 (marginal)")
axes[1, 2].set_xlabel("assist ratio  kappa"); axes[1, 2].set_title("regime coverage")
axes[1, 2].legend()

for ax in axes.flat:
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0] and ax is not axes[1, 2]:
        ax.legend(fontsize=8)
for ax in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
    ax.axvline(1.0, color="r", ls="--", lw=1)

fig.suptitle(f"Wind error analysis - {os.path.basename(DATA_FOLDER)}")
fig.tight_layout()
fig.savefig(OUT_PREFIX + "_scatter.png", dpi=150)
print(f"\nSaved figure to {OUT_PREFIX}_scatter.png")

keys = sorted({k for r in rows for k in r})
with open(OUT_PREFIX + "_per_traj.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval="")
    w.writeheader(); w.writerows(rows)
print(f"Saved per-trajectory CSV to {OUT_PREFIX}_per_traj.csv")

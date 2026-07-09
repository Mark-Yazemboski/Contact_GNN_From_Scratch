"""
impact_slope_analysis.py  (T2 + T3)

One evaluation pass over an existing test set + checkpoint that produces:

  T3 - toss vs sliding split: overall + per-phase metrics and contact slopes,
       accumulated separately by params['type'] read from each trajectory file.
  T2 - slope vs entry-speed scatter: per-trajectory contact-phase error slope
       plotted against the COM speed entering contact, with a linear fit
       slope = a + b*v on the tosses. The intercept `a` is the
       speed-independent (friction-tracking) component and should match where
       the sliding trajectories sit; the b*v part is the impact-impulse
       component. Fractional impulse resolution = delta_v / v_entry is
       reported per trajectory (delta_v = slope * BLOCK_WIDTH / DT).

Run it once per dataset (DATASET = "paper" or "mojoco"). Put this file in the
same folder as train_gnn_multi_step.py etc. on the VM.

Outputs:
  - printed tables (per-type metrics, slopes, fractional resolution)
  - <OUT_PREFIX>_per_traj.csv        one row per trajectory
  - <OUT_PREFIX>_scatter.png         the T2 figure
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
from evaluate_metrics import (compute_metrics, compute_phase_boundaries,
                              BLOCK_WIDTH)
from display_results import rollout_trajectory_feedback_shape_match

# ======================================================================
# CONFIG - flip DATASET and check the paths/ranges against your run files
# ======================================================================
DATASET = "paper"          # "paper" or "mojoco"

script_dir = os.path.dirname(os.path.abspath(__file__))

if DATASET == "paper":
    TRAJECTORY_FOLDER = os.path.join(script_dir, "data/tosses_processed")
    MODEL_PATH  = os.path.join(script_dir, "models/Paper_Error_Plots/256_train_gns_model_best_model.pt")
    NORMS_PATH  = os.path.join(script_dir, "models/Paper_Error_Plots/256_train_gns_model_norms.pt")
    TEST_INDICES = range(454, 568)      # match run_paper_traj.py test_range
    WEIGHTS_ONLY = True
    UNSCALE      = True
    USE_WIND     = False
    DT           = 1.0 / 148.0          # paper capture rate
else:
    TRAJECTORY_FOLDER = os.path.join(script_dir, "data/mojoco_trajectories_2.5_wind_2048")
    MODEL_PATH  = os.path.join(script_dir, "models/mojoco_pure_sliding_no_wind_fixed_noise/1024_train_gns_model_best_model.pt")
    NORMS_PATH  = os.path.join(script_dir, "models/mojoco_pure_sliding_no_wind_fixed_noise/1024_train_gns_model_norms.pt")
    TEST_INDICES = range(1638, 2047)    # match run_multi_step.py test_range
    WEIGHTS_ONLY = False
    UNSCALE      = False
    USE_WIND     = False
    DT           = 0.0001348 * 50       # MuJoCo timestep * substeps

NODES_PER_EDGE      = 2
K_NEAREST_NEIGHBORS = 3
H                   = 3
LATENT_DIM          = 128
MP_LAYERS           = 5
REPEAT_BLOCKS       = 1

MIN_CONTACT_FRAMES  = 8     # need at least this many frames to fit a slope
OUT_PREFIX          = os.path.join(script_dir, f"analysis_{DATASET}")

# ======================================================================
# Setup (mirrors the run scripts)
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

nodes_body = torch.tensor(
    mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE), dtype=torch.float32)

norms = torch.load(NORMS_PATH, weights_only=False)
x_mean, x_std = norms["x_mean"], norms["x_std"]
e_mean, e_std = norms["e_mean"], norms["e_std"]
acc_mean, acc_std = norms["acc_mean"], norms["acc_std"]

node_dim = x_mean.shape[0]
edge_dim = e_mean.shape[0]
model = GNSModel(node_dim, edge_dim, latent_dim=LATENT_DIM,
                 L=MP_LAYERS, K=REPEAT_BLOCKS)
loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(loaded, dict) and "model_state_dict" in loaded:
    loaded = loaded["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in loaded.keys()):
    loaded = {k.replace("_orig_mod.", "", 1): v for k, v in loaded.items()}
model.load_state_dict(loaded)
model.to(device).eval()


def load_traj_type(idx):
    """'toss' / 'sliding' from the saved params dict; paper files have none."""
    try:
        raw = torch.load(os.path.join(TRAJECTORY_FOLDER, f"{idx}.pt"),
                         weights_only=False)
        if isinstance(raw, (list, tuple)) and len(raw) > 3 and isinstance(raw[3], dict):
            return raw[3].get("type", "toss")
    except Exception:
        pass
    return "toss"


def entry_speed_mps(true_positions, t_contact):
    """COM speed (m/s) entering contact. Tosses: last airborne step(s).
    Sliding (t_contact == 0): initial speed."""
    cm = true_positions.mean(dim=1)                       # (T,3)
    if t_contact >= 3:
        v1 = (cm[t_contact - 1] - cm[t_contact - 2]).norm()
        v2 = (cm[t_contact - 2] - cm[t_contact - 3]).norm()
        v = 0.5 * (v1 + v2)
    elif t_contact >= 2:
        v = (cm[t_contact - 1] - cm[t_contact - 2]).norm()
    else:                                                  # starts in contact
        v = (cm[1] - cm[0]).norm()
    return float(v) / DT


def fit_slope(err_t, a, b):
    """Least-squares slope of err_t[a:b] vs frame index (units: err/frame)."""
    seg = err_t[a:b]
    if len(seg) < MIN_CONTACT_FRAMES:
        return float("nan")
    x = np.arange(len(seg), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(seg, dtype=np.float64), 1)[0])


# ======================================================================
# Main loop: one rollout per test trajectory
# ======================================================================
rows = []
for n_done, idx in enumerate(TEST_INDICES, 1):
    ttype = load_traj_type(idx)
    pred, true, _ = rollout_trajectory_feedback_shape_match(
        TRAJECTORY_FOLDER, model, Floor, throw_number=idx,
        nodes_per_edge=NODES_PER_EDGE, nearest_neighbors=K_NEAREST_NEIGHBORS,
        rest_positions=nodes_body, accel_std=acc_std, accel_mean=acc_mean,
        x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
        do_shape_match=True, shape_alpha=1.0, return_edge_info=True,
        weights_only_load=WEIGHTS_ONLY, unscale_trajectory_data=UNSCALE,
        h=H, use_wind=USE_WIND,
    )
    m = compute_metrics(pred, true, nodes_body)
    tc, ts = m["t_contact"], m["t_settle"]
    T = len(m["center_error_t"])
    if tc >= T:            # never contacts - skip (pure freefall)
        continue

    v_entry = entry_speed_mps(true, tc)
    s_center = fit_slope(m["center_error_t"].numpy(), tc, ts)
    s_angle  = fit_slope(m["angle_error_t_deg"].numpy(), tc, ts)
    dv = s_center * BLOCK_WIDTH / DT if np.isfinite(s_center) else float("nan")
    frac = dv / v_entry if (np.isfinite(dv) and v_entry > 1e-6) else float("nan")

    rows.append(dict(
        idx=idx, type=ttype, T=T, t_contact=tc, t_settle=ts,
        v_entry_mps=v_entry, slope_center=s_center, slope_angle=s_angle,
        dv_mps=dv, frac_dv=frac,
        center_mean=m["center_error"], angle_mean=m["angle_error_deg"],
        center_air=m["center_error_airborne"], center_con=m["center_error_contact"],
        center_set=m["center_error_settled"],
        angle_air=m["angle_error_airborne"], angle_con=m["angle_error_contact"],
        angle_set=m["angle_error_settled"],
    ))
    print(f"[{n_done:>3}] traj {idx} ({ttype:>7}): tc={tc:>3} ts={ts:>3} "
          f"v_in={v_entry:5.2f} m/s  slope={s_center:.5f}  frac={100*frac:5.1f}%",
          flush=True)

# ======================================================================
# T3 tables: split by type
# ======================================================================
def _fmt(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return (f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            if vals else "n/a")

print("\n" + "=" * 74)
print(f"T3 - per-type breakdown ({DATASET}, n={len(rows)} trajectories)")
print("=" * 74)
for ttype in sorted(set(r["type"] for r in rows)):
    sel = [r for r in rows if r["type"] == ttype]
    print(f"\n--- {ttype.upper()}  (n={len(sel)}) ---")
    print(f"Center Error (/width):  {_fmt([r['center_mean'] for r in sel])}")
    print(f"Angle Error (deg):      {_fmt([r['angle_mean'] for r in sel])}")
    print(f"Phase center (a/c/s):   "
          f"{np.nanmean([r['center_air'] for r in sel]):.4f} / "
          f"{np.nanmean([r['center_con'] for r in sel]):.4f} / "
          f"{np.nanmean([r['center_set'] for r in sel]):.4f}")
    sl = [r["slope_center"] for r in sel if np.isfinite(r["slope_center"])]
    fr = [r["frac_dv"] for r in sel if np.isfinite(r["frac_dv"])]
    ve = [r["v_entry_mps"] for r in sel]
    print(f"Contact slope (/w/frame): median {np.median(sl):.5f}  "
          f"(IQR {np.percentile(sl,25):.5f}-{np.percentile(sl,75):.5f})")
    print(f"Entry speed (m/s):        median {np.median(ve):.2f}")
    print(f"Fractional dv/v:          median {100*np.median(fr):.1f}%")

# ======================================================================
# T2 fit + scatter
# ======================================================================
toss = [r for r in rows if r["type"] == "toss" and np.isfinite(r["slope_center"])]
slide = [r for r in rows if r["type"] == "sliding" and np.isfinite(r["slope_center"])]

print("\n" + "=" * 74)
print("T2 - slope vs entry speed")
print("=" * 74)
a = b = float("nan")
if len(toss) >= 5:
    v = np.array([r["v_entry_mps"] for r in toss])
    s = np.array([r["slope_center"] for r in toss])
    b, a = np.polyfit(v, s, 1)          # slope = a + b*v
    resid = s - (a + b * v)
    r2 = 1 - resid.var() / s.var() if s.var() > 0 else float("nan")
    print(f"Toss fit: slope = {a:.5f} + {b:.5f} * v   (R^2 = {r2:.2f})")
    print(f"  intercept a  -> speed-independent (friction) component")
    print(f"  b * W / DT   -> fractional impulse resolution = {100*b*BLOCK_WIDTH/DT:.1f}%")
if slide:
    med = np.median([r["slope_center"] for r in slide])
    print(f"Sliding median slope = {med:.5f}   "
          f"(compare against toss intercept a = {a:.5f})")

fig, ax = plt.subplots(figsize=(7, 5))
if toss:
    ax.scatter([r["v_entry_mps"] for r in toss],
               [r["slope_center"] for r in toss],
               s=22, alpha=0.6, label=f"toss (n={len(toss)})")
    if np.isfinite(b):
        vx = np.linspace(0, max(r["v_entry_mps"] for r in toss) * 1.05, 50)
        ax.plot(vx, a + b * vx, "C0--", lw=1.5,
                label=f"fit: {a:.4f} + {b:.4f} v")
if slide:
    ax.scatter([r["v_entry_mps"] for r in slide],
               [r["slope_center"] for r in slide],
               s=30, alpha=0.7, marker="^", color="C3",
               label=f"sliding (n={len(slide)})")
ax.set_xlabel("COM speed entering contact (m/s)")
ax.set_ylabel("contact-phase center-error slope (/width/frame)")
ax.set_title(f"Error growth rate vs contact-entry speed - {DATASET}")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(OUT_PREFIX + "_scatter.png", dpi=150)
print(f"\nSaved figure to {OUT_PREFIX}_scatter.png")

with open(OUT_PREFIX + "_per_traj.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Saved per-trajectory CSV to {OUT_PREFIX}_per_traj.csv")
"""
characterize_dataset.py

Measures everything needed to replicate the paper cube-toss dataset in MuJoCo:

  launch state   : initial COM height z0, per-component launch velocity,
                   launch angle, angular-velocity magnitude at launch
  units check    : gravity recovered from airborne parabola fits
                   (validates dt = 1/148 s and the x0.0524 unscaling at once)
  friction       : (a) clean-slide estimator  mu = decel_h / g   [-> cube.xml]
                   (b) stopping-distance estimator mu_eff = v^2/(2 g d)
                       (tumbling-inclusive; effective, NOT a material value)
  restitution    : vertical-velocity ratio across the first impact + rebound
                   height ratio  [-> tune MuJoCo solref damping to match]
  diagnostics    : resting-height offset (floor calibration), phase durations
  similarity     : N_slide = v_h^2 / (2 mu g L)  and  H/L = v_z^2 / (2 g L)

Run once with PRESET="paper", then regenerate MuJoCo data, then run again with
PRESET="mojoco" on the new folder and diff the printed tables. When they agree,
train the identical pipeline on both.

Outputs: printed summary, <prefix>_per_traj.csv, <prefix>_histograms.png,
and a paste-ready parameter block for capture_mojoco_traj.py.
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from generate_node_states import (mesh_cube_surface, quat_to_rotmat,
                                  unscale_position_velocity)
from evaluate_metrics import compute_phase_boundaries

BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2 * BLOCK_HALF_WIDTH

# ======================================================================
# CONFIG
# ======================================================================
PRESET = "mojoco"            # "paper" or "mojoco"
script_dir = os.path.dirname(os.path.abspath(__file__))

if PRESET == "paper":
    FOLDER       = os.path.join(script_dir, "data/tosses_processed")
    INDICES      = range(0, 569)        # characterize the whole set
    WEIGHTS_ONLY = True
    UNSCALE      = True
    DT           = 1.0 / 148.0
else:
    FOLDER       = os.path.join(script_dir, "data/mojoco_paper_matched_toss")
    INDICES      = range(0, 569)
    WEIGHTS_ONLY = False
    UNSCALE      = False
    DT           = 0.0001348 * 50

G = 9.81
L = BLOCK_WIDTH                         # cube side length (m)

# friction slide-window filters
SLIDE_MIN_NODES   = 3                   # corner nodes within CONTACT_Z of floor
SLIDE_CONTACT_Z   = 0.2 * BLOCK_HALF_WIDTH
SLIDE_MIN_VH      = 0.05                # m/s   horizontal speed to count as sliding
SLIDE_MAX_OMEGA   = 2.0                 # rad/s exclude tumbling frames
SLIDE_MAX_VZ      = 0.05                # m/s   exclude bounce frames
SLIDE_MIN_FRAMES  = 6                   # window length needed for a decel fit

MIN_AIRBORNE_FIT  = 5                   # frames needed for the launch parabola
OUT_PREFIX = os.path.join(script_dir, f"characterize_{PRESET}")

# ======================================================================
# Loading / kinematics helpers
# ======================================================================
def load_states(idx):
    """(T, >=7) [pos(3), quat wxyz(4), maybe vel(3), ...] in meters."""
    raw = torch.load(os.path.join(FOLDER, f"{idx}.pt"), weights_only=WEIGHTS_ONLY)
    states = raw[0].float()
    if UNSCALE:
        states = unscale_position_velocity(states)
    return states


NODES_BODY = torch.tensor(mesh_cube_surface(L, 2), dtype=torch.float32)  # 8 corners

def corner_positions(states):
    """(T, 8, 3) world corner positions from COM + quaternion."""
    out = []
    for t in range(states.shape[0]):
        R = quat_to_rotmat(states[t, 3:7])
        out.append((R @ NODES_BODY.T).T + states[t, :3])
    return torch.stack(out)


def omega_series(states):
    """|angular velocity| per frame (rad/s) from quaternion finite differences."""
    q_wxyz = states[:, 3:7].numpy()
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    R = Rotation.from_quat(q_xyzw)
    rel = R[:-1].inv() * R[1:]
    return np.linalg.norm(rel.as_rotvec(), axis=1) / DT      # (T-1,)


def airborne_fit(cm, tc):
    """Quadratic z / linear xy fit over airborne frames.
    Returns dict or None if too short."""
    if tc < MIN_AIRBORNE_FIT:
        return None
    k = np.arange(tc, dtype=np.float64)
    x, y, z = (cm[:tc, i].numpy().astype(np.float64) for i in range(3))
    az, bz, cz = np.polyfit(k, z, 2)              # z = az k^2 + bz k + cz
    vx = np.polyfit(k, x, 1)[0] / DT
    vy = np.polyfit(k, y, 1)[0] / DT
    vz0 = bz / DT
    vz_impact = (2 * az * (tc - 1) + bz) / DT     # derivative at last airborne frame
    return dict(g_est=-2 * az / DT ** 2, z0=cz,
                vx0=vx, vy0=vy, vz0=vz0, vz_impact=vz_impact)


def slide_windows(cm, corners, omega, tc, ts):
    """Maximal runs of clean face-sliding frames inside [tc, ts).
    Returns list of (mu, n_frames, v_start)."""
    T = cm.shape[0]
    vh = (cm[1:, :2] - cm[:-1, :2]).norm(dim=1).numpy() / DT     # (T-1,)
    vz = ((cm[1:, 2] - cm[:-1, 2]).numpy() / DT)
    n_low = (corners[..., 2] < SLIDE_CONTACT_Z).sum(dim=1).numpy()

    ok = np.zeros(T - 1, dtype=bool)
    hi = min(ts, T - 1)
    for t in range(tc, hi):
        ok[t] = (n_low[t] >= SLIDE_MIN_NODES and vh[t] > SLIDE_MIN_VH
                 and abs(vz[t]) < SLIDE_MAX_VZ
                 and (t < len(omega) and omega[t] < SLIDE_MAX_OMEGA))

    out, t = [], tc
    while t < hi:
        if ok[t]:
            t0 = t
            while t < hi and ok[t]:
                t += 1
            if t - t0 >= SLIDE_MIN_FRAMES:
                k = np.arange(t - t0, dtype=np.float64)
                decel = -np.polyfit(k, vh[t0:t], 1)[0] / DT      # m/s^2
                if decel > 0:
                    out.append((decel / G, t - t0, float(vh[t0])))
        else:
            t += 1
    return out


def first_bounce(cm, fit, tc, ts):
    """Vertical restitution proxy + rebound height ratio at first impact."""
    if fit is None or fit["vz_impact"] >= -0.05:
        return None, None
    T = cm.shape[0]
    vz = (cm[1:, 2] - cm[:-1, 2]).numpy() / DT
    hi = min(tc + 6, T - 1)
    vz_out = vz[tc:hi].max() if hi > tc else 0.0
    e_z = max(0.0, vz_out / -fit["vz_impact"])
    z_rest = cm[ts:, 2].median().item() if ts < T else BLOCK_HALF_WIDTH
    drop = max(cm[:tc, 2].max().item() - z_rest, 1e-6)
    reb_hi = min(ts, T)
    rebound = max(cm[tc:reb_hi, 2].max().item() - z_rest, 0.0) if reb_hi > tc else 0.0
    return e_z, rebound / drop


# ======================================================================
# Main loop
# ======================================================================
rows, all_mu_windows = [], []
for n_done, idx in enumerate(INDICES, 1):
    try:
        states = load_states(idx)
    except FileNotFoundError:
        continue
    T = states.shape[0]
    cm = states[:, :3]
    corners = corner_positions(states)
    tc, ts = compute_phase_boundaries(corners)
    if tc >= T:                                   # never contacts: skip
        continue
    omega = omega_series(states)
    fit = airborne_fit(cm, tc)

    wins = slide_windows(cm, corners, omega, tc, ts)
    all_mu_windows.extend(wins)
    mu_slide = float(np.median([w[0] for w in wins])) if wins else float("nan")

    # stopping-distance effective mu (tumbling-inclusive)
    vh_in = (cm[tc, :2] - cm[tc - 1, :2]).norm().item() / DT if tc >= 1 else float("nan")
    stop_hi = min(ts, T)
    d_contact = (cm[tc + 1:stop_hi, :2] - cm[tc:stop_hi - 1, :2]).norm(dim=1).sum().item() \
        if stop_hi > tc + 1 else float("nan")
    mu_eff = (vh_in ** 2 / (2 * G * d_contact)) if (d_contact and d_contact > 1e-4) else float("nan")

    e_z, rebound_ratio = first_bounce(cm, fit, tc, ts)
    z_rest = cm[ts:, 2].median().item() if ts < T else float("nan")
    omega0 = float(np.median(omega[:max(1, min(5, tc))]))

    r = dict(idx=idx, T=T, t_contact=tc, t_settle=ts,
             frames_air=tc, frames_contact=ts - tc, frames_settled=T - ts,
             z_rest_offset=(z_rest - BLOCK_HALF_WIDTH) if np.isfinite(z_rest) else float("nan"),
             omega0=omega0, mu_slide=mu_slide, n_slide_windows=len(wins),
             mu_eff=mu_eff, e_z=e_z if e_z is not None else float("nan"),
             rebound_ratio=rebound_ratio if rebound_ratio is not None else float("nan"),
             vh_impact=vh_in)
    if fit is not None:
        r.update(g_est=fit["g_est"], z0=fit["z0"], vx0=fit["vx0"], vy0=fit["vy0"],
                 vz0=fit["vz0"], vz_impact=fit["vz_impact"],
                 vh0=float(np.hypot(fit["vx0"], fit["vy0"])),
                 launch_angle_deg=float(np.degrees(np.arctan2(
                     fit["vz0"], np.hypot(fit["vx0"], fit["vy0"]) + 1e-9))))
        r["N_slide"] = vh_in ** 2 / (2 * max(mu_slide if np.isfinite(mu_slide) else 0.2, 1e-3) * G * L)
        r["H_over_L"] = fit["vz_impact"] ** 2 / (2 * G * L)
    rows.append(r)
    if n_done % 50 == 0:
        print(f"  processed {n_done} trajectories...", flush=True)

# ======================================================================
# Summary
# ======================================================================
def col(name):
    return np.array([r[name] for r in rows if name in r and np.isfinite(r[name])])

def stat(label, arr, unit=""):
    if len(arr) == 0:
        print(f"{label:<34} n=0"); return
    print(f"{label:<34} median {np.median(arr):9.4f}  "
          f"IQR [{np.percentile(arr,25):8.4f},{np.percentile(arr,75):8.4f}]  "
          f"5-95% [{np.percentile(arr,5):8.4f},{np.percentile(arr,95):8.4f}] {unit} (n={len(arr)})")

print("\n" + "=" * 100)
print(f"DATASET CHARACTERIZATION - {PRESET}  ({len(rows)} trajectories)")
print("=" * 100)

print("\n--- Units / calibration checks (do these FIRST) ---")
g_arr = col("g_est")
stat("recovered gravity g_est", g_arr, "m/s^2")
if len(g_arr):
    dt_implied = DT * np.sqrt(np.median(g_arr) / G)
    print(f"{'':<34} -> if this is not ~9.81, implied dt = {dt_implied:.6f} s "
          f"(assumed {DT:.6f})")
stat("resting COM height - halfwidth", col("z_rest_offset"), "m  (floor offset; want ~0)")

print("\n--- Launch state (frame 0) ---")
stat("z0 (COM height)", col("z0"), "m")
stat("vx0", col("vx0"), "m/s")
stat("vy0", col("vy0"), "m/s")
stat("vz0", col("vz0"), "m/s")
stat("horizontal speed |vh0|", col("vh0"), "m/s")
stat("launch angle", col("launch_angle_deg"), "deg")
stat("|omega| at launch", col("omega0"), "rad/s")

print("\n--- Impact / contact ---")
stat("vz at impact", col("vz_impact"), "m/s")
stat("vh at impact", col("vh_impact"), "m/s")
stat("restitution proxy e_z", col("e_z"))
stat("rebound height ratio", col("rebound_ratio"))

print("\n--- Friction ---")
mu_w = np.array([w[0] for w in all_mu_windows])
print(f"clean slide windows found: {len(mu_w)} across "
      f"{sum(1 for r in rows if r['n_slide_windows'] > 0)} trajectories")
stat("mu (clean-slide, PRIMARY)", mu_w, " -> cube.xml friction")
stat("mu_eff (stopping-distance)", col("mu_eff"), " (tumbling-inclusive, context only)")
if len(mu_w) < 20:
    print("  WARNING: few clean slide windows - real tosses tumble; treat mu_eff as a "
          "sanity band\n  and consider tuning mu so MuJoCo's mu_eff distribution matches "
          "the paper's instead.")

print("\n--- Phase structure / recording convention ---")
stat("frames airborne", col("frames_air"), "frames")
stat("frames contact", col("frames_contact"), "frames")
stat("frames settled (post-stop tail)", col("frames_settled"), "frames")
stat("total frames T", col("T"), "frames")

print("\n--- Dimensionless similarity targets ---")
stat("N_slide = vh^2/(2 mu g L)", col("N_slide"))
stat("H/L = vz_imp^2/(2 g L)", col("H_over_L"))

# ---------------------------------------------------------------------
# Paste-ready parameter block
# ---------------------------------------------------------------------
def rng(name, pad=0.0):
    a = col(name)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(a, 5)) - pad, float(np.percentile(a, 95)) + pad)

vh_comp = np.concatenate([np.abs(col("vx0")), np.abs(col("vy0"))])
vh95 = float(np.percentile(vh_comp, 95)) if len(vh_comp) else float("nan")
w95 = float(np.percentile(col("omega0"), 95)) if len(col("omega0")) else float("nan")
mu_rec = float(np.median(mu_w)) if len(mu_w) else float(np.median(col("mu_eff")))

print("\n" + "=" * 100)
print("PASTE-READY BLOCK for capture_mojoco_traj.py  (5-95% ranges)")
print("=" * 100)
print(f"""
#----- Paper-matched toss parameters (measured from {FOLDER}) -----
toss_vertical_pos_range     = ({rng('z0')[0]:.3f}, {rng('z0')[1]:.3f})     # COM start height (m)
toss_horizontal_speed_range = ({-vh95:.3f}, {vh95:.3f})   # per-component (m/s)
toss_vertical_speed_range   = ({rng('vz0')[0]:.3f}, {rng('vz0')[1]:.3f})   # signed vz0 (m/s)
toss_angvel_range           = ({-w95:.3f}, {w95:.3f})   # rad/s
sliding_percentage          = 0.0        # paper data has no sliding-only trajectories
n_steps                     = {int(np.median(col('T'))) if len(col('T')) else 'CHECK'}       # median paper length;
                                         # faithful option: trim each traj at t_settle + {int(np.median(col('frames_settled'))) if len(col('frames_settled')) else 20}

# cube.xml: set BOTH floor and cube geom friction to "{mu_rec:.3f} 0.005 0.0001"
# restitution: no direct XML field - sweep solref damping (2nd entry) until the
#   generated e_z distribution matches: target median e_z = {np.median(col('e_z')) if len(col('e_z')) else float('nan'):.3f}
# match criteria (rerun this script on the generated folder and compare):
#   N_slide median  {np.median(col('N_slide')) if len(col('N_slide')) else float('nan'):.2f}
#   H/L median      {np.median(col('H_over_L')) if len(col('H_over_L')) else float('nan'):.2f}
""")

# ---------------------------------------------------------------------
# CSV + histograms
# ---------------------------------------------------------------------
keys = sorted({k for r in rows for k in r})
with open(OUT_PREFIX + "_per_traj.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"Saved per-trajectory CSV to {OUT_PREFIX}_per_traj.csv")

panels = [("z0", "z0 (m)"), ("vh0", "|vh0| (m/s)"), ("vz0", "vz0 (m/s)"),
          ("omega0", "|omega0| (rad/s)"), ("e_z", "restitution e_z"),
          ("vz_impact", "vz impact (m/s)")]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, (name, label) in zip(axes.flat, panels):
    a = col(name)
    if len(a):
        ax.hist(a, bins=30, alpha=0.8)
    ax.set_title(label); ax.grid(alpha=0.3)
if len(mu_w):
    axes.flat[4].hist(mu_w, bins=30, alpha=0.5, color="C3", label="mu windows")
    axes.flat[4].legend()
fig.suptitle(f"Dataset characterization - {PRESET}")
fig.tight_layout()
fig.savefig(OUT_PREFIX + "_histograms.png", dpi=150)
print(f"Saved histograms to {OUT_PREFIX}_histograms.png")

"""
replicate_paper_tosses.py

Trajectory-for-trajectory MuJoCo replica of the paper cube-toss dataset.

For every real trajectory i:
  1. extract the frame-0 state: position, quaternion, linear velocity
     (airborne parabola fit), angular velocity (rotation-vector least squares,
     converted world -> BODY frame: MuJoCo free-joint qvel[3:6] is body-frame,
     verified empirically)
  2. re-simulate that exact toss in MuJoCo with measured physics
     (mu = 0.224, elliptic cone, g = 9.615 data-units, dt = 1/148)
  3. save the twin as data/<OUT_NAME>/i.pt in the standard training format
  4. grade the match in two tiers:
       Tier 1 (kinematic validation) - airborne COM/angle deviation, impact
         frame offset. This is where MuJoCo must match tightly.
       Tier 2 (statistical contact agreement) - contact duration, stopping
         distance, rest offset, e_z, compared as DISTRIBUTIONS. Per-frame
         post-bounce agreement is NOT expected: contact is chaotic (our
         perturbation-floor analysis: mm-level input differences produce
         large post-impact spread). Divergent bounces do not indict MuJoCo.

All physics parameters are set IN CODE and printed at launch - this dataset
cannot silently inherit a stale cube.xml.
"""

import os
import csv
import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Shared ground-truth wrench bookkeeping, defined once in capture_mojoco_traj.py
# so both generators write identical label schemas. Importing that module is
# safe - its dataset-generation code is under `if __name__ == "__main__"`.
from capture_mojoco_traj import (ContactFrameSelector, new_wrench_interval,
                                 accumulate_substep_wrench, verify_wrench,
                                 make_wrench_save_dict)

from generate_node_states import (mesh_cube_surface, quat_to_rotmat,
                                  unscale_position_velocity, BLOCK_HALF_WIDTH)
from evaluate_metrics import compute_phase_boundaries, BLOCK_WIDTH

# ======================================================================
# CONFIG
# ======================================================================
script_dir   = os.path.dirname(os.path.abspath(__file__))
PAPER_FOLDER = os.path.join(script_dir, "data/tosses_processed")
INDICES      = range(0, 569)
OUT_NAME     = "mojoco_paper_replica_0_wind"
OUT_DIR      = os.path.join(script_dir, "data", OUT_NAME)
XML_PATH     = os.path.join(script_dir, "cube.xml")

DT       = 1.0 / 148.0          # recorded frame time of the paper data
SUBSTEPS = 50

# Record MuJoCo's true contact/fluid wrenches alongside each replica. Costs a
# few % of generation time and removes the need for a separate
# add_wrench_labels.py pass. Labels are validation-only - never trained on.
RECORD_WRENCH = True

# --- physics (overridden in code regardless of what cube.xml says) ---
MATCH_DATA_UNITS = True
if MATCH_DATA_UNITS:
    GRAVITY = 9.615            # recovered g of the paper recordings
    MU      = 1.9/GRAVITY          # measured slide decel (2.154 m/s^2) / 9.615
else:
    GRAVITY = 9.81              # textbook physics (expect ~1 frame tc drift)
    MU      = 0.220
CONE        = "elliptic"        # isotropic friction (learnable + physical)
SOLREF_DAMP = 0.5               # sweep {1.0, 0.7, 0.5, 0.35} to hit e_z ~ 0.15
SOLREF_TIME = 0.02

MIN_AIRBORNE_FIT = 5            # frames needed for the launch fits
MASS = 0.37

WIND_RANGE     = (0.0, 0)      # uniform |wind| range, horizontal (m/s)
FIX_WIND_DIR   = False
WIND_DIR_FIXED = (1.0, 0.0, 0.0)
MEASURE_WIND_EFFECT = True       # also sim a no-wind twin per traj, report deviation
WIND_ON = WIND_RANGE[1] > 0

# ======================================================================
# Model setup with loud receipts
# ======================================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
model.opt.gravity[:]      = [0.0, 0.0, -GRAVITY]
model.opt.timestep        = DT / SUBSTEPS
model.opt.cone            = (mujoco.mjtCone.mjCONE_ELLIPTIC if CONE == "elliptic"
                             else mujoco.mjtCone.mjCONE_PYRAMIDAL)
model.geom_friction[:, 0] = MU
model.geom_solref[:]      = [SOLREF_TIME, SOLREF_DAMP]
model.body_mass[1]        = MASS

print("=" * 70)
print("REPLICA PHYSICS (set in code - cube.xml values are overridden):")
print(f"  gravity   = {model.opt.gravity[2]:.4f}   timestep = {model.opt.timestep:.7f}"
      f"  (dt_record = {DT:.6f})")
print(f"  friction  = {model.geom_friction[:, 0].tolist()}")
print(f"  cone      = {'elliptic' if int(model.opt.cone) == 1 else 'pyramidal'}")
print(f"  solref    = {model.geom_solref[0].tolist()}   mass = {model.body_mass[1]:.3f}")
print("=" * 70)

NODES_BODY = torch.tensor(mesh_cube_surface(BLOCK_WIDTH, 2), dtype=torch.float32)


# ======================================================================
# Initial-condition extraction from a real trajectory
# ======================================================================
def corner_positions(states):
    out = []
    for t in range(states.shape[0]):
        R = quat_to_rotmat(states[t, 3:7])
        out.append((R @ NODES_BODY.T).T + states[t, :3])
    return torch.stack(out)


def extract_ics(states):
    """Frame-0 state of a real trajectory.
    Returns pos0, quat0(wxyz), v0(world), w_body, tc, ts, quality."""
    T = states.shape[0]
    corners = corner_positions(states)
    tc, ts = compute_phase_boundaries(corners)
    cm = states[:, :3]

    pos0 = cm[0].numpy().astype(np.float64)
    q0_wxyz = states[0, 3:7].numpy().astype(np.float64)
    q0_wxyz /= np.linalg.norm(q0_wxyz)
    R0 = Rotation.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])

    # --- linear velocity at k=0 ---
    n_fit = min(tc, T)
    if n_fit >= MIN_AIRBORNE_FIT:
        k = np.arange(n_fit, dtype=np.float64)
        vx = np.polyfit(k, cm[:n_fit, 0].numpy(), 1)[0] / DT
        vy = np.polyfit(k, cm[:n_fit, 1].numpy(), 1)[0] / DT
        az, bz, _ = np.polyfit(k, cm[:n_fit, 2].numpy(), 2)
        vz = bz / DT                       # derivative of the parabola at k=0
        quality = "fit"
    else:                                  # short airborne phase: crude FD
        v = (cm[1] - cm[0]).numpy() / DT
        vx, vy, vz = v
        quality = "fallback"
    v0 = np.array([vx, vy, vz])

    # --- angular velocity: world-frame LSQ on rotation vectors, then -> body ---
    K = max(1, min(n_fit - 1, 8))
    qs = states[:K + 1, 3:7].numpy()
    Rk = Rotation.from_quat(qs[:, [1, 2, 3, 0]])
    num = np.zeros(3); den = 0.0
    for k in range(1, K + 1):
        rv = (Rk[k] * R0.inv()).as_rotvec()      # world-frame increment
        num += rv * k; den += k * k * DT
    w_world = num / den if den > 0 else np.zeros(3)
    w_body = R0.inv().apply(w_world)             # MuJoCo qvel[3:6] is BODY frame

    return pos0, q0_wxyz, v0, w_body, w_world, int(tc), int(ts), quality


# ======================================================================
# Simulation (records frame 0 BEFORE stepping -> exact frame alignment)
# ======================================================================
def simulate_replica(pos0, quat0_wxyz, v0, w_body, T, wind=np.zeros(3),
                     record_wrench=False):
    """Returns (states, wrench).  wrench is None unless record_wrench=True.

    MuJoCo already computes the true contact and fluid forces while stepping;
    recording them here means the dataset ships with its own force labels and
    no separate re-simulation pass is needed. Accumulated at SUBSTEP resolution
    because an impact spike lives entirely inside the SUBSTEPS between recorded
    frames - sampling qfrc at frame boundaries would alias it away.

    ALIGNMENT: this generator records frame 0 BEFORE stepping, so substep block
    i is exactly the transition frame i -> frame i+1 and every block labels an
    interval (T-1 of them). Labels are VALIDATION ONLY; the force model is
    never trained on them.
    """
    model.opt.wind[:] = wind
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3]  = pos0
    data.qpos[3:7] = quat0_wxyz
    data.qvel[:3]  = v0
    data.qvel[3:6] = w_body
    mujoco.mj_forward(model, data)

    states = [np.concatenate([data.qpos[:3].copy(), data.qpos[3:7].copy()])]
    qvel_lin = [data.qvel[:3].copy()]
    qvel_ang_body = [data.qvel[3:6].copy()]
    intervals = []
    selector = ContactFrameSelector()
    dt = model.opt.timestep

    for _ in range(T - 1):
        acc = new_wrench_interval() if record_wrench else None
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
            if record_wrench:
                accumulate_substep_wrench(model, data, acc, dt, selector)
        states.append(np.concatenate([data.qpos[:3].copy(), data.qpos[3:7].copy()]))
        if record_wrench:
            intervals.append(acc)
            qvel_lin.append(data.qvel[:3].copy())
            qvel_ang_body.append(data.qvel[3:6].copy())

    states_np = np.stack(states)
    if not record_wrench:
        return states_np, None

    stack = lambda key: np.stack([iv[key] for iv in intervals])
    wrench = dict(
        J_contact=stack("Jc"), J_fluid=stack("Jf"),
        tau_contact=stack("Tc"), tau_contact_qfrc=stack("Tc_q"),
        tau_fluid=stack("Tf"),
        qvel_lin=np.stack(qvel_lin), qvel_ang_body=np.stack(qvel_ang_body),
        contact_xcheck=selector.max_residual,
    )
    return states_np, wrench

def sample_wind():
    if not WIND_ON:
        return np.zeros(3)
    if FIX_WIND_DIR:
        d = np.array(WIND_DIR_FIXED, float); d[2] = 0.0
    else:
        d = np.random.randn(3); d[2] = 0.0
    d /= (np.linalg.norm(d) + 1e-8)
    return d * np.random.uniform(*WIND_RANGE)

# ======================================================================
# Pair comparison helpers
# ======================================================================
def angle_series_deg(qa, qb):
    Ra = Rotation.from_quat(qa[:, [1, 2, 3, 0]])
    Rb = Rotation.from_quat(qb[:, [1, 2, 3, 0]])
    return np.degrees((Ra * Rb.inv()).magnitude())


def bounce_ez(cm, tc, T):
    vz = np.diff(cm[:, 2]) / DT
    if tc < 2 or tc >= T - 1:
        return float("nan")
    vz_in = vz[tc - 1]
    if vz_in >= -0.05:
        return float("nan")
    vz_out = vz[tc:min(tc + 6, T - 1)].max()
    return max(0.0, vz_out / -vz_in)


def contact_stats(states_np, tc, ts):
    cm = states_np[:, :3]
    T = len(cm)
    hi = min(ts, T)
    stop_dist = float(np.linalg.norm(np.diff(cm[tc:hi, :2], axis=0), axis=1).sum()) \
        if hi > tc + 1 else float("nan")
    return dict(contact_frames=hi - tc, stop_dist=stop_dist,
                e_z=bounce_ez(cm, tc, T),
                rest_xy=cm[hi - 1, :2] if hi > tc else cm[-1, :2])


# ======================================================================
# Main loop
# ======================================================================
os.makedirs(OUT_DIR, exist_ok=True)
rows = []
wrench_verif = []

for n_done, idx in enumerate(INDICES, 1):
    try:
        raw = torch.load(os.path.join(PAPER_FOLDER, f"{idx}.pt"), weights_only=True)
    except FileNotFoundError:
        continue
    states_real = unscale_position_velocity(raw[0].float())
    T = states_real.shape[0]
    if T < 4:
        continue

    pos0, quat0, v0, w_body, w_world, tc_r, ts_r, quality = extract_ics(states_real)
    wind = sample_wind()
    twin, twin_wrench = simulate_replica(pos0, quat0, v0, w_body, T,
                                         wind=wind, record_wrench=RECORD_WRENCH)
    if WIND_ON and MEASURE_WIND_EFFECT:
        twin0, _ = simulate_replica(pos0, quat0, v0, w_body, T)   # no-wind baseline
        path0 = float(np.linalg.norm(np.diff(twin0[:, :3], axis=0), axis=1).sum())
        w_off = float(np.linalg.norm(twin[-1, :3] - twin0[-1, :3]))

    # twin phase boundaries (same detector, same node geometry)
    twin_t = torch.tensor(twin, dtype=torch.float32)
    tc_t, ts_t = compute_phase_boundaries(corner_positions(twin_t))
    tc_t, ts_t = int(tc_t), int(ts_t)

    # --- Tier 1: airborne kinematic match ---
    n_air = max(1, min(tc_r, tc_t, T))
    com_dev = np.linalg.norm(states_real[:n_air, :3].numpy() - twin[:n_air, :3], axis=1)
    ang_dev = angle_series_deg(states_real[:n_air, 3:7].numpy(), twin[:n_air, 3:7])

    # --- Tier 2: contact statistics ---
    s_real = contact_stats(states_real.numpy(), tc_r, ts_r)
    s_twin = contact_stats(twin, tc_t, ts_t)
    rest_offset = float(np.linalg.norm(s_real["rest_xy"] - s_twin["rest_xy"])) / BLOCK_WIDTH
    if not WIND_ON:
        rows.append(dict(
            idx=idx, T=T, quality=quality,
            v0=float(np.linalg.norm(v0)), w0=float(np.linalg.norm(w_world)),
            tc_real=tc_r, tc_twin=tc_t, dtc=tc_t - tc_r,
            air_com_dev_mm=float(com_dev.max() * 1000.0),
            air_ang_dev_deg=float(ang_dev.max()),
            contact_frames_real=s_real["contact_frames"],
            contact_frames_twin=s_twin["contact_frames"],
            stop_dist_real=s_real["stop_dist"], stop_dist_twin=s_twin["stop_dist"],
            e_z_real=s_real["e_z"], e_z_twin=s_twin["e_z"],
            rest_offset_w=rest_offset,
        ))
    else:
        rows.append(dict(
                    idx=idx, T=T, quality=quality,
                    v0=float(np.linalg.norm(v0)), w0=float(np.linalg.norm(w_world)),
                    tc_real=tc_r, tc_twin=tc_t, dtc=tc_t - tc_r,
                    air_com_dev_mm=float(com_dev.max() * 1000.0),
                    air_ang_dev_deg=float(ang_dev.max()),
                    contact_frames_real=s_real["contact_frames"],
                    contact_frames_twin=s_twin["contact_frames"],
                    stop_dist_real=s_real["stop_dist"], stop_dist_twin=s_twin["stop_dist"],
                    e_z_real=s_real["e_z"], e_z_twin=s_twin["e_z"],
                    rest_offset_w=rest_offset,
                    wind_mag=float(np.linalg.norm(wind)),
                    wind_offset_m=w_off, 
                    wind_offset_w=w_off/BLOCK_WIDTH, 
                    wind_offset_pct=100*w_off/(path0+1e-9),
                ))


    # --- save twin in the standard training format ---
    params = dict(wind=wind, pos=pos0, quat=quat0, vel=v0,
                  angvel=w_world, mass=MASS, type="toss",
                  source_idx=idx, ic_quality=quality,
                  replica_physics=dict(mu=MU, g=GRAVITY, cone=CONE,
                                       solref=[SOLREF_TIME, SOLREF_DAMP]))
    save_data = [torch.tensor(twin, dtype=torch.float32),
                 torch.tensor(wind, dtype=torch.float32), MASS, params]

    if twin_wrench is not None:
        dt_record = model.opt.timestep * SUBSTEPS
        v = verify_wrench(twin_wrench["qvel_lin"], twin_wrench, MASS,
                          model.opt.gravity.copy(), dt_record)
        wrench_verif.append(v)
        save_data.append(make_wrench_save_dict(
            twin_wrench, dt_record, SUBSTEPS,
            verification=dict(momentum_resid_rel=v["rel_max"],
                              momentum_resid_air=v["rel_max_air"],
                              momentum_resid_contact=v["rel_max_contact"],
                              contact_xcheck=twin_wrench["contact_xcheck"])))
        if len(wrench_verif) == 1:
            print("\n  --- wrench label diagnostics (first replica) ---")
            print(f"  integrator={model.opt.integrator}  cone={model.opt.cone}  "
                  f"timestep={model.opt.timestep:.3e}  dt_record={dt_record:.6f}")
            print(f"  per-frame gravity impulse = {v['grav_imp']:.6f} N*s")
            print(f"  momentum residual (% of that): max {100*v['rel_max']:.4f}%  "
                  f"| airborne {100*v['rel_max_air']:.4f}%  "
                  f"| contact {100*v['rel_max_contact']:.4f}%")
            print(f"  mean residual vector = {v['mean_resid']} N*s")
            print(f"     (systematic bias along one axis = missing force term;")
            print(f"      zero-mean scatter = integrator round-off, harmless)")
            print(f"  contact-frame cross-check = {twin_wrench['contact_xcheck']:.3e} N\n")

    torch.save(save_data, os.path.join(OUT_DIR, f"{idx}.pt"))

    if n_done % 50 == 0:
        print(f"  replicated {n_done} trajectories...", flush=True)

# ======================================================================
# Report
# ======================================================================
def med_iqr(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return "n/a"
    return f"{np.median(v):8.3f}  IQR [{np.percentile(v,25):7.3f},{np.percentile(v,75):7.3f}] (n={len(v)})"

fit_rows = [r for r in rows if r["quality"] == "fit"]
print("\n" + "=" * 70)
print(f"TIER 1 - kinematic validation (airborne phase, {len(fit_rows)} fit-quality"
      f" / {len(rows)} total)")
print("=" * 70)
print(f"max airborne COM deviation (mm): {med_iqr([r['air_com_dev_mm'] for r in fit_rows])}")
print(f"max airborne angle dev (deg):    {med_iqr([r['air_ang_dev_deg'] for r in fit_rows])}")
print(f"impact frame offset (frames):    {med_iqr([r['dtc'] for r in fit_rows])}")
print("PASS looks like: COM dev of a few mm, angle dev ~1 deg, |dtc| <= 1.")

print("\n" + "=" * 70)
print("TIER 2 - contact statistics, real vs twin (compare as distributions;")
print("per-bounce agreement is NOT expected - contact is chaotic)")
print("=" * 70)
print(f"contact duration real (frames):  {med_iqr([r['contact_frames_real'] for r in rows])}")
print(f"contact duration twin (frames):  {med_iqr([r['contact_frames_twin'] for r in rows])}")
print(f"stopping distance real (m):      {med_iqr([r['stop_dist_real'] for r in rows])}")
print(f"stopping distance twin (m):      {med_iqr([r['stop_dist_twin'] for r in rows])}")
print(f"e_z real:                        {med_iqr([r['e_z_real'] for r in rows])}")
print(f"e_z twin:                        {med_iqr([r['e_z_twin'] for r in rows])}")
print(f"rest-position offset (/width):   {med_iqr([r['rest_offset_w'] for r in rows])}")
print("(rest offset reflects chaos, not error - compare against the")
print(" perturbation-floor spread, and check the medians above instead)")

with open(os.path.join(script_dir, f"replica_{OUT_NAME}_pairs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys() if k != "rest_xy"])
    for r in rows:
        r.pop("rest_xy", None)
    w.writeheader(); w.writerows(rows)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [
    ("air_com_dev_mm", "airborne COM dev (mm)", None),
    ("air_ang_dev_deg", "airborne angle dev (deg)", None),
    ("dtc", "impact frame offset", None),
    ("contact_frames_real", "contact frames", "contact_frames_twin"),
    ("stop_dist_real", "stopping distance (m)", "stop_dist_twin"),
    ("e_z_real", "restitution e_z", "e_z_twin"),
]
for ax, (key, label, twin_key) in zip(axes.flat, panels):
    a = np.array([r[key] for r in rows if np.isfinite(r[key])])
    ax.hist(a, bins=30, alpha=0.6, label="real" if twin_key else None)
    if twin_key:
        b = np.array([r[twin_key] for r in rows if np.isfinite(r[twin_key])])
        ax.hist(b, bins=30, alpha=0.6, label="twin")
        ax.legend()
    ax.set_title(label); ax.grid(alpha=0.3)
fig.suptitle(f"Paper vs MuJoCo twin - {OUT_NAME}")
fig.tight_layout()
fig.savefig(os.path.join(script_dir, f"replica_{OUT_NAME}_report.png"), dpi=150)
if wrench_verif:
    rel = np.array([v["rel_max"] for v in wrench_verif])
    air = np.array([v["rel_max_air"] for v in wrench_verif])
    con = np.array([v["rel_max_contact"] for v in wrench_verif])
    print(f"\nWrench labels written for all {len(wrench_verif)} replicas.")
    print(f"  Momentum identity (% of per-frame gravity impulse):")
    print(f"     overall  median {100*np.median(rel):.4f}%   max {100*rel.max():.4f}%")
    print(f"     airborne median {100*np.median(air):.4f}%   max {100*air.max():.4f}%")
    print(f"     contact  median {100*np.median(con):.4f}%   max {100*con.max():.4f}%")
    print(f"  Under ~1% is bookkeeping-clean; a real frame/sign/alignment bug"
          f" shows up near 100%.")

print(f"\nSaved twins to {OUT_DIR}/, CSV + report PNG next to this script.")

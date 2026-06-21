import mujoco
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation

# Reuse the EXACT phase-split logic + constants the GNN eval uses, so the floor
# numbers this script prints are directly comparable to evaluate_model's output.
from generate_node_states import mesh_cube_surface, quat_to_rotmat, BLOCK_HALF_WIDTH
from evaluate_metrics import compute_phase_boundaries, BLOCK_WIDTH

# ---------------------------------------------------------------------------
# What this script measures
# ---------------------------------------------------------------------------
# The "irreducible error floor": take a reference MuJoCo toss, perturb its
# INITIAL state by a tiny epsilon, re-simulate, and measure how far the two
# (both ground-truth) trajectories diverge -- broken into airborne / contact /
# settled using the same boundaries as the GNN metric.
#
# Because freefall is non-chaotic, epsilon arrives at the floor un-amplified, so
# the airborne column should be ~epsilon (a sanity check) and any blowup in the
# contact/settled columns is the contact chaos. Sweep epsilon to get a CURVE of
# floor-vs-input-precision; compare your GNN's per-phase error against the floor
# at the epsilon matching your model's effective precision.
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_XML   = os.path.join(SCRIPT_DIR, "cube.xml")

# ---- match the dataset you're comparing against ----
N_STEPS  = 200
SUBSTEPS = 50
MASS     = 0.37
NODES_PER_EDGE = 2
# Wind is held identical between a reference and its perturbations (we're probing
# sensitivity to initial STATE, not to wind). The floor is ~insensitive to this;
# set it to whatever the dataset you're comparing to used.
WIND = np.array([0.0, 0.0, 0.0])

# ---- experiment size ----
N_REFERENCE = 100     # number of reference tosses
N_PERTURB   = 10      # perturbations per reference
SEED        = 0

# ---- toss sampling ranges (same as capture_mojoco_traj.py) ----
TOSS_HPOS = (-0.2, 0.2)
TOSS_VPOS = (0.3, 0.8)
TOSS_HVEL = (-1.25, 1.25)
TOSS_VVEL = (-0.3, 0.3)
TOSS_ANGVEL = (-3, 3)

# ---- perturbation scales: one "unit" of epsilon, swept by the multipliers ----
EPS_POS    = 1e-3   # meters
EPS_VEL    = 1e-3   # m/s
EPS_ANG    = 1e-3   # radians of orientation
EPS_ANGVEL = 1e-3   # rad/s
EPS_MULTIPLIERS = [0.1, 0.3, 1.0, 3.0, 10.0]

# "toss_start" perturbs the launch state (recommended -- comparable to GNN eval).
# "contact" perturbs the last airborne state instead, to isolate the contact map
# with no freefall transport in front of it.
PERTURB_MODE = "toss_start"

# ---- visual sanity check ----
# When True, skip the sweep and instead replay ONE reference toss (blue) against
# its N_PERTURB perturbations (red, transparent) in a single MuJoCo viewer, so you
# can eyeball that the perturbed trajectories really are nearly on top of each other.
VISUALIZE     = False
VIS_EPS_MULT  = 10.0    # which epsilon level to visualize (one of EPS_MULTIPLIERS)
VIS_PLAYBACK  = 0.02   # seconds between replayed frames


# ---------------------------------------------------------------------------
# Simulation (same stepping convention as capture_mojoco_traj.collect_trajectory)
# ---------------------------------------------------------------------------
def simulate(model, qpos0, qvel0, wind, n_steps=N_STEPS, substeps=SUBSTEPS, record_vel=False):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model.opt.wind[:] = wind
    model.body_mass[1] = MASS
    data.qpos[:7] = qpos0
    data.qvel[:6] = qvel0
    mujoco.mj_forward(model, data)

    qpos_states, qvel_states = [], []
    for _ in range(n_steps):
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        qpos_states.append(data.qpos[:7].copy())
        if record_vel:
            qvel_states.append(data.qvel[:6].copy())

    qpos_traj = np.stack(qpos_states)                 # (T, 7) = [pos(3), quat(4)]
    if record_vel:
        return qpos_traj, np.stack(qvel_states)       # (T, 6) = [v(3), w(3)]
    return qpos_traj


# ---------------------------------------------------------------------------
# Reference toss + perturbation
# ---------------------------------------------------------------------------
def sample_reference_toss(rng):
    pos = np.array([rng.uniform(*TOSS_HPOS), rng.uniform(*TOSS_HPOS), rng.uniform(*TOSS_VPOS)])
    quat = Rotation.random(random_state=rng).as_quat()          # xyzw
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])   # -> wxyz (MuJoCo)
    vel = np.array([rng.uniform(*TOSS_HVEL), rng.uniform(*TOSS_HVEL), rng.uniform(*TOSS_VVEL)])
    angvel = rng.uniform(TOSS_ANGVEL[0], TOSS_ANGVEL[1], size=3)
    qpos0 = np.concatenate([pos, quat_wxyz])
    qvel0 = np.concatenate([vel, angvel])
    return qpos0, qvel0


def perturb_state(qpos0, qvel0, mult, rng):
    """Tiny perturbation of (pos, quat, vel, angvel). All four scaled by `mult`."""
    pos  = qpos0[:3] + rng.normal(0.0, EPS_POS * mult, size=3)

    # orientation: compose a small random rotation onto the quaternion
    delta_rotvec = rng.normal(0.0, EPS_ANG * mult, size=3)
    q_xyzw = np.array([qpos0[4], qpos0[5], qpos0[6], qpos0[3]])
    q_new = (Rotation.from_rotvec(delta_rotvec) * Rotation.from_quat(q_xyzw)).as_quat()
    quat_wxyz = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])

    vel    = qvel0[:3] + rng.normal(0.0, EPS_VEL * mult, size=3)
    angvel = qvel0[3:] + rng.normal(0.0, EPS_ANGVEL * mult, size=3)

    return np.concatenate([pos, quat_wxyz]), np.concatenate([vel, angvel])


# ---------------------------------------------------------------------------
# Metrics (same definitions as compute_metrics: COM dist / width, angle in deg)
# ---------------------------------------------------------------------------
def node_positions_from_qpos(qpos_traj):
    """(T,7) qpos -> (T,N,3) corner-node world positions, for phase boundaries."""
    nodes_body = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, NODES_PER_EDGE),
                              dtype=torch.float32)
    out = []
    for t in range(qpos_traj.shape[0]):
        q = torch.tensor(qpos_traj[t, 3:7], dtype=torch.float32)   # wxyz
        R = quat_to_rotmat(q)
        pos = torch.tensor(qpos_traj[t, :3], dtype=torch.float32)
        out.append((R @ nodes_body.T).T + pos)
    return torch.stack(out)


def center_error_per_frame(qa, qb):
    T = min(len(qa), len(qb))
    return np.linalg.norm(qa[:T, :3] - qb[:T, :3], axis=1) / BLOCK_WIDTH   # (T,)


def angle_error_per_frame_deg(qa, qb):
    T = min(len(qa), len(qb))
    Ra = Rotation.from_quat(qa[:T, [4, 5, 6, 3]])   # wxyz -> xyzw
    Rb = Rotation.from_quat(qb[:T, [4, 5, 6, 3]])
    return np.degrees((Ra * Rb.inv()).magnitude())  # (T,)


def phase_average(vec, t_contact, t_settle):
    T = len(vec)
    out = {}
    for name, sl in (('airborne', slice(0, t_contact)),
                     ('contact',  slice(t_contact, t_settle)),
                     ('settled',  slice(t_settle, T))):
        seg = vec[sl]
        out[name] = float(np.mean(seg)) if len(seg) > 0 else float('nan')
    return out


# ---------------------------------------------------------------------------
# One epsilon level
# ---------------------------------------------------------------------------
def run_epsilon(model, mult, rng):
    phases = ['airborne', 'contact', 'settled']
    acc = {f'center_{p}': [] for p in phases}
    acc.update({f'angle_{p}': [] for p in phases})

    for _ in range(N_REFERENCE):
        qpos0, qvel0 = sample_reference_toss(rng)

        if PERTURB_MODE == "toss_start":
            ref_qpos = simulate(model, qpos0, qvel0, WIND)
            ref_nodes = node_positions_from_qpos(ref_qpos)
            t_contact, t_settle = compute_phase_boundaries(ref_nodes)
            launch_qpos, launch_qvel = qpos0, qvel0
            n_steps = N_STEPS

        elif PERTURB_MODE == "contact":
            # Run the reference fully (recording velocity), grab the last airborne
            # state, and perturb THAT -- isolates the contact map.
            ref_full_qpos, ref_full_qvel = simulate(model, qpos0, qvel0, WIND, record_vel=True)
            full_nodes = node_positions_from_qpos(ref_full_qpos)
            tc, _ = compute_phase_boundaries(full_nodes)
            t_pre = tc - 1
            if t_pre < 1 or tc >= N_STEPS:
                continue   # never cleanly contacts; skip
            launch_qpos, launch_qvel = ref_full_qpos[t_pre], ref_full_qvel[t_pre]
            n_steps = N_STEPS - t_pre
            ref_qpos = simulate(model, launch_qpos, launch_qvel, WIND, n_steps=n_steps)
            ref_nodes = node_positions_from_qpos(ref_qpos)
            t_contact, t_settle = compute_phase_boundaries(ref_nodes)
        else:
            raise ValueError(f"unknown PERTURB_MODE {PERTURB_MODE}")

        for _ in range(N_PERTURB):
            p_qpos, p_qvel = perturb_state(launch_qpos, launch_qvel, mult, rng)
            pert_qpos = simulate(model, p_qpos, p_qvel, WIND, n_steps=n_steps)

            c = center_error_per_frame(ref_qpos, pert_qpos)
            a = angle_error_per_frame_deg(ref_qpos, pert_qpos)
            cp = phase_average(c, t_contact, t_settle)
            ap = phase_average(a, t_contact, t_settle)
            for p in phases:
                acc[f'center_{p}'].append(cp[p])
                acc[f'angle_{p}'].append(ap[p])

    return {k: np.nanmean(v) if len(v) else float('nan') for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Visual sanity check: overlay one reference + its perturbations in one viewer
# ---------------------------------------------------------------------------
def _replay_xml(n_perturb):
    """One blue reference cube + n_perturb transparent red cubes, all free bodies
    we pose by hand (gravity off -- we're only replaying recorded poses)."""
    bodies = []
    bodies.append("""
        <body name="ref" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.0524 0.0524 0.0524" material="blue"/>
        </body>""")
    for i in range(n_perturb):
        bodies.append(f"""
        <body name="pert{i}" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.0524 0.0524 0.0524" material="red"/>
        </body>""")
    return f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <asset>
        <material name="blue" rgba="0.2 0.4 0.9 1.0"/>
        <material name="red"  rgba="0.9 0.2 0.2 0.35"/>
      </asset>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="3 3 0.01" rgba="0.85 0.85 0.85 1"/>
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """


def visualize_overlay(model, mult, rng):
    import mujoco.viewer
    import time

    # one reference toss + its perturbations, same logic as run_epsilon
    qpos0, qvel0 = sample_reference_toss(rng)
    ref_qpos = simulate(model, qpos0, qvel0, WIND)

    pert_qpos_list = []
    for _ in range(N_PERTURB):
        p_qpos, p_qvel = perturb_state(qpos0, qvel0, mult, rng)
        pert_qpos_list.append(simulate(model, p_qpos, p_qvel, WIND))

    # report the spread so you get a number alongside the picture
    finals = np.array([p[-1, :3] for p in pert_qpos_list])
    spread = np.linalg.norm(finals - ref_qpos[-1, :3], axis=1).mean() / BLOCK_WIDTH
    print(f"Visualizing 1 reference + {N_PERTURB} perturbations at eps_mult={mult} "
          f"(pos eps={EPS_POS*mult:.1e} m)")
    print(f"  mean final-position spread vs reference: {spread:.4f} block widths")
    print("  blue = reference, red (transparent) = perturbations")

    vmodel = mujoco.MjModel.from_xml_string(_replay_xml(N_PERTURB))
    vdata = mujoco.MjData(vmodel)
    mujoco.mj_forward(vmodel, vdata)

    T = min([len(ref_qpos)] + [len(p) for p in pert_qpos_list])

    with mujoco.viewer.launch_passive(vmodel, vdata) as viewer:
        time.sleep(2)
        for t in range(T):
            # ref occupies qpos[0:7], pert i occupies qpos[7*(i+1):7*(i+2)]
            vdata.qpos[0:3] = ref_qpos[t, :3]
            vdata.qpos[3:7] = ref_qpos[t, 3:7]
            for i, p in enumerate(pert_qpos_list):
                base = 7 * (i + 1)
                vdata.qpos[base:base + 3] = p[t, :3]
                vdata.qpos[base + 3:base + 7] = p[t, 3:7]
            mujoco.mj_forward(vmodel, vdata)
            viewer.sync()
            time.sleep(VIS_PLAYBACK)
        input("Press Enter to close viewer...")


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    model = mujoco.MjModel.from_xml_path(CUBE_XML)

    if VISUALIZE:
        visualize_overlay(model, VIS_EPS_MULT, rng)
        return

    print("=" * 78)
    print(f"Perturbation floor  |  mode={PERTURB_MODE}  |  wind={WIND.tolist()}  "
          f"|  N_ref={N_REFERENCE} x K={N_PERTURB}")
    print(f"base eps: pos={EPS_POS} m, vel={EPS_VEL} m/s, ang={EPS_ANG} rad, "
          f"angvel={EPS_ANGVEL} rad/s")
    print("=" * 78)
    print(f"{'eps_mult':>8} | {'pos(m)':>9} | "
          f"{'center floor (/width)':>32} | {'angle floor (deg)':>26}")
    print(f"{'':>8} | {'':>9} | {'airborne':>10} {'contact':>10} {'settled':>10} | "
          f"{'airborne':>8} {'contact':>8} {'settled':>8}")
    print("-" * 78)

    for mult in EPS_MULTIPLIERS:
        r = run_epsilon(model, mult, rng)
        print(f"{mult:>8.2f} | {EPS_POS*mult:>9.1e} | "
              f"{r['center_airborne']:>10.4f} {r['center_contact']:>10.4f} {r['center_settled']:>10.4f} | "
              f"{r['angle_airborne']:>8.3f} {r['angle_contact']:>8.3f} {r['angle_settled']:>8.3f}")

    print("-" * 78)
    print("Read: airborne column should track epsilon (freefall is non-chaotic).")
    print("Compare each column against your GNN's per-phase error at the eps that")
    print("matches your model's effective input precision. If your GNN's contact/")
    print("settled error >> the floor, it's reducible; if it sits ON the floor, it's chaos.")


if __name__ == "__main__":
    main()
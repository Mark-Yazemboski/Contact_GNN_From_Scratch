"""
add_wrench_labels.py

NEW FILE - Stage 0 of the force-model transition. Augments existing trajectory
datasets with MuJoCo's ground-truth CONTACT and FLUID wrench labels, without
touching the originals (writes to <folder>_wrench/).

Method: deterministic re-simulation from the saved initial conditions, exactly
like recover_wind_labels.py - determinism means an exact re-run reproduces the
saved trajectory to float precision, and that match is verified per file. The
wrench must be accumulated at SUBSTEP resolution: contact impulses during
impact live inside the 50 substeps between recorded frames, so sampling
qfrc at recorded frames would alias away the spike. We therefore sum
force * dt over the substeps of each recorded interval, giving the IMPULSE
over interval t -> t+1, which is precisely the quantity conjugate to the
per-step velocity change the model predicts.

Frame conventions (empirically verified in this codebase):
  - free-joint qvel[0:3] is WORLD linear velocity
  - free-joint qvel[3:6] is BODY-frame angular velocity (see
    replicate_paper_tosses.py docstring), so generalized angular forces are
    BODY-frame torques; we rotate them to world with R(t) per substep.
  - contact forces from mj_contactForce are in the contact frame; the
    world-rotation orientation is selected EMPIRICALLY per run by matching the
    contact-force sum against qfrc_constraint[0:3] (which is unambiguous), and
    the residual of that match is reported as a cross-check.

Built-in verification per file (any failure -> file flagged, not written):
  1. trajectory match:  max |resim qpos - saved qpos|  (determinism gate)
  2. linear momentum:   m dv = m g DT + J_contact + J_fluid, exact for
     MuJoCo's semi-implicit Euler -> residual should be ~float precision.
     This check also empirically validates every frame convention above.

Labels are for VALIDATION, not training - the force model is never shown them.
That is the claim the evaluation makes: the model recovers the contact/fluid
split it was never trained on.

USAGE: set FOLDERS below, run once per dataset. Set MAX_TRAJ = 5 first as a
smoke test and read the verification columns before doing the full run.
"""

import os
import csv
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

# ======================================================================
# CONFIG
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(script_dir, "cube.xml")

FOLDERS = [
    os.path.join(script_dir, "data/mojoco_paper_replica_0_wind"),
    # os.path.join(script_dir, "data/mojoco_paper_replica_20_wind"),
]

MAX_TRAJ = None            # None = all files; set 5 for a smoke test
SUBSTEPS_REPLICA = 50      # matches replicate_paper_tosses.py
DT_REPLICA = 1.0 / 148.0
SUBSTEPS_CAPTURE = 50      # matches capture_mojoco_traj.py __main__

TOL_TRAJ = 1e-4            # m; recover_wind_labels saw ~1e-7 on exact matches
TOL_MOMENTUM = 1e-3        # N*s; semi-implicit Euler makes the identity exact

# ======================================================================


def load_file(path):
    raw = torch.load(path, weights_only=False)
    states = raw[0].numpy().astype(np.float64)
    wind = np.asarray(raw[1], dtype=np.float64).reshape(3) if len(raw) > 1 else np.zeros(3)
    mass = float(raw[2]) if len(raw) > 2 else 0.37
    params = raw[3] if len(raw) > 3 and isinstance(raw[3], dict) else {}
    return raw, states, wind, mass, params


def configure_model(model, params, mass, is_replica):
    """Reproduce the generator's in-code physics overrides exactly."""
    if is_replica:
        rp = params["replica_physics"]
        model.opt.gravity[:] = [0.0, 0.0, -float(rp["g"])]
        model.opt.timestep = DT_REPLICA / SUBSTEPS_REPLICA
        model.opt.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC
                          if rp.get("cone", "elliptic") == "elliptic"
                          else mujoco.mjtCone.mjCONE_PYRAMIDAL)
        model.geom_friction[:, 0] = float(rp["mu"])
        model.geom_solref[:] = list(rp.get("solref", [0.02, 0.5]))
    # capture-style datasets inherit cube.xml physics untouched
    model.body_mass[1] = mass
    model.opt.wind[:] = np.asarray(params.get("wind", np.zeros(3)), dtype=np.float64)


def seed_state(data, params, is_replica):
    """Set qpos/qvel from the saved params. Replica stored angvel in WORLD
    frame (extract_ics), so convert to body; capture seeded qvel directly."""
    pos0 = np.asarray(params["pos"], dtype=np.float64)
    quat0 = np.asarray(params["quat"], dtype=np.float64)          # wxyz
    vel0 = np.asarray(params["vel"], dtype=np.float64)
    ang0 = np.asarray(params["angvel"], dtype=np.float64)
    if is_replica:
        R0 = Rotation.from_quat(quat0[[1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
        ang0 = R0.T @ ang0                                        # world -> body
    data.qpos[:3] = pos0
    data.qpos[3:7] = quat0
    data.qvel[:3] = vel0
    data.qvel[3:6] = ang0


class ContactFrameSelector:
    """Picks (once, empirically) whether contact-frame -> world uses frame^T or
    frame, by matching the summed contact forces to qfrc_constraint[0:3]."""

    def __init__(self):
        self.use_transpose = None
        self.max_residual = 0.0

    def world_forces(self, model, data):
        forces, points = [], []
        f6 = np.zeros(6)
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, f6)
            F = f6[:3].copy()
            Rc = data.contact[i].frame.reshape(3, 3)
            forces.append((Rc.T @ F, Rc @ F))
            points.append(data.contact[i].pos.copy())
        if not forces:
            return [], []
        target = data.qfrc_constraint[:3].copy()
        if self.use_transpose is None:
            s_t = sum(f[0] for f in forces)
            s_n = sum(f[1] for f in forces)
            self.use_transpose = (np.linalg.norm(s_t - target)
                                  <= np.linalg.norm(s_n - target))
        chosen = [f[0] if self.use_transpose else f[1] for f in forces]
        self.max_residual = max(self.max_residual,
                                float(np.linalg.norm(sum(chosen) - target)))
        return chosen, points


def resimulate_with_wrench(model, params, mass, T, is_replica, selector):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    seed_state(data, params, is_replica)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    substeps = SUBSTEPS_REPLICA if is_replica else SUBSTEPS_CAPTURE

    qpos_frames, qvel_lin, qvel_ang_body = [], [], []
    intervals = []          # one dict of impulse accumulators per recorded interval

    def record_frame():
        qpos_frames.append(np.concatenate([data.qpos[:3].copy(), data.qpos[3:7].copy()]))
        qvel_lin.append(data.qvel[:3].copy())
        qvel_ang_body.append(data.qvel[3:6].copy())

    def run_interval():
        acc = dict(Jc=np.zeros(3), Jf=np.zeros(3), Tc=np.zeros(3),
                   Tc_q=np.zeros(3), Tf=np.zeros(3))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            R = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]]).as_matrix()
            com = data.xipos[1].copy()
            acc["Jc"] += data.qfrc_constraint[:3] * dt
            acc["Jf"] += data.qfrc_passive[:3] * dt
            acc["Tc_q"] += (R @ data.qfrc_constraint[3:6]) * dt   # body->world
            acc["Tf"] += (R @ data.qfrc_passive[3:6]) * dt
            fw, pts = selector.world_forces(model, data)
            for F, p in zip(fw, pts):
                acc["Tc"] += np.cross(p - com, F) * dt
        return acc

    if is_replica:
        # replicate_paper_tosses.py records frame 0 BEFORE stepping
        record_frame()
        for _ in range(T - 1):
            intervals.append(run_interval())
            record_frame()
    else:
        # capture_mojoco_traj.py records the first frame AFTER one substep
        # block; labels start at recorded frame 0 (the pre-frame-0 block is
        # dropped so labels align with training frames).
        run_interval()
        record_frame()
        for _ in range(T - 1):
            intervals.append(run_interval())
            record_frame()

    stack = lambda key: np.stack([iv[key] for iv in intervals])
    return (np.stack(qpos_frames), np.stack(qvel_lin), np.stack(qvel_ang_body),
            {k: stack(k) for k in ("Jc", "Jf", "Tc", "Tc_q", "Tf")})


def verify(states_saved, qpos_resim, qvel_lin, wrench, mass, g_vec, DT):
    traj_dev = float(np.abs(qpos_resim - states_saved[:, :7]).max())
    dv = mass * (qvel_lin[1:] - qvel_lin[:-1])
    grav = mass * g_vec * DT
    resid = dv - (grav + wrench["Jc"] + wrench["Jf"])
    mom_resid = float(np.abs(resid).max())
    return traj_dev, mom_resid


# ======================================================================
# Main
# ======================================================================
for folder in FOLDERS:
    out_dir = folder.rstrip("/\\") + "_wrench"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")],
                   key=lambda s: int(os.path.splitext(s)[0]))
    if MAX_TRAJ is not None:
        files = files[:MAX_TRAJ]
    print(f"\n=== {folder} -> {out_dir}  ({len(files)} files) ===")

    rows, failures = [], 0
    selector = ContactFrameSelector()

    for n, fn in enumerate(files, 1):
        raw, states, wind, mass, params = load_file(os.path.join(folder, fn))
        is_replica = "replica_physics" in params
        if "pos" not in params:
            print(f"  {fn}: no params dict with ICs - skipped"); failures += 1
            continue

        model = mujoco.MjModel.from_xml_path(XML_PATH)
        configure_model(model, params, mass, is_replica)
        T = states.shape[0]
        DT = (DT_REPLICA if is_replica
              else model.opt.timestep * SUBSTEPS_CAPTURE)
        g_vec = model.opt.gravity.copy()

        qpos_resim, qvel_lin, qvel_ang_body, wr = resimulate_with_wrench(
            model, params, mass, T, is_replica, selector)
        traj_dev, mom_resid = verify(states, qpos_resim, qvel_lin, wr, mass, g_vec, DT)

        ok = traj_dev < TOL_TRAJ and mom_resid < TOL_MOMENTUM
        rows.append(dict(file=fn, replica=int(is_replica), T=T,
                         wind_mag=float(np.linalg.norm(wind)),
                         traj_dev=traj_dev, momentum_resid=mom_resid,
                         contact_xcheck=selector.max_residual, ok=int(ok)))
        if not ok:
            failures += 1
            print(f"  {fn}: FAILED verification "
                  f"(traj_dev={traj_dev:.2e}, mom_resid={mom_resid:.2e}) - not written")
            continue

        wrench_dict = dict(
            J_contact=torch.tensor(wr["Jc"], dtype=torch.float32),
            J_fluid=torch.tensor(wr["Jf"], dtype=torch.float32),
            tau_contact=torch.tensor(wr["Tc"], dtype=torch.float32),
            tau_contact_qfrc=torch.tensor(wr["Tc_q"], dtype=torch.float32),
            tau_fluid=torch.tensor(wr["Tf"], dtype=torch.float32),
            qvel_lin=torch.tensor(qvel_lin, dtype=torch.float32),
            qvel_ang_body=torch.tensor(qvel_ang_body, dtype=torch.float32),
            dt_record=DT, substeps=(SUBSTEPS_REPLICA if is_replica else SUBSTEPS_CAPTURE),
            convention=("J*/tau* are impulses over recorded interval t->t+1, world "
                        "frame, torque about the body COM. J[t] pairs with the model "
                        "prediction whose input window ends at frame t."),
            verification=dict(traj_dev=traj_dev, momentum_resid=mom_resid),
        )
        torch.save(list(raw) + [wrench_dict], os.path.join(out_dir, fn))

        if n % 50 == 0 or n == len(files):
            print(f"  {n}/{len(files)}  "
                  f"(traj_dev max so far ~{max(r['traj_dev'] for r in rows):.1e}, "
                  f"failures: {failures})", flush=True)

    csv_path = os.path.join(script_dir, f"wrench_labels_{os.path.basename(folder)}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    devs = np.array([r["traj_dev"] for r in rows])
    mres = np.array([r["momentum_resid"] for r in rows])
    print(f"\n  SUMMARY: written {len(rows) - failures}/{len(rows)}"
          + (f"   *** {failures} FAILED ***" if failures else ""))
    print(f"  trajectory match:   median {np.median(devs):.2e} m, max {devs.max():.2e} m")
    print(f"  momentum identity:  median {np.median(mres):.2e} N*s, max {mres.max():.2e} N*s")
    print(f"  contact-frame cross-check residual (vs qfrc): {selector.max_residual:.2e} N")
    print(f"  per-file CSV -> {csv_path}")

print("\nNext: point run_force_multi_step.py / evaluate_force_model.py at the "
      "*_wrench folders. Old loaders still work - the wrench dict is appended "
      "as element [4], elements [0..3] are unchanged.")

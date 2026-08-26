import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation
import time


# ======================================================================
# GROUND-TRUTH WRENCH CAPTURE
# ----------------------------------------------------------------------
# MuJoCo already computes the true contact and fluid forces while stepping;
# it just does not save them. We record them here, inline, so the dataset
# carries its own force labels and no separate re-simulation pass is needed.
#
# Accumulated at SUBSTEP resolution: an impact spike lives entirely inside the
# `substeps` between recorded frames, so sampling qfrc only at frames would
# alias it away. We sum force * dt over each recorded interval, giving the
# IMPULSE over frame t -> t+1 - precisely the quantity conjugate to the
# per-step velocity change the force model predicts.
#
# These labels are for VALIDATION ONLY. The force GNN is never shown them;
# that is what lets us claim it recovered the contact/fluid split from motion
# alone.
# ======================================================================

class ContactFrameSelector:
    """mj_contactForce returns forces in the CONTACT frame. Which orientation
    maps them to world (frame^T vs frame) is settled empirically on the first
    contact by matching their sum against qfrc_constraint[0:3], which is
    unambiguous. The residual of that match is reported as a cross-check."""

    def __init__(self):
        self.use_transpose = None
        self.max_residual = 0.0

    def world_forces(self, model, data):
        if data.ncon == 0:
            return [], []
        f6 = np.zeros(6)
        cand, points = [], []
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, f6)
            F = f6[:3].copy()
            Rc = data.contact[i].frame.reshape(3, 3)
            cand.append((Rc.T @ F, Rc @ F))
            points.append(data.contact[i].pos.copy())
        target = data.qfrc_constraint[:3].copy()
        if self.use_transpose is None:
            s_t = sum(c[0] for c in cand)
            s_n = sum(c[1] for c in cand)
            self.use_transpose = (np.linalg.norm(s_t - target)
                                  <= np.linalg.norm(s_n - target))
        chosen = [c[0] if self.use_transpose else c[1] for c in cand]
        self.max_residual = max(self.max_residual,
                                float(np.linalg.norm(sum(chosen) - target)))
        return chosen, points


def new_wrench_interval():
    return dict(Jc=np.zeros(3), Jf=np.zeros(3), Tc=np.zeros(3),
                Tc_q=np.zeros(3), Tf=np.zeros(3))


# Running audit of the "passive force == fluid force" assumption. qfrc_passive
# is the TOTAL passive force (spring + damper + gravcomp + fluid); labelling it
# J_fluid is only correct if nothing but fluid contributes. MuJoCo >= 3.0 splits
# it into qfrc_{spring,damper,gravcomp,fluid}, so we read the fluid term
# DIRECTLY and additionally record how far the two ever diverge.
PASSIVE_AUDIT = dict(max_nonfluid=0.0, have_qfrc_fluid=None, substeps=0)


def fluid_qfrc(data):
    """The fluid part of the passive generalized force. Correct by
    construction on mujoco>=3.0; falls back to qfrc_passive (the old, unproven
    assumption) on older versions, and audit_passive_terms() says so loudly."""
    if PASSIVE_AUDIT["have_qfrc_fluid"] is None:
        PASSIVE_AUDIT["have_qfrc_fluid"] = hasattr(data, "qfrc_fluid")
    if not PASSIVE_AUDIT["have_qfrc_fluid"]:
        return data.qfrc_passive
    PASSIVE_AUDIT["substeps"] += 1
    PASSIVE_AUDIT["max_nonfluid"] = max(
        PASSIVE_AUDIT["max_nonfluid"],
        float(np.abs(data.qfrc_passive - data.qfrc_fluid).max()))
    return data.qfrc_fluid


def accumulate_substep_wrench(model, data, acc, dt, selector):
    """Add one substep's worth of impulse to the current recorded interval.
    free-joint qvel[3:6] is BODY-frame angular velocity, so the generalized
    angular forces are BODY-frame torques and are rotated to world with R(t)."""
    R = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]]).as_matrix()
    com = data.xipos[1].copy()
    qf = fluid_qfrc(data)
    acc["Jc"] += data.qfrc_constraint[:3] * dt
    acc["Jf"] += qf[:3] * dt
    acc["Tc_q"] += (R @ data.qfrc_constraint[3:6]) * dt
    acc["Tf"] += (R @ qf[3:6]) * dt
    fw, pts = selector.world_forces(model, data)
    for F, p in zip(fw, pts):
        acc["Tc"] += np.cross(p - com, F) * dt


def make_wrench_save_dict(wrench, dt_record, substeps, verification):
    """Element [4] of a saved trajectory. Shared by capture_mojoco_traj.py and
    replicate_paper_tosses.py so both datasets carry byte-identical label
    schemas and evaluate_force_model.py reads either without special cases."""
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(
        J_contact=t(wrench["J_contact"]), J_fluid=t(wrench["J_fluid"]),
        tau_contact=t(wrench["tau_contact"]),
        tau_contact_qfrc=t(wrench["tau_contact_qfrc"]),
        tau_fluid=t(wrench["tau_fluid"]),
        qvel_lin=t(wrench["qvel_lin"]), qvel_ang_body=t(wrench["qvel_ang_body"]),
        dt_record=dt_record, substeps=substeps,
        convention=("J*/tau* are impulses over recorded interval t->t+1, world "
                    "frame, torque about the body COM. J[t] pairs with the model "
                    "prediction whose input window ends at frame t."),
        verification=verification,
    )


def verify_wrench(qvel_lin, wrench, mass, g_vec, dt_record):
    """Momentum identity m*dv = m*g*DT + J_contact + J_fluid. This checks OUR
    BOOKKEEPING, not MuJoCo's physics: a wrong array, sign, frame, or substep
    alignment breaks it by ~100%, while integrator details leave ~0.01%.
    Returned residual is RELATIVE to the per-frame gravity impulse."""
    dv = mass * (qvel_lin[1:] - qvel_lin[:-1])
    grav = mass * g_vec * dt_record
    resid = dv - (grav + wrench["J_contact"] + wrench["J_fluid"])
    grav_imp = float(np.linalg.norm(mass * g_vec * dt_record)) or 1.0
    rel = np.linalg.norm(resid, axis=1) / grav_imp
    contact = np.linalg.norm(wrench["J_contact"], axis=1) > 1e-12
    return dict(rel_max=float(rel.max()), rel_mean=float(rel.mean()),
                rel_max_air=float(rel[~contact].max()) if (~contact).any() else 0.0,
                rel_max_contact=float(rel[contact].max()) if contact.any() else 0.0,
                mean_resid=resid.mean(axis=0), grav_imp=grav_imp)


def audit_passive_terms(model, data=None):
    """ONE-TIME model-level check that the passive force contains only fluid.
    Every source listed must read 0, otherwise J_fluid / tau_fluid are polluted
    with spring, damper, or gravity-compensation force and the "fluid ground
    truth" claim in the thesis is false. Call once after loading the model."""
    print("\n" + "=" * 70)
    print("PASSIVE-FORCE AUDIT - is the passive force really the FLUID force?")
    print("=" * 70)
    print(f"  mujoco version          = {mujoco.__version__}")
    have = data is not None and hasattr(data, "qfrc_fluid")
    print(f"  data.qfrc_fluid present = {have}"
          + ("   (reading the fluid term directly)" if have else
             "   <-- needs mujoco>=3.0; FALLING BACK to qfrc_passive"))

    sources = [
        ("joint stiffness (springs)", model.jnt_stiffness),
        ("dof damping",               model.dof_damping),
        ("dof frictionloss",          model.dof_frictionloss),
        ("body gravcomp",             model.body_gravcomp),
        ("tendon stiffness",          model.tendon_stiffness),
        ("tendon damping",            model.tendon_damping),
    ]
    dirty = False
    for name, arr in sources:
        v = float(np.abs(np.asarray(arr)).max()) if np.asarray(arr).size else 0.0
        dirty |= (v != 0.0)
        print(f"  max {name:<26s} = {v:<12.6g}"
              + ("  <-- POLLUTES the fluid label" if v != 0.0 else ""))
    for name, n in (("tendons", model.ntendon), ("flexes", model.nflex),
                    ("plugins", model.nplugin)):
        dirty |= (n != 0)
        print(f"  n_{name:<27s} = {n}" + ("  <-- POLLUTES the fluid label" if n else ""))

    print(f"  medium: density={model.opt.density:.6g} viscosity={model.opt.viscosity:.6g} "
          f"wind={np.asarray(model.opt.wind).round(3).tolist()}")
    gf = np.asarray(model.geom_fluid).reshape(model.ngeom, -1)
    ell = [i for i in range(model.ngeom) if gf[i, 0] != 0.0]
    print(f"  geoms using the ELLIPSOID fluid model: {ell}"
          + ("   (empty -> inertia-based / Stokes drag only)" if not ell else ""))
    print("  " + ("CLEAN: passive force can only be fluid." if not dirty else
                  "DIRTY: fix the flagged entries before trusting J_fluid."))
    print("  Runtime max |qfrc_passive - qfrc_fluid| is printed with the wrench")
    print("  diagnostics below; it must be exactly 0.0.")
    print("=" * 70 + "\n")


#This file will generate all of the training data we will use to train the contact and fluid GNN. Using mojoco's built in
# physics engine, it will simulate the trajectory of a cube under the influence of different wind vectors, initial positions,
#orientations, velocities, and angular velocities. The trajectories are saved as .pt files, which contain the trajectory data 
#as well as the initial conditions and parameters for each simulation. We will use this data to train our GNN to predict the effect
# of wind on the trajectory of the cube, and to learn the underlying physics of the system.

#Generates a random quaternion
def random_quat():
    return Rotation.random().as_quat(scalar_first=True) 


#Takes in the initial conditions and parameters for a MuJoCo simulation, runs the simulation for a specified number of steps, 
#and returns the trajectory of the cube's position and orientation over time.
def collect_trajectory(model, wind_vector, initial_pos, initial_quat, initial_vel,
                       initial_angvel, mass, n_steps=1000,substeps= 10, visualize=False,
                       record_wrench=True):
    
    #Loads the model into MuJoCo, sets the initial conditions and parameters and resets the simulation data.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    model.opt.wind[:] = wind_vector
    model.body_mass[1] = mass
    data.qpos[:3] = initial_pos
    data.qpos[3:7] = initial_quat
    data.qvel[:3] = initial_vel
    data.qvel[3:6] = initial_angvel

    mujoco.mj_forward(model, data)

    states = []
    qvel_lin, qvel_ang_body = [], []
    intervals = []
    selector = ContactFrameSelector()
    dt = model.opt.timestep

    def _record_frame():
        states.append(np.concatenate([
            data.qpos[:3].copy(),
            data.qpos[3:7].copy(),
        ]))
        if record_wrench:
            qvel_lin.append(data.qvel[:3].copy())
            qvel_ang_body.append(data.qvel[3:6].copy())

    def _run_block():
        """One recorded interval = `substeps` physics steps, with the contact
        and fluid impulses accumulated across all of them."""
        acc = new_wrench_interval() if record_wrench else None
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            if record_wrench:
                accumulate_substep_wrench(model, data, acc, dt, selector)
        return acc

    #If visualize is True, it will launch the MuJoCo viewer and step through the simulation, 
    #rendering the cube's motion
    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            time.sleep(10)  # wait for viewer to initialize
            for i in range(n_steps):
                # Take multiple small physics steps per recorded frame
                acc = _run_block()
                # The first block precedes frame 0, so it labels no interval.
                # Every later block is the transition frame i-1 -> frame i.
                if record_wrench and i > 0:
                    intervals.append(acc)
                _record_frame()
                viewer.sync()
                time.sleep(.05)
            # Keep window open after sim finishes
            input("Press Enter to close viewer...")
    
    #If not visualizing, it will just run the simulation and record the trajectory data without rendering.
    else:
        for i in range(n_steps):
            acc = _run_block()
            if record_wrench and i > 0:
                intervals.append(acc)
            _record_frame()

    trajectory = np.stack(states)

    if not record_wrench:
        return trajectory, None

    stack = lambda key: np.stack([iv[key] for iv in intervals])
    wrench = dict(
        J_contact=stack("Jc"), J_fluid=stack("Jf"),
        tau_contact=stack("Tc"), tau_contact_qfrc=stack("Tc_q"),
        tau_fluid=stack("Tf"),
        qvel_lin=np.stack(qvel_lin), qvel_ang_body=np.stack(qvel_ang_body),
        contact_xcheck=selector.max_residual,
    )
    return trajectory, wrench

def generate_paper_matched_toss(mass):
    pos = np.array([np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(0.119, 0.166)])      # measured z0 5-95%
    quat = random_quat()
    # horizontal: fixed-ish SPEED, uniform direction (measured: launcher-style)
    speed = np.random.uniform(0.99, 1.27)                  # |vh0| 5-95%
    theta = np.random.uniform(0, 2*np.pi)
    vz    = np.random.uniform(-0.329, 0.173)               # signed vz0 5-95%
    vel   = np.array([speed*np.cos(theta), speed*np.sin(theta), vz])
    # spin: magnitude x random axis (measured |omega| 5-95%)
    w_mag = np.random.uniform(1.6, 6.4)
    axis  = np.random.randn(3); axis /= np.linalg.norm(axis) + 1e-8
    angvel = w_mag * axis
    return {'wind': np.zeros(3), 'pos': pos, 'quat': quat,
            'vel': vel, 'angvel': angvel, 'mass': mass, 'type': 'toss'}


#This function generates random initial conditions and parameters for the MuJoCo simulation, including a random wind vector,
#initial position, orientation, velocity, and angular velocity of the cube, and returns them in a dictionary.
def generate_toss_params(mass, wind_range, horizontal_pos_range, vertical_pos_range,
                         horizontal_speed_range, vertical_speed_range, angvel_range,
                         fix_wind_dir=False, wind_dir_fixed=(1.0, 0.0, 0.0),
                         fix_toss_dir=False, toss_dir_fixed=(1.0, 0.0, 0.0)):
    # --- wind ---
    if fix_wind_dir:
        wind_dir = np.array(wind_dir_fixed, dtype=float); wind_dir[2] = 0.0
        wind_dir /= (np.linalg.norm(wind_dir) + 1e-8)
    else:
        wind_dir = np.random.randn(3); wind_dir[2] = 0
        wind_dir /= (np.linalg.norm(wind_dir) + 1e-8)
    wind_vec = wind_dir * np.random.uniform(wind_range[0], wind_range[1])

    pos = np.array([
        np.random.uniform(horizontal_pos_range[0], horizontal_pos_range[1]),
        np.random.uniform(horizontal_pos_range[0], horizontal_pos_range[1]),
        np.random.uniform(vertical_pos_range[0], vertical_pos_range[1]),
    ])
    quat = random_quat()

    # --- horizontal toss velocity ---
    if fix_toss_dir:
        d = np.array(toss_dir_fixed, dtype=float); d[2] = 0.0
        d /= (np.linalg.norm(d) + 1e-8)
        speed = np.random.uniform(0.0, abs(horizontal_speed_range[1]))   # magnitude only
        vx, vy = speed * d[0], speed * d[1]
    else:
        vx = np.random.uniform(horizontal_speed_range[0], horizontal_speed_range[1])
        vy = np.random.uniform(horizontal_speed_range[0], horizontal_speed_range[1])
    vz = np.random.uniform(vertical_speed_range[0], vertical_speed_range[1])
    vel = np.array([vx, vy, vz])

    angvel = np.random.uniform(angvel_range[0], angvel_range[1], size=3)
    return {'wind': wind_vec, 'pos': pos, 'quat': quat, 'vel': vel,
            'angvel': angvel, 'mass': mass, 'type': 'toss'}

def generate_sliding_params(mass, wind_range, sliding_speed_range, angvel_z_range,
                            half_width=0.0524,
                            fix_wind_dir=False, wind_dir_fixed=(1.0, 0.0, 0.0),
                            fix_slide_dir=False, slide_dir_fixed=(1.0, 0.0, 0.0)):
    # --- wind (same logic as toss) ---
    if fix_wind_dir:
        wind_dir = np.array(wind_dir_fixed, dtype=float); wind_dir[2] = 0.0
        wind_dir /= (np.linalg.norm(wind_dir) + 1e-8)
    else:
        wind_dir = np.random.randn(3); wind_dir[2] = 0
        wind_dir /= (np.linalg.norm(wind_dir) + 1e-8)
    wind_vec = wind_dir * np.random.uniform(wind_range[0], wind_range[1])

    pos = np.array([
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(-0.2, 0.2),
        half_width,
    ])

    theta = np.random.uniform(0, 2 * np.pi)
    quat = Rotation.from_euler('z', theta).as_quat(scalar_first=True)

    # --- horizontal slide velocity ---
    speed = np.random.uniform(sliding_speed_range[0], sliding_speed_range[1])
    if fix_slide_dir:
        d = np.array(slide_dir_fixed, dtype=float); d[2] = 0.0
        d /= (np.linalg.norm(d) + 1e-8)
        vel = np.array([speed * d[0], speed * d[1], 0.0])
    else:
        angle = np.random.uniform(0, 2 * np.pi)
        vel = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    angvel = np.array([0.0, 0.0, np.random.uniform(angvel_z_range[0], angvel_z_range[1])])
    return {'wind': wind_vec, 'pos': pos, 'quat': quat, 'vel': vel,
            'angvel': angvel, 'mass': mass, 'type': 'sliding'}


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = os.path.join(script_dir, "data", "mojoco_paper_matched_toss")
    os.makedirs(save_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(os.path.join(script_dir, "cube.xml"))
    print("Friction parameters for the cube and floor geoms:")
    print(model.geom_friction)
    audit_passive_terms(model, mujoco.MjData(model))

    #----- Dataset parameters -----

    # Record MuJoCo's true contact/fluid wrenches alongside each trajectory.
    # Costs a few % of generation time and removes the need for a separate
    # add_wrench_labels.py pass. Labels are validation-only - never trained on.
    RECORD_WRENCH = True

    Match_paper_traj = True

    n_trajectories = 570
    n_steps = 100
    substeps = 50
    visualize_first = True

    #What percentage of the total trajectories should be sliding-only (0.0 to 1.0)
    sliding_percentage = 0

    #----- Shared parameters -----
    wind_range = (0, 0)
    mass = 0.37

    FIX_WIND_DIR = False
    FIX_TOSS_DIR = False

    WIND_DIR = (0.0, 1.0, 0.0)
    TOSS_DIR = (1.0, 0.0, 0.0)

    #----- Toss-specific parameters -----
    toss_horizontal_pos_range = (-0.2, 0.2)
    toss_vertical_pos_range = (11, 11) #CHANGED THIS TO FIXED START HEIGHT SO IT NEVER CONTACTS
    toss_horizontal_speed_range = (-1.25, 1.25)
    toss_vertical_speed_range = (-0.3, 0.3)
    toss_angvel_range = (-3, 3)

    #----- Sliding-specific parameters -----
    sliding_speed_range = (0.0, 2.0)
    sliding_angvel_z_range = (-3, 3)

    #----- Build the schedule of which indices are sliding vs toss -----
    n_sliding = int(n_trajectories * sliding_percentage)
    n_toss = n_trajectories - n_sliding

    #Spread sliding trajectories evenly throughout the dataset so that 
    #training batches always contain a mix of both types.
    sliding_indices = set()
    if n_sliding > 0:
        spacing = n_trajectories / n_sliding
        for k in range(n_sliding):
            sliding_indices.add(int(k * spacing))

    print(f"Total: {n_trajectories} | Toss: {n_toss} | Sliding: {n_sliding} ({sliding_percentage*100:.0f}%)")

    verif_stats = []

    for i in range(n_trajectories):
        if not Match_paper_traj:
            if i in sliding_indices:
                params = generate_sliding_params(
                    mass=mass, wind_range=wind_range,
                    sliding_speed_range=sliding_speed_range,
                    angvel_z_range=sliding_angvel_z_range,
                    fix_wind_dir=FIX_WIND_DIR, wind_dir_fixed=WIND_DIR,
                    fix_slide_dir=FIX_TOSS_DIR, slide_dir_fixed=TOSS_DIR,   # or separate SLIDE_DIR flags
                )
            else:
                params = generate_toss_params(
                    mass=mass, wind_range=wind_range,
                    horizontal_pos_range=toss_horizontal_pos_range,
                    vertical_pos_range=toss_vertical_pos_range,
                    horizontal_speed_range=toss_horizontal_speed_range,
                    vertical_speed_range=toss_vertical_speed_range,
                    angvel_range=toss_angvel_range,
                    fix_wind_dir=FIX_WIND_DIR, wind_dir_fixed=WIND_DIR,
                    fix_toss_dir=FIX_TOSS_DIR, toss_dir_fixed=TOSS_DIR,
                )
        else:
            params = generate_paper_matched_toss(mass=mass)

        print(f"Trajectory {i} [{params['type']}]: wind={params['wind'].round(2)}, "
              f"mass={params['mass']:.2f}, "
              f"pos={params['pos'].round(2)}, "
              f"vel={params['vel'].round(3)}, "
              f"angvel={params['angvel'].round(3)}")

        traj, wrench = collect_trajectory(
            model=model,
            wind_vector=params['wind'],
            initial_pos=params['pos'],
            initial_quat=params['quat'],
            initial_vel=params['vel'],
            initial_angvel=params['angvel'],
            mass=params['mass'],
            n_steps=n_steps,
            substeps=substeps,
            visualize=True if i == 0 and visualize_first else False,
            record_wrench=RECORD_WRENCH,
        )

        traj_tensor = torch.tensor(traj, dtype=torch.float32)
        save_data = [
            traj_tensor,
            torch.tensor(params['wind'], dtype=torch.float32),
            params['mass'],
            params,
        ]

        if wrench is not None:
            dt_record = model.opt.timestep * substeps
            v = verify_wrench(wrench["qvel_lin"], wrench, params['mass'],
                              model.opt.gravity.copy(), dt_record)
            verif_stats.append(v)
            # Element [4]: same schema add_wrench_labels.py produced, so
            # evaluate_force_model.py reads these files unchanged.
            save_data.append(make_wrench_save_dict(
                wrench, dt_record, substeps,
                verification=dict(momentum_resid_rel=v["rel_max"],
                                  momentum_resid_air=v["rel_max_air"],
                                  momentum_resid_contact=v["rel_max_contact"],
                                  contact_xcheck=wrench["contact_xcheck"])))

            if i == 0:
                print("\n  --- wrench label diagnostics (trajectory 0) ---")
                print(f"  integrator={model.opt.integrator}  cone={model.opt.cone}  "
                      f"timestep={model.opt.timestep:.3e}  dt_record={dt_record:.6f}")
                print(f"  per-frame gravity impulse = {v['grav_imp']:.6f} N*s")
                print(f"  momentum residual (% of that): max {100*v['rel_max']:.4f}%  "
                      f"| airborne {100*v['rel_max_air']:.4f}%  "
                      f"| contact {100*v['rel_max_contact']:.4f}%")
                print(f"  mean residual vector = {v['mean_resid']} N*s")
                print(f"     (systematic bias along one axis = missing force term;")
                print(f"      zero-mean scatter = integrator round-off, harmless)")
                print(f"  contact-frame cross-check = {wrench['contact_xcheck']:.3e} N")
                print(f"  max |qfrc_passive - qfrc_fluid| = "
                      f"{PASSIVE_AUDIT['max_nonfluid']:.3e}  "
                      f"({PASSIVE_AUDIT['substeps']} substeps; MUST be 0.0)\n")

        save_path = os.path.join(save_dir, f"{i}.pt")
        torch.save(save_data, save_path)
        print(f"Saved trajectory {i}")
    
    print(f"\nSaved {n_trajectories} trajectories to {save_dir}/")
    print(f"  Toss: {n_toss} | Sliding: {n_sliding}")

    if verif_stats:
        rel = np.array([v["rel_max"] for v in verif_stats])
        air = np.array([v["rel_max_air"] for v in verif_stats])
        con = np.array([v["rel_max_contact"] for v in verif_stats])
        print(f"\n  Wrench labels written for all {len(verif_stats)} trajectories.")
        print(f"  Momentum identity (% of per-frame gravity impulse):")
        print(f"     overall  median {100*np.median(rel):.4f}%   max {100*rel.max():.4f}%")
        print(f"     airborne median {100*np.median(air):.4f}%   max {100*air.max():.4f}%")
        print(f"     contact  median {100*np.median(con):.4f}%   max {100*con.max():.4f}%")
        print(f"  Anything under ~1% is bookkeeping-clean; a real frame/sign bug"
              f" shows up near 100%.")
        print(f"  Fluid-label purity: max |qfrc_passive - qfrc_fluid| = "
              f"{PASSIVE_AUDIT['max_nonfluid']:.3e} over "
              f"{PASSIVE_AUDIT['substeps']} substeps (0.0 = labels are pure fluid).")
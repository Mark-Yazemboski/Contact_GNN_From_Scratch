"""
run_force_multi_step.py

NEW FILE - entry point for training the FORCE-based GNS. Mirrors the config
style of run_multi_step.py so switching between the two pipelines is a
one-file change. Evaluation/figures live in evaluate_force_model.py.

THE STAGED PLAN (one variable at a time):
  Stage 1 (parity gate): use_drag_baseline=False, all physics-loss weights 0,
          no-wind replica dataset, use_wind_feature=False. Success bar: match
          the acceleration model's rollout numbers. This isolates "does the
          force representation cost anything" before adding physics.
  Stage 2: use_drag_baseline=True on the wind datasets; run the extrapolation
          four-cell (train 0-5 m/s, test at 8) against the accel baseline.
  Stage 3: turn on w_diss (then w_sparse / w_fluid_reg) ONE AT A TIME,
          watching the raw physics-term magnitudes printed each epoch. h_diss
          matters most: it is what disambiguates friction from horizontal
          fluid force during a slide.
"""

import os
import torch
import wall
from train_force_gns import train_force_gnn
from evaluate_force_model import evaluate_force_model
from visualize_force_model import visualize_force_rollout
from run_report import save_run_report
from generate_node_states import BLOCK_HALF_WIDTH


#This function will take in the number of trajectories we are training on, and the 
#number of optimizer steps we want to hit, and some other perameters, and calculate
#how many epochs we need to train for.
def compute_epochs(num_trajectories, target_steps, batch_size, accumulation_steps, traj_timesteps=100, history=2):
    usable_per_traj = traj_timesteps - history - 1
    total_samples = num_trajectories * usable_per_traj
    num_batches = (total_samples + batch_size - 1) // batch_size  # ceil division
    effective_accum = min(accumulation_steps, num_batches)
    steps_per_epoch = num_batches // effective_accum
    steps_per_epoch = max(steps_per_epoch, 1)
    epochs = (target_steps + steps_per_epoch - 1) // steps_per_epoch
    return epochs

torch.set_float32_matmul_precision('high')

script_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

# Point this at a *_wrench folder once add_wrench_labels.py has run, so the
# same files carry ground-truth wrenches for evaluate_force_model.py.
trajectory_folder = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")

#This is the number of timesteps in each trajectory. 
traj_timesteps = 200

Num_total_trajectories = 569
training_percentage = 0.5
validation_percentage = 0.3

Num_train = int(training_percentage * Num_total_trajectories)
Num_val = int(validation_percentage * Num_total_trajectories)

#---------------------------------------------------------------------------------------------------------
#This is the number of training trajectories to actually use.
# Override for experiments with smaller training sets
Used_Num_train_trajectories = 256
#---------------------------------------------------------------------------------------------------------


train_range = range(0, Used_Num_train_trajectories)
val_range = range(Num_train, Num_train + Num_val)
test_range = range(Num_train + Num_val, Num_total_trajectories - 1)

print(f"Training range: {train_range}")
print(f"Validation range: {val_range}")
print(f"Test range: {test_range}")

# ----------------------------------------------------------------------
# Architecture / recipe (matches the current best acceleration recipe)
# ----------------------------------------------------------------------
nodes_per_edge = 2
K_nearest_neighbors = 3
message_passing_layers = 5
repeat_blocks = 1
Latent_dimension = 128
pos_history = 3

batch_size = 512
learning_rate = 1e-4
steps = 1000000
noise_scale = 3e-4 * BLOCK_HALF_WIDTH            # meters/step, same as accel runs
rot_noise_scale = None                           # None -> noise_scale / half_width (rad)

multistep = 1                                    # the K=8 recipe
curriculum_epochs = 50                           # epochs per phase of [1,2,4,8]
curriculum_schedule = None                       # None -> powers of 2 up to multistep
Learning_Rate_Scheduler = None                # "decay", "cosine", or None
accumulation_steps = 1

validation_check_interval = 10
epoch_checkpoint_interval = 100

weights_only_load = False                        # MuJoCo-generated data
unscale_trajectory_data = False

epochs = compute_epochs(Used_Num_train_trajectories, steps, batch_size, accumulation_steps, traj_timesteps=traj_timesteps, history=pos_history)


# ----------------------------------------------------------------------
# Force-model physics (the new knobs)
# ----------------------------------------------------------------------
DT = 1.0 / 148.0        # replica record rate. NOTE: generate_node_states.DT_RECORD
                        # is 0.00674 (0.0001348*50) while the replica records at
                        # 1/148 = 0.006757 - the known small mismatch. The wind
                        # FEATURE (imported builder) keeps the old constant for
                        # parity with existing runs; the DYNAMICS here uses the
                        # correct 1/148.
GRAVITY = None          # None -> read from replica_physics in the data (9.615)
MASS = 0.37

use_wind_feature = False        # Stage 1: off. Stage 2+: on for wind datasets.
use_drag_baseline = True        # analytic drag at COM (calibrated k/m); the
                                # anchor term assumes this is the fluid center
K_OVER_M = 0.0285               # from wind_error_analysis.py drag calibration

contact_d0 = 0.02               # soft geometric contact gate center (m)
contact_tau = 0.005             # gate width (m)

# Loss: "accel" = per-node acceleration MSE, the SAME objective as the
# acceleration model's _unroll_chain_loss_accel, so the parity comparison is
# exact and the logged loss number is directly comparable. "position" = the
# block-width position MSE. Keep "accel" until parity is established.
loss_mode = "accel"

# Training budget: max_steps counts OPTIMIZER steps (the paper's 1M-step
# convention) and stays comparable across batch size and multistep K, unlike
# epochs. When set, it OVERRIDES `epochs`. Set to None to use `epochs`.
MAX_STEPS = 1_000_000

# ----------------------------------------------------------------------
# PHYSICS-INFORMED LOSS (proposal Eq. 5-6; one function per term in
# physics_losses.py). Raw magnitudes print every epoch - calibrate each
# gamma so (gamma * raw) is ~1-10%% of the position loss after epoch 1.
#
# What the wrench labels showed and which term answers it:
#   fluid channel carried ~0.2 mg of friction (= mu m g), at every wind level
#     -> w_fluid_anchor pins fluid to the analytic drag law
#     -> w_fluid_smooth forbids the jumpy, contact-synchronized compensation
#        (the chaotic pink arrow) - NEEDS multistep >= 2
#     -> w_diss gives the displaced friction a correctly-structured home:
#        anti-parallel to slip, proportional to the local normal force,
#        one global mu. mu is LEARNABLE by default, so the model recovers
#        the friction coefficient the same way it recovered the drag
#        coefficient (replica ground truth: mu = 0.198).
#   h_pen needs no weight: normal forces are >= 0 by construction (softplus). Set each so its weighted term
# is ~1-10% of the position loss at init; raw magnitudes print every epoch.
w_diss = 1e-3          # gamma_1: Coulomb dissipation on sliding contact
w_sparse = 0.0         # contact sparsity - leave off initially (shrinks
                       # legitimate resting normal forces too)
w_fluid_anchor = 1e-2  # gamma_3a: fluid FORCE == analytic drag law
w_fluid_torque = 1e-2  # gamma_3c: fluid TORQUE == 0 (all rotation from contact)
w_fluid_smooth = 1e-2  # gamma_3b: fluid force smooth in time (K >= 2 only)

MU_INIT = 0.2          # friction coefficient init
LEARN_MU = True        # recover mu from data (the drag-coefficient story)
FIX_MU = None          # or e.g. 1.9/9.615 to hard-fix it (ablation arm)

# ----------------------------------------------------------------------
# Naming / paths
# ----------------------------------------------------------------------
extra_name = "force_stage1"      # CHANGE PER EXPERIMENT
model_folder_path = os.path.join(script_dir, "models", extra_name)
os.makedirs(model_folder_path, exist_ok=True)
save_model_path = os.path.join(
    model_folder_path, f"{Used_Num_train_trajectories}_force_gns_model.pt")

# Everything runs from this one file so a single batch job on ROAR trains,
# evaluates, and renders the GIFs without a second submission.
Train_model = True
Evaluate_model = True
Visualize_model = True

# Which test trajectories to render as GIFs. Keep this short - each one is a
# full rollout plus a matplotlib animation, so ~10-30 s apiece.
VISUALIZE_TRAJECTORIES = [test_range[0], test_range[len(test_range) // 2]]
VISUALIZE_SHOW = False          # False on a compute node (no display)

# One canonical row per force run, kept in its OWN master file so the force
# architecture's numbers never get mixed into the acceleration model's
# all_runs_master.csv. Opens directly in Excel.
Save_run_report = True
FORCE_MASTER_CSV = os.path.join(script_dir, "models", "all_force_runs_master.csv")

# ----------------------------------------------------------------------
if Train_model:
    train_force_gnn(
        Wall=Floor,
        train_range=train_range,
        val_range=val_range,
        save_model_path=save_model_path,
        trajectory_folder=trajectory_folder,
        epochs=epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        h=pos_history,
        message_passing_layers=message_passing_layers,
        repeat_blocks=repeat_blocks,
        latent_dim=Latent_dimension,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data,
        noise_scale=noise_scale,
        rot_noise_scale=rot_noise_scale,
        multistep=multistep,
        curriculum_epochs=curriculum_epochs,
        curriculum_schedule=curriculum_schedule,
        Learning_Rate_Scheduler=Learning_Rate_Scheduler,
        use_wind=use_wind_feature,
        dt=DT,
        gravity=GRAVITY,
        mass=MASS,
        use_drag_baseline=use_drag_baseline,
        k_over_m=K_OVER_M,
        contact_d0=contact_d0,
        contact_tau=contact_tau,
        loss_mode=loss_mode,
        w_diss=w_diss, w_sparse=w_sparse,
        w_fluid_anchor=w_fluid_anchor, w_fluid_torque=w_fluid_torque,
        w_fluid_smooth=w_fluid_smooth,
        mu_init=MU_INIT, learn_mu=LEARN_MU, fix_mu=FIX_MU,
        max_steps=MAX_STEPS,
        validation_check_interval=validation_check_interval,
        epoch_checkpoint_interval=epoch_checkpoint_interval,
    )

# ----------------------------------------------------------------------
if Evaluate_model:
    print("\n" + "#" * 70)
    print("# EVALUATION")
    print("#" * 70)
    metrics = evaluate_force_model(
        model_folder=model_folder_path,
        data_folder=trajectory_folder,
        test_indices=test_range,
        weights_only=weights_only_load,
        unscale=unscale_trajectory_data,
    )
    print("\nSummary:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items()})

    if Save_run_report:
        settings = dict(
            architecture="force",           # distinguishes these rows at a glance
            dataset=trajectory_folder,
            n_train=Used_Num_train_trajectories,
            train_range=f"{train_range.start}-{train_range.stop}",
            val_range=f"{val_range.start}-{val_range.stop}",
            test_range=f"{test_range.start}-{test_range.stop}",
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=K_nearest_neighbors,
            message_passing_layers=message_passing_layers,
            repeat_blocks=repeat_blocks,
            latent_dim=Latent_dimension,
            pos_history=pos_history,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            noise_scale=noise_scale,
            rot_noise_scale=rot_noise_scale,
            multistep=multistep,
            curriculum_epochs=curriculum_epochs,
            scheduler=Learning_Rate_Scheduler,
            loss_mode=loss_mode,
            use_wind=use_wind_feature,
            use_drag_baseline=use_drag_baseline,
            k_over_m=K_OVER_M,
            contact_d0=contact_d0,
            contact_tau=contact_tau,
            dt=DT, gravity=GRAVITY, mass=MASS,
            max_steps=MAX_STEPS,
            w_diss=w_diss, w_sparse=w_sparse,
            w_fluid_anchor=w_fluid_anchor, w_fluid_torque=w_fluid_torque,
            w_fluid_smooth=w_fluid_smooth,
            learn_mu=LEARN_MU, fix_mu=FIX_MU,
        )
        save_run_report(model_folder_path, settings, metrics, slopes=[],
                        run_name=extra_name, master_csv=FORCE_MASTER_CSV)

# ----------------------------------------------------------------------
if Visualize_model:
    print("\n" + "#" * 70)
    print("# VISUALIZATION")
    print("#" * 70)
    for traj_idx in VISUALIZE_TRAJECTORIES:
        try:
            out = visualize_force_rollout(
                model_folder=model_folder_path,
                data_folder=trajectory_folder,
                trajectory=int(traj_idx),
                show=VISUALIZE_SHOW,
                weights_only=weights_only_load,
                unscale=unscale_trajectory_data,
            )
            print(f"  wrote {out}")
        except Exception as e:
            # A failed GIF must never take down a finished training run.
            print(f"  visualization of traj {traj_idx} FAILED: {type(e).__name__}: {e}")

print("\nAll done.")

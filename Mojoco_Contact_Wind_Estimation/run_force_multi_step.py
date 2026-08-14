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
from generate_node_states import BLOCK_HALF_WIDTH

torch.set_float32_matmul_precision('high')

script_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

# Point this at a *_wrench folder once add_wrench_labels.py has run, so the
# same files carry ground-truth wrenches for evaluate_force_model.py.
trajectory_folder = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")

Num_total_trajectories = 569
training_percentage = 0.5
validation_percentage = 0.3

Num_train = int(training_percentage * Num_total_trajectories)
Num_val = int(validation_percentage * Num_total_trajectories)

Used_Num_train_trajectories = Num_train          # override for data-scaling runs

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
epochs = 400
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
use_drag_baseline = False       # Stage 2: True (analytic drag at COM, calibrated k/m)
K_OVER_M = 0.0285               # from wind_error_analysis.py drag calibration

contact_d0 = 0.02               # soft geometric contact gate center (m)
contact_tau = 0.005             # gate width (m)

# Loss: "accel" = per-node acceleration MSE, the SAME objective as the
# acceleration model's _unroll_chain_loss_accel, so the parity comparison is
# exact and the logged loss number is directly comparable. "position" = the
# block-width position MSE. Keep "accel" until parity is established.
loss_mode = "accel"

# Physics-loss weights (Stage 3, one at a time). Set each so its weighted term
# is ~1-10% of the position loss at init; raw magnitudes print every epoch.
w_diss = 0.0
w_sparse = 0.0
w_fluid_reg = 0.0    # L2 on the raw fluid-head outputs (keep residual small)

# ----------------------------------------------------------------------
# Naming / paths
# ----------------------------------------------------------------------
extra_name = "force_stage1"      # CHANGE PER EXPERIMENT
model_folder_path = os.path.join(script_dir, "models", extra_name)
os.makedirs(model_folder_path, exist_ok=True)
save_model_path = os.path.join(
    model_folder_path, f"{Used_Num_train_trajectories}_force_gns_model.pt")

Train_model = True

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
        w_diss=w_diss, w_sparse=w_sparse, w_fluid_reg=w_fluid_reg,
        validation_check_interval=validation_check_interval,
        epoch_checkpoint_interval=epoch_checkpoint_interval,
    )

print("\nTraining done. Next:")
print(f"  python evaluate_force_model.py   (set MODEL_FOLDER = models/{extra_name})")

"""
run_multi_step.py   —  entry point for the ACCELERATION (old) architecture.

Updated to log into the same master CSV as the force pipeline so the two
architectures can be compared in one place.

WHAT CHANGED, and why
  1. save_run_report() was called WITHOUT run_name / master_csv, so rows never
     reached all_force_runs_master.csv. Both are passed now, and settings
     carries architecture="accel" so you can tell the rows apart.
  2. train_range / val_range were not recorded. For the sample-efficiency
     sweep the training SUBSET is the thing that varies between runs, so it
     has to be in the row or the provenance is gone.
  3. `metrics` and `slopes` were used at the bottom but only assigned inside
     `if display_stats:` / `if plot_phase_curves:`. Turning either off raised
     NameError after a full training run. Both are initialised up front.
  4. model_folder_path was hardcoded to models/Testing, so every run
     overwrote the last one. It is derived from RUN_NAME now.

Everything you don't need for the comparison (GIFs, augmentation previews,
phase curves, loss curves) is off by default. Turn them back on below.

SET RUN_NAME, N_TRAIN and SEED_INDEX, then run.
"""

import os
import random

import torch

import wall
import generate_node_states
import evaluate_metrics
import display_results
import train_gnn_multi_step
from train_gnn_multi_step import GNSModel
from run_report import save_run_report

torch.set_float32_matmul_precision('high')
script_dir = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# 1.  WHAT THIS RUN IS
# ======================================================================

# Goes in the CSV as run_name, and names the model folder.
RUN_NAME = "Paper_Arc_256_train_1"      # CHANGE PER RUN

# Sample-efficiency sweep: how many training trajectories, and which seed.
N_TRAIN = 64                            # 4 / 16 / 64 / 128 / 256
train_start = 64
SEED_INDEX = 1                           

# Optimizer-step budget per sweep point. These are the totals the FORCE arm
# actually reached (read out of metrics.total_optimizer_steps), so matching
# them here gives both architectures equal training at every point. If you
# change the budget, change it for both arms or the comparison is not paired.
Epochs_BY_N_TRAIN = {
    4:     60_000,
    16:   60_000,
    64:   20,
    128:  20000,
    256:  10000,
}
# TARGET_STEPS = STEPS_BY_N_TRAIN.get(N_TRAIN, 490_150)

MASTER_CSV = os.path.join(script_dir, "models", "all_force_runs_master.csv")
model_folder_path = os.path.join(script_dir, "models", RUN_NAME)
save_model_path = os.path.join(model_folder_path, f"{N_TRAIN}_train_gns_model.pt")


# ======================================================================
# 2.  DATA
# ======================================================================

Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))
trajectory_folder = os.path.join(script_dir, "data/tosses_processed")

BLOCK_HALF_WIDTH = 0.0524
Num_total_trajectories = 569

Num_train_trajectories = int(0.5 * Num_total_trajectories)      # 284
Num_validation_trajectories = int(0.3 * Num_total_trajectories)  # 170
Num_test_trajectories = int(0.2 * Num_total_trajectories)

# Each seed trains on a DIFFERENT subset, so the error bars include
# training-subset variance and not just weight initialisation. Blocks are
# disjoint where they fit; past that they overlap and the caption has to say so.

train_range = range(train_start, train_start + N_TRAIN)
val_range = range(Num_train_trajectories,
                  Num_train_trajectories + Num_validation_trajectories)
test_range = range(Num_train_trajectories + Num_validation_trajectories,
                   Num_total_trajectories - 1)

if train_range.stop > Num_train_trajectories:
    raise SystemExit(
        f"train_range {train_range.start}-{train_range.stop} runs past the "
        f"training pool (0-{Num_train_trajectories}). Lower SEED_INDEX.")

print(f"run          : {RUN_NAME}")
print(f"train        : {N_TRAIN} trajectories, indices "
      f"{train_range.start}-{train_range.stop}  ")
print(f"val / test   : {val_range.start}-{val_range.stop} / "
      f"{test_range.start}-{test_range.stop}")


# ======================================================================
# 3.  MODEL AND TRAINING
# ======================================================================

nodes_per_edge = 2
K_nearest_neighbors = 3
message_passing_layers = 5
repeat_blocks = 1
Latent_dimension = 128
pos_history = 3

batch_size = 512
accumulation_steps = 1
learning_rate = 1e-4
noise_scale = 3e-4 * BLOCK_HALF_WIDTH

multistep = 1                 # 1 = single-step, the Allen et al. setting
impact_weight = 1
Learning_Rate_Scheduler = None
curriculum_epochs = 100

use_wind_feature = False      # real tosses have no wind

weights_only_load = True
unscale_trajectory_data = True

# !! CHECK THIS AGAINST YOUR DATA !!
# compute_epochs() converts a step budget into epochs, and it needs the real
# usable-samples-per-trajectory. The force runs imply about 98 usable samples
# per trajectory on tosses_processed (25,096 samples over 256 trajectories),
# which means traj_timesteps is near 102, NOT the 200 this file used to carry.
# At 200 you get roughly half the epochs you asked for and silently miss the
# step budget. Set it to whatever your trajectories actually are.
traj_timesteps = 102


def compute_epochs(num_trajectories, target_steps, batch_size,
                   accumulation_steps, traj_timesteps=100, history=2):
    usable_per_traj = traj_timesteps - history - 1
    total_samples = num_trajectories * usable_per_traj
    num_batches = (total_samples + batch_size - 1) // batch_size
    effective_accum = min(accumulation_steps, num_batches)
    steps_per_epoch = max(num_batches // effective_accum, 1)
    return (target_steps + steps_per_epoch - 1) // steps_per_epoch


# epochs = compute_epochs(N_TRAIN, TARGET_STEPS, batch_size, accumulation_steps,
#                         traj_timesteps=traj_timesteps, history=pos_history)
epochs = Epochs_BY_N_TRAIN[N_TRAIN]
usable = traj_timesteps - pos_history - 1
steps_per_epoch = max(((N_TRAIN * usable + batch_size - 1) // batch_size), 1)
print(f"               ~{steps_per_epoch} steps/epoch  ->  {epochs:,} epochs")

epoch_checkpoint_interval = 100
validation_check_interval = 10


# ======================================================================
# 4.  WHAT TO DO
# ======================================================================

Train = True
Evaluate = True
Save_run_report = True

# Off by default: none of this is needed for the centre / angle / penetration
# comparison, and the GIFs in particular cost real time on a compute node.
display_loss_curves = False
show_meshed_cube = False
show_augmentation = False
show_rollout = False
plot_phase_curves = False

rebuild_datasets = True
resume_training_checkpoint_path = None
copy_weights_only_path = None
inference_model_path = os.path.join(
    model_folder_path, f"{N_TRAIN}_train_gns_model_best_model.pt")


# ======================================================================
# 5.  TRAIN
# ======================================================================

if Train:
    if resume_training_checkpoint_path is not None and rebuild_datasets:
        print("Resume checkpoint detected. Forcing rebuild_datasets=False.")
        rebuild_datasets = False

    os.makedirs(model_folder_path, exist_ok=True)
    train_gnn_multi_step.train_gnn(
        Floor,
        train_range=train_range,
        val_range=val_range,
        save_train_dataset_path=os.path.join(
            script_dir, "data/pytorch_datasets/gns_train_dataset.pt"),
        save_val_dataset_path=os.path.join(
            script_dir, "data/pytorch_datasets/gns_val_dataset.pt"),
        save_model_path=save_model_path,
        rebuild_datasets=rebuild_datasets,
        epochs=epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        trajectory_folder=trajectory_folder,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        h=pos_history,
        message_passing_layers=message_passing_layers,
        repeat_blocks=repeat_blocks,
        copy_weights_only_path=copy_weights_only_path,
        resume_checkpoint_path=resume_training_checkpoint_path,
        epoch_checkpoint_interval=epoch_checkpoint_interval,
        validation_check_interval=validation_check_interval,
        noise_scale=noise_scale,
        multistep=multistep,
        latent_dim=Latent_dimension,
        use_rollout_validation=True,
        use_wind=use_wind_feature,
        impact_weight=impact_weight,
        Learning_Rate_Scheduler=Learning_Rate_Scheduler,
        curriculum_epochs=curriculum_epochs,
    )


# ======================================================================
# 6.  LOAD THE TRAINED MODEL
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_feat, edge_feat, edge_index, _ = generate_node_states.get_gns_features(
    Floor, throw_number=0, nodes_per_edge=nodes_per_edge,
    nearest_neighbors=K_nearest_neighbors, data_folder=trajectory_folder,
    weights_only=weights_only_load, unscale_data=unscale_trajectory_data,
    h=pos_history, use_wind=use_wind_feature)
node_dim = node_feat.shape[2]
edge_dim = edge_feat.shape[2]

nodes_body = torch.tensor(
    generate_node_states.mesh_cube_surface(BLOCK_HALF_WIDTH * 2, nodes_per_edge),
    dtype=torch.float32)

model = GNSModel(node_dim, edge_dim, latent_dim=Latent_dimension,
                 L=message_passing_layers, K=repeat_blocks)

load_model_path = inference_model_path or (
    os.path.splitext(save_model_path)[0] + "_best_model.pt")
print(f"Loading model from {load_model_path} for evaluation")

loaded_obj = torch.load(load_model_path, map_location=device)
if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
    loaded_obj = loaded_obj["model_state_dict"]
if any(k.startswith('_orig_mod.') for k in loaded_obj.keys()):
    loaded_obj = {k.replace('_orig_mod.', '', 1): v for k, v in loaded_obj.items()}
model.load_state_dict(loaded_obj)
model.to(device).eval()

norm_stats = torch.load(os.path.splitext(save_model_path)[0] + "_norms.pt",
                        map_location=device)
x_mean, x_std = norm_stats["x_mean"], norm_stats["x_std"]
e_mean, e_std = norm_stats["e_mean"], norm_stats["e_std"]
accel_std, accel_mean = norm_stats["acc_std"], norm_stats["acc_mean"]


# ======================================================================
# 7.  EVALUATE   (centre / angle / penetration)
# ======================================================================

# Initialised up front. These used to be assigned only inside the optional
# blocks below, so turning a flag off raised NameError after a full run.
metrics = {}
slopes = []

if Evaluate:
    print("\n" + "#" * 70)
    print("# EVALUATION")
    print("#" * 70)
    metrics = evaluate_metrics.evaluate_model(
        trajectory_folder, model, Floor, test_range, nodes_per_edge,
        K_nearest_neighbors, nodes_body, accel_std, accel_mean,
        x_mean, x_std, e_mean, e_std, weights_only_load,
        unscale_trajectory_data, pos_history, use_wind=use_wind_feature)

    if not isinstance(metrics, dict):
        print(f"  !! evaluate_model returned {type(metrics).__name__}, "
              f"not a dict — nothing to log")
        metrics = {}
    else:
        print("\n  metrics being written to the CSV:")
        for k, v in sorted(metrics.items()):
            print(f"    {k:<28} {v:.6g}" if isinstance(v, (int, float))
                  else f"    {k:<28} {v}")


# ======================================================================
# 8.  OPTIONAL EXTRAS  (all off by default)
# ======================================================================

if display_loss_curves:
    loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"
    if os.path.exists(loss_history_path):
        hist = torch.load(loss_history_path, map_location="cpu", weights_only=False)
        display_results.plot_loss_curves(
            train_loss_epochs=list(hist.get("train_loss_epochs", [])),
            train_loss_values=list(hist.get("train_loss_values", [])),
            val_loss_epochs=list(hist.get("val_loss_epochs", [])),
            val_loss_values=list(hist.get("val_loss_values", [])),
            title="Training and Validation Loss",
            save_path=os.path.join(model_folder_path, "loss_curve.png"),
            show_plot=False)
    else:
        print(f"Loss history file not found: {loss_history_path}")

if show_meshed_cube:
    display_results.display_meshed_cube(nodes_body, edge_index=edge_index)

if show_augmentation:
    throw_number = random.choice(test_range)
    display_results.animate_augmented_data(
        Floor, throw_number=throw_number,
        save_path=os.path.join(model_folder_path, "augmented_data.gif"),
        nodes_per_edge=nodes_per_edge, nearest_neighbors=K_nearest_neighbors)

if plot_phase_curves:
    slopes = evaluate_metrics.plot_phase_error_curves(
        trajectory_folder, model, Floor, test_range, nodes_per_edge,
        K_nearest_neighbors, nodes_body, accel_std, accel_mean,
        x_mean, x_std, e_mean, e_std, weights_only_load,
        unscale_trajectory_data, pos_history, use_wind=use_wind_feature,
        zero_at_phase_start=True,
        save_path=os.path.join(model_folder_path, "phase_error_curves.png"))

if show_rollout:
    throw_number = random.choice(test_range)
    print("Showing rollout for trajectory number:", throw_number)
    pred_positions, true_positions, edge_info = \
        display_results.rollout_trajectory_feedback_shape_match(
            trajectory_folder=trajectory_folder, model=model, Wall=Floor,
            throw_number=throw_number, nodes_per_edge=nodes_per_edge,
            nearest_neighbors=K_nearest_neighbors, rest_positions=nodes_body,
            accel_std=accel_std, accel_mean=accel_mean, x_mean=x_mean,
            x_std=x_std, e_mean=e_mean, e_std=e_std, do_shape_match=True,
            shape_alpha=1.0, return_edge_info=True,
            weights_only_load=weights_only_load,
            unscale_trajectory_data=unscale_trajectory_data,
            h=pos_history, use_wind=use_wind_feature)
    evaluate_metrics.compute_metrics(pred_positions, true_positions, nodes_body)
    display_results.animate_cube(
        pred_positions, true_positions, edge_info=edge_info,
        save_path=os.path.join(model_folder_path, "rollout_trajectory.gif"))


# ======================================================================
# 9.  WRITE THE ROW
# ======================================================================

if Save_run_report:
    settings = dict(
        architecture="accel",          # <- how you tell these rows from force rows
        dataset=trajectory_folder,
        n_train=N_TRAIN,
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
        multistep=multistep,
        scheduler=Learning_Rate_Scheduler,
        curriculum_epochs=curriculum_epochs,
        use_wind=use_wind_feature,
    )
    save_run_report(model_folder_path, settings, metrics, slopes,
                    run_name=RUN_NAME, master_csv=MASTER_CSV)
    print(f"\n  row appended to {MASTER_CSV}")

print("\nAll done.")

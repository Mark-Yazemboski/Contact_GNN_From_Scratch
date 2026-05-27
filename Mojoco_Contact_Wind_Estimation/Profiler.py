"""
Profile _build_timestep_samples to find what's actually slow in the rebuild step.

Run on Roar:
    salloc --gres=gpu:v100:1 --cpus-per-task=4 --mem=16G --time=00:20:00
    source /storage/work/mby5170/miniconda3/etc/profile.d/conda.sh
    conda activate gnn
    cd /storage/work/mby5170/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation
    python profile_rebuild.py

The script does three things:
  1. Times one full rebuild (one epoch's worth) end-to-end.
  2. cProfile breakdown of where time goes inside _build_timestep_samples.
  3. Coarse breakdown of construction vs. Data() instantiation cost.
"""

import os
import sys
import time
import cProfile
import pstats
import io

import torch
from torch_geometric.data import Data

# Match the imports used by train_gnn_multi_step.py
from train_gnn_multi_step import _build_timestep_samples, build_dataset, _compute_node_stats,_compute_edge_stats,_compute_accel_stats
from generate_node_states import get_clean_positions, add_random_walk_noise


# ----------------------------------------------------------------------
# Config — match what your training script uses
# ----------------------------------------------------------------------
TRAIN_DATASET_PATH = "/home/ari/Desktop/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation/data/pytorch_datasets/gns_train_dataset.pt"
NORM_STATS_PATH    = "/home/ari/Desktop/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation/models/mojoco_pure_sliding_no_wind_fixed_noise/1024_train_gns_model_norms.pt"

H = 3                  # history window — match training config
NOISE_SCALE = 3e-4     # match training config
N_TRAJECTORIES = 1024  # how many to profile (cap to dataset size)






def load_wall_object():
    """
    The Wall object has .normal and .center_position attributes used inside _build_timestep_samples.
    We don't need the real one — a stand-in with the right attributes is fine for profiling.
    """
    class FakeWall:
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        center_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    return FakeWall()


def main():
    Used_Num_train_trajectories = 1024
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_model_path = os.path.join(script_dir, f"models/mojoco_pure_sliding_no_wind_fixed_noise/{Used_Num_train_trajectories}_train_gns_model.pt")


    

    Wall = load_wall_object()

    print("Building training dataset...")
    train_range = range(0, Used_Num_train_trajectories)
    dataset_train = build_dataset(
        Wall,
        train_range,
        nodes_per_edge=2,
        nearest_neighbors=3,
        h=3,
        trajectory_folder="/home/ari/Desktop/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation/data/mojoco_pure_sliding_2048_no_wind",
        weights_only=False,
        unscale_data=False,
    )
    torch.save(dataset_train, "data/pytorch_datasets/gns_train_dataset.pt")
    print(f"Training dataset saved to {"data/pytorch_datasets/gns_train_dataset.pt"}")



    #Compute normalization stats from one clean pass (no noise) over training trajectories.
    clean_samples = []
    for traj in dataset_train:
        clean_samples.extend(_build_timestep_samples(traj, Wall, h=3, noise_scale=3e-4))

    x_mean, x_std = _compute_node_stats(clean_samples)
    e_mean, e_std = _compute_edge_stats(clean_samples)
    acc_mean, acc_std = _compute_accel_stats(clean_samples)
    del clean_samples


    # saves the normalization statistics to a file, which can be used later for normalizing new data during inference.
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_norms.pt"
    torch.save({"x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std, "acc_mean": acc_mean, "acc_std": acc_std}, norm_stats_path)

    print(f"Saved normalization stats to {norm_stats_path}")


    print("=" * 70)
    print("Profiling _build_timestep_samples")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from {TRAIN_DATASET_PATH}")
    dataset_train = torch.load(TRAIN_DATASET_PATH, weights_only=False)
    n_traj = min(N_TRAJECTORIES, len(dataset_train))
    print(f"Using {n_traj} trajectories")

    # Load normalization stats
    print(f"Loading norm stats from {NORM_STATS_PATH}")
    norms = torch.load(NORM_STATS_PATH, weights_only=False)
    x_mean, x_std = norms["x_mean"], norms["x_std"]
    e_mean, e_std = norms["e_mean"], norms["e_std"]
    acc_mean, acc_std = norms["acc_mean"], norms["acc_std"]


    # ----------------------------------------------------------------------
    # 1. End-to-end timing (one full rebuild — matches one training epoch)
    # ----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. End-to-end rebuild timing (one full epoch's worth)")
    print("-" * 70)

    # Warmup (first call is often slower due to allocator setup)
    _ = _build_timestep_samples(
        dataset_train[0], Wall, h=H, noise_scale=NOISE_SCALE,
        x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
        acc_mean=acc_mean, acc_std=acc_std,
    )

    t0 = time.time()
    all_samples = []
    for traj in dataset_train[:n_traj]:
        all_samples.extend(_build_timestep_samples(
            traj, Wall, h=H, noise_scale=NOISE_SCALE,
            x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
            acc_mean=acc_mean, acc_std=acc_std,
        ))
    elapsed = time.time() - t0

    n_samples = len(all_samples)
    print(f"  Built {n_samples:,} samples from {n_traj} trajectories in {elapsed:.2f}s")
    print(f"  ({elapsed / n_traj * 1000:.2f} ms per trajectory)")
    print(f"  ({elapsed / n_samples * 1000:.3f} ms per sample)")

    # ----------------------------------------------------------------------
    # 2. cProfile — what's actually slow
    # ----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. cProfile breakdown (top 25 by cumulative time)")
    print("-" * 70)

    profiler = cProfile.Profile()
    profiler.enable()
    for traj in dataset_train[:n_traj]:
        _build_timestep_samples(
            traj, Wall, h=H, noise_scale=NOISE_SCALE,
            x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
            acc_mean=acc_mean, acc_std=acc_std,
        )
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(25)
    print(s.getvalue())

    # Also sort by tottime (self time, not cumulative)
    print("-" * 70)
    print("2b. cProfile — sorted by self time (tottime)")
    print("-" * 70)
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)
    print(s2.getvalue())

    # ----------------------------------------------------------------------
    # 3. Coarse split: tensor work vs Data() object creation
    # ----------------------------------------------------------------------
    print("-" * 70)
    print("3. Tensor work vs Data() construction breakdown")
    print("-" * 70)

    
    # Pick one trajectory, run the inside of _build_timestep_samples manually
    # so we can time each chunk separately.
    traj = dataset_train[0]
    clean_positions = traj["positions"]
    edge_index = traj["edge_index"]
    nodes_body = traj["nodes_body"]
    wind_vector = traj["wind_vector"]

    # Time the tensor math (just the math, no Data construction)
    REPEATS = 20
    t_math = 0.0
    for _ in range(REPEATS):
        t = time.time()
        noisy_positions, noise = add_random_walk_noise(clean_positions, noise_scale=NOISE_SCALE)
        sender = edge_index[0]
        receiver = edge_index[1]
        dU = nodes_body[sender] - nodes_body[receiver]
        dU_norm = torch.norm(dU, dim=1, keepdim=True)
        wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32)
        wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32)
        T = clean_positions.shape[0]
        N = noisy_positions.shape[1]
        M = T - 1 - H

        v_fd_list = []
        for k in range(H):
            v_fd_list.append(noisy_positions[H-k : T-1-k] - noisy_positions[H-k-1 : T-2-k])
        v_fd_all = torch.cat(v_fd_list, dim=-1)

        rel_pos = noisy_positions[H : T-1] - wall_c
        dist_all = torch.sum(rel_pos * wall_n, dim=-1, keepdim=True).clamp(0.0, 0.5)
        wind_broadcast = wind_vector.view(1, 1, -1).expand(M, N, -1)
        x_node_all = torch.cat([v_fd_all, wind_broadcast, dist_all], dim=-1)

        pos_at_t = noisy_positions[H : T-1]
        d_all = pos_at_t[:, sender] - pos_at_t[:, receiver]
        d_norm_all = torch.norm(d_all, dim=-1, keepdim=True)
        dU_broadcast = dU.unsqueeze(0).expand(M, -1, -1)
        dU_norm_broadcast = dU_norm.unsqueeze(0).expand(M, -1, -1)
        e_attr_all = torch.cat([d_all, d_norm_all, dU_broadcast, dU_norm_broadcast], dim=-1)

        accel_clean = clean_positions[H+1 : T] - 2.0 * clean_positions[H : T-1] + clean_positions[H-1 : T-2]
        accel_corrected = accel_clean - noise[H-1 : T-2]

        x_node_all = (x_node_all - x_mean) / x_std
        e_attr_all = (e_attr_all - e_mean) / e_std
        accel_corrected = (accel_corrected - acc_mean) / acc_std
        t_math += time.time() - t

    # Time JUST the Data() construction (math results from last iter)
    t_data = 0.0
    for _ in range(REPEATS):
        t = time.time()
        samples = [
            Data(x=x_node_all[i], edge_index=edge_index, edge_attr=e_attr_all[i], y=accel_corrected[i])
            for i in range(M)
        ]
        t_data += time.time() - t

    t_math /= REPEATS
    t_data /= REPEATS

    print(f"  Per-trajectory tensor math:    {t_math*1000:.2f} ms")
    print(f"  Per-trajectory Data() build:   {t_data*1000:.2f} ms ({M} Data objects)")
    print(f"  Ratio:                          tensor={t_math/(t_math+t_data)*100:.0f}%  data={t_data/(t_math+t_data)*100:.0f}%")
    print(f"\n  Extrapolated for {n_traj} trajectories:")
    print(f"    Tensor math:   {t_math * n_traj:.2f}s")
    print(f"    Data() build:  {t_data * n_traj:.2f}s")
    print(f"    Total:         {(t_math + t_data) * n_traj:.2f}s")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    pct_data = t_data / (t_math + t_data) * 100
    if pct_data > 50:
        print(f"Data() construction is {pct_data:.0f}% of rebuild — refactoring to fewer Data")
        print("objects would meaningfully speed up training.")
    elif pct_data > 25:
        print(f"Data() construction is {pct_data:.0f}% of rebuild — refactor would help moderately.")
    else:
        print(f"Data() construction is only {pct_data:.0f}% of rebuild — refactor not worth it.")
        print("The tensor math itself dominates, and you'd need to optimize that instead.")


if __name__ == "__main__":

    main()
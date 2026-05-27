"""
sanity_check_fast_batch.py

Verifies that fast_batch.build_epoch_tensors + iterate_batches produces
EXACTLY the same results as the existing _build_timestep_samples + DataLoader
pipeline. If this fails, do NOT use fast_batch — there's a bug.

Run with the same env that train_gnn_multi_step.py uses:
    python sanity_check_fast_batch.py

Update DATASET_PATH and NORM_STATS_PATH below to point to your saved files.
"""

import os
import sys
import torch
import numpy as np
import random

from torch_geometric.loader import DataLoader as PyGDataLoader

from train_gnn_multi_step import _build_timestep_samples
from fast_batch import build_epoch_tensors, iterate_batches, n_batches


# ----------------------------------------------------------------------
# Config — update paths to match your machine
# ----------------------------------------------------------------------
DATASET_PATH = "/home/ari/Desktop/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation/data/pytorch_datasets/gns_train_dataset.pt"
NORM_STATS_PATH = "/home/ari/Desktop/Contact_GNN_From_Scratch/Mojoco_Contact_Wind_Estimation/models/mojoco_pure_sliding_no_wind_fixed_noise/1024_train_gns_model_norms.pt"

H = 3
NOISE_SCALE = 3e-4
N_TRAJECTORIES = 8        # small for sanity — verify correctness, not perf
BATCH_SIZE = 32
TOLERANCE = 1e-5          # max abs difference allowed


class FakeWall:
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    center_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_tensors_close(a, b, name, tol=TOLERANCE):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    diff = (a - b).abs().max().item()
    if diff > tol:
        raise AssertionError(f"{name}: max abs diff {diff:.3e} > tol {tol:.0e}")
    print(f"  ✓ {name}: shape {tuple(a.shape)}, max diff {diff:.2e}")


def main():
    print("=" * 70)
    print("Sanity check: fast_batch vs _build_timestep_samples + DataLoader")
    print("=" * 70)

    # ---- Load data ----
    print(f"\nLoading dataset from {DATASET_PATH}")
    dataset_full = torch.load(DATASET_PATH, weights_only=False)
    dataset = dataset_full[:N_TRAJECTORIES]
    print(f"Using {len(dataset)} trajectories")

    print(f"Loading norm stats from {NORM_STATS_PATH}")
    norms = torch.load(NORM_STATS_PATH, weights_only=False)
    x_mean, x_std = norms["x_mean"], norms["x_std"]
    e_mean, e_std = norms["e_mean"], norms["e_std"]
    acc_mean, acc_std = norms["acc_mean"], norms["acc_std"]

    Wall = FakeWall()

    # ============================================================
    # TEST 1: identical samples produced by both build pipelines
    # ============================================================
    print("\n" + "-" * 70)
    print("TEST 1: per-sample feature equality (build pipelines)")
    print("-" * 70)

    # OLD pipeline
    set_all_seeds(42)
    old_samples = []
    for traj in dataset:
        old_samples.extend(_build_timestep_samples(
            traj, Wall, h=H, noise_scale=NOISE_SCALE,
            x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
            acc_mean=acc_mean, acc_std=acc_std,
        ))

    # NEW pipeline (same seed → same noise → same features)
    set_all_seeds(42)
    epoch_data = build_epoch_tensors(
        dataset, Wall, h=H, noise_scale=NOISE_SCALE,
        x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
        acc_mean=acc_mean, acc_std=acc_std,
    )

    n_samples = len(old_samples)
    print(f"  old: {n_samples} Data objects")
    print(f"  new: {epoch_data['x'].shape[0]} flat rows")

    if n_samples != epoch_data['x'].shape[0]:
        raise AssertionError(
            f"Sample count mismatch: old={n_samples} new={epoch_data['x'].shape[0]}"
        )

    # Stack old samples to compare against new flat tensors
    old_x = torch.stack([s.x for s in old_samples], dim=0)            # (n, N, F)
    old_e = torch.stack([s.edge_attr for s in old_samples], dim=0)    # (n, E, F)
    old_y = torch.stack([s.y for s in old_samples], dim=0)            # (n, N, 3)

    assert_tensors_close(old_x, epoch_data['x'], "x")
    assert_tensors_close(old_e, epoch_data['edge_attr'], "edge_attr")
    assert_tensors_close(old_y, epoch_data['y'], "y")
    assert_tensors_close(
        old_samples[0].edge_index, epoch_data['edge_index'], "edge_index"
    )
    print("  TEST 1 PASSED")

    # ============================================================
    # TEST 2: batched outputs match (no shuffle, deterministic order)
    # ============================================================
    print("\n" + "-" * 70)
    print("TEST 2: batched output equality (no shuffle)")
    print("-" * 70)

    # OLD loader, no shuffle
    old_loader = PyGDataLoader(old_samples, batch_size=BATCH_SIZE, shuffle=False)

    # NEW iterator, no shuffle
    new_batches = list(iterate_batches(epoch_data, batch_size=BATCH_SIZE, shuffle=False))

    old_batches = list(old_loader)
    print(f"  old: {len(old_batches)} batches")
    print(f"  new: {len(new_batches)} batches")

    if len(old_batches) != len(new_batches):
        raise AssertionError(f"Batch count mismatch")

    for bi, (old_b, new_b) in enumerate(zip(old_batches, new_batches)):
        if bi < 3 or bi == len(old_batches) - 1:
            print(f"  batch {bi}:")
            assert_tensors_close(old_b.x, new_b.x, f"    batch{bi}.x")
            assert_tensors_close(old_b.edge_attr, new_b.edge_attr, f"    batch{bi}.edge_attr")
            assert_tensors_close(old_b.y, new_b.y, f"    batch{bi}.y")
            assert_tensors_close(old_b.edge_index, new_b.edge_index, f"    batch{bi}.edge_index")
        else:
            # Silent check on middle batches
            if (old_b.x - new_b.x).abs().max() > TOLERANCE:
                raise AssertionError(f"batch {bi} x mismatch")
            if (old_b.edge_index - new_b.edge_index).abs().max() > 0:
                raise AssertionError(f"batch {bi} edge_index mismatch")
    print("  TEST 2 PASSED")

    # ============================================================
    # TEST 3: model forward produces same output
    # ============================================================
    print("\n" + "-" * 70)
    print("TEST 3: model forward output equality")
    print("-" * 70)

    from train_gnn_multi_step import GNSModel

    node_dim = epoch_data['x'].shape[-1]
    edge_dim = epoch_data['edge_attr'].shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_all_seeds(1234)
    model = GNSModel(node_dim, edge_dim, latent_dim=128, L=5, K=1).to(device)
    model.eval()  # disable dropout etc

    with torch.no_grad():
        old_b = old_batches[0].to(device)
        new_b = new_batches[0]
        # Move new batch to device manually
        new_b.x = new_b.x.to(device)
        new_b.edge_index = new_b.edge_index.to(device)
        new_b.edge_attr = new_b.edge_attr.to(device)
        new_b.y = new_b.y.to(device)

        out_old = model(old_b)
        out_new = model(new_b)

    assert_tensors_close(out_old, out_new, "model output", tol=1e-4)

    # Loss comparison
    loss_fn = torch.nn.MSELoss()
    loss_old = loss_fn(out_old, old_b.y)
    loss_new = loss_fn(out_new, new_b.y)
    diff = abs(loss_old.item() - loss_new.item())
    print(f"  loss old: {loss_old.item():.8f}")
    print(f"  loss new: {loss_new.item():.8f}")
    print(f"  diff:     {diff:.2e}")
    if diff > 1e-5:
        raise AssertionError(f"Loss mismatch: {diff:.2e}")
    print("  TEST 3 PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED. fast_batch is a faithful drop-in replacement.")
    print("=" * 70)


if __name__ == "__main__":
    main()
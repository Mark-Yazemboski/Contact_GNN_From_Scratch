"""
fast_batch.py

Drop-in replacement for the per-epoch rebuild step in train_gnn_multi_step.py.
Skips PyG Data() construction entirely — the model only needs .x, .edge_attr,
.edge_index, .y attributes, so we use a lightweight namespace object.

Two public entry points:
  build_epoch_tensors(...)  — replaces the for-traj loop that builds Data objects
  iterate_batches(...)       — replaces DataLoader iteration

Use:
    epoch_data = build_epoch_tensors(
        dataset_train, Wall, h, noise_scale,
        x_mean, x_std, e_mean, e_std, acc_mean, acc_std,
    )
    for batch in iterate_batches(epoch_data, batch_size=1024, shuffle=True, device=device):
        # batch.x, batch.edge_attr, batch.edge_index, batch.y all live on `device`
        ...
"""

import torch
from generate_node_states import add_random_walk_noise


# ---- The tiny batch object that replaces PyG's Data/Batch ----
class FastBatch:
    """Minimal stand-in for PyG Data — exposes the four attributes the model uses."""
    __slots__ = ('x', 'edge_index', 'edge_attr', 'y')

    def __init__(self, x, edge_index, edge_attr, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y


def build_epoch_tensors(dataset_train, Wall, h, noise_scale=3e-4,
                        x_mean=None, x_std=None, e_mean=None, e_std=None,
                        acc_mean=None, acc_std=None):
    """
    Build all features for one training epoch as flat stacked tensors.
    Same math as _build_timestep_samples — just produces tensors instead of Data objects.

    Returns dict with:
        x:           (total_M, N, F_node)
        edge_attr:   (total_M, E, F_edge)
        y:           (total_M, N, 3)
        edge_index:  (2, E)  — shared template (same for all trajectories)
        N, E:        int     — node/edge counts per graph
    """
    x_list = []
    edge_attr_list = []
    y_list = []

    # Hoist Wall tensors out of the loop
    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32)

    # edge_index is identical across trajectories (confirmed by user) — keep one copy
    edge_index_template = dataset_train[0]["edge_index"]
    sender = edge_index_template[0]
    receiver = edge_index_template[1]
    nodes_body = dataset_train[0]["nodes_body"]
    dU = nodes_body[sender] - nodes_body[receiver]
    dU_norm = torch.norm(dU, dim=1, keepdim=True)

    for traj in dataset_train:
        clean_positions = traj["positions"]
        noisy_positions, noise = add_random_walk_noise(clean_positions, noise_scale=noise_scale)

        T = clean_positions.shape[0]
        N = noisy_positions.shape[1]
        M = T - 1 - h
        if M <= 0:
            continue

        wind_vector = traj["wind_vector"]

        # ---- Velocity history features (vectorized over t) ----
        v_fd_list = []
        for k in range(h):
            v_k = noisy_positions[h-k : T-1-k] - noisy_positions[h-k-1 : T-2-k]  # (M, N, 3)
            v_fd_list.append(v_k)
        v_fd_all = torch.cat(v_fd_list, dim=-1)  # (M, N, 3h)

        # ---- Wall distance ----
        rel_pos = noisy_positions[h : T-1] - wall_c
        dist_all = torch.sum(rel_pos * wall_n, dim=-1, keepdim=True).clamp(0.0, 0.5)

        # ---- Wind broadcast ----
        wind_broadcast = wind_vector.view(1, 1, -1).expand(M, N, -1)

        # ---- Node features ----
        x_node_all = torch.cat([v_fd_all, wind_broadcast, dist_all], dim=-1)  # (M, N, 3h+4)

        # ---- Edge features ----
        pos_at_t = noisy_positions[h : T-1]
        d_all = pos_at_t[:, sender] - pos_at_t[:, receiver]
        d_norm_all = torch.norm(d_all, dim=-1, keepdim=True)
        dU_broadcast = dU.unsqueeze(0).expand(M, -1, -1)
        dU_norm_broadcast = dU_norm.unsqueeze(0).expand(M, -1, -1)
        e_attr_all = torch.cat([d_all, d_norm_all, dU_broadcast, dU_norm_broadcast], dim=-1)

        # ---- Acceleration targets (noise-corrected) ----
        accel_clean = clean_positions[h+1 : T] - 2.0 * clean_positions[h : T-1] + clean_positions[h-1 : T-2]
        accel_corrected = accel_clean - noise[h-1 : T-2]

        # ---- Normalization folded in ----
        if x_mean is not None:
            x_node_all = (x_node_all - x_mean) / x_std
            e_attr_all = (e_attr_all - e_mean) / e_std
        if acc_mean is not None:
            accel_corrected = (accel_corrected - acc_mean) / acc_std

        x_list.append(x_node_all)
        edge_attr_list.append(e_attr_all)
        y_list.append(accel_corrected)

    return {
        'x': torch.cat(x_list, dim=0),                  # (total_M, N, F_node)
        'edge_attr': torch.cat(edge_attr_list, dim=0),  # (total_M, E, F_edge)
        'y': torch.cat(y_list, dim=0),                  # (total_M, N, 3)
        'edge_index': edge_index_template,              # (2, E)
        'N': N,
        'E': edge_index_template.shape[1],
    }


def iterate_batches(epoch_data, batch_size, shuffle=True, device=None):
    """
    Iterate over the flat epoch tensors in batches of size `batch_size`.
    Yields FastBatch objects whose tensors are already on `device` (if given).

    For B graphs in a batch:
      x:          (B*N, F_node)
      edge_index: (2, B*E)       — offsets applied per graph
      edge_attr:  (B*E, F_edge)
      y:          (B*N, 3)
    """
    x_all = epoch_data['x']
    e_all = epoch_data['edge_attr']
    y_all = epoch_data['y']
    edge_index_template = epoch_data['edge_index']
    N = epoch_data['N']
    E = epoch_data['E']

    total = x_all.shape[0]

    if shuffle:
        perm = torch.randperm(total)
    else:
        perm = torch.arange(total)

    for start in range(0, total, batch_size):
        idx = perm[start : start + batch_size]
        B = len(idx)

        # Vectorized gather (one indexed op each)
        x_b = x_all[idx].reshape(B * N, -1)          # (B*N, F_node)
        e_b = e_all[idx].reshape(B * E, -1)          # (B*E, F_edge)
        y_b = y_all[idx].reshape(B * N, -1)          # (B*N, 3)

        # Offset edge_index for batching:
        # graph b's nodes occupy rows [b*N : (b+1)*N] in x_b, so add b*N to its edge_index
        offsets = (torch.arange(B) * N).view(B, 1, 1)               # (B, 1, 1)
        ei_b = (edge_index_template.unsqueeze(0) + offsets)         # (B, 2, E)
        ei_b = ei_b.permute(1, 0, 2).reshape(2, B * E)              # (2, B*E)

        if device is not None:
            x_b = x_b.to(device, non_blocking=True)
            e_b = e_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            ei_b = ei_b.to(device, non_blocking=True)

        yield FastBatch(x=x_b, edge_index=ei_b, edge_attr=e_b, y=y_b)


def n_batches(epoch_data, batch_size):
    """Number of batches per epoch — equivalent to len(loader)."""
    total = epoch_data['x'].shape[0]
    return (total + batch_size - 1) // batch_size
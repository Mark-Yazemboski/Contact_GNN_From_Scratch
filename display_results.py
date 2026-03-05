# display_results.py

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import torch_geometric
from generate_node_states import get_gns_features, BLOCK_HALF_WIDTH, knn_adjacency
from train_gnn import GNSModel  # import your model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HZ = 144
DT = 1.0 / HZ

# -----------------------------
# Load trained model
# -----------------------------
def load_model(model_path, node_dim, edge_dim, device=device):
    model = GNSModel(node_dim, edge_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def shape_match(pred_positions, rest_positions, masses=None, alpha=1.0):
    """
    pred_positions: (N,3) predicted positions at current timestep
    rest_positions: (N,3) undeformed rest shape
    masses: optional (N,) tensor
    alpha: stiffness [0..1]
    """
    N = pred_positions.shape[0]

    if masses is None:
        masses = torch.ones(N, device=pred_positions.device)

    # compute centers of mass
    x0_cm = (rest_positions * masses[:, None]).sum(0) / masses.sum()
    x_cm = (pred_positions * masses[:, None]).sum(0) / masses.sum()

    # relative positions
    q = rest_positions - x0_cm   # (N,3)
    p = pred_positions - x_cm    # (N,3)

    # compute Apq
    # sum_i m_i * p_i * q_i^T
    Apq = (masses[:, None, None] * p[:, :, None] @ q[:, None, :]).sum(0)  # (3,3)

    # polar decomposition to get rotation
    U, S, Vh = torch.linalg.svd(Apq)
    R = U @ Vh

    # goal positions
    g = (R @ q.T).T + x_cm  # (N,3)

    # move toward goal
    new_positions = pred_positions + alpha * (g - pred_positions)
    return new_positions


# -----------------------------
# Run predictions & rollout with feedback
# -----------------------------
def rollout_trajectory_feedback_shape_match(
    model,
    Wall,
    throw_number,
    nodes_per_edge=2,
    h=2,
    rest_positions=None,
    accel_std=None,
    accel_mean=None,
    x_mean=None,
    x_std=None,
    e_mean=None,
    e_std=None,
    do_shape_match=True,
    shape_alpha=1.0,
    return_edge_info=False
):
    """
    Closed-loop rollout:
      1) predict a_t from current predicted state
      2) integrate x_{t+1} = a_t + 2 x_t - x_{t-1}
      3) shape-match x_{t+1}
      4) feed updated positions into next step feature construction

        If return_edge_info=True, also returns a dict with:
            - edge_index: (2, E)
            - edge_feat:  (T-h, E, F)
    """
    node_feat, edge_feat, edge_index, true_positions = get_gns_features(
        Wall, throw_number, nodes_per_edge=nodes_per_edge, h=h
    )

    edge_index = edge_index.long().to(device)
    true_positions = true_positions.to(device)
    node_dim_target = node_feat.shape[-1]
    edge_dim_target = edge_feat.shape[-1]

    if rest_positions is None:
        rest_positions = true_positions[0].clone()
    rest_positions = rest_positions.to(device)
    
    if accel_std is not None and accel_mean is not None:
        accel_std = accel_std.to(device)
        accel_mean = accel_mean.to(device)
    if x_mean is not None and x_std is not None:
        x_mean = x_mean.to(device)
        x_std = x_std.to(device)
    if e_mean is not None and e_std is not None:
        e_mean = e_mean.to(device)
        e_std = e_std.to(device)

    # Need 3 frames for h=2 finite-difference velocity history
    if true_positions.shape[0] < 3:
        raise ValueError("Need at least 3 frames for feedback rollout with h=2.")

    pred_positions = [
        true_positions[0].clone(),
        true_positions[1].clone(),
        true_positions[2].clone(),
    ]

    # Build/predict from t=2 -> predict x_{3}, then onward
    for _ in range(2, true_positions.shape[0] - 1):
        x_tm2 = pred_positions[-3]
        x_tm1 = pred_positions[-2]
        x_t = pred_positions[-1]

        x_node, e_attr = _build_feedback_features(
            x_t, x_tm1, x_tm2, edge_index, rest_positions, Wall,
            node_dim_target=node_dim_target,
            edge_dim_target=edge_dim_target,
            x_mean=x_mean,
            x_std=x_std,
            e_mean=e_mean,
            e_std=e_std
        )

        data = torch_geometric.data.Data(
            x=x_node,
            edge_index=edge_index,
            edge_attr=e_attr
        ).to(device)

        with torch.no_grad():
            a_t = model(data)

        if accel_std is not None and accel_mean is not None:
            a_t = a_t * accel_std + accel_mean

        x_next = a_t + 2.0 * x_t - x_tm1

        if do_shape_match:  
            x_next = shape_match(x_next, rest_positions, alpha=shape_alpha)

        pred_positions.append(x_next)

    pred_positions = torch.stack(pred_positions, dim=0).cpu()
    true_positions_cpu = true_positions.cpu()

    if return_edge_info:
        edge_info = {
            "edge_index": edge_index.cpu(),
            "edge_feat": edge_feat.cpu(),
        }
        return pred_positions, true_positions_cpu, edge_info

    return pred_positions, true_positions_cpu

# -----------------------------
# Plot RMSE
# -----------------------------
def plot_rmse(pred_acc, true_acc):
    rmse = torch.sqrt(((pred_acc - true_acc) ** 2).mean(dim=0))  # per node
    plt.figure(figsize=(8,6))
    plt.plot(rmse.numpy())
    plt.title("Per-node RMSE (x,y,z)")
    plt.xlabel("Node index")
    plt.ylabel("RMSE")
    plt.show()

# -----------------------------
# Animation function
# -----------------------------
def animate_cube(
    pred_positions,
    true_positions=None,
    edge_info=None,
    interval=50,
    save_path=None
):
    """
    Animate predicted node positions and optionally ground truth with edges.
    
    pred_positions: Tensor (T, N, 3)
    true_positions: Tensor (T, N, 3) (optional)
    edge_info: dict with key "edge_index" from rollout_trajectory_feedback_shape_match
    interval: ms between frames
    save_path: string path to save GIF (optional)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Move tensors to CPU if needed
    pred_positions = pred_positions.cpu()
    if true_positions is not None:
        true_positions = true_positions.cpu()

    # Intentionally require precomputed edge info from rollout.
    edge_index = edge_info["edge_index"]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    # Axis limits (include GT if available)
    all_pos = pred_positions
    if true_positions is not None:
        all_pos = torch.cat([pred_positions, true_positions], dim=0)

    x_min, x_max = all_pos[:, :, 0].min(), all_pos[:, :, 0].max()
    y_min, y_max = all_pos[:, :, 1].min(), all_pos[:, :, 1].max()
    z_min, z_max = all_pos[:, :, 2].min(), all_pos[:, :, 2].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Red = Prediction | Blue = Ground Truth')

    pred_scatter = ax.scatter([], [], [], c='r', label='Pred')

    if true_positions is not None:
        gt_scatter = ax.scatter([], [], [], c='b', alpha=0.5, label='GT')
        ax.legend()
    else:
        gt_scatter = None

    pred_edge_lines = [
        ax.plot([], [], [], c='r', alpha=0.25, linewidth=1.0)[0]
        for _ in range(edge_index.shape[1])
    ]

    if true_positions is not None:
        gt_edge_lines = [
            ax.plot([], [], [], c='b', alpha=0.2, linewidth=1.0)[0]
            for _ in range(edge_index.shape[1])
        ]
    else:
        gt_edge_lines = []

    def update(frame):
        pred_scatter._offsets3d = (
            pred_positions[frame, :, 0].numpy(),
            pred_positions[frame, :, 1].numpy(),
            pred_positions[frame, :, 2].numpy()
        )

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            pred_edge_lines[i].set_data(
                [pred_positions[frame, src, 0].item(), pred_positions[frame, dst, 0].item()],
                [pred_positions[frame, src, 1].item(), pred_positions[frame, dst, 1].item()]
            )
            pred_edge_lines[i].set_3d_properties(
                [pred_positions[frame, src, 2].item(), pred_positions[frame, dst, 2].item()]
            )

        if true_positions is not None:
            gt_scatter._offsets3d = (
                true_positions[frame, :, 0].numpy(),
                true_positions[frame, :, 1].numpy(),
                true_positions[frame, :, 2].numpy()
            )

            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])

                gt_edge_lines[i].set_data(
                    [true_positions[frame, src, 0].item(), true_positions[frame, dst, 0].item()],
                    [true_positions[frame, src, 1].item(), true_positions[frame, dst, 1].item()]
                )
                gt_edge_lines[i].set_3d_properties(
                    [true_positions[frame, src, 2].item(), true_positions[frame, dst, 2].item()]
                )

        artists = [pred_scatter] + pred_edge_lines
        if gt_scatter is not None:
            artists = artists + [gt_scatter] + gt_edge_lines
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=pred_positions.shape[0],
        interval=interval,
        blit=False
    )

    # -------------------------
    # Optional Save
    # -------------------------
    ax.set_aspect('equal')
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=1000 // interval)
        print("Saved successfully.")
    
    plt.show()


def display_meshed_cube(points, edge_index=None, nearest_neighbors=4):
    """
    Display mesh points and edges.

    points: (N, 3) torch.Tensor or np.ndarray
    edge_index: optional (2, E) connectivity. If None, KNN edges are built.
    nearest_neighbors: used only when edge_index is None.
    """
    if torch.is_tensor(points):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    if edge_index is None:
        edge_index = knn_adjacency(points_np, k=nearest_neighbors)
    elif torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='r')

    # Draw directed edges in edge_index.
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])

        ax.plot(
            [points_np[src, 0], points_np[dst, 0]],
            [points_np[src, 1], points_np[dst, 1]],
            [points_np[src, 2], points_np[dst, 2]],
            c='k',
            alpha=0.35,
            linewidth=1.0
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Meshed Cube Surface Points + Edges')
    plt.show()

def _match_feature_dim(feat, target_dim):
    """Pad/truncate feature dimension to match trained model input size."""
    d = feat.shape[-1]
    if d == target_dim:
        return feat
    if d < target_dim:
        pad = torch.zeros(feat.shape[0], target_dim - d, device=feat.device, dtype=feat.dtype)
        return torch.cat([feat, pad], dim=-1)
    return feat[:, :target_dim]


def _build_feedback_features(x_t, x_tm1, x_tm2, edge_index, rest_positions, Wall,
                             node_dim_target, edge_dim_target,
                             x_mean=None, x_std=None, e_mean=None, e_std=None):
    """
    Build node/edge features from predicted positions for closed-loop rollout.
    Node base: [v_t, v_{t-1}, boundary_dist]
    Edge base: [d_ij, ||d_ij||, dU_ij, ||dU_ij||]
    """
    # Node features
    v_t = x_t - x_tm1
    v_tm1 = x_tm1 - x_tm2

    if Wall is not None:
        wall_n = torch.as_tensor(Wall.get_normal(), dtype=x_t.dtype, device=x_t.device)
        wall_c = torch.as_tensor(Wall.center_position, dtype=x_t.dtype, device=x_t.device)
        b = torch.sum((x_t - wall_c) * wall_n, dim=-1, keepdim=True)
        b = torch.clamp(b, 0.0, 0.5)
        x_node = torch.cat([v_t, v_tm1, b], dim=-1)
    else:
        x_node = torch.cat([v_t, v_tm1], dim=-1)

    x_node = _match_feature_dim(x_node, node_dim_target)

    # Edge features
    src, dst = edge_index[0], edge_index[1]
    d = x_t[src] - x_t[dst]
    d_norm = torch.norm(d, dim=-1, keepdim=True)

    if rest_positions is not None:
        d_u = rest_positions[src] - rest_positions[dst]
        d_u_norm = torch.norm(d_u, dim=-1, keepdim=True)
    else:
        d_u = torch.zeros_like(d)
        d_u_norm = torch.zeros_like(d_norm)

    e_attr = torch.cat([d, d_norm, d_u, d_u_norm], dim=-1)
    e_attr = _match_feature_dim(e_attr, edge_dim_target)

    # -------------------------
    # Apply normalization
    # -------------------------
    if x_mean is not None and x_std is not None:
        x_node = (x_node - x_mean) / x_std

    if e_mean is not None and e_std is not None:
        e_attr = (e_attr - e_mean) / e_std

    return x_node, e_attr


def animate_rotated_with_velocities_and_edges(
    Wall,
    throw_number,
    nodes_per_edge=2,
    h=2,
    interval=50,
    save_path=None
):
    """
        Debug animation that mirrors train-time augmentation:
            - rotate node velocity vectors like train_gnn.rotate_batch
            - rotate edge features [d, d_norm, dU, dU_norm] like train_gnn.rotate_batch
            - draw edges from edge_feat displacement d (not from node-node geometry)
    """

    node_feat, edge_feat, edge_index, positions = get_gns_features(
        Wall,
        throw_number,
        nodes_per_edge=nodes_per_edge,
        h=h
    )

    positions = positions.cpu()
    edge_index = edge_index.cpu()
    node_feat = node_feat.cpu()
    edge_feat = edge_feat.cpu()

    # ---- Velocities from node features ----
    # train_gnn.rotate_batch rotates only first 6 dims (two 3D vectors)
    if node_feat.shape[-1] >= 6:
        velocities = node_feat[:, :, 0:6].reshape(
            node_feat.shape[0],
            node_feat.shape[1],
            2,
            3
        ).sum(dim=2)
    else:
        velocities = torch.zeros_like(positions)

    # ---- Rotation ----
    theta = torch.rand(1) * 2 * torch.pi
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ]).squeeze()

    pos_rot = positions @ R.T
    vel_rot = velocities @ R.T

    # ---- Rotate edge features exactly like train_gnn.rotate_batch ----
    # edge_feat = [d, d_norm, dU, dU_norm]
    if edge_feat.shape[-1] >= 8:
        d = edge_feat[:, :, 0:3]
        d_norm = edge_feat[:, :, 3:4]
        d_u = edge_feat[:, :, 4:7]
        d_u_norm = edge_feat[:, :, 7:8]

        d_rot = d @ R.T
        d_u_rot = d_u @ R.T

        edge_feat_rot = torch.cat([d_rot, d_norm, d_u_rot, d_u_norm], dim=-1)
    else:
        edge_feat_rot = edge_feat

    # Edge vectors used for drawing (must come from edge_feat, not node positions)
    edge_d = edge_feat[:, :, 0:3] if edge_feat.shape[-1] >= 3 else None
    edge_d_rot = edge_feat_rot[:, :, 0:3] if edge_feat_rot.shape[-1] >= 3 else None

    # Convert to numpy
    positions = positions.numpy()
    pos_rot = pos_rot.numpy()
    velocities = velocities.numpy()
    vel_rot = vel_rot.numpy()
    edge_d = edge_d.numpy() if edge_d is not None else None
    edge_d_rot = edge_d_rot.numpy() if edge_d_rot is not None else None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set limits
    all_pos = torch.cat([torch.tensor(positions),
                         torch.tensor(pos_rot)], dim=0)

    ax.set_xlim(all_pos[:, :, 0].min(), all_pos[:, :, 0].max())
    ax.set_ylim(all_pos[:, :, 1].min(), all_pos[:, :, 1].max())
    ax.set_zlim(all_pos[:, :, 2].min(), all_pos[:, :, 2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Blue = Original | Red = Rotated")

    def update(frame):
        ax.cla()  # 🔥 Proper 3D clear

        # Re-set limits after clearing
        ax.set_xlim(all_pos[:, :, 0].min(), all_pos[:, :, 0].max())
        ax.set_ylim(all_pos[:, :, 1].min(), all_pos[:, :, 1].max())
        ax.set_zlim(all_pos[:, :, 2].min(), all_pos[:, :, 2].max())

        # Plot nodes
        ax.scatter(
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
            c='b'
        )

        ax.scatter(
            pos_rot[frame, :, 0],
            pos_rot[frame, :, 1],
            pos_rot[frame, :, 2],
            c='r'
        )

        # Plot edges from edge_feat displacement d using sender anchor.
        # For each edge: d = x_sender - x_receiver, so receiver = sender - d.
        if edge_d is not None and edge_d_rot is not None:
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])

                src_pos = positions[frame, src]
                dst_from_feat = src_pos - edge_d[frame, i]

                src_pos_rot = pos_rot[frame, src]
                dst_from_feat_rot = src_pos_rot - edge_d_rot[frame, i]

                ax.plot(
                    [src_pos[0], dst_from_feat[0]],
                    [src_pos[1], dst_from_feat[1]],
                    [src_pos[2], dst_from_feat[2]],
                    c='b',
                    alpha=0.3
                )

                ax.plot(
                    [src_pos_rot[0], dst_from_feat_rot[0]],
                    [src_pos_rot[1], dst_from_feat_rot[1]],
                    [src_pos_rot[2], dst_from_feat_rot[2]],
                    c='r',
                    alpha=0.3
                )

        # Velocity vectors
        ax.quiver(
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
            velocities[frame, :, 0],
            velocities[frame, :, 1],
            velocities[frame, :, 2],
            color='b',
            length=0.05,
            normalize=True
        )

        ax.quiver(
            pos_rot[frame, :, 0],
            pos_rot[frame, :, 1],
            pos_rot[frame, :, 2],
            vel_rot[frame, :, 0],
            vel_rot[frame, :, 1],
            vel_rot[frame, :, 2],
            color='r',
            length=0.05,
            normalize=True
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=positions.shape[0],
        interval=interval,
        blit=False
    )

    ax.set_aspect('equal')
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=1000 // interval)
        print("Saved successfully.")
    
    plt.show()

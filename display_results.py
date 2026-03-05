# display_results.py

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import torch_geometric
from generate_node_states import get_gns_features, BLOCK_HALF_WIDTH
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
    nodes_per_edge=5,
    h=2,
    rest_positions=None,
    accel_std=None,
    accel_mean=None,
    shape_alpha=1.0
):
    """
    Closed-loop rollout:
      1) predict a_t from current predicted state
      2) integrate x_{t+1} = a_t + 2 x_t - x_{t-1}
      3) shape-match x_{t+1}
      4) feed updated positions into next step feature construction
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
            edge_dim_target=edge_dim_target
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
        x_next = shape_match(x_next, rest_positions, alpha=shape_alpha)

        pred_positions.append(x_next)

    pred_positions = torch.stack(pred_positions, dim=0).cpu()
    return pred_positions, true_positions.cpu()

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
def animate_cube(pred_positions, true_positions=None, interval=50, save_path=None):
    """
    Animate predicted node positions and optionally ground truth using scatter points.
    
    pred_positions: Tensor (T, N, 3)
    true_positions: Tensor (T, N, 3) (optional)
    interval: ms between frames
    save_path: string path to save GIF (optional)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Move tensors to CPU if needed
    pred_positions = pred_positions.cpu()
    if true_positions is not None:
        true_positions = true_positions.cpu()

    # Axis limits
    x_min, x_max = pred_positions[:, :, 0].min(), pred_positions[:, :, 0].max()
    y_min, y_max = pred_positions[:, :, 1].min(), pred_positions[:, :, 1].max()
    z_min, z_max = pred_positions[:, :, 2].min(), pred_positions[:, :, 2].max()
    
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

    def update(frame):
        pred_scatter._offsets3d = (
            pred_positions[frame, :, 0].numpy(),
            pred_positions[frame, :, 1].numpy(),
            pred_positions[frame, :, 2].numpy()
        )

        if true_positions is not None:
            gt_scatter._offsets3d = (
                true_positions[frame, :, 0].numpy(),
                true_positions[frame, :, 1].numpy(),
                true_positions[frame, :, 2].numpy()
            )

        return (pred_scatter,) if gt_scatter is None else (pred_scatter, gt_scatter)

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


def display_meshed_cube(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Meshed Cube Surface Points')
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
                             node_dim_target, edge_dim_target):
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

    return x_node, e_attr
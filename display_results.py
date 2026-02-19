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
# Run predictions & rollout
# -----------------------------
def rollout_trajectory(model, Wall, throw_number, h=2, rest_positions=None):
    node_feat, edge_feat, edge_index, true_positions  = get_gns_features(Wall, throw_number, h=h)
    edge_index = edge_index.long()
    
    predictions = []
    
    for t in range(node_feat.shape[0]):
        data = torch_geometric.data.Data(
            x=node_feat[t],
            edge_index=edge_index,
            edge_attr=edge_feat[t]
        )
        data = data.to(device)
        
        with torch.no_grad():
            acc_pred = model(data)
        predictions.append(acc_pred.cpu())
    
    predictions = torch.stack(predictions)
    
    # ------------------------------
    # Correct Euler rollout
    # ------------------------------
    pred_positions = [true_positions[0], true_positions[1]]  # use positions, not velocities
    print("Mean Z position of last 10 timesteps in prediction:")
    print(predictions[-10:, :, 2].mean())
    for t in range(2, true_positions.shape[0]):
        x_prev = pred_positions[-2]
        x_curr = pred_positions[-1]
        a_pred = predictions[t-2]  # predicted acceleration
        
        x_next = 2*x_curr - x_prev + a_pred * DT**2

        # x_next[:, 2] = torch.clamp(x_next[:, 2], min=0)
        pred_positions.append(x_next)

        pred_positions[-1] = shape_match(pred_positions[-1], rest_positions, alpha=0.9)
    
    pred_positions = torch.stack(pred_positions)
    
    return pred_positions, true_positions

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

    ax.set_box_aspect([1, 1, 1])
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
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=1000 // interval)
        print("Saved successfully.")

    plt.show()
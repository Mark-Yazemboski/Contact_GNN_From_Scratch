import torch
import numpy as np
import matplotlib.pyplot as plt
from display_results import rollout_trajectory_feedback_shape_match

BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2 * BLOCK_HALF_WIDTH



def angle_between_rotations(R_pred, R_true):
    """
    Computes the absolute angle difference between two rotation matrices.
    angle = arccos((trace(R_pred^T @ R_true) - 1) / 2)
    """
    R_rel = R_pred.T @ R_true
    trace = torch.trace(R_rel)
    # Clamp to valid range for arccos
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos_angle)


def compute_metrics(pred_positions, true_positions, rest_positions):
    """
    Vectorized version: batched SVD over all timesteps.
    pred_positions: (T,N,3)
    true_positions: (T,N,3)
    rest_positions: (N,3)
    """
    device = pred_positions.device
    rest_positions = rest_positions.to(device)

    T, N, _ = pred_positions.shape

    # Centers of mass per timestep
    pred_cm = pred_positions.mean(dim=1)  # (T,3)
    true_cm = true_positions.mean(dim=1)  # (T,3)

    # Rest relative positions (constant)
    q = rest_positions - rest_positions.mean(dim=0)  # (N,3)

    # Pred/true relative positions per timestep
    p_pred = pred_positions - pred_cm[:, None, :]  # (T,N,3)
    p_true = true_positions - true_cm[:, None, :]  # (T,N,3)

    # Apq per timestep: (T,3,3)
    # Apq[t] = sum_n p[t,n]^T outer q[n]
    Apq_pred = torch.einsum('tni,nj->tij', p_pred, q)  # (T,3,3)
    Apq_true = torch.einsum('tni,nj->tij', p_true, q)  # (T,3,3)

    # Batched SVD
    U_p, S_p, Vh_p = torch.linalg.svd(Apq_pred)  # U:(T,3,3), Vh:(T,3,3)
    U_t, S_t, Vh_t = torch.linalg.svd(Apq_true)

    # Rotation fix to ensure a proper rotation (det = +1)
    # d = det(U @ Vh) for each timestep
    d_p = torch.linalg.det(U_p @ Vh_p)  # (T,)
    d_t = torch.linalg.det(U_t @ Vh_t)  # (T,)

    # Build D matrices batched: (T,3,3)
    D_p = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_t = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_p[:, 2, 2] = d_p
    D_t[:, 2, 2] = d_t

    R_pred = U_p @ D_p @ Vh_p  # (T,3,3)
    R_true = U_t @ D_t @ Vh_t  # (T,3,3)

    # --- Metric 1: Relative center position error (vectorized) ---
    center_errors = torch.norm(pred_cm - true_cm, dim=1) / BLOCK_WIDTH  # (T,)

    # --- Metric 2: Absolute angle difference (vectorized) ---
    # R_rel[t] = R_pred[t]^T @ R_true[t]
    R_rel = R_pred.transpose(-1, -2) @ R_true  # (T,3,3)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (T,)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_errors = torch.arccos(cos_angle)  # (T,)

    # --- Metric 3: Floor penetration (vectorized) ---
    # If you intended "max over nodes" penetration per timestep:
    z = pred_positions[..., 2]  # (T,N)
    floor_penetrations = torch.clamp(-z, min=0.0).amax(dim=1) / BLOCK_WIDTH  # (T,)

    return {
        'center_error':            center_errors.mean().item(),
        'angle_error_deg':         torch.rad2deg(angle_errors).mean().item(),
        'floor_penetration':       floor_penetrations.mean().item(),
        'center_error_t':          center_errors.detach().cpu(),
        'angle_error_t_deg':       torch.rad2deg(angle_errors).detach().cpu(),
        'floor_penetration_t':     floor_penetrations.detach().cpu(),
    }


def evaluate_model(model, Wall, test_trajectory_indices, nodes_per_edge,
                   nearest_neighbors,
                   rest_positions, accel_std, accel_mean,
                   x_mean, x_std, e_mean, e_std):
    """
    Runs rollout on all test trajectories and averages metrics across them.
    """
    

    all_center_errors = []
    all_angle_errors = []
    all_floor_penetrations = []

    for throw_number in test_trajectory_indices:
        print(f"Evaluating trajectory {throw_number}...")

        pred_positions, true_positions, _ = rollout_trajectory_feedback_shape_match(
            model, Wall,
            throw_number=throw_number,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            rest_positions=rest_positions,
            accel_std=accel_std,
            accel_mean=accel_mean,
            x_mean=x_mean,
            x_std=x_std,
            e_mean=e_mean,
            e_std=e_std,
            do_shape_match=True,
            shape_alpha=1.0,
            return_edge_info=True
        )

        metrics = compute_metrics(pred_positions, true_positions, rest_positions)
        all_center_errors.append(metrics['center_error'])
        all_angle_errors.append(metrics['angle_error_deg'])
        all_floor_penetrations.append(metrics['floor_penetration'])

    print("\n--- Test Set Metrics ---")
    print(f"Center Error (relative):   {np.mean(all_center_errors):.4f} ± {np.std(all_center_errors):.4f}")
    print(f"Angle Error (degrees):     {np.mean(all_angle_errors):.4f} ± {np.std(all_angle_errors):.4f}")
    print(f"Floor Penetration:         {np.mean(all_floor_penetrations):.4f} ± {np.std(all_floor_penetrations):.4f}")

    return {
        'center_error':      np.mean(all_center_errors),
        'angle_error_deg':   np.mean(all_angle_errors),
        'floor_penetration': np.mean(all_floor_penetrations),
    }


def plot_loss_curves(
    train_loss_epochs,
    train_loss_values,
    val_loss_epochs=None,
    val_loss_values=None,
    title="Training and Validation Loss",
    save_path=None,
    show_plot=True,
):
    """
    Plots loss vs epoch curves.

    Validation can be logged less frequently than training, so it has its own
    epoch list (val_loss_epochs).
    """
    if len(train_loss_epochs) != len(train_loss_values):
        raise ValueError("train_loss_epochs and train_loss_values must have same length")

    if val_loss_epochs is not None and val_loss_values is not None:
        if len(val_loss_epochs) != len(val_loss_values):
            raise ValueError("val_loss_epochs and val_loss_values must have same length")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_loss_epochs, train_loss_values, label="Train Loss", linewidth=2.0)

    if val_loss_epochs is not None and val_loss_values is not None and len(val_loss_epochs) > 0:
        ax.plot(
            val_loss_epochs,
            val_loss_values,
            label="Validation Loss",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved loss curve to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
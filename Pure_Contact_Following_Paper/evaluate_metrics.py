import torch
import numpy as np
import matplotlib.pyplot as plt
from display_results import rollout_trajectory_feedback_shape_match

#This file is used to compute all of the different metrics that are used to compare the truth cube toss, to the
# predicted cube toss from the GNN. The metrics we compute are:
# 1. Relative center position error: the distance between the predicted and true center of mass, normalized
#    by the block width.
# 2. Absolute angle difference: the angle difference between the predicted and true rotation matrices,
#    converted to degrees.
# 3. Floor penetration: the maximum depth that any part of the cube goes below the
#    floor (z=0), normalized by the block width.

BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2 * BLOCK_HALF_WIDTH


#Computes the angle difference between two rotation matrices.
def angle_between_rotations(R_pred, R_true):

    R_rel = R_pred.T @ R_true
    trace = torch.trace(R_rel)
    # Clamp to valid range for arccos
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos_angle)


#Computes the three metrics for a single trajectory, given the predicted and true positions over time, 
#as well as the rest positions of the cube's nodes.
def compute_metrics(pred_positions, true_positions, rest_positions):

    #Makes sure everything is on the same device for computation.
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

    # --- Metric 1: Relative center position error
    center_errors = torch.norm(pred_cm - true_cm, dim=1) / BLOCK_WIDTH  # (T,)

    # --- Metric 2: Absolute angle difference ---
    R_rel = R_pred.transpose(-1, -2) @ R_true  # (T,3,3)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (T,)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_errors = torch.arccos(cos_angle)  # (T,)

    # --- Metric 3: Floor penetration  ---
    z = pred_positions[..., 2]  # (T,N)
    floor_penetrations = torch.clamp(-z, min=0.0).amax(dim=1) / BLOCK_WIDTH  # (T,)

    #Returns all of the metrics averaged across the trajectory, 
    #as well as the per-timestep values for each metric for further analysis if desired.
    return {
        'center_error':            center_errors.mean().item(),
        'angle_error_deg':         torch.rad2deg(angle_errors).mean().item(),
        'floor_penetration':       floor_penetrations.mean().item(),
        'center_error_t':          center_errors.detach().cpu(),
        'angle_error_t_deg':       torch.rad2deg(angle_errors).detach().cpu(),
        'floor_penetration_t':     floor_penetrations.detach().cpu(),
    }


#This function runs a rollout of the GNN model on the specified test trajectories, 
#computes the metrics for each trajectory, and averages them across all trajectories to get an overall 
#performance evaluation of the model. It prints out the average and standard deviation for each metric across the test set.
def evaluate_model(trajectory_folder, model, Wall, test_trajectory_indices, nodes_per_edge,
                   nearest_neighbors,
                   rest_positions, accel_std, accel_mean,
                   x_mean, x_std, e_mean, e_std):
    

    all_center_errors = []
    all_angle_errors = []
    all_floor_penetrations = []

    #runs through each trajectory in the test set
    for throw_number in test_trajectory_indices:
        print(f"Evaluating trajectory {throw_number}...")

        #Simulates a rollout of the GNN model on the current trajectory, getting the predicted and true positions over time.
        pred_positions, true_positions, _ = rollout_trajectory_feedback_shape_match(
            trajectory_folder,
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

        #Computes the metrics for the current trajectory and appends them to the lists for averaging later.
        metrics = compute_metrics(pred_positions, true_positions, rest_positions)
        all_center_errors.append(metrics['center_error'])
        all_angle_errors.append(metrics['angle_error_deg'])
        all_floor_penetrations.append(metrics['floor_penetration'])

    #Prints the average and standard deviation of each metric across the test set.
    print("\n--- Test Set Metrics ---")
    print(f"Center Error ( / width):   {np.mean(all_center_errors):.4f} ± {np.std(all_center_errors):.4f}")
    print(f"Angle Error (degrees):     {np.mean(all_angle_errors):.4f} ± {np.std(all_angle_errors):.4f}")
    print(f"Floor Penetration (/ width):         {np.mean(all_floor_penetrations):.4f} ± {np.std(all_floor_penetrations):.4f}")

    return {
        'center_error':      np.mean(all_center_errors),
        'angle_error_deg':   np.mean(all_angle_errors),
        'floor_penetration': np.mean(all_floor_penetrations),
    }



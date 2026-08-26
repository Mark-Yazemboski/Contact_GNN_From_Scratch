"""
train_force_gns.py

NEW FILE - does not modify any existing code. Imports the feature builder and
normalization-stat helpers from train_gnn_multi_step.py so the force model sees
EXACTLY the same inputs as the acceleration model (feature parity is what makes
the Stage-1 comparison a clean one-variable experiment).

What changes vs train_gnn_multi_step.py:
  - The dataset keeps the RIGID state (COM + rotation matrix per frame) instead
    of only node positions, because the dynamics layer integrates a 6-DOF pose.
  - The model outputs per-node CONTACT forces plus ONE body-level FLUID wrench
    (force + torque at the COM) from pooled node latents (see force_gns.py for
    why fluid is a COM wrench at this mesh resolution); a Newton-Euler step
    turns them into the next pose. Rigidity is exact - no shape matching.
  - Training noise is RIGID (random walk on COM + rotation) instead of
    per-node. Rationale: the force model's control authority is a rigid wrench,
    so noise is injected in the space the model can actually correct. Per-node
    noise would ask it to fix deformations it structurally cannot produce.
  - Optional physics terms, adapted from the proposal's Eq. (5):
      h_diss      - contact tangential forces may not do positive work
                    (friction dissipates). Valid here BECAUSE fluid force lives
                    in its own COM head: the contact-tangential residual IS
                    friction. This term is also the identifiability mechanism
                    for the slide regime, where contact-tangential and
                    horizontal fluid force are otherwise confounded.
      h_sparse    - L1 on contact force magnitudes (concentrated contact).
      h_fluid_reg - L2 on the RAW fluid-head outputs, keeping the learned
                    fluid wrench a small residual on the analytic drag
                    baseline (PIROM practice). Replaces the per-node fluid
                    smoothness term, which has no object in the COM-head
                    design.
    h_pen needs no term: normal forces are >= 0 by construction (softplus).
  - The supervision itself is UNCHANGED: multistep position MSE in block
    widths, chain sampling, curriculum, z-rotation augmentation, rollout-based
    validation and best-model selection. No force labels are ever used.

Everything is driven from run_force_multi_step.py.
"""

import os
import re
import time
import math
import random
import numpy as np
import torch
import torch.optim as optim

# ---- reused, unmodified, from the existing codebase ----
from Mojoco_Contact_Wind_Estimation.train_gnn_multi_step import (_build_features_for_unroll,
                                  _build_timestep_samples,
                                  _compute_node_stats,
                                  _compute_edge_stats,
                                  _compute_accel_stats)
from generate_node_states import (mesh_cube_surface, knn_adjacency,
                                  unscale_position_velocity, BLOCK_HALF_WIDTH)
from evaluate_metrics import compute_metrics

# ---- the new force model + dynamics layer ----
from force_gns import (ForceGNSModel, quat_wxyz_to_R, so3_exp, so3_log,
                       rigid_step, nodes_from_state, contact_weight,
                       assemble_contact_forces, fluid_wrench_from_raw,
                       drag_accel_step, BLOCK_WIDTH, I_OVER_M)
from physics_losses import PhysicsLosses

BLOCK_WIDTH_FOR_LOSS = BLOCK_WIDTH


def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def prune_old_checkpoints(save_model_path, keep_last_n):
    """Delete rotating '<stem>_epoch<N>.pt' checkpoints beyond the newest
    `keep_last_n`. A 10k-epoch run at interval 100 otherwise leaves 100 files,
    each carrying model AND optimizer state - the reason model folders blow up
    on ROAR.

    Touches ONLY files matching the _epoch<digits>.pt pattern. _best_model.pt,
    _final.pt, _norms.pt, _physics.pt and _loss_history.pt are never
    candidates, so the artifacts evaluation needs cannot be pruned by accident.
    keep_last_n <= 0 disables pruning.
    """
    if keep_last_n is None or keep_last_n <= 0:
        return
    stem = os.path.splitext(save_model_path)[0]
    folder = os.path.dirname(stem) or "."
    base = os.path.basename(stem)
    pattern = re.compile(r"^" + re.escape(base) + r"_epoch(\d+)\.pt$")

    found = []
    for fn in os.listdir(folder):
        m = pattern.match(fn)
        if m:
            found.append((int(m.group(1)), os.path.join(folder, fn)))
    found.sort()                                   # oldest epoch first
    for _, path in found[:-keep_last_n]:
        try:
            os.remove(path)
        except OSError as e:
            print(f"  (could not remove {os.path.basename(path)}: {e})")


def _random_z_rotation():
    """Random rotation about z (same convention as the existing augmentation)."""
    th = torch.rand(()) * 2.0 * math.pi
    c, s = torch.cos(th), torch.sin(th)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ======================================================================
# Dataset: keep the rigid state, not just node positions
# ======================================================================

def build_force_dataset(traj_range, trajectory_folder,
                        weights_only=False, unscale_data=False, verbose_every=200):
    """
    Loads raw trajectory files, keeping per-frame COM and rotation matrix (the
    state the force model integrates) plus the wind vector. Also pulls the
    replica_physics dict (gravity, mu, ...) out of the first file that has one,
    so training can use the SAME gravity the data was generated with - a wrong
    g would be silently absorbed into the learned normal forces otherwise.

    Returns (dataset, meta) where dataset is a list of dicts
    {com (T,3), R (T,3,3), wind (3,), T} and meta holds replica_physics if found.
    """
    dataset, meta = [], {}
    for n, throw_number in enumerate(traj_range):
        path = os.path.join(trajectory_folder, f"{throw_number}.pt")
        raw = torch.load(path, weights_only=weights_only)
        states = raw[0].float()
        if unscale_data:
            states = unscale_position_velocity(states)

        com = states[:, 0:3].contiguous()
        quat = states[:, 3:7]
        R = quat_wxyz_to_R(quat)

        wind = torch.zeros(3)
        if len(raw) > 1:
            try:
                wind = torch.as_tensor(raw[1], dtype=torch.float32).reshape(3)
            except Exception:
                pass

        if not meta and len(raw) > 3 and isinstance(raw[3], dict):
            rp = raw[3].get("replica_physics", None)
            if isinstance(rp, dict):
                meta = dict(rp)

        dataset.append({"com": com, "R": R, "wind": wind, "T": com.shape[0]})
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"  loaded {n + 1} trajectories...", flush=True)
    return dataset, meta


def _positions_view(dataset, rest_nodes, edge_index):
    """
    A node-position view of the rigid dataset shaped exactly like the dicts
    build_dataset() produces, so the EXISTING _build_timestep_samples can
    compute the feature/target normalization stats. Guarantees stat parity
    with the acceleration pipeline - same features, same numbers.
    """
    view = []
    for d in dataset:
        pos = d["com"].unsqueeze(1) + torch.einsum('tij,nj->tni', d["R"], rest_nodes)
        view.append({"positions": pos, "edge_index": edge_index,
                     "nodes_body": rest_nodes, "wind_vector": d["wind"]})
    return view


def _compute_angular_stats(dataset):
    """Empirical per-step^2 angular-acceleration std over the training set -
    the rotational analog of the acceleration-target stats, used as the output
    scale of the fluid torque head. Symmetrized in x/y so the scaling is
    z-rotation equivariant (matching the augmentation), and clamped away from
    zero for degenerate (torque-free) datasets."""
    alphas = []
    for d in dataset:
        R = d["R"]
        w = so3_log(R[1:] @ R[:-1].transpose(-1, -2))           # (T-1, 3) rad/step
        if w.shape[0] >= 2:
            alphas.append(w[1:] - w[:-1])                       # (T-2, 3) rad/step^2
    a = torch.cat(alphas, dim=0)
    std = a.std(dim=0)
    s_xy = float(std[:2].mean())
    return torch.tensor([s_xy, s_xy, float(std[2])]).clamp_min(1e-8)


# ======================================================================
# Chain sampling with RIGID random-walk noise
# ======================================================================

def build_chain_index(dataset, h, multistep, stride=1):
    """List of (traj_idx, start_frame) for every valid chain window."""
    span = h + 1 + multistep
    index = []
    for ti, d in enumerate(dataset):
        last_start = d["T"] - span
        for s in range(0, last_start + 1, stride):
            index.append((ti, s))
    return index


def n_chain_batches(chain_index, batch_size):
    return (len(chain_index) + batch_size - 1) // batch_size


def iterate_force_chains(dataset, chain_index, batch_size, h, multistep, N,
                         device, shuffle=True, noise_scale=0.0, rot_noise_scale=0.0):
    """
    Yields chain batches of rigid state. Rigid random-walk noise is applied to
    the INPUT window only (frame 0 clean, targets clean), mirroring the
    existing chain-noise convention but in (COM, rotation) space:
      COM:      i.i.d. velocity noise per transition, cumsum'd (same as before)
      rotation: i.i.d. rotation-vector noise per transition, cumsum'd and
                applied as a left perturbation  R_noisy = exp(w_cum) R.
    """
    order = torch.randperm(len(chain_index)) if shuffle else torch.arange(len(chain_index))

    for start in range(0, len(chain_index), batch_size):
        sel = order[start:start + batch_size].tolist()
        B = len(sel)

        com_win = torch.empty(B, h + 1, 3)
        R_win = torch.empty(B, h + 1, 3, 3)
        tgt_com = torch.empty(B, multistep, 3)
        tgt_R = torch.empty(B, multistep, 3, 3)
        winds = torch.empty(B, 3)

        for b, idx in enumerate(sel):
            ti, s = chain_index[idx]
            d = dataset[ti]
            com_win[b] = d["com"][s: s + h + 1]
            R_win[b] = d["R"][s: s + h + 1]
            tgt_com[b] = d["com"][s + h + 1: s + h + 1 + multistep]
            tgt_R[b] = d["R"][s + h + 1: s + h + 1 + multistep]
            winds[b] = d["wind"]

        com_win = com_win.to(device)
        R_win = R_win.to(device)
        tgt_com = tgt_com.to(device)
        tgt_R = tgt_R.to(device)
        winds = winds.to(device)

        if noise_scale > 0:
            vel_noise = torch.randn(B, h, 3, device=device) * noise_scale
            com_win[:, 1:] = com_win[:, 1:] + torch.cumsum(vel_noise, dim=1)
        if rot_noise_scale > 0:
            w_noise = torch.randn(B, h, 3, device=device) * rot_noise_scale
            w_cum = torch.cumsum(w_noise, dim=1).reshape(B * h, 3)
            R_win[:, 1:] = so3_exp(w_cum).reshape(B, h, 3, 3) @ R_win[:, 1:]

        yield {"com_win": com_win, "R_win": R_win, "tgt_com": tgt_com,
               "tgt_R": tgt_R, "wind": winds, "B": B}


def rotate_force_chain(batch):
    """z-rotation augmentation on the rigid state. The rest mesh stays
    CANONICAL: the eval-time feature map always builds dU from the canonical
    mesh, so augmented features must too - rotating (COM, R, wind, targets)
    while keeping rest fixed produces exactly the features a genuinely rotated
    trajectory would produce."""
    Rz = _random_z_rotation().to(batch["com_win"].device)
    batch["com_win"] = batch["com_win"] @ Rz.T
    batch["tgt_com"] = batch["tgt_com"] @ Rz.T
    batch["wind"] = batch["wind"] @ Rz.T
    batch["R_win"] = Rz @ batch["R_win"]
    batch["tgt_R"] = Rz @ batch["tgt_R"]
    return batch


# ======================================================================
# Multistep unroll loss through the force decoder + dynamics layer
# ======================================================================

def _unroll_force_loss(model, batch, multistep, Wall, h, rest_nodes,
                       edge_index_b, N,
                       x_mean, x_std, e_mean, e_std, scale_vec, ang_scale_vec,
                       acc_mean, acc_std, g_step, dt,
                       use_wind=False, use_drag_baseline=False, k_over_m=0.0285,
                       contact_d0=0.02, contact_tau=0.005,
                       loss_mode="accel",
                       phys=None, phys_weights=None):
    """
    Unroll `multistep` steps: features -> contact forces + COM fluid wrench ->
    rigid step -> loss vs truth.

    loss_mode="accel"    -> PER-NODE ACCELERATION MSE, normalized by acc_std.
        This is byte-for-byte the same objective as _unroll_chain_loss_accel in
        train_gnn_multi_step.py: at each step the predicted and true per-node
        accelerations are both measured against the SAME (possibly drifted)
        window, then normalized identically. acc_mean cancels in the difference.
        Use this for the parity experiment - the printed loss number is
        directly comparable to the acceleration model's.

    loss_mode="position" -> position MSE in block widths, matching
        _unroll_chain_loss.

    NOTE these two differ ONLY in the normalizer. Because both the predicted
    and target accelerations share the term (-2 w[-1] + w[-2]), their
    difference is exactly (predicted next position - true next position). So
    "accel" divides that per-node position error by acc_std COMPONENTWISE,
    while "position" divides it by the scalar block width. Same residual,
    different axis weighting.

    Optional violation terms are accumulated per step and returned RAW
    (unweighted) so their magnitudes can be logged honestly - a physics loss
    that reaches zero while rollout error is flat means it bought nothing.
    """
    com_win, R_win = batch["com_win"], batch["R_win"]
    tgt_com, tgt_R = batch["tgt_com"], batch["tgt_R"]
    wind = batch["wind"]
    B = batch["B"]
    device = com_win.device

    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32, device=device)
    wall_n = wall_n / wall_n.norm().clamp_min(1e-12)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32, device=device)

    pos_window = [nodes_from_state(com_win[:, j], R_win[:, j], rest_nodes)
                  for j in range(h + 1)]
    com_prev, com_curr = com_win[:, -2], com_win[:, -1]
    R_prev, R_curr = R_win[:, -2], R_win[:, -1]
    rest_b = rest_nodes.unsqueeze(0).expand(B, -1, -1)

    phys_weights = phys_weights or {}
    any_phys = phys is not None and any(v > 0 for v in phys_weights.values())
    raw_accum = {}
    fluid_series = []      # total fluid accel per step, for temporal smoothness
    torque_series = []     # fluid angular accel per step, same purpose

    step_losses = []
    for k in range(multistep):
        x_node, e_attr = _build_features_for_unroll(
            pos_window, edge_index_b, rest_b, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N, use_wind=use_wind)
        contact_raw, fluid_raw = model(x_node, edge_index_b, e_attr, B)

        cur_nodes = pos_window[-1]
        dist = ((cur_nodes - wall_c) * wall_n).sum(-1, keepdim=True)   # unclamped
        c_w = contact_weight(dist, d0=contact_d0, tau=contact_tau)
        phi_c, parts = assemble_contact_forces(contact_raw, c_w, wall_n, scale_vec)
        a_fluid, alpha_fluid = fluid_wrench_from_raw(fluid_raw, scale_vec,
                                                     ang_scale_vec)

        extra_accel = a_fluid
        if use_drag_baseline: 
            extra_accel = extra_accel + drag_accel_step(
                wind, com_curr - com_prev, dt, k_over_m)
        com_next, R_next = rigid_step(phi_c, com_prev, com_curr, R_prev, R_curr,
                                      rest_nodes, g_step,
                                      extra_accel=extra_accel,
                                      extra_alpha=alpha_fluid)
        pred_nodes = nodes_from_state(com_next, R_next, rest_nodes)
        true_nodes = nodes_from_state(tgt_com[:, k], tgt_R[:, k], rest_nodes)

        if loss_mode == "accel":
            # Both accelerations are measured against the SAME window, so the
            # shared (-2 w[-1] + w[-2]) term cancels and acc_mean cancels too.
            # Written out in full for clarity / auditability.
            a_pred = pred_nodes - 2.0 * pos_window[-1] + pos_window[-2]
            a_true = true_nodes - 2.0 * pos_window[-1] + pos_window[-2]
            a_pred_norm = (a_pred - acc_mean) / acc_std
            a_true_norm = (a_true - acc_mean) / acc_std
            step_losses.append((a_pred_norm - a_true_norm).pow(2).mean())
        else:
            step_losses.append(((pred_nodes - true_nodes)
                                / BLOCK_WIDTH_FOR_LOSS).pow(2).mean())

        if any_phys:
            # See physics_losses.py for the full documentation of each term
            # and the mapping to the proposal's Eq. (5).
            v_node = pos_window[-1] - pos_window[-2]              # m/step
            drag_target = drag_accel_step(
                wind, (com_curr - com_prev).detach(), dt, k_over_m).detach()
            # For the physics terms, build the fluid total with the DETACHED
            # baseline: the terms then constrain only the fluid head, never
            # the motion. With the baseline on, the anchor reduces exactly to
            # "keep the learned residual small".
            fluid_total_phys = (a_fluid + drag_target if use_drag_baseline
                                else a_fluid)
            fluid_series.append(fluid_total_phys)
            torque_series.append(alpha_fluid)
            step_raws = phys.compute_step_terms(
                parts["phi_contact"], c_w, v_node, wall_n,
                fluid_total_phys, alpha_fluid, drag_target, phys_weights)
            for kname, v in step_raws.items():
                raw_accum[kname] = raw_accum.get(kname, 0.0) + v

        pos_window = pos_window[1:] + [pred_nodes]
        com_prev, com_curr = com_curr, com_next
        R_prev, R_curr = R_curr, R_next

    pos_loss = torch.stack(step_losses).mean()

    raw_terms = {k: v / multistep for k, v in raw_accum.items()}
    if any_phys and phys_weights.get("w_fluid_smooth", 0) > 0:
        raw_terms["fluid_smooth"] = phys.h_fluid_temporal_smooth(
            fluid_series, torque_series)
    total = pos_loss + PhysicsLosses.weighted_total(raw_terms, phys_weights)
    return total, pos_loss.detach(), {k: float(v.detach()) for k, v in raw_terms.items()}


# ======================================================================
# Batched rollout (validation + evaluation). Mirrors
# _rollout_validation_batched: pad to max length, roll in lockstep, score each
# trajectory over its own real frames. No shape matching - rigidity is exact.
# ======================================================================

def rollout_force_batched(model, trajs, Wall, h, rest_nodes,
                          x_mean, x_std, e_mean, e_std, scale_vec, ang_scale_vec,
                          g_step, dt,
                          device, use_wind=False, use_drag_baseline=False,
                          k_over_m=0.0285, contact_d0=0.02, contact_tau=0.005,
                          mass=0.37, return_forces=False, return_per_traj=False):
    model.eval()
    B = len(trajs)
    N = rest_nodes.shape[0]

    lengths = [t["T"] for t in trajs]
    T_max = max(lengths)

    def _pad(x, T):
        if x.shape[0] == T:
            return x
        tail = x[-1:].expand(T - x.shape[0], *([-1] * (x.dim() - 1)))
        return torch.cat([x, tail], dim=0)

    com_all = torch.stack([_pad(t["com"], T_max) for t in trajs]).to(device)
    R_all = torch.stack([_pad(t["R"], T_max) for t in trajs]).to(device)
    wind = torch.stack([t["wind"] for t in trajs]).to(device)
    rest_nodes = rest_nodes.to(device)
    rest_b = rest_nodes.unsqueeze(0).expand(B, -1, -1)

    # Callers attach the shared edge_index to each trajectory dict.
    ei = trajs[0]["edge_index"].to(device)
    edge_index_b = torch.cat([ei + b * N for b in range(B)], dim=1)

    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32, device=device)
    wall_n = wall_n / wall_n.norm().clamp_min(1e-12)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32, device=device)

    to_dev = lambda x: None if x is None else x.to(device)
    x_mean, x_std, e_mean, e_std = map(to_dev, (x_mean, x_std, e_mean, e_std))
    scale_vec = scale_vec.to(device)
    ang_scale_vec = ang_scale_vec.to(device)
    g_step = g_step.to(device)

    # Same frame convention as _rollout_validation_batched: work from frame h,
    # seed the window with the first h+1 true frames of that view.
    com_fh = com_all[:, h:]
    R_fh = R_all[:, h:]
    L_max = com_fh.shape[1]

    true_nodes_fh = com_fh.unsqueeze(2) + torch.einsum('btij,nj->btni', R_fh, rest_nodes)

    pred_nodes = [true_nodes_fh[:, i].clone() for i in range(h + 1)]
    pos_window = [true_nodes_fh[:, i] for i in range(h + 1)]
    com_prev, com_curr = com_fh[:, h - 1].clone(), com_fh[:, h].clone()
    R_prev, R_curr = R_fh[:, h - 1].clone(), R_fh[:, h].clone()

    forces = {"F_contact": [], "F_fluid": [], "tau_contact": [],
              "tau_fluid": [], "c_weight": [],
              # per-node, split for visualization (Newtons)
              "node_normal": [], "node_tangent": []} if return_forces else None
    dt2 = dt * dt
    m_I = mass * I_OVER_M                            # = I, the cube's inertia

    with torch.no_grad():
        for _ in range(h, L_max - 1):
            x_node, e_attr = _build_features_for_unroll(
                pos_window, edge_index_b, rest_b, Wall, wind,
                x_mean, x_std, e_mean, e_std, B, N, use_wind=use_wind)
            contact_raw, fluid_raw = model(x_node, edge_index_b, e_attr, B)

            cur_nodes = pos_window[-1]
            dist = ((cur_nodes - wall_c) * wall_n).sum(-1, keepdim=True)
            c_w = contact_weight(dist, d0=contact_d0, tau=contact_tau)
            phi_c, parts = assemble_contact_forces(contact_raw, c_w, wall_n, scale_vec)
            a_fluid, alpha_fluid = fluid_wrench_from_raw(fluid_raw, scale_vec,
                                                         ang_scale_vec)

            extra_accel = a_fluid
            if use_drag_baseline:
                extra_accel = extra_accel + drag_accel_step(
                    wind, com_curr - com_prev, dt, k_over_m)

            if return_forces:
                # Physical units: F = m * a / dt^2 (avg force over the step),
                # tau = I * alpha / dt^2. extra_accel is the FULL fluid accel
                # (learned + drag baseline), so F_fluid compares directly with
                # the logged J_fluid/DT.
                F_c = mass * phi_c.sum(dim=1) / dt2
                F_f = mass * extra_accel / dt2
                r = torch.einsum('bij,nj->bni', R_curr, rest_nodes)
                tau_c = mass * torch.cross(r, phi_c, dim=-1).sum(dim=1) / dt2
                tau_f = m_I * alpha_fluid / dt2
                # per-node split against the wall normal (Newtons)
                phi_n = (phi_c * wall_n).sum(-1, keepdim=True) * wall_n
                forces["node_normal"].append(mass * phi_n / dt2)
                forces["node_tangent"].append(mass * (phi_c - phi_n) / dt2)
                forces["F_contact"].append(F_c)
                forces["F_fluid"].append(F_f)
                forces["tau_contact"].append(tau_c)
                forces["tau_fluid"].append(tau_f)
                forces["c_weight"].append(c_w.squeeze(-1))

            com_next, R_next = rigid_step(phi_c, com_prev, com_curr, R_prev,
                                          R_curr, rest_nodes, g_step,
                                          extra_accel=extra_accel,
                                          extra_alpha=alpha_fluid)
            nxt = nodes_from_state(com_next, R_next, rest_nodes)
            pred_nodes.append(nxt)
            pos_window = pos_window[1:] + [nxt]
            com_prev, com_curr = com_curr, com_next
            R_prev, R_curr = R_curr, R_next

    pred_nodes = torch.stack(pred_nodes, dim=1)                # (B, L_max, N, 3)

    center_errors, angle_errors, per_traj = [], [], []
    rest_cpu = rest_nodes.cpu()
    for b in range(B):
        Lb = lengths[b] - h
        m = compute_metrics(pred_nodes[b, :Lb].cpu(), true_nodes_fh[b, :Lb].cpu(), rest_cpu)
        center_errors.append(m["center_error"])
        angle_errors.append(m["angle_error_deg"])
        if return_per_traj:
            per_traj.append(m)

    out = [float(np.mean(center_errors)), float(np.mean(angle_errors))]
    if return_forces:
        forces = {k: torch.stack(v, dim=1).cpu() for k, v in forces.items()}
        # forces["F_contact"][b, k] is the predicted avg force over ORIGINAL
        # frame interval t -> t+1 with t = 2h + k (window-end alignment).
        out.append(forces)
    if return_per_traj:
        out.append((per_traj, pred_nodes.cpu(), true_nodes_fh.cpu(), lengths))
    return tuple(out)


# ======================================================================
# Main training function
# ======================================================================

def train_force_gnn(Wall,
                    train_range,
                    val_range,
                    save_model_path,
                    trajectory_folder,
                    epochs=200,
                    batch_size=512,
                    accumulation_steps=1,
                    lr=1e-4,
                    nodes_per_edge=2,
                    nearest_neighbors=3,
                    h=3,
                    message_passing_layers=5,
                    repeat_blocks=1,
                    latent_dim=128,
                    weights_only=False,
                    unscale_data=False,
                    noise_scale=3e-4 * BLOCK_HALF_WIDTH,
                    rot_noise_scale=None,          # None -> noise_scale / half_width (rad)
                    multistep=8,
                    curriculum_epochs=0,           # 0 = off; else epochs per ramp phase
                    curriculum_schedule=None,      # e.g. [1,2,4,8]; None -> powers of 2
                    Learning_Rate_Scheduler=None,  # "decay", "cosine", or None
                    use_wind=False,
                    dt=1.0 / 148.0,
                    gravity=None,                  # None -> read from replica_physics, else 9.615
                    mass=0.37,
                    use_drag_baseline=False,
                    k_over_m=0.0285,
                    contact_d0=0.02,
                    contact_tau=0.005,
                    loss_mode="accel",     # "accel" (parity) or "position"
                    # --- physics-informed loss weights (proposal Eq. 6 gammas;
                    #     see physics_losses.py for each term) ---
                    w_diss=0.0,            # gamma_1: Coulomb dissipation
                    w_sparse=0.0,          # contact sparsity (Fig. 1)
                    w_fluid_anchor=0.0,    # gamma_3a: fluid force == analytic drag law
                    w_fluid_torque=0.0,    # gamma_3c: fluid torque == 0
                    w_fluid_smooth=0.0,    # gamma_3b: fluid smooth in time (K>=2)
                    mu_init=0.2,           # friction coefficient init
                    learn_mu=True,         # recover mu from data
                    fix_mu=None,           # set to a float to hard-fix mu
                    slip_v0=1e-3, slip_tau=1e-4,   # slip gate (m/step)
                    max_steps=None,        # optimizer-step budget (paper: 1e6);
                                           # overrides `epochs` when set
                    validation_check_interval=10,
                    epoch_checkpoint_interval=100,
                    keep_last_n_checkpoints=2,   # rotate; 0/None = keep all
                    resume_checkpoint_path=None,
                    compile_model=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    print("Building force training dataset (COM + rotation state)...")
    dataset_train, meta = build_force_dataset(train_range, trajectory_folder,
                                              weights_only=weights_only,
                                              unscale_data=unscale_data)
    print("Building force validation dataset...")
    dataset_val, _ = build_force_dataset(val_range, trajectory_folder,
                                         weights_only=weights_only,
                                         unscale_data=unscale_data)

    if gravity is None:
        gravity = float(meta.get("g", 9.615))
    print("=" * 70)
    print("FORCE-MODEL PHYSICS (must match the data generator):")
    print(f"  gravity = {gravity:.4f} m/s^2   dt = {dt:.6f} s   mass = {mass:.3f} kg")
    print(f"  drag baseline = {use_drag_baseline} (k/m = {k_over_m})   "
          f"contact gate d0/tau = {contact_d0}/{contact_tau} m")
    print(f"  loss_mode = {loss_mode}"
          + ("   (per-node acceleration MSE, same objective as "
             "_unroll_chain_loss_accel)" if loss_mode == "accel"
             else "   (position MSE in block widths)"))
    print(f"  physics-loss weights: diss={w_diss} sparse={w_sparse} "
          f"fluid_anchor={w_fluid_anchor} fluid_torque={w_fluid_torque} "
          f"fluid_smooth={w_fluid_smooth}")
    if meta:
        print(f"  replica_physics found in data: {meta}")
    print("=" * 70)

    rest_nodes = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, nodes_per_edge),
                              dtype=torch.float32)
    N = rest_nodes.shape[0]
    edge_index = torch.tensor(knn_adjacency(rest_nodes.numpy(), k=nearest_neighbors),
                              dtype=torch.long)
    for d in dataset_val:
        d["edge_index"] = edge_index          # rollout helper reads it from here

    # ---------------- normalization stats via the EXISTING builder ----------------
    view = _positions_view(dataset_train, rest_nodes, edge_index)
    clean_samples = []
    for traj in view:
        clean_samples.extend(_build_timestep_samples(traj, Wall, h=h,
                                                     noise_scale=noise_scale,
                                                     use_wind=use_wind))
    x_mean, x_std = _compute_node_stats(clean_samples)
    e_mean, e_std = _compute_edge_stats(clean_samples)
    acc_mean, acc_std = _compute_accel_stats(clean_samples)
    del clean_samples, view

    # Output scales, both [s_xy, s_xy, s_z] (equal x/y keeps them z-rotation
    # equivariant, so the rotation augmentation stays valid):
    #   scale_vec     - linear, from the acceleration-target stats
    #   ang_scale_vec - angular, from the empirical angular-acceleration stats
    s_xy = float(acc_std[:2].mean())
    scale_vec = torch.tensor([s_xy, s_xy, float(acc_std[2])])
    ang_scale_vec = _compute_angular_stats(dataset_train)
    print(f"  output scales: linear {scale_vec.tolist()} m/step^2 | "
          f"angular {ang_scale_vec.tolist()} rad/step^2")

    if rot_noise_scale is None:
        # the rotation that moves a corner about as far as the COM noise does
        rot_noise_scale = noise_scale / BLOCK_HALF_WIDTH

    # ---------------- physics-informed loss module ----------------
    phys_weights = dict(w_diss=w_diss, w_sparse=w_sparse,
                        w_fluid_anchor=w_fluid_anchor,
                        w_fluid_torque=w_fluid_torque,
                        w_fluid_smooth=w_fluid_smooth)
    phys = PhysicsLosses(phi_g=gravity * dt * dt, ang_scale_vec=ang_scale_vec,
                         mu_init=mu_init, learn_mu=learn_mu, fixed_mu=fix_mu,
                         slip_v0=slip_v0, slip_tau=slip_tau)
    any_phys = any(v > 0 for v in phys_weights.values())
    if any_phys:
        print(f"  physics losses ON: {[k for k, v in phys_weights.items() if v > 0]}"
              f"   mu: " + (f"FIXED {fix_mu}" if fix_mu is not None else
                            f"learnable, init {mu_init}" if learn_mu else
                            f"frozen at {mu_init}"))
        if phys_weights["w_fluid_smooth"] > 0 and multistep < 2:
            print("  WARNING: w_fluid_smooth needs multistep >= 2 for "
                  "consecutive predictions - it will contribute ZERO at K=1.")

    force_cfg = dict(dt=dt, gravity=gravity, mass=mass,
                     use_drag_baseline=use_drag_baseline, k_over_m=k_over_m,
                     contact_d0=contact_d0, contact_tau=contact_tau,
                     h=h, use_wind=use_wind, latent_dim=latent_dim,
                     L=message_passing_layers, K=repeat_blocks,
                     nodes_per_edge=nodes_per_edge,
                     nearest_neighbors=nearest_neighbors,
                     multistep=multistep, epochs=epochs,
                     scale_vec=scale_vec, ang_scale_vec=ang_scale_vec,
                     loss_mode=loss_mode,
                     w_diss=w_diss, w_sparse=w_sparse,
                     w_fluid_anchor=w_fluid_anchor, w_fluid_torque=w_fluid_torque,
                     w_fluid_smooth=w_fluid_smooth,
                     mu_init=mu_init, learn_mu=learn_mu, fix_mu=fix_mu,
                     slip_v0=slip_v0, slip_tau=slip_tau, max_steps=max_steps,
                     noise_scale=noise_scale, rot_noise_scale=rot_noise_scale)
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_norms.pt"
    torch.save({"x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std,
                "acc_mean": acc_mean, "acc_std": acc_std, "force_cfg": force_cfg},
               norm_stats_path)
    print(f"Saved normalization stats + force config to {norm_stats_path}")

    x_mean_g, x_std_g = x_mean.to(device), x_std.to(device)
    e_mean_g, e_std_g = e_mean.to(device), e_std.to(device)
    scale_vec_g = scale_vec.to(device)
    ang_scale_vec_g = ang_scale_vec.to(device)
    acc_mean_g, acc_std_g = acc_mean.to(device), acc_std.to(device)
    g_step = torch.tensor([0.0, 0.0, -gravity]) * dt * dt
    g_step_g = g_step.to(device)
    rest_g = rest_nodes.to(device)
    ei_g = edge_index.to(device)

    node_dim = x_mean.shape[0]
    edge_dim = e_mean.shape[0]
    model = ForceGNSModel(node_dim, edge_dim, latent_dim=latent_dim,
                          L=message_passing_layers, K=repeat_blocks).to(device)
    if compile_model and torch.cuda.is_available() and _triton_available():
        model = torch.compile(model)

    phys = phys.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(phys.parameters()), lr=lr)
    scheduler = None
    if Learning_Rate_Scheduler == "decay":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(0.1) ** (1.0 / max(epochs, 1)))   # 10x decay over the run
    elif Learning_Rate_Scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if resume_checkpoint_path is not None:
        ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_checkpoint_path} at epoch {start_epoch}")

    # ---------------- curriculum ----------------
    if curriculum_epochs > 0 and multistep > 1:
        if curriculum_schedule is None:
            curriculum_schedule = []
            k = 1
            while k < multistep:
                curriculum_schedule.append(k)
                k *= 2
            curriculum_schedule.append(multistep)
        print(f"Curriculum: {curriculum_schedule} x {curriculum_epochs} epochs/phase")
    else:
        curriculum_schedule = None

    def _K_for_epoch(ep):
        if curriculum_schedule is None:
            return multistep
        phase = min(ep // curriculum_epochs, len(curriculum_schedule) - 1)
        return curriculum_schedule[phase]

    train_loss_epochs, train_loss_values = [], []
    val_loss_epochs, val_loss_values = [], []
    mu_trace = []
    best_val_loss, best_val_epoch = float("inf"), -1
    loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"

    # ---------------- step budget ----------------
    # `max_steps` counts OPTIMIZER steps (the paper's 1M-step convention),
    # which stays comparable across batch size, K, and curriculum - unlike
    # epochs, where one K=8 epoch costs ~8x a K=1 epoch. When set, it
    # overrides `epochs`; training stops mid-epoch once the budget is spent.
    global_step = 0
    budget_spent = False
    if max_steps is not None:
        epochs = 10 ** 9        # effectively unbounded; the budget terminates

    # ---------------- epochs ----------------
    for epoch in range(start_epoch, epochs):
        if budget_spent:
            break
        t0 = time.time()
        _K_now = _K_for_epoch(epoch)
        chain_index = build_chain_index(dataset_train, h, _K_now, stride=1)
        t1 = time.time()

        model.train()
        total_loss = 0.0
        phys_accum = {}
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for bi, batch in enumerate(iterate_force_chains(
                dataset_train, chain_index, batch_size, h, _K_now, N, device,
                shuffle=True, noise_scale=noise_scale,
                rot_noise_scale=rot_noise_scale)):
            batch = rotate_force_chain(batch)
            B = batch["B"]
            edge_index_b = torch.cat([ei_g + b * N for b in range(B)], dim=1)

            loss, pos_loss, raw_terms = _unroll_force_loss(
                model, batch, _K_now, Wall, h, rest_g, edge_index_b, N,
                x_mean_g, x_std_g, e_mean_g, e_std_g, scale_vec_g,
                ang_scale_vec_g, acc_mean_g, acc_std_g, g_step_g, dt,
                use_wind=use_wind, use_drag_baseline=use_drag_baseline,
                k_over_m=k_over_m, contact_d0=contact_d0, contact_tau=contact_tau,
                loss_mode=loss_mode,
                phys=phys, phys_weights=phys_weights)

            (loss / accumulation_steps).backward()
            if (bi + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += float(loss.detach())
            for key, v in raw_terms.items():
                phys_accum[key] = phys_accum.get(key, 0.0) + v
            num_batches += 1
            if max_steps is not None and global_step >= max_steps:
                budget_spent = True
                print(f"  step budget reached: {global_step}/{max_steps} "
                      f"optimizer steps")
                break

        if num_batches % accumulation_steps != 0:      # flush the remainder
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_loss / max(num_batches, 1)
        epoch_num = epoch + 1
        train_loss_epochs.append(epoch_num)
        train_loss_values.append(float(avg_train_loss))
        t2 = time.time()
        if scheduler is not None:
            scheduler.step()

        if any_phys and phys_accum:
            nb = max(num_batches, 1)
            line = " | ".join(f"{k}: {v/nb:.3e}" for k, v in sorted(phys_accum.items()))
            print(f"  Physics terms (raw) | {line}")
            if fix_mu is None and learn_mu:
                mu_trace.append((epoch + 1, float(phys.mu)))
                print(f"  recovered mu = {float(phys.mu):.4f}"
                      f"   (data generator used 0.198 for the replica sets)")

        if epoch % validation_check_interval == 0:
            rollout_center, rollout_angle = rollout_force_batched(
                model, dataset_val, Wall, h, rest_g,
                x_mean_g, x_std_g, e_mean_g, e_std_g, scale_vec_g,
                ang_scale_vec_g, g_step_g, dt,
                device, use_wind=use_wind, use_drag_baseline=use_drag_baseline,
                k_over_m=k_over_m, contact_d0=contact_d0, contact_tau=contact_tau,
                mass=mass)
            print(f"  Rollout val | center: {rollout_center:.4f} | angle: {rollout_angle:.2f}")
            avg_val_loss = rollout_center
            val_loss_epochs.append(epoch_num)
            val_loss_values.append(float(avg_val_loss))

            best_eligible = (multistep <= 1) or (curriculum_schedule is None) or (_K_now == multistep)
            if avg_val_loss < best_val_loss and best_eligible:
                best_val_loss = float(avg_val_loss)
                best_val_epoch = epoch_num
                best_model_path = os.path.splitext(save_model_path)[0] + "_best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} at epoch {best_val_epoch}")
            elif avg_val_loss < best_val_loss:
                print(f"  (val {avg_val_loss:.6f} beats best, but curriculum K={_K_now} "
                      f"< final K={multistep} -- not saved)")

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.9f} | "
                  f"Val Loss: {avg_val_loss:.9f}")
            torch.save({"train_loss_epochs": train_loss_epochs,
                        "train_loss_values": train_loss_values,
                        "val_loss_epochs": val_loss_epochs,
                        "val_loss_values": val_loss_values,
                        "validation_check_interval": validation_check_interval,
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch,
                        "mu_trace": mu_trace,
                        "global_step": global_step}, loss_history_path)
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.9f}")
        print(f"Epoch {epoch+1}: build={t1-t0:.1f}s, train={t2-t1:.1f}s (K={_K_now})",
              flush=True)

        if (epoch + 1) % epoch_checkpoint_interval == 0:
            checkpoint_path = os.path.splitext(save_model_path)[0] + f"_epoch{epoch+1}.pt"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss_epochs": train_loss_epochs,
                        "train_loss_values": train_loss_values,
                        "val_loss_epochs": val_loss_epochs,
                        "val_loss_values": val_loss_values,
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch}, checkpoint_path)
            prune_old_checkpoints(save_model_path, keep_last_n_checkpoints)
            kept = ("all" if not keep_last_n_checkpoints
                    else f"last {keep_last_n_checkpoints}")
            print(f"Checkpoint saved to {checkpoint_path}  (keeping {kept})")

    final_path = os.path.splitext(save_model_path)[0] + "_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    phys_path = os.path.splitext(save_model_path)[0] + "_physics.pt"
    torch.save({"state_dict": phys.state_dict(),
                "recovered_mu": float(phys.mu),
                "mu_mode": ("fixed" if fix_mu is not None
                            else "learnable" if learn_mu else "frozen"),
                "weights": phys_weights}, phys_path)
    if any_phys and fix_mu is None and learn_mu:
        print(f"Recovered friction coefficient mu = {float(phys.mu):.4f} "
              f"(saved to {phys_path})")
    torch.save({"train_loss_epochs": train_loss_epochs,
                "train_loss_values": train_loss_values,
                "val_loss_epochs": val_loss_epochs,
                "val_loss_values": val_loss_values,
                "validation_check_interval": validation_check_interval,
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
                "mu_trace": mu_trace,
                "global_step": global_step}, loss_history_path)
    print(f"Loss history saved to {loss_history_path} "
          f"(total optimizer steps: {global_step})")
    return model

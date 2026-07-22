import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from generate_node_states import get_clean_positions, add_random_walk_noise, relative_wind
from fast_batch import build_epoch_tensors, iterate_batches, n_batches
import time
import math

BLOCK_WIDTH_FOR_LOSS = 2 * 0.0524

def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False
 



#This is the GNS layer, which performs message passing and updates node features based on edge features.
#It uses a multi-layer perceptron (MLP) to process the edge features and node features.
#The layer performs edge updates, aggregates messages from incoming edges, and updates node features.
class GNSLayer(nn.Module):

    #This layer is responsible for processing the node and edge features in the GNN.
    #It takes the dimensions of the node features, edge features, and a hidden dimension for the MLPs.
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        # Edge update MLP that takes the concatenated sender node features, receiver node features, and edge features as input,
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Node update MLP. Takes the concatenated node features and aggregated messages as input, and outputs the 
        # updated node features.
        # There is no RELU at the end of the node MLP because we want the model to be able
        # to output negative accelerations if needed.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        #Normalization layers for the edge messages and node features, which help stabilize training.
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(node_dim)


    #The forward function defines how the data flows through the layer. It takes the node features x, edge indices,
    #and edge attributes as input, and outputs the updated node features.
    def forward(self, x, edge_index, edge_attr):
        
        #Gets the sender and receiver node indices from the edge_index, which is a tensor of shape [2, num_edges]
        #  where the first row contains the sender node indices and the second row contains the receiver node indices for each edge.
        senders = edge_index[0]
        receivers = edge_index[1]

        #Extracts the sender node features and receiver node features based on the sender and receiver indices,
        sender_features = x[senders]
        receiver_features = x[receivers]

        #Concatenates the sender node features, receiver node features, and edge attributes to form the input for the edge MLP.
        edge_input = torch.cat(
            [sender_features, receiver_features, edge_attr], dim=-1
        )

        #Processes the edge input through the edge MLP to get the edge updates, and applies layer normalization to the edge updates.
        edge_update = self.edge_mlp(edge_input)
        edge_update = self.edge_norm(edge_update)

        #Updates the edge attributes by adding the edge updates to the original edge attributes.
        edge_attr = edge_attr + edge_update

        #Gets the number of nodes and the hidden dimension from the input tensors. 
        num_nodes = x.size(0)
        hidden_dim = edge_attr.size(1)

        #Preallocates a tensor for aggregating messages for each node, initialized to zeros.
        node_agg = torch.zeros(num_nodes, hidden_dim, device=x.device)

        #Aggregates the edge updates for each receiver node by adding the edge updates to the corresponding receiver nodes in the
        #node_agg tensor.
        node_agg.index_add_(0, receivers, edge_attr)

        #Concatenates the original node features with the aggregated messages to form the input for the node MLP.
        node_input = torch.cat([x, node_agg], dim=-1)

        #Processes the node input through the node MLP to get the node updates.
        node_update = self.node_mlp(node_input)
        node_update = self.node_norm(node_update)

        #Updates the node features by adding the normalized node update to the original node features.
        x = x + node_update

        return x, edge_attr


#This is the full encoder-processor-decoder model
class GNSModel(nn.Module):

    #Initializes the GNS model with the dimensions of the node features, edge features, a latent dimension for the MLPs,
    # and the number of message passing layers.
    def __init__(self, node_in_dim, edge_in_dim,
                 latent_dim=128, L = 5, K = 1):
        super().__init__()

        #Initialize the number of message passing steps (L) and number of repeated blocks (K)
        self.K = K
        self.L = L


        #Node encoder. Takes in the raw node features and encodes them into a latent space using an MLP. 
        #The output of the encoder is used as the initial node features for the message passing layers.
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        #Edge encoder. Similar to the node encoder, it takes the raw edge features and encodes them into a latent space 
        #using an MLP. 
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        #ADD COMMENT -------------------------------------------------------------------------------------
        self.processor_layers = nn.ModuleList([
            GNSLayer(latent_dim, latent_dim, latent_dim)
            for _ in range(L)
        ])

        #Decoder (NO LayerNorm per paper) Takes the 128 dimentional node features and decodes them into 
        #x,y,z accelration, using a MLP.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )

    #This is the forward function for the GNSModel, which defines how the data flows through the model. 
    #It takes a Data object as input, which contains the node features, edge indices, and edge attributes.
    def forward(self, data):

        #First the raw node features and edge features are encoded into the latent space 
        #using the node encoder and edge encoder MLPs.
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        #This is the main message passing loop, where the node features are updated based on the edge features and 
        #neighboring node features. The number of iterations of message passing is determined by the parameter L, 
        #and the number of times the blocks are repeated is determined by K.
        for _ in range(self.K):  
            for layer in self.processor_layers: 
                x, edge_attr = layer(x, data.edge_index, edge_attr)

        #Finally, the processed node features are passed through the decoder MLP to get the predicted accelerations for each node.
        decoded_state = self.decoder(x)

        return decoded_state
    
def _shape_match_batched(pred, rest, alpha=1.0):
    """Batched uniform-mass shape_match. pred:(B,N,3) rest:(N,3). Mirrors shape_match()."""
    x0_cm = rest.mean(dim=0)                                  # (3,)
    x_cm  = pred.mean(dim=1, keepdim=True)                    # (B,1,3)
    q = rest - x0_cm                                          # (N,3)
    p = pred - x_cm                                           # (B,N,3)
    Apq = torch.einsum('bni,nj->bij', p, q)                   # (B,3,3)
    U, S, Vh = torch.linalg.svd(Apq)
    d = torch.linalg.det(U @ Vh)                              # (B,)
    D = torch.eye(3, device=pred.device, dtype=pred.dtype).expand(d.shape[0], 3, 3).clone()
    D[:, 2, 2] = d
    R = U @ D @ Vh                                            # (B,3,3)
    g = torch.einsum('bij,nj->bni', R, q) + x_cm             # (B,N,3)
    return pred + alpha * (g - pred)


def _rollout_validation_batched(model, val_trajs, Wall, h,
                                x_mean, x_std, e_mean, e_std, acc_mean, acc_std,
                                device, do_shape_match=True, use_wind=False):
    """
    Batched rollout validation, faithful for VARIABLE-LENGTH trajectories.
    Pads every trajectory to the max length and rolls them all forward in lockstep,
    but scores each trajectory only over its own real frames -> matches the per-traj
    rollout / evaluate_model. Padded frames are never scored and don't affect other
    trajectories (block-diagonal graph).
    """
    from evaluate_metrics import compute_metrics  # deferred: avoids circular import

    model.eval()
    B = len(val_trajs)
    N = val_trajs[0]["positions"].shape[1]

    lengths = [t["positions"].shape[0] for t in val_trajs]      # real T_i per trajectory
    T_max = max(lengths)

    def _pad_to(p, T_max):
        if p.shape[0] == T_max:
            return p
        tail = p[-1:].expand(T_max - p.shape[0], -1, -1)        # repeat last frame
        return torch.cat([p, tail], dim=0)

    clean = torch.stack([_pad_to(t["positions"], T_max) for t in val_trajs], dim=0).to(device)
    wind  = torch.stack([t["wind_vector"] for t in val_trajs], dim=0).to(device)
    nodes_body = torch.stack([t["nodes_body"] for t in val_trajs], dim=0).to(device)
    rest_positions = val_trajs[0]["nodes_body"].to(device)

    ei = val_trajs[0]["edge_index"].to(device)
    edge_index_b = torch.cat([ei + b * N for b in range(B)], dim=1)

    to_dev = lambda x: None if x is None else x.to(device)
    x_mean, x_std, e_mean, e_std = map(to_dev, (x_mean, x_std, e_mean, e_std))
    acc_mean, acc_std = to_dev(acc_mean), to_dev(acc_std)

    true_from_h = clean[:, h:]                   # (B, L_max, N, 3), L_max = T_max - h
    L_max = true_from_h.shape[1]

    pos_window = [true_from_h[:, i] for i in range(h + 1)]
    pred = [true_from_h[:, i].clone() for i in range(h + 1)]

    with torch.no_grad():
        for _ in range(h, L_max - 1):
            x_node, e_attr = _build_features_for_unroll(
                pos_window, edge_index_b, nodes_body, Wall, wind,
                x_mean, x_std, e_mean, e_std, B, N, use_wind=use_wind
            )
            data = Data(x=x_node, edge_index=edge_index_b, edge_attr=e_attr)
            a = (model(data) * acc_std + acc_mean).reshape(B, N, 3)
            x_next = a + 2.0 * pos_window[-1] - pos_window[-2]
            if do_shape_match:
                x_next = _shape_match_batched(x_next, rest_positions, alpha=1.0)
            pred.append(x_next)
            pos_window = pos_window[1:] + [x_next]

    pred = torch.stack(pred, dim=1)              # (B, L_max, N, 3)

    # Score each trajectory over ONLY its real frames (L_i = T_i - h).
    center_errors, angle_errors = [], []
    for b in range(B):
        Lb = lengths[b] - h
        m = compute_metrics(pred[b, :Lb], true_from_h[b, :Lb], rest_positions)
        center_errors.append(m['center_error'])
        angle_errors.append(m['angle_error_deg'])
    return float(np.mean(center_errors)), float(np.mean(angle_errors))
    


#This function builds a dataset from a range of trajectories. It processes each trajectory to extract the graph 
#features and constructs a list of Data objects,
def build_dataset(Wall, traj_range,
                  nodes_per_edge=5,
                  nearest_neighbors=3,
                  h=2,
                  trajectory_folder="data/tosses_processed",
                  weights_only=True,
                  unscale_data=True):

    dataset = []

    #Runs through each trajectory in the specified range
    for throw_number in traj_range:
        print(f"Processing trajectory {throw_number}")
        positions, edge_index, nodes_body, wind_vector  = get_clean_positions(
            Wall,
            throw_number,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            data_folder=trajectory_folder,
            weights_only=weights_only,
            unscale_data=unscale_data,
        )
        dataset.append({"positions": positions, "edge_index": edge_index, "nodes_body": nodes_body, "wind_vector": wind_vector})

    return dataset

#This function takes a raw trajectory dictionary, applies random-walk noise to the node positions, 
#and returns a list of Data objects which include the node features, edge features, and target accelerations for each timestep.
def _build_timestep_samples(traj, Wall, h, noise_scale=3e-4,
                            x_mean=None, x_std=None, e_mean=None, e_std=None,
                            acc_mean=None, acc_std=None, use_wind=False):
    clean_positions = traj["positions"]
    noisy_positions, noise = add_random_walk_noise(clean_positions, noise_scale=noise_scale)
 
    edge_index = traj["edge_index"]
    sender = edge_index[0]
    receiver = edge_index[1]
 
    nodes_body = traj["nodes_body"]
    dU = nodes_body[sender] - nodes_body[receiver]
    dU_norm = torch.norm(dU, dim=1, keepdim=True)
 
    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32)
    wind_vector = traj["wind_vector"]
 
    T = clean_positions.shape[0]
    N = noisy_positions.shape[1]
    M = T - 1 - h  # number of timestep samples (matches range(h, T-1))
    if M <= 0:
        return []
 
    # ---- Velocity history features (vectorized over t) ----
    # Original: for t in [h, T-2], v_k = noisy[t-k] - noisy[t-k-1] for k in [0, h-1]
    v_fd_list = []
    for k in range(h):
        v_k = noisy_positions[h-k : T-1-k] - noisy_positions[h-k-1 : T-2-k]  # (M, N, 3)
        v_fd_list.append(v_k)
    v_fd_all = torch.cat(v_fd_list, dim=-1)  # (M, N, 3h)
 
    # ---- Wall distance (vectorized over t) ----
    rel_pos = noisy_positions[h : T-1] - wall_c                        # (M, N, 3)
    dist_all = torch.sum(rel_pos * wall_n, dim=-1, keepdim=True).clamp(-0.05, 0.5)  # (M, N, 1)
 
    # ---- Node features ----
    v_curr = v_fd_list[0]
    node_parts = [v_fd_all]
    if use_wind:
        u, u_norm = relative_wind(wind_vector.view(1, 1, 3), v_curr)
        node_parts += [u, u_norm]
    node_parts.append(dist_all)
    x_node_all = torch.cat(node_parts, dim=-1)
 
    # ---- Edge features (vectorized over t) ----
    pos_at_t = noisy_positions[h : T-1]                                # (M, N, 3)
    d_all = pos_at_t[:, sender] - pos_at_t[:, receiver]                # (M, E, 3)
    d_norm_all = torch.norm(d_all, dim=-1, keepdim=True)               # (M, E, 1)
    dU_broadcast = dU.unsqueeze(0).expand(M, -1, -1)
    dU_norm_broadcast = dU_norm.unsqueeze(0).expand(M, -1, -1)
    e_attr_all = torch.cat([d_all, d_norm_all, dU_broadcast, dU_norm_broadcast], dim=-1)  # (M, E, 8)
 
    # ---- Acceleration targets ----
    accel_clean = clean_positions[h+1 : T] - 2.0 * clean_positions[h : T-1] + clean_positions[h-1 : T-2]
    accel_corrected = accel_clean - noise[h-1 : T-2]                   # (M, N, 3)
 
    # ---- Apply normalization to batched tensors (folded in for speed) ----
    if x_mean is not None:
        x_node_all = (x_node_all - x_mean) / x_std
        e_attr_all = (e_attr_all - e_mean) / e_std
    if acc_mean is not None:
        accel_corrected = (accel_corrected - acc_mean) / acc_std
 
    # ---- Build Data objects (cheap now — just object construction) ----
    samples = [
        Data(x=x_node_all[i], edge_index=edge_index, edge_attr=e_attr_all[i], y=accel_corrected[i])
        for i in range(M)
    ]
    return samples


#This function generates a random rotation matrix for rotating the data around the z-axis.
def random_z_rotation():
    theta = torch.rand(1) * 2 * torch.pi
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ]).squeeze()

    return R

#This function applies a random rotation to the node features, edge features, and target accelerations in a batch of data.
def rotate_batch(batch, h, use_wind=False):
    device = batch.x.device
    R = random_z_rotation().to(device)

    # Node layout:
    #   use_wind : [v_fd(3h), u(3), ||u||(1), dist(1)] -> (h+1) 3-vectors, 2 scalars
    #   no wind  : [v_fd(3h),                 dist(1)] ->  h    3-vectors, 1 scalar
    num_vec3 = (h + 1) if use_wind else h
    vecs = batch.x[:, :num_vec3 * 3].view(-1, num_vec3, 3)
    vecs = torch.matmul(vecs, R.T).view(-1, num_vec3 * 3)
    scalars = batch.x[:, num_vec3 * 3:]                 # ||u||, dist (or just dist) — rotation-invariant
    batch.x = torch.cat([vecs, scalars], dim=1)

    # Edge features unchanged: [d(3), d_norm(1), dU(3), dU_norm(1)]
    d       = batch.edge_attr[:, 0:3] @ R.T
    d_norm  = batch.edge_attr[:, 3:4]
    dU      = batch.edge_attr[:, 4:7] @ R.T
    dU_norm = batch.edge_attr[:, 7:8]
    batch.edge_attr = torch.cat([d, d_norm, dU, dU_norm], dim=1)

    batch.y = batch.y @ R.T
    return batch



def _build_features_for_unroll(pos_window, edge_index, nodes_body, Wall, wind,
                               x_mean, x_std, e_mean, e_std, B, N, use_wind=False):
    """
    pos_window: list of h+1 tensors, each (B, N, 3), most recent last.
    Returns flat (B*N, node_dim) and (B*E, edge_dim) ready for the model.
    """
    device = pos_window[0].device

    # Velocity history via finite differences — same convention as get_gns_features
    v_fd_list = []
    for k in range(len(pos_window) - 1):
        v_fd_list.append(pos_window[-(k+1)] - pos_window[-(k+2)])
    v_fd = torch.cat(v_fd_list, dim=-1)
    v_curr = v_fd_list[0]                                   # (B, N, 3)

    x_t = pos_window[-1]
    wall_n = torch.as_tensor(Wall.normal,           dtype=x_t.dtype, device=device)
    wall_c = torch.as_tensor(Wall.center_position,  dtype=x_t.dtype, device=device)
    dist = torch.sum((x_t - wall_c) * wall_n, dim=-1, keepdim=True).clamp(-0.05, 0.5)  # (B, N, 1)

    node_parts = [v_fd]
    if use_wind:
        u, u_norm = relative_wind(wind.unsqueeze(1), v_curr)   # (B,3)->(B,1,3) broadcasts
        node_parts += [u, u_norm]
    node_parts.append(dist)
    x_node = torch.cat(node_parts, dim=-1).reshape(B * N, -1)

    # Edge features — operate on flattened positions so the offset edge_index works
    x_t_flat = x_t.reshape(B * N, 3)
    nodes_body_flat = nodes_body.reshape(B * N, 3)
    src, dst = edge_index[0], edge_index[1]

    d  = x_t_flat[src]        - x_t_flat[dst]
    dU = nodes_body_flat[src] - nodes_body_flat[dst]
    e_attr = torch.cat([d, torch.norm(d, dim=-1, keepdim=True),
                        dU, torch.norm(dU, dim=-1, keepdim=True)], dim=-1)

    if x_mean is not None:
        x_node = (x_node - x_mean) / x_std
    if e_mean is not None:
        e_attr = (e_attr - e_mean) / e_std

    return x_node, e_attr



def rotate_chain_batch(batch):
    R = random_z_rotation().to(batch["history"].device)
    batch["history"]       = batch["history"]       @ R.T
    batch["clean_history"] = batch["clean_history"] @ R.T   # NEW
    batch["targets"]       = batch["targets"]       @ R.T
    batch["nodes_body"]    = batch["nodes_body"]    @ R.T
    batch["wind"]          = batch["wind"]          @ R.T
    batch["noise_at_input"] = batch["noise_at_input"] @ R.T
    return batch

def build_chain_index(dataset_train, h, multistep, stride=1,
                      t_contact_list=None, impact_weight=1):
    """List of (traj_idx, start_t). Chains whose PREDICTED span
    [s+h+1, s+h+multistep] contains the impact frame are appended
    impact_weight times (importance sampling on the impulse)."""
    span = h + 1 + multistep
    index = []
    for ti, traj in enumerate(dataset_train):
        last_start = traj["positions"].shape[0] - span
        tc = t_contact_list[ti] if t_contact_list is not None else None
        for s in range(0, last_start + 1, stride):
            straddles = (tc is not None) and (s + h < tc <= s + h + multistep)
            reps = impact_weight if straddles else 1
            index.extend([(ti, s)] * reps)
    return index


def n_chain_batches(chain_index, batch_size):
    return (len(chain_index) + batch_size - 1) // batch_size


def iterate_chain_batches(dataset_train, chain_index, batch_size, h, multistep,
                          device, shuffle=True, noise_scale=0.0):
    """Yields chain batches as dicts. Gathers windows on the fly (light on memory)."""
    ei_cpu = dataset_train[0]["edge_index"]
    N = dataset_train[0]["positions"].shape[1]
    nodes_body_single = dataset_train[0]["nodes_body"]

    order = torch.randperm(len(chain_index)) if shuffle else torch.arange(len(chain_index))
    for start in range(0, len(chain_index), batch_size):
        sel = order[start:start + batch_size].tolist()
        B = len(sel)

        win_frames = [[] for _ in range(h + 1)]
        targ, winds = [], []
        for idx in sel:
            ti, s = chain_index[idx]
            pos = dataset_train[ti]["positions"]            # (T, N, 3) on CPU
            for j in range(h + 1):
                win_frames[j].append(pos[s + j])
            targ.append(pos[s + h + 1 : s + h + 1 + multistep])   # (multistep, N, 3)
            winds.append(dataset_train[ti]["wind_vector"])

        window = [torch.stack(f, 0).to(device) for f in win_frames]   # h+1 x (B,N,3)

        # --- Random-walk noise on the INPUT window only (targets stay clean) ---
        # Same convention as add_random_walk_noise: i.i.d. velocity noise per
        # transition, cumsum'd into position offsets; frame 0 stays clean.
        if noise_scale > 0:
            vel_noise = torch.randn(h, B, N, 3, device=device) * noise_scale
            pos_noise = torch.cumsum(vel_noise, dim=0)        # (h, B, N, 3)
            for j in range(h):
                window[j + 1] = window[j + 1] + pos_noise[j]


        targets = torch.stack(targ, 0).to(device)                     # (B, multistep, N, 3)
        wind = torch.stack(winds, 0).to(device)                       # (B, 3)
        nodes_body = nodes_body_single.unsqueeze(0).expand(B, -1, -1).to(device)
        edge_index_b = torch.cat([ei_cpu.to(device) + b * N for b in range(B)], dim=1)

        yield {"window": window, "targets": targets, "wind": wind,
               "nodes_body": nodes_body, "edge_index_b": edge_index_b, "B": B, "N": N}


def rotate_chain(batch):
    """z-rotation augmentation for a chain batch. Positions, targets, wind, and the rest
    mesh all rotate by the same R; the wall-distance feature is z-invariant, so it stays
    consistent. (Supersedes the old rotate_chain_batch, which used a different key set.)"""
    R = random_z_rotation().to(batch["wind"].device)
    batch["window"]     = [w @ R.T for w in batch["window"]]
    batch["targets"]    = batch["targets"]    @ R.T
    batch["nodes_body"] = batch["nodes_body"] @ R.T
    batch["wind"]       = batch["wind"]       @ R.T
    return batch


def _unroll_chain_loss(model, batch, multistep, Wall, h,
                       x_mean, x_std, e_mean, e_std, acc_mean, acc_std,
                       use_wind=False, block_width=BLOCK_WIDTH_FOR_LOSS):
    """Unroll the model `multistep` steps from a true window; POSITION MSE vs truth,
    full backprop through the rollout. Reuses _build_features_for_unroll + the exact
    integration from _rollout_validation_batched. No shape-matching during training
    (matches the paper; loss is on raw network predictions)."""
    window  = list(batch["window"])         # h+1 x (B,N,3)
    targets = batch["targets"]              # (B, multistep, N, 3)
    wind    = batch["wind"]
    nodes_body   = batch["nodes_body"]
    edge_index_b = batch["edge_index_b"]
    B, N = batch["B"], batch["N"]

    preds = []
    for _ in range(multistep):
        x_node, e_attr = _build_features_for_unroll(
            window, edge_index_b, nodes_body, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N, use_wind=use_wind,
        )
        data = Data(x=x_node, edge_index=edge_index_b, edge_attr=e_attr)
        a = (model(data) * acc_std + acc_mean).reshape(B, N, 3)
        x_next = a + 2.0 * window[-1] - window[-2]      # gradients flow through window
        preds.append(x_next)
        window = window[1:] + [x_next]

    preds = torch.stack(preds, dim=1)                   # (B, multistep, N, 3)
    return ((preds - targets) / block_width).pow(2).mean()

def _unroll_chain_loss_accel(model, batch, multistep, Wall, h,
                             x_mean, x_std, e_mean, e_std, acc_mean, acc_std,
                             use_wind=False):
    window  = list(batch["window"])         # h+1 x (B,N,3)
    targets = batch["targets"]              # (B, multistep, N, 3) TRUE positions
    wind, nodes_body = batch["wind"], batch["nodes_body"]
    edge_index_b, B, N = batch["edge_index_b"], batch["B"], batch["N"]

    total = 0.0
    for k in range(multistep):
        x_node, e_attr = _build_features_for_unroll(
            window, edge_index_b, nodes_body, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N, use_wind=use_wind,
        )
        data = Data(x=x_node, edge_index=edge_index_b, edge_attr=e_attr)
        pred_norm = model(data).reshape(B, N, 3)                 # normalized accel

        # true accel for THIS step, from the true positions, normalized the same way
        true_prev = window[-2]
        true_curr = window[-1]
        true_next = targets[:, k]
        true_accel = true_next - 2.0 * true_curr + true_prev
        true_accel_norm = (true_accel - acc_mean) / acc_std

        total = total + (pred_norm - true_accel_norm).pow(2).mean()

        # advance the window using the model's own prediction (this is what injects drift)
        a = (pred_norm * acc_std + acc_mean)
        x_next = a + 2.0 * true_curr - true_prev
        window = window[1:] + [x_next]

    return total / multistep


#This function trains the GNS model
def train_gnn(Wall,
              train_range,
              val_range,
              save_train_dataset_path,
              save_val_dataset_path,
              save_model_path,
              rebuild_datasets=False,
              epochs=50,
              batch_size=64,
              accumulation_steps=8,
              lr=1e-4,
              nodes_per_edge=5,
              nearest_neighbors=3,
              h=2,
              message_passing_layers=5,
              repeat_blocks=1,
              trajectory_folder="data/tosses_processed",
              weights_only=True,
              unscale_data=True,
              copy_weights_only_path = None,
              resume_checkpoint_path=None,
              epoch_checkpoint_interval=500,
              validation_check_interval=20,
              noise_scale=3e-4,
              multistep = 1,
              latent_dim = 128,
              use_rollout_validation=False,
              use_wind=False,
              impact_weight= 1,
              Learning_Rate_Scheduler=None,
              curriculum_epochs = 0,          # 0 = off; else epochs per ramp phase
              curriculum_schedule = None,     # e.g. [1,2,4,6,8]; None -> auto powers-of-2
              ):
    
    #Sets device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resume_checkpoint_path is not None and rebuild_datasets:
        print("Warning: resuming with rebuild_datasets=True can cause large loss jumps due to new data/noise.")

    #If the user wants to rebuild the datasets, it processes the trajectories to build the training and test datasets, 
    #and saves them.
    if rebuild_datasets:
        print("Building training dataset...")
        dataset_train = build_dataset(
            Wall,
            train_range,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            h=h,
            trajectory_folder=trajectory_folder,
            weights_only=weights_only,
            unscale_data=unscale_data,
        )
        torch.save(dataset_train, save_train_dataset_path)
        print(f"Training dataset saved to {save_train_dataset_path}")

        print("Building validation dataset...")
        dataset_val = build_dataset(
            Wall,
            val_range,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            h=h,
            trajectory_folder=trajectory_folder,
            weights_only=weights_only,
            unscale_data=unscale_data,
        )
        torch.save(dataset_val, save_val_dataset_path)
        print(f"Validation dataset saved to {save_val_dataset_path}")

    else:
        print("Loading training dataset...")
        dataset_train = torch.load(save_train_dataset_path, weights_only=False)
        print(f"Training dataset loaded from {save_train_dataset_path}")

        print("Loading validation dataset...")
        dataset_val = torch.load(save_val_dataset_path, weights_only=False)
        print(f"Validation dataset loaded from {save_val_dataset_path}")


    #Compute normalization stats from one clean pass (no noise) over training trajectories.
    clean_samples = []
    for traj in dataset_train:
        clean_samples.extend(_build_timestep_samples(traj, Wall, h=h, noise_scale=noise_scale, use_wind=use_wind))

    x_mean, x_std = _compute_node_stats(clean_samples)
    e_mean, e_std = _compute_edge_stats(clean_samples)
    acc_mean, acc_std = _compute_accel_stats(clean_samples)
    del clean_samples


    #saves the normalization statistics to a file, which can be used later for normalizing new data during inference.
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_norms.pt"
    torch.save({"x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std, "acc_mean": acc_mean, "acc_std": acc_std}, norm_stats_path)

    print(f"Saved normalization stats to {norm_stats_path}")


    x_mean_gpu = x_mean.to(device); x_std_gpu = x_std.to(device)
    e_mean_gpu = e_mean.to(device); e_std_gpu = e_std.to(device)
    acc_mean_gpu = acc_mean.to(device); acc_std_gpu = acc_std.to(device)

    #Build a clean validation sample set once (no training noise).
    val_samples = []
    for traj in dataset_val:
        val_samples.extend(_build_timestep_samples(traj, Wall, h=h, noise_scale=0.0, use_wind=use_wind))
    _apply_input_normalization(val_samples, x_mean, x_std, e_mean, e_std)
    _apply_accel_normalization(val_samples, acc_mean, acc_std)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    #Get input dimensions from one clean sample.
    sample_for_dims = _build_timestep_samples(dataset_train[0], Wall, h=h, noise_scale=0.0, use_wind=use_wind)[0]
    node_dim = sample_for_dims.x.shape[1]
    edge_dim = sample_for_dims.edge_attr.shape[1]

    #Initializes the GNS model, which consists of an encoder for the node features, an encoder for the edge features, 
    #multiple GNS layers for message passing,
    model = GNSModel(node_dim, edge_dim, latent_dim=latent_dim, L=message_passing_layers, K = repeat_blocks).to(device)
    if torch.cuda.is_available() and _triton_available():
        model = torch.compile(model)
        print("torch.compile enabled.")
    else:
        print("torch.compile disabled (no CUDA/Triton) — running eager mode.")
    #Sets the optimizer to Adam, which will be used to update the model parameters during training based on the computed gradients.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if Learning_Rate_Scheduler == "decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(1e-2) ** (1.0 / max(1, epochs)))
    elif Learning_Rate_Scheduler == "cosine":
        # Cosine cycles with decaying peaks (exploration), then a final anneal.
        n_cycles, peak_decay, final_frac = 4, 0.5, 1e-2
        explore_epochs = int(0.7 * epochs)
        cycle_len = max(1, explore_epochs // n_cycles)
        def _lam(ep):
            if ep < explore_epochs:
                cyc, pos = divmod(ep, cycle_len)
                peak = peak_decay ** cyc
                return final_frac + (peak - final_frac) * 0.5 * (1 + math.cos(math.pi * pos / cycle_len))
            pos = (ep - explore_epochs) / max(1, epochs - explore_epochs)
            peak = peak_decay ** n_cycles
            return final_frac + (peak - final_frac) * 0.5 * (1 + math.cos(math.pi * pos))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lam)
    elif Learning_Rate_Scheduler is not None:
        raise ValueError(f"Unknown Learning_Rate_Scheduler: {Learning_Rate_Scheduler!r}")
    else:
        print("No learning rate scheduler used.")
        scheduler = None

    def _coerce_cpu_byte_tensor(state):
        if state is None:
            return None
        if torch.is_tensor(state):
            return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()

    #Optionally resume training from a previous checkpoint.
    start_epoch = 0
    
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ckpt_state = checkpoint["model_state_dict"]
            # Strip or add _orig_mod prefix as needed for torch.compile compatibility
            if any(k.startswith('_orig_mod.') for k in ckpt_state.keys()):
                if not hasattr(model, '_orig_mod'):
                    ckpt_state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt_state.items()}
            elif hasattr(model, '_orig_mod'):
                ckpt_state = {'_orig_mod.' + k: v for k, v in ckpt_state.items()}
            model.load_state_dict(ckpt_state)

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore RNG states when available for closer continuation from checkpoint.
            if "torch_rng_state" in checkpoint:
                try:
                    torch_state = _coerce_cpu_byte_tensor(checkpoint["torch_rng_state"])
                    if torch_state is not None:
                        torch.set_rng_state(torch_state)
                except Exception as exc:
                    print(f"Warning: could not restore torch RNG state: {exc}")

            if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint and checkpoint["cuda_rng_state_all"] is not None:
                try:
                    cuda_states = [
                        _coerce_cpu_byte_tensor(s) for s in checkpoint["cuda_rng_state_all"]
                    ]
                    torch.cuda.set_rng_state_all(cuda_states)
                except Exception as exc:
                    print(f"Warning: could not restore CUDA RNG states: {exc}")

            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])

            if "python_rng_state" in checkpoint:
                random.setstate(checkpoint["python_rng_state"])

            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            print(f"Resumed training from checkpoint {resume_checkpoint_path} at epoch {start_epoch}")
            for _ in range(start_epoch):
                if scheduler is not None:
                    scheduler.step()
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {resume_checkpoint_path}")

    # Warm-start: load weights only, no optimizer/RNG/epoch/history restore.
    # Mutually exclusive with resume_checkpoint_path — resume takes precedence.
    if copy_weights_only_path is not None and resume_checkpoint_path is None:
        print(f"Warm-starting model weights from {copy_weights_only_path}")
        loaded = torch.load(copy_weights_only_path, map_location=device, weights_only=False)

        # Handle both raw state_dict files and full checkpoint dicts.
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        # Handle _orig_mod prefix from torch.compile
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        elif hasattr(model, '_orig_mod'):
            state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("Weights loaded. Starting fresh: epoch=0, new optimizer, empty loss history.")

    elif copy_weights_only_path is not None and resume_checkpoint_path is not None:
        print("Warning: copy_weights_only_path ignored because resume_checkpoint_path is set.")

    #Sets the loss function to mean squared error loss, which will be used to compute the loss between the
    # predicted accelerations and the target accelerations during training.
    loss_fn = nn.MSELoss()

    #Track losses over training. Validation is sparse, so keep separate epoch/loss lists.
    train_loss_epochs = []
    train_loss_values = []
    val_loss_epochs = []
    val_loss_values = []
    best_val_loss = float("inf")
    best_rollout_error = float("inf")
    best_val_epoch = None

    #If resuming and history exists in checkpoint, continue appending.
    if resume_checkpoint_path is not None and isinstance(checkpoint, dict):
        train_loss_epochs = list(checkpoint.get("train_loss_epochs", train_loss_epochs))
        train_loss_values = list(checkpoint.get("train_loss_values", train_loss_values))
        val_loss_epochs = list(checkpoint.get("val_loss_epochs", val_loss_epochs))
        val_loss_values = list(checkpoint.get("val_loss_values", val_loss_values))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_rollout_error = float(checkpoint.get("best_rollout_error", best_rollout_error))
        best_val_epoch = checkpoint.get("best_val_epoch", best_val_epoch)

    print("Starting training...")
    print(f"  Train samples: {len(dataset_train)} | Val samples: {len(dataset_val)}")
    print(f"  Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")

    #Training loop
    if multistep > 1:
        from evaluate_metrics import compute_phase_boundaries   # lazy: avoids circular import
        t_contact_list = [compute_phase_boundaries(traj["positions"])[0]
                        for traj in dataset_train]              # K-independent: compute once
        if curriculum_epochs > 0 and curriculum_schedule is None:
            ks, k = [], 1
            while k < multistep:
                ks.append(k); k *= 2
            curriculum_schedule = ks + [multistep]              # e.g. 8 -> [1,2,4,8]
        if curriculum_epochs > 0:
            print(f"Multistep curriculum: {curriculum_schedule}, "
                  f"{curriculum_epochs} epochs/phase, final K for the remainder")
        _K_now, chain_index = None, None
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_loss_tensor = torch.zeros((), device=device)
        t0 = time.time()
        if multistep > 1:
            if curriculum_epochs > 0:
                phase = min(epoch // curriculum_epochs, len(curriculum_schedule) - 1)
                K_epoch = curriculum_schedule[phase]
            else:
                K_epoch = multistep
            if K_epoch != _K_now:               # phase transition: rebuild the index
                _K_now = K_epoch
                chain_index = build_chain_index(dataset_train, h=h, multistep=_K_now,
                                                stride=1, t_contact_list=t_contact_list,
                                                impact_weight=impact_weight)
                print(f"[curriculum] epoch {epoch}: multistep K={_K_now} "
                      f"({len(chain_index)} chains)")
            num_batches = n_chain_batches(chain_index, batch_size)
            batch_iter = enumerate(iterate_chain_batches(
                dataset_train, chain_index, batch_size=batch_size,
                h=h, multistep=_K_now, device=device, shuffle=True, noise_scale=noise_scale
            ))
        else:
            # Single-step path: use fast_batch to skip PyG Data overhead.
            # Builds flat (total_M, N, F) tensors once and gathers slices per batch.
            epoch_data = build_epoch_tensors(
                dataset_train, Wall, h=h, noise_scale=noise_scale,
                x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
                acc_mean=acc_mean, acc_std=acc_std, use_wind=use_wind,
            )
            num_batches = n_batches(epoch_data, batch_size)
            # Note: iterate_batches moves tensors to device internally, so the
            # batch.to(device) call in the inner loop is a no-op (idempotent).
            batch_iter = enumerate(iterate_batches(
                epoch_data, batch_size=batch_size, shuffle=True, device=device,
            ))
        t1 = time.time()
        optimizer.zero_grad()
        effective_accumulation = min(accumulation_steps, num_batches)
 
        for i, batch in batch_iter:
            torch.cuda.synchronize(); b0 = time.time()
 
            if multistep > 1:
                batch = rotate_chain(batch)
                torch.cuda.synchronize(); b1 = time.time()
                loss = _unroll_chain_loss_accel(
                    model, batch, multistep=_K_now, Wall=Wall, h=h,
                    x_mean=x_mean_gpu, x_std=x_std_gpu, e_mean=e_mean_gpu, e_std=e_std_gpu,
                    acc_mean=acc_mean_gpu, acc_std=acc_std_gpu, use_wind=use_wind,
                )
                torch.cuda.synchronize(); b2 = time.time()
            else:
                batch = rotate_batch(batch, h, use_wind)
                torch.cuda.synchronize(); b1 = time.time()
                pred_accel = model(batch)
                loss = loss_fn(pred_accel, batch.y)
                torch.cuda.synchronize(); b2 = time.time()
 
            scaled_loss = loss / effective_accumulation
            scaled_loss.backward()
            torch.cuda.synchronize(); b3 = time.time()
 
            total_loss_tensor += loss.detach()
 
            if (i + 1) % effective_accumulation == 0 or (i + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize(); b4 = time.time()
 
            if epoch == 1 and i < 5:
                print(f"  batch {i}: to_gpu+rotate={b1-b0:.3f}s  fwd+loss={b2-b1:.3f}s  "
                    f"bwd={b3-b2:.3f}s  opt={b4-b3:.3f}s  total={b4-b0:.3f}s", flush=True)
 
        #Averages loss over the epoch
        total_loss = total_loss_tensor.item()
        avg_train_loss = total_loss / num_batches
        epoch_num = epoch + 1
        train_loss_epochs.append(epoch_num)
        train_loss_values.append(float(avg_train_loss))
        t2 = time.time()
        if scheduler is not None:
            scheduler.step()
        if epoch % validation_check_interval == 0:
            # --- Validation ---
            if not use_rollout_validation:
                model.eval()
                total_val_loss = 0.0
    
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        # No rotation augmentation at validation time
                        pred_accel = model(batch)
                        loss = loss_fn(pred_accel, batch.y)
                        total_val_loss += loss.item()
                
    
                avg_val_loss = total_val_loss / len(val_loader)

            # --- Rollout validation (paper-faithful checkpoint selection) ---
            if use_rollout_validation:
                rollout_center, rollout_angle = _rollout_validation_batched(
                    model, dataset_val, Wall, h,
                    x_mean, x_std, e_mean, e_std, acc_mean, acc_std,
                    device, do_shape_match=True, use_wind=use_wind
                )

                
                avg_val_loss = rollout_center
                print(f"  Rollout val | center: {rollout_center:.4f} | angle: {rollout_angle:.2f}")

            val_loss_epochs.append(epoch_num)
            val_loss_values.append(float(avg_val_loss))

 
            best_eligible = (multistep <= 1) or (curriculum_epochs == 0) or (_K_now == multistep)
            if avg_val_loss < best_val_loss and best_eligible:
                best_val_loss = float(avg_val_loss)
                best_val_epoch = epoch_num
                best_model_path = os.path.splitext(save_model_path)[0] + "_best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} at epoch {best_val_epoch}")
            elif avg_val_loss < best_val_loss:
                print(f"  (val {avg_val_loss:.6f} beats best, but curriculum K={_K_now} "
                      f"< final K={multistep} -- not saved)")
                
 
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.9f} | "
                f"Val Loss: {avg_val_loss:.9f}")
            tval = time.time()
            print("Validation time: {:.1f}s".format(tval - t2), flush=True)
            loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"
            torch.save(
                {
                    "train_loss_epochs": train_loss_epochs,
                    "train_loss_values": train_loss_values,
                    "val_loss_epochs": val_loss_epochs,
                    "val_loss_values": val_loss_values,
                    "validation_check_interval": validation_check_interval,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch,
                    "best_rollout_error": best_rollout_error,
                },
                loss_history_path,
            )
            print(f"Loss history saved to {loss_history_path}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.9f}")
        print(f"Epoch {epoch+1}: rebuild={t1-t0:.1f}s, train={t2-t1:.1f}s", flush=True)
            
        if (epoch + 1) % epoch_checkpoint_interval == 0:
            checkpoint_path = os.path.splitext(save_model_path)[0] + f"_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "train_loss_epochs": train_loss_epochs,
                    "train_loss_values": train_loss_values,
                    "val_loss_epochs": val_loss_epochs,
                    "val_loss_values": val_loss_values,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch,
                    "best_rollout_error": best_rollout_error,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")



    #Saves the trained model
    checkpoint_path = os.path.splitext(save_model_path)[0] + f"_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    #Persist full loss history for plotting after training.
    loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"
    torch.save(
        {
            "train_loss_epochs": train_loss_epochs,
            "train_loss_values": train_loss_values,
            "val_loss_epochs": val_loss_epochs,
            "val_loss_values": val_loss_values,
            "validation_check_interval": validation_check_interval,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "best_rollout_error": best_rollout_error,
        },
        loss_history_path,
    )
    print(f"Loss history saved to {loss_history_path}")

    return model


#The following functions compute the mean and standard deviation for the accelerations, 
#node features, and edge features across the entire dataset.

#The _compute_accel_stats function concatenates the target accelerations from all data points in the dataset,
#computes the mean and standard deviation for each acceleration dimension, and returns these statistics.
def _compute_accel_stats(dataset):
    y_all = torch.cat([d.y for d in dataset], dim=0)

    acc_mean = y_all.mean(dim=0)
    acc_std = y_all.std(dim=0).clamp_min(1e-8)

    return acc_mean, acc_std

#The _compute_node_stats function concatenates the node features from all data points in the dataset, 
#computes the mean and standard deviation for each feature dimension, and returns these statistics.
def _compute_node_stats(dataset):
    x_all = torch.cat([d.x for d in dataset], dim=0)

    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0).clamp_min(1e-8)

    return mean, std

#The _compute_edge_stats function concatenates the edge features from all data points in the dataset,
#computes the mean and standard deviation for each feature dimension, and returns these statistics.
def _compute_edge_stats(dataset):
    e_all = torch.cat([d.edge_attr for d in dataset], dim=0)

    mean = e_all.mean(dim=0)
    std = e_all.std(dim=0).clamp_min(1e-8)

    return mean, std


#The _apply_input_normalization function takes the dataset and the computed mean and standard deviation 
#for the node features and edge features, and applies normalization to the node features and edge features 
#in each data point in the dataset.
def _apply_input_normalization(dataset, x_mean, x_std, e_mean, e_std):

    for d in dataset:
        d.x = (d.x - x_mean) / x_std
        d.edge_attr = (d.edge_attr - e_mean) / e_std


#The _apply_accel_normalization function takes the dataset and the computed mean and standard deviation for 
#the target accelerations, and applies normalization to the target accelerations in each data point in the dataset.
def _apply_accel_normalization(dataset, acc_mean, acc_std):
    for d in dataset:
        d.y = (d.y - acc_mean) / acc_std

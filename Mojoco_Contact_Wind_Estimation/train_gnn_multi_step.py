import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from generate_node_states import get_clean_positions, add_random_walk_noise
from fast_batch import build_epoch_tensors, iterate_batches, n_batches
import time

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
                            acc_mean=None, acc_std=None):
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
    dist_all = torch.sum(rel_pos * wall_n, dim=-1, keepdim=True).clamp(0.0, 0.5)  # (M, N, 1)
 
    # ---- Wind broadcast ----
    wind_broadcast = wind_vector.view(1, 1, -1).expand(M, N, -1)       # (M, N, 3)
 
    # ---- Node features ----
    x_node_all = torch.cat([v_fd_all, wind_broadcast, dist_all], dim=-1)  # (M, N, 3h+4)
 
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
def rotate_batch(batch):
    
    #Gets the device of the input data
    device = batch.x.device

    #Generates a random rotation matrix for rotating the data around the z-axis.
    R = random_z_rotation().to(device)

    #Gets the length of the state vector. From this we can tell how many velocity history steps there are 
    # (since we know the distance feature is always 1 dim and at the end), which allows us to correctly reshape 
    # the velocity features for rotation. We also know there is a wind vector in the -4:-1 dims,
    state_vector_length = len(batch.x[0])

    #Calculates the number of velocity vectors in the node features by taking the total length of the state vector,
    # subtracting 1 for the distance feature and 3 for the wind vector, and then dividing by 3 (the dimension of each velocity vector).
    num_vel_vectors = (state_vector_length - 1 - 3) // 3


    # ---- Rotate node features ----
    # x contains [v_t, v_tm1, v_{t-2},..., wind_vector, dist]


    #Extracts the velocity features, wind vector, and distance feature from the node features in batch.x
    # based on the calculated number of velocity vectors.
    v = batch.x[:, :num_vel_vectors*3] 
    wind_vector = batch.x[:, num_vel_vectors*3:num_vel_vectors*3 + 3]
    dist = batch.x[:, num_vel_vectors*3 + 3:num_vel_vectors*3 + 4]

    #The velocity features are reshaped to separate the current and previous velocity vectors,
    #then rotated using the random rotation matrix R, and finally reshaped back to the original format.
    v = v.view(-1, num_vel_vectors, 3)
    v = torch.matmul(v, R.T)
    v = v.view(-1, num_vel_vectors * 3)

    #Rotates the wind vector using the same random rotation matrix R to ensure consistency in the data augmentation.
    wind_vector = wind_vector @ R.T

    #The rotated velocity features are concatenated with the distance features to form the new node features, 
    #which are then assigned back to batch.x.
    batch.x = torch.cat([v, wind_vector, dist], dim=1)

    # ---- Rotate edge features ----
    # edge_attr = [d, d_norm, dU, dU_norm]

    #Extracts the distance and relative velocity features from the edge attributes.
    d = batch.edge_attr[:, 0:3]
    d_norm = batch.edge_attr[:, 3:4]
    dU = batch.edge_attr[:, 4:7]
    dU_norm = batch.edge_attr[:, 7:8]

    #The distance and relative velocity features are rotated using the same random rotation matrix R.
    d = d @ R.T
    dU = dU @ R.T

    #The rotated distance and relative velocity features are concatenated with the normalized distance and 
    #relative velocity features
    batch.edge_attr = torch.cat(
        [d, d_norm, dU, dU_norm],
        dim=1
    )

    #The target accelerations in batch.y are also rotated using the same random rotation matrix R.
    batch.y = batch.y @ R.T



    return batch


#This function withh build all of the chains possible from a trajectory. The position history of the first position in the chain
#Will be recorded, and then the true positions of the next K steps in the chain will be recorded as the target.
def _build_chain_samples(traj, Wall, h, K, noise_scale=3e-4):
    clean_positions = traj["positions"]
    noisy_positions, noise = add_random_walk_noise(clean_positions, noise_scale=noise_scale)

    T = clean_positions.shape[0]
    samples = []

    for t in range(h, T - K):
        history_positions = noisy_positions[t-h : t+1]   # (h+1, N, 3) — model input
        clean_history     = clean_positions[t-h : t+1]   # (h+1, N, 3) — for true accel targets
        target_positions  = clean_positions[t+1 : t+1+K] # (K, N, 3) — clean
        noise_at_input    = noise[t-1]

        samples.append({
            "history_positions": history_positions,
            "clean_history":     clean_history,
            "target_positions":  target_positions,
            "noise_at_input":    noise_at_input,
            "edge_index":  traj["edge_index"],
            "nodes_body":  traj["nodes_body"],
            "wind_vector": traj["wind_vector"],
        })

    return samples

def _collate_chain_batch(chain_samples):
    B = len(chain_samples)
    N = chain_samples[0]["history_positions"].shape[1]
    E = chain_samples[0]["edge_index"].shape[1]

    history       = torch.stack([s["history_positions"] for s in chain_samples], dim=0)
    clean_history = torch.stack([s["clean_history"]     for s in chain_samples], dim=0)  # NEW
    targets       = torch.stack([s["target_positions"]  for s in chain_samples], dim=0)
    wind          = torch.stack([s["wind_vector"]       for s in chain_samples], dim=0)
    nodes_body    = torch.stack([s["nodes_body"]        for s in chain_samples], dim=0)
    noise_at_input = torch.stack([s["noise_at_input"]   for s in chain_samples], dim=0)  # (B, N, 3)

    edge_index_single = chain_samples[0]["edge_index"]
    edge_index_batched = torch.cat([edge_index_single + b * N for b in range(B)], dim=1)

    return {
        "history": history, "clean_history": clean_history,    # NEW key
        "targets": targets, "wind": wind, "nodes_body": nodes_body,
        "edge_index": edge_index_batched,
        "B": B, "N": N, "E": E,
        "noise_at_input": noise_at_input,
    }


def _build_features_for_unroll(pos_window, edge_index, nodes_body, Wall, wind,
                               x_mean, x_std, e_mean, e_std, B, N):
    """
    pos_window: list of h+1 tensors, each (B, N, 3), most recent last.
    Returns flat (B*N, node_dim) and (B*E, edge_dim) ready for the model.
    """
    device = pos_window[0].device

    # Velocity history via finite differences — same convention as get_gns_features
    v_fd = []
    for k in range(len(pos_window) - 1):
        v_fd.append(pos_window[-(k+1)] - pos_window[-(k+2)])
    v_fd = torch.cat(v_fd, dim=-1)  # (B, N, 3*h)

    x_t = pos_window[-1]
    wall_n = torch.as_tensor(Wall.normal,           dtype=x_t.dtype, device=device)
    wall_c = torch.as_tensor(Wall.center_position,  dtype=x_t.dtype, device=device)
    dist = torch.sum((x_t - wall_c) * wall_n, dim=-1, keepdim=True).clamp(0.0, 0.5)  # (B, N, 1)

    wind_expanded = wind.unsqueeze(1).expand(-1, N, -1)  # (B, N, 3)
    x_node = torch.cat([v_fd, wind_expanded, dist], dim=-1).reshape(B * N, -1)

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


def _unroll_chain_loss_accel(model, chain_batch, K, Wall, h,
                             x_mean, x_std, e_mean, e_std, accel_mean, accel_std):
    history       = chain_batch["history"]        # (B, h+1, N, 3) noisy — model input
    clean_history = chain_batch["clean_history"]  # (B, h+1, N, 3) clean — for targets
    targets       = chain_batch["targets"]        # (B, K, N, 3) clean
    wind          = chain_batch["wind"]
    nodes_body    = chain_batch["nodes_body"]
    edge_index    = chain_batch["edge_index"]
    noise_at_input = chain_batch["noise_at_input"]
    B, N = chain_batch["B"], chain_batch["N"]

    # Build one long clean trajectory: [clean[t-h], ..., clean[t], clean[t+1], ..., clean[t+K]]
    # Index h is clean[t], index h+k+1 is clean[t+k+1].
    clean_full = torch.cat([clean_history, targets], dim=1)  # (B, h+1+K, N, 3)

    # Sliding window of positions used as model INPUT (mix of noisy and own predictions)
    pos_window = [history[:, i] for i in range(h + 1)]

    step_losses = []
    for k in range(K):
        x_node, e_attr = _build_features_for_unroll(
            pos_window, edge_index, nodes_body, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N,
        )
        data = Data(x=x_node, edge_index=edge_index, edge_attr=e_attr)
        pred_accel_norm = model(data)

        # TRUE accel at this physical timestep — same definition as single-step training.
        # At unroll step k, model is predicting accel "for time t+k+1".
        # true_accel = clean[t+k+1] - 2*clean[t+k] + clean[t+k-1]
        true_accel = (clean_full[:, h + k + 1]
                      - 2.0 * clean_full[:, h + k]
                      +       clean_full[:, h + k - 1])
        true_accel_norm = (true_accel.reshape(B * N, 3) - accel_mean) / accel_std

        # Noise correction only at k=0 (inputs are noisy positions).
        # For k >= 1, inputs are the model's own predictions and there's
        # no random-walk noise to correct.
        if k == 0:
            true_accel = true_accel - noise_at_input

        step_losses.append(((pred_accel_norm - true_accel_norm) ** 2).mean())

        # Integrate using INPUT window (which has noise/predictions), not clean — this is what
        # creates the compounding-error signal that multi-step loss actually trains on.
        pred_accel = (pred_accel_norm * accel_std + accel_mean).reshape(B, N, 3)
        pred_pos = pred_accel + 2.0 * pos_window[-1] - pos_window[-2]
        pos_window = pos_window[1:] + [pred_pos]

    return torch.stack(step_losses).mean()

def _unroll_chain_loss_position(model, chain_batch, K, Wall, h,
                                x_mean, x_std, e_mean, e_std, accel_mean, accel_std):
    history    = chain_batch["history"]
    targets    = chain_batch["targets"]    # (B, K, N, 3) clean
    wind       = chain_batch["wind"]
    nodes_body = chain_batch["nodes_body"]
    edge_index = chain_batch["edge_index"]
    B, N = chain_batch["B"], chain_batch["N"]

    pos_window = [history[:, i] for i in range(h + 1)]

    step_losses = []
    for k in range(K):
        x_node, e_attr = _build_features_for_unroll(
            pos_window, edge_index, nodes_body, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N,
        )
        data = Data(x=x_node, edge_index=edge_index, edge_attr=e_attr)
        pred_accel_norm = model(data)

        # Unnormalize accel and integrate to position
        pred_accel = (pred_accel_norm * accel_std + accel_mean).reshape(B, N, 3)
        pred_pos = pred_accel + 2.0 * pos_window[-1] - pos_window[-2]

        # Position-space MSE against the clean target
        position_loss = ((pred_pos - targets[:, k]) ** 2).mean()
        step_losses.append(position_loss * 1e5)  # rough rescale to accel-loss range

        pos_window = pos_window[1:] + [pred_pos]

    return torch.stack(step_losses).mean()

def rotate_chain_batch(batch):
    R = random_z_rotation().to(batch["history"].device)
    batch["history"]       = batch["history"]       @ R.T
    batch["clean_history"] = batch["clean_history"] @ R.T   # NEW
    batch["targets"]       = batch["targets"]       @ R.T
    batch["nodes_body"]    = batch["nodes_body"]    @ R.T
    batch["wind"]          = batch["wind"]          @ R.T
    batch["noise_at_input"] = batch["noise_at_input"] @ R.T
    return batch


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
              latent_dim = 128,):
    
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
        clean_samples.extend(_build_timestep_samples(traj, Wall, h=h, noise_scale=noise_scale))

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
        val_samples.extend(_build_timestep_samples(traj, Wall, h=h, noise_scale=0.0))
    _apply_input_normalization(val_samples, x_mean, x_std, e_mean, e_std)
    _apply_accel_normalization(val_samples, acc_mean, acc_std)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    #Get input dimensions from one clean sample.
    sample_for_dims = _build_timestep_samples(dataset_train[0], Wall, h=h, noise_scale=0.0)[0]
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
    best_val_epoch = None

    #If resuming and history exists in checkpoint, continue appending.
    if resume_checkpoint_path is not None and isinstance(checkpoint, dict):
        train_loss_epochs = list(checkpoint.get("train_loss_epochs", train_loss_epochs))
        train_loss_values = list(checkpoint.get("train_loss_values", train_loss_values))
        val_loss_epochs = list(checkpoint.get("val_loss_epochs", val_loss_epochs))
        val_loss_values = list(checkpoint.get("val_loss_values", val_loss_values))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_val_epoch = checkpoint.get("best_val_epoch", best_val_epoch)

    print("Starting training...")
    print(f"  Train samples: {len(dataset_train)} | Val samples: {len(dataset_val)}")
    print(f"  Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")

    #Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_loss_tensor = torch.zeros((), device=device)
        t0 = time.time()
        if multistep > 1:
            # Build fresh chain samples this epoch (fresh noise on history)
            chain_samples = []
            for traj in dataset_train:
                chain_samples.extend(
                    _build_chain_samples(traj, Wall, h=h, K=multistep, noise_scale=noise_scale)
                )
            loader = torch.utils.data.DataLoader(
                chain_samples, batch_size=batch_size, shuffle=True,
                collate_fn=_collate_chain_batch,
                num_workers=4, pin_memory=True,
            )
            num_batches = len(loader)
            batch_iter = enumerate(loader)
        else:
            # Single-step path: use fast_batch to skip PyG Data overhead.
            # Builds flat (total_M, N, F) tensors once and gathers slices per batch.
            epoch_data = build_epoch_tensors(
                dataset_train, Wall, h=h, noise_scale=noise_scale,
                x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
                acc_mean=acc_mean, acc_std=acc_std,
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
                # Move dict to device, rotate, unroll
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch = rotate_chain_batch(batch)
                torch.cuda.synchronize(); b1 = time.time()
                loss = _unroll_chain_loss_accel(
                    model, batch, K=multistep, Wall=Wall, h=h,
                    x_mean=x_mean_gpu, x_std=x_std_gpu, e_mean=e_mean_gpu, e_std=e_std_gpu,
                    accel_mean=acc_mean_gpu, accel_std=acc_std_gpu,
                )
                torch.cuda.synchronize(); b2 = time.time()
            else:
                batch = rotate_batch(batch)
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
        if epoch % validation_check_interval == 0:
            # --- Validation ---
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
            val_loss_epochs.append(epoch_num)
            val_loss_values.append(float(avg_val_loss))
 
            if avg_val_loss < best_val_loss:
                best_val_loss = float(avg_val_loss)
                best_val_epoch = epoch_num
                best_model_path = os.path.splitext(save_model_path)[0] + "_best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} at epoch {best_val_epoch}")
 
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.9f} | "
                f"Val Loss: {avg_val_loss:.9f}")
            
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

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from generate_node_states import get_gns_features
from torch_geometric.nn import MessagePassing


#The Hz used in the collected data
HZ = 144
DT = 1.0 / HZ

# Graph Network Simulator (GNS) Layer
class GNSLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # LayerNorm (important for stability)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x:           [num_nodes, node_dim]
        edge_index:  [2, num_edges]
                     edge_index[0] = senders
                     edge_index[1] = receivers
        edge_attr:   [num_edges, edge_dim]
        """

        senders = edge_index[0]
        receivers = edge_index[1]

        # --------------------------------------------------
        # 1️⃣ EDGE UPDATE
        # --------------------------------------------------
        sender_features = x[senders]
        receiver_features = x[receivers]

        edge_input = torch.cat(
            [sender_features, receiver_features, edge_attr], dim=-1
        )

        edge_messages = self.edge_mlp(edge_input)
        edge_messages = self.edge_norm(edge_messages)

        # --------------------------------------------------
        # 2️⃣ AGGREGATION (SUM over incoming edges)
        # --------------------------------------------------
        num_nodes = x.size(0)
        hidden_dim = edge_messages.size(1)

        node_agg = torch.zeros(
            num_nodes, hidden_dim, device=x.device
        )

        node_agg.index_add_(0, receivers, edge_messages)

        # --------------------------------------------------
        # 3️⃣ NODE UPDATE
        # --------------------------------------------------
        node_input = torch.cat([x, node_agg], dim=-1)

        node_update = self.node_mlp(node_input)

        # Residual connection (VERY important)
        x = x + node_update

        x = self.node_norm(x)

        return x


#This is the full encoder-processor-decoder model
class GNSModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim,
                 latent_dim=128, num_layers=1):
        super().__init__()

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            self.processor.append(
                GNSLayer(
                    node_dim=latent_dim,
                    edge_dim=latent_dim,
                    hidden_dim=latent_dim
                )
            )

        # Decoder (NO LayerNorm per paper)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        # Fully dynamic message passing
        for layer in self.processor:
            x = layer(x, data.edge_index, edge_attr)

        return self.decoder(x)


#This function builds a dataset from multiple trajectories
def build_dataset(Wall, traj_range,
                  nodes_per_edge=5,
                  nearest_neighbors=4,
                  h=2):

    dataset = []

    #Runs through each trajectory in the specified range
    for throw_number in traj_range:
        print(f"Processing trajectory {throw_number}")

        #Gets the graph features for the current trajectory
        node_feat, edge_feat, edge_index, all_positions = get_gns_features(
            Wall,
            throw_number,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            h=h,
            training=True
        )

        #Dimensions of the data
        T = node_feat.shape[0]
        assert all_positions.shape[0] == T, "node_feat and all_positions must be time-aligned"

        # For each time t, predict a_t from features at t:
        # a_t = x_{t+1} - 2 x_t + x_{t-1}
        for t in range(1, T - 1):
            y_t = all_positions[t + 1] - 2.0 * all_positions[t] + all_positions[t - 1]

            data = Data(
                x=node_feat[t],
                edge_index=edge_index,
                edge_attr=edge_feat[t],
                y=y_t,
                pos_prev=all_positions[t - 1],
                pos_curr=all_positions[t],
                pos_next=all_positions[t + 1],
            )
            dataset.append(data)

    return dataset

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

def rotate_batch(batch):
    device = batch.x.device
    R = random_z_rotation().to(device)

    # ---- Rotate node features ----
    # x contains [v_t, v_tm1, dist]
    # So rotate only first 6 dims (2 velocity vectors)

    node_dim = batch.x.shape[1]
    if node_dim >= 6:
        v = batch.x[:, :6]  # first two 3D vectors
        dist = batch.x[:, 6:]

        v = v.view(-1, 2, 3)
        v = torch.matmul(v, R.T)
        v = v.view(-1, 6)

        batch.x = torch.cat([v, dist], dim=1)

    # ---- Rotate edge features ----
    # edge_attr = [d, d_norm, dU, dU_norm]

    edge_dim = batch.edge_attr.shape[1]

    if edge_dim >= 8:
        d = batch.edge_attr[:, 0:3]
        d_norm = batch.edge_attr[:, 3:4]
        dU = batch.edge_attr[:, 4:7]
        dU_norm = batch.edge_attr[:, 7:8]

        d = d @ R.T
        dU = dU @ R.T

        batch.edge_attr = torch.cat(
            [d, d_norm, dU, dU_norm],
            dim=1
        )

    # ---- Rotate acceleration target ----
    batch.y = batch.y @ R.T

    return batch


#This function trains the GNS model
def train_gnn(Wall,
              num_trajectories_train,
              num_trajectories_test,
              save_train_dataset_path,
              save_test_dataset_path,
              save_model_path,
              rebuild_datasets = False,
              epochs=50,
              batch_size=8,
              lr=1e-4,
              nodes_per_edge=5,
              message_passing_layers=2):
    
    #Sets device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rebuild_datasets:
        #BUILDS TRAINING DATASET
        print("Building training dataset...")
        traj_range_train = range(num_trajectories_train)
        dataset_train = build_dataset(Wall, traj_range_train,nodes_per_edge=nodes_per_edge)
        torch.save(dataset_train, save_train_dataset_path)
        print(f"Dataset saved to {save_train_dataset_path}")

        #BUILDS TEST DATASET
        print("Building test dataset...")
        #Test data set is just what's left over
        traj_range_test = range(num_trajectories_train+1, num_trajectories_train + num_trajectories_test+1)
        dataset_test = build_dataset(Wall, traj_range_test, nodes_per_edge=nodes_per_edge)
        torch.save(dataset_test, save_test_dataset_path)
        print(f"Test dataset saved to {save_test_dataset_path}")

    #Loads the training dataset
    else:
        print("Loading training dataset...")
        dataset_train = torch.load(save_train_dataset_path, weights_only=False)
        print(f"Training dataset loaded from {save_train_dataset_path}")
        print("Loading test dataset...")
        dataset_test = torch.load(save_test_dataset_path, weights_only=False)
        print(f"Test dataset loaded from {save_test_dataset_path}")


    acc_mean, acc_std = _compute_accel_stats(dataset_train)

    _apply_accel_normalization(dataset_train, acc_mean, acc_std)
    _apply_accel_normalization(dataset_test, acc_mean, acc_std)

    # Save stats
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_accel_norm.pt"
    torch.save({"acc_mean": acc_mean, "acc_std": acc_std}, norm_stats_path)

    print(f"Saved accel normalization stats to {norm_stats_path}")


    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    #Gets input dimensions from the first data point
    node_dim = dataset_train[0].x.shape[1]
    edge_dim = dataset_train[0].edge_attr.shape[1]

    #Initializes the GNS model, optimizer, and loss function
    model = GNSModel(node_dim, edge_dim, latent_dim=128, num_layers=message_passing_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("Starting training...")

    #Training loop
    disp = True
    for epoch in range(epochs):
        
        #Resets total loss for the epoch
        total_loss = 0.0

        #Iterates through batches of data
        for batch in loader:
            batch = batch.to(device)
            batch = rotate_batch(batch)
            


            #Zeroes the gradients
            optimizer.zero_grad()

            #Forward pass
            pred_accel = model(batch)

            loss = loss_fn(pred_accel, batch.y)

            #Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            #Accumulates loss
            total_loss += loss.item()

        #Averages loss over the epoch
        avg_loss = total_loss / len(loader)

        #Prints epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.9f}")

    #Saves the trained model
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    return model

def _compute_accel_stats(dataset):
    y_all = torch.cat([d.y for d in dataset], dim=0)

    acc_mean = y_all.mean(dim=0)
    acc_std = y_all.std(dim=0).clamp_min(1e-8)

    return acc_mean, acc_std


def _apply_accel_normalization(dataset, acc_mean, acc_std):
    for d in dataset:
        d.y = (d.y - acc_mean) / acc_std

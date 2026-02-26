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
class GNSLayer(MessagePassing):
    def __init__(self, latent_dim):

        #Sets the aggregation method to 'add', which means that messages from neighboring nodes will be summed together
        super().__init__(aggr='add')

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    # Forward step
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    #Passes the target and source node features along with edge attributes into the edge MLP to generate a message
    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(m)

    #Updates node features by concatenating the aggregated messages with the original node features and passing them through the node MLP
    def update(self, aggr_out, x):
        h = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(h)


#This is the full encoder-processor-decoder model
class GNSModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim,
                 latent_dim=128, num_layers=1):
        super().__init__()

        # Encoder for Nodes (Two-layer MLPs with ReLU activations)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Encoder for Edges (Two-layer MLPs with ReLU activations)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        #Processor: This will run through multiple layers of the graph network using the GNSLayer defined above
        self.processor = nn.ModuleList()
        for i in range(num_layers):
            layer = GNSLayer(latent_dim)
            self.processor.append(layer)

        # Decoder for Nodes (Two-layer MLPs with ReLU activations)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )

    # Forward pass through the entire model
    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        #adds the output of each GNS layer to the input (residual connection)
        for layer in self.processor:
            x = x + layer(x, data.edge_index, edge_attr)

        #Decodes the final node features to produce the output (predicted accelerations)
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
            h=h
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


    # Min-max normalize acceleration targets using TRAIN stats only
    acc_min, acc_max, acc_span = _compute_accel_minmax(dataset_train)
    _apply_accel_minmax(dataset_train, acc_min, acc_span)
    _apply_accel_minmax(dataset_test, acc_min, acc_span)

    # Save normalization stats next to model
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_accel_minmax.pt"
    torch.save({"acc_min": acc_min, "acc_max": acc_max}, norm_stats_path)
    print(f"Saved accel min/max stats to {norm_stats_path}")

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

            

            #Moves batch to the appropriate device
            batch = batch.to(device)

            #Zeroes the gradients
            optimizer.zero_grad()

            #Forward pass
            pred_accel = model(batch)

            #Computes loss
            # floor_z = 0.0
            # lambda_floor = 250000.0  # weight for the floor penalty
            # lambda_velocity = 1000.0  # weight for the downward velocity penalty

            # x_curr = batch.pos_curr
            # x_prev = batch.pos_prev
            # pred_positions = 2 * x_curr - x_prev + pred_accel

            # # Compute how far each node is below the floor
            # floor_penalty = torch.relu(floor_z - pred_positions[:, 2])
            # floor_penalty_loss = (abs(floor_penalty)).mean()

            # v_pred = (pred_positions - x_curr)
            # # Contact mask (only near or below floor)
            # contact_mask = (pred_positions[:, 2] <= floor_z)

            # # velocityz
            # velocity_z = v_pred[:, 2]

            # # Apply mask
            # vel_violation = velocity_z * contact_mask.float()
            # floor_velocity_loss = (vel_violation ** 2).mean()

            # if epoch == 0 and disp:  # Print penalties for the first batch of the first epoch
            #     print("Initial acceleration loss:", loss_fn(pred_accel, batch.y).item())
            #     print("Initial floor penalty loss:", lambda_floor*floor_penalty_loss.item())
            #     print("Initial floor velocity loss:", lambda_velocity*floor_velocity_loss.item())
            #     disp = False

            # print(pred_accel.shape, batch.y.shape)
            # print(x_curr.shape, x_prev.shape, pred_positions.shape)
            # print(floor_penalty.min(), floor_penalty.max())

            # loss = loss_fn(pred_accel, batch.y) + lambda_floor * floor_penalty_loss  + lambda_velocity * floor_velocity_loss
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

def _compute_accel_minmax(dataset):
    y_all = torch.cat([d.y for d in dataset], dim=0)  # (total_nodes_over_all_samples, 3)
    acc_min = y_all.min(dim=0).values
    acc_max = y_all.max(dim=0).values
    acc_span = (acc_max - acc_min).clamp_min(1e-8)
    return acc_min, acc_max, acc_span

def _apply_accel_minmax(dataset, acc_min, acc_span):
    for d in dataset:
        d.y = (d.y - acc_min) / acc_span

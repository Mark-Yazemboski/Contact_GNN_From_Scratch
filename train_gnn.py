import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from generate_node_states import get_gns_features



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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
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
        """
        x:           [num_nodes, node_dim]
        edge_index:  [2, num_edges]
                    edge_index[0] = senders
                    edge_index[1] = receivers
        edge_attr:   [num_edges, edge_dim]
        """

        #Extracts the sender and receiver node indices from the edge_index tensor.
        senders = edge_index[0]
        receivers = edge_index[1]

        # --------------------------------------------------
        # 1️⃣ EDGE UPDATE
        # --------------------------------------------------

        #For each edge, the sender and receiver node features are gathered based on the sender and receiver indices.
        sender_features = x[senders]
        receiver_features = x[receivers]

        #The edge features are updated by concatenating the sender node features, receiver node features, and edge attributes for
        #each edge, and passing this through the edge MLP. 
        edge_input = torch.cat(
            [sender_features, receiver_features, edge_attr], dim=-1
        )
        edge_messages = self.edge_mlp(edge_input)

        #The output is then normalized using layer normalization.
        edge_messages = self.edge_norm(edge_messages)

        # --------------------------------------------------
        # 2️⃣ AGGREGATION (SUM over incoming edges)
        # --------------------------------------------------

        #Gets the number of nodes from the input node features, and initializes a tensor to hold the aggregated messages
        # for each node.
        num_nodes = x.size(0)

        hidden_dim = edge_messages.size(1)

        node_agg = torch.zeros(
            num_nodes, hidden_dim, device=x.device
        )

        #The edge messages are aggregated for each receiver node by summing the messages from all incoming edges.
        node_agg.index_add_(0, receivers, edge_messages)

        # --------------------------------------------------
        # 3️⃣ NODE UPDATE
        # --------------------------------------------------

        #The node features are updated by concatenating the original node features with the aggregated messages
        node_input = torch.cat([x, node_agg], dim=-1)

        #The concatenated features are passed through the node MLP to get the node updates
        node_update = self.node_mlp(node_input)

        #Node features are updated by adding the node updates to the original node features, 
        #and then normalized using layer normalization.
        x = x + node_update
        x = self.node_norm(x)

        return x


#This is the full encoder-processor-decoder model
class GNSModel(nn.Module):

    #Initializes the GNS model with the dimensions of the node features, edge features, a latent dimension for the MLPs,
    # and the number of message passing layers.
    def __init__(self, node_in_dim, edge_in_dim,
                 latent_dim=128, num_layers=1):
        super().__init__()

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

        #This is the processor, which consists of multiple GNS layers stacked together. 
        #Each layer takes the node features and edge features, and performs message passing to update the node features. 
        #The number of layers is determined by the num_layers parameter.
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            self.processor.append(
                GNSLayer(
                    node_dim=latent_dim,
                    edge_dim=latent_dim,
                    hidden_dim=latent_dim
                )
            )

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

        #Next the encoded node features and edge features are passed through the processor, 
        #which consists of multiple GNS layers.
        for layer in self.processor:
            x = layer(x, data.edge_index, edge_attr)

        #Finally, the processed node features are passed through the decoder MLP to get the predicted accelerations for each node.
        decoded_state = self.decoder(x)

        return decoded_state


#This function builds a dataset from a range of trajectories. It processes each trajectory to extract the graph 
#features and constructs a list of Data objects,
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

            #Saves the node features, edge features, edge indices, and target accelerations for time t into a Data object, 
            #which is then added to the dataset list.
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

    # ---- Rotate node features ----
    # x contains [v_t, v_tm1, dist]
    # So rotate only first 6 dims (2 velocity vectors)


    #Extracts the velocity and distance features from the node features. 
    #The velocity features are the first 6 dimensions (v_t and v_{t-1}),
    #and the distance features are the remaining dimensions.
    v = batch.x[:, :6] 
    dist = batch.x[:, 6:]

    #The velocity features are reshaped to separate the current and previous velocity vectors,
    #then rotated using the random rotation matrix R, and finally reshaped back to the original format.
    v = v.view(-1, 2, 3)
    v = torch.matmul(v, R.T)
    v = v.view(-1, 6)

    #The rotated velocity features are concatenated with the distance features to form the new node features, 
    #which are then assigned back to batch.x.
    batch.x = torch.cat([v, dist], dim=1)

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

    #If the user wants to rebuild the datasets, it processes the trajectories to build the training and test datasets, 
    #and saves them.
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

    else:
        #If the user does not want to rebuild the datasets, it loads the training and test datasets from the specified paths.
        print("Loading training dataset...")
        dataset_train = torch.load(save_train_dataset_path, weights_only=False)
        print(f"Training dataset loaded from {save_train_dataset_path}")
        print("Loading test dataset...")
        dataset_test = torch.load(save_test_dataset_path, weights_only=False)
        print(f"Test dataset loaded from {save_test_dataset_path}")

    #Computes the mean and standard deviation for the node features, edge features, and target accelerations 
    #across the training dataset.
    x_mean, x_std = _compute_node_stats(dataset_train)
    e_mean, e_std = _compute_edge_stats(dataset_train)
    acc_mean, acc_std = _compute_accel_stats(dataset_train)

    #Applies normalization to the node features, edge features, and target accelerations in both the training and 
    #test datasets using the computed statistics.
    _apply_input_normalization(dataset_train, x_mean, x_std, e_mean, e_std)
    _apply_input_normalization(dataset_test, x_mean, x_std, e_mean, e_std)

    _apply_accel_normalization(dataset_train, acc_mean, acc_std)
    _apply_accel_normalization(dataset_test, acc_mean, acc_std)

    #saves the normalization statistics to a file, which can be used later for normalizing new data during inference.
    norm_stats_path = os.path.splitext(save_model_path)[0] + "_norms.pt"
    torch.save({"x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std, "acc_mean": acc_mean, "acc_std": acc_std}, norm_stats_path)

    print(f"Saved normalization stats to {norm_stats_path}")

    #Creates a DataLoader for the training dataset, which will handle batching and shuffling of the data during training.
    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    #Gets input dimensions from the first data point
    node_dim = dataset_train[0].x.shape[1]
    edge_dim = dataset_train[0].edge_attr.shape[1]

    #Initializes the GNS model, which consists of an encoder for the node features, an encoder for the edge features, 
    #multiple GNS layers for message passing,
    model = GNSModel(node_dim, edge_dim, latent_dim=128, num_layers=message_passing_layers).to(device)

    #Sets the optimizer to Adam, which will be used to update the model parameters during training based on the computed gradients.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Sets the loss function to mean squared error loss, which will be used to compute the loss between the
    # predicted accelerations and the target accelerations during training.
    loss_fn = nn.MSELoss()

    print("Starting training...")

    #Training loop
    for epoch in range(epochs):
        
        #Resets total loss for the epoch
        total_loss = 0.0

        #Iterates through batches of data
        for batch in loader:

            #Moves the batch to the GPU
            batch = batch.to(device)

            #Applies a random rotation to the node features, edge features, and target accelerations in the batch 
            #for data augmentation.
            batch = rotate_batch(batch)
            
            #Zeroes the gradients
            optimizer.zero_grad()

            #Forward pass
            pred_accel = model(batch)

            #Computes the loss between the predicted accelerations and the target accelerations in the batch 
            #using mean squared error loss.
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

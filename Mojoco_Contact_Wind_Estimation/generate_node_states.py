import os
import torch
import numpy as np

#The half width of the block used in the making of the dataset
BLOCK_HALF_WIDTH = 0.0524

#takes in the raw data and converts it back to SI units using the conversion described in the paper
def unscale_position_velocity(scaled_tensor):
    unscaled_tensor = scaled_tensor.clone()
    unscaled_tensor[..., :3] *= BLOCK_HALF_WIDTH     # position
    unscaled_tensor[..., 7:10] *= BLOCK_HALF_WIDTH   # velocity
    return unscaled_tensor


#Creates a cube mesh surface centered at the origin with specified side length and number of nodes per edge
def mesh_cube_surface(side_length, nodes_per_edge):


    L = side_length / 2.0
    lin = np.linspace(-L, L, nodes_per_edge)

    nodes = []

    # ±X faces
    for x in [-L, L]:
        for y in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Y faces
    for y in [-L, L]:
        for x in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Z faces
    for z in [-L, L]:
        for x in lin:
            for y in lin:
                nodes.append([x, y, z])

    #removes all of the duplicates points, and returns the unique set of nodes
    return np.unique(np.array(nodes), axis=0)

#Computes the normal vector for each node using the PCA method
def compute_normals_pca(nodes_body, k=8):

    N = nodes_body.shape[0]
    normals = torch.zeros_like(nodes_body)

    for i in range(N):
        # Distances to all nodes
        diff = nodes_body - nodes_body[i]
        dist = torch.norm(diff, dim=1)

        # k nearest neighbors (excluding self)
        knn_idx = torch.argsort(dist)[1:k+1]
        neighbors = nodes_body[knn_idx]

        # Center neighbors
        centered = neighbors - neighbors.mean(dim=0)

        # Covariance
        cov = centered.T @ centered

        # Smallest eigenvector = normal
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = eigvecs[:, 0]

        # Ensure outward direction
        if torch.dot(normal, nodes_body[i]) < 0:
            normal = -normal

        normals[i] = normal / torch.norm(normal)

    return normals

#Finds the k-nearest neighbors of each node to create an adjacency list
def knn_adjacency(nodes, k=4):
    N = nodes.shape[0]
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)
    edge_list = []
    for i in range(N):
        knn_idx = np.argsort(dist[i])[1:k+1]  # exclude self
        for j in knn_idx:
            edge_list.append([i, j])
    edge_index = np.array(edge_list).T  # shape (2, num_edges)
    return edge_index

#Converts a quaternion into a rotation matrix
def quat_to_rotmat(q):
    w, x, y, z = q

    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)]),
        torch.stack([2*(x*y + z*w),         1 - 2*(x*x + z*z), 2*(y*z - x*w)]),
        torch.stack([2*(x*z - y*w),         2*(y*z + x*w),     1 - 2*(x*x + y*y)])
    ])

    return R

#This function adds random walk noise to the positions of the nodes over time, which can be used to ensure the model 
#is robust to small perturbations in the trajectories during training. The noise is added cumulatively over time, 
#creating a random walk effect.
def add_random_walk_noise(positions, noise_scale=3e-4):
    
    #Creates a copy of the input positions to avoid modifying the original tensor, and initializes a 
    #noise tensor to accumulate the random walk noise over time.
    noisy = positions.clone()
    T, N, _ = positions.shape
    noise = torch.zeros(N, 3, device=positions.device)

    #For each timestep, random noise is generated and added to the cumulative noise. 
    #This cumulative noise is then added to the positions for that timestep, creating a random walk effect over time.
    for t in range(T):
        step_noise = torch.randn(N, 3, device=positions.device) * noise_scale
        noise = noise + step_noise
        noisy[t] = noisy[t] + noise

    return noisy

#This function returns the full clean positions of the nodes for each timestep in the trajectory, as well as the edge indices 
#and the body-centered node positions.
def get_clean_positions(Wall, throw_number, nodes_per_edge=2, nearest_neighbors=4, data_folder="data/tosses_processed", weights_only=True, unscale_data=True):

    data_path = os.path.join(data_folder, f"{throw_number}.pt")
    data = torch.load(data_path, weights_only=weights_only)
    states = data[0].float()
    if unscale_data:
        unscaled_states = unscale_position_velocity(states)
    else:
        unscaled_states = states
    T = unscaled_states.shape[0]

    nodes_body = torch.tensor(
        mesh_cube_surface(BLOCK_HALF_WIDTH * 2, nodes_per_edge), dtype=torch.float32
    )
    edge_index = torch.tensor(
        knn_adjacency(nodes_body.numpy(), k=nearest_neighbors), dtype=torch.long
    )

    all_positions = []
    for t in range(T):
        state = unscaled_states[t]
        R = quat_to_rotmat(state[3:7])
        all_positions.append((R @ nodes_body.T).T + state[:3])

    wind_vector = data[1]
    mass = data[2]

    return torch.stack(all_positions), edge_index, nodes_body, wind_vector

#This function will take in a given trajectory from the dataset, and will compute the node and edge states for each timestep
#of the trajectory. The node features include finite difference velocity features and distance to the wall, while the edge features
#include the relative position between connected nodes, the undeformed edge displacement, and their norms. The function returns
#the node features, edge features, edge indices, and the true positions of the nodes for each timestep, which can be used for 
#training the GNN model.
def get_gns_features(Wall, throw_number, nodes_per_edge=2, nearest_neighbors=4, h=2, training=False, data_folder="data/tosses_processed", weights_only=True, unscale_data=True):

    #This is the data path for the raw trajectories from the paper. The function loads the data for the specified throw number, 
    #unscales the position and velocity data back to SI units,
    data_path = os.path.join(data_folder, f"{throw_number}.pt")
    data = torch.load(data_path, weights_only=weights_only)
    states = data[0].float()
    wind_vector = data[1]
    if unscale_data:
        unscaled_states = unscale_position_velocity(states)
    else:
        unscaled_states = states

    #Gets the number of timesteps in the trajectory
    T = unscaled_states.shape[0]

    #Builds a mesh of nodes on the surface of the cube
    nodes_body = torch.tensor(
        mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )

    #Given the nodes of the meshed cube, the function computes the k-nearest neighbor adjacency to 
    #create the edge indices for the GNN.
    edge_index = knn_adjacency(nodes_body.numpy(), k=nearest_neighbors)
    edge_index = torch.tensor(edge_index, dtype=torch.long)


    #This computes the undeformed edge displacements dU and their norms, which are used as part of the edge features for the GNN.
    sender = edge_index[0]
    receiver = edge_index[1]
    dU = nodes_body[sender] - nodes_body[receiver]      
    dU_norm = torch.norm(dU, dim=1, keepdim=True)        

    #Store world positions for all timesteps
    all_positions = []

    #For each timestep, the function extracts the position and orientation of the cube, computes the world positions of the nodes
    #by applying the rotation and translation to the body-centered nodes, and stores these positions in a list. 
    for t in range(T):

        #Extracts pos and orientation from the COM data for the current time step
        state = unscaled_states[t]
        pos = state[:3]
        quat = state[3:7]

        #Makes a rotation matrix from the quaternion
        R = quat_to_rotmat(quat)

        #Applies the rotation and translation to the body-centered nodes to get the world positions of the nodes for
        #the current time step, and appends these positions to the list of all positions.
        nodes_world = (R @ nodes_body.T).T + pos
        all_positions.append(nodes_world)

    #Stacks the list of all positions into a single tensor with shape (T, N, 3), where T is the number of timesteps, 
    #N is the number of nodes, and 3 corresponds to the x,y,z coordinates.
    all_positions = torch.stack(all_positions)


    node_features = []
    edge_features = []

    #This will compute the node features and edge features for each timestep starting from h to T, where h is the number of 
    #history steps used for the finite difference velocity features.
    for t in range(h, T):

        v_fd = []

        #Finds the current velocity and h previous velocities using finite differences over the history of positions, 
        #and concatenates them to create the velocity features for the nodes at the current timestep.
        for k in range(h):
            v = (all_positions[t-k] - all_positions[t-k-1])
            v_fd.append(v)

        v_fd = torch.cat(v_fd, dim=1)  # (N, 3h)

        #Computes the distance from each node to the wall using the get_distance_to_point function of the Wall object, and clamps
        #the value between 0 and 0.5
        dist = Wall.get_distance_to_point(all_positions[t].numpy())
        dist = torch.tensor(dist, dtype=torch.float32).unsqueeze(1)
        dist = torch.clamp(dist, 0.0, 0.5)

        #Adds the distance to the wall as an additional feature to the velocity features for each node, 
        #creating the final node features for the current timestep.
        wind_expanded = wind_vector.unsqueeze(0).expand(len(all_positions[t]), -1) 
        node_feat = torch.cat([v_fd,wind_expanded, dist], dim=1)  
        node_features.append(node_feat)

        #For each edge, the function computes the relative position d between the sender and receiver nodes, 
        #as well as the norm of this relative position.
        pos_t = all_positions[t]
        d = pos_t[sender] - pos_t[receiver]        # (E,3)
        d_norm = torch.norm(d, dim=1, keepdim=True)

        #The edge features for the current timestep are created by concatenating the relative position d, 
        #its norm d_norm, the undeformed edge displacement dU, and its norm dU_norm for each edge. 
        #These edge features are stored in a list for each timestep.
        edge_feat = torch.cat([d, d_norm, dU, dU_norm], dim=1)
        edge_features.append(edge_feat)

    #Finally, the function stacks the list of node features and edge features into tensors with shapes 
    #(T-h, N, 3h+1) and (T-h, E, 8) respectively,
    node_features = torch.stack(node_features)   # (T-h, N, 3h+1)
    edge_features = torch.stack(edge_features)   # (T-h, E, 8)

    #The function returns the node features, edge features, edge indices, and the true positions 
    #of the nodes for each timestep starting from h to T.
    return node_features, edge_features, edge_index, all_positions[h:]
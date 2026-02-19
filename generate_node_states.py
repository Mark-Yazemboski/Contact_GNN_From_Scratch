import torch
import numpy as np

#The half width of the block used in the making of the dataset
BLOCK_HALF_WIDTH = 0.0524
HZ = 144
DT = 1.0 / HZ

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


def get_gns_features(Wall, throw_number, nodes_per_edge=5, nearest_neighbors=4, h=2):
    """
    Returns:
        node_features: (T-h, N, 3h+1)
        edge_features: (T-h, E, 8)
        edge_index: (2, E)
    """

    data_path = f"data/tosses_processed/{throw_number}.pt"
    data = torch.load(data_path)
    unscaled_states = unscale_position_velocity(data[0].float())
    T = unscaled_states.shape[0]

    # Build body mesh (undeformed mesh)
    nodes_body = torch.tensor(
        mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )

    edge_index = knn_adjacency(nodes_body.numpy(), k=nearest_neighbors)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    N = nodes_body.shape[0]
    E = edge_index.shape[1]

    # Precompute undeformed edge displacements dU_ij
    sender = edge_index[0]
    receiver = edge_index[1]
    dU = nodes_body[sender] - nodes_body[receiver]       # (E,3)
    dU_norm = torch.norm(dU, dim=1, keepdim=True)        # (E,1)

    # Store world positions for all timesteps
    all_positions = []

    for t in range(T):
        state = unscaled_states[t]
        pos = state[:3]
        quat = state[3:7]

        R = quat_to_rotmat(quat)
        nodes_world = (R @ nodes_body.T).T + pos
        all_positions.append(nodes_world)

    all_positions = torch.stack(all_positions)  # (T,N,3)

    node_features = []
    edge_features = []

    for t in range(h, T):

        # -------- Node features --------
        v_fd = []

        for k in range(h):
            v = (all_positions[t-k] - all_positions[t-k-1]) / DT
            v_fd.append(v)

        v_fd = torch.cat(v_fd, dim=1)  # (N, 3h)

        # boundary distance
        dist = Wall.get_distance_to_point(all_positions[t].numpy())
        dist = torch.tensor(dist, dtype=torch.float32).unsqueeze(1)
        dist = torch.clamp(dist, 0.0, 0.5)

        node_feat = torch.cat([v_fd, dist], dim=1)  # (N, 3h+1)
        node_features.append(node_feat)

        # -------- Edge features --------
        pos_t = all_positions[t]
        d = pos_t[sender] - pos_t[receiver]        # (E,3)
        d_norm = torch.norm(d, dim=1, keepdim=True)

        edge_feat = torch.cat([d, d_norm, dU, dU_norm], dim=1)
        edge_features.append(edge_feat)

    node_features = torch.stack(node_features)   # (T-h, N, 3h+1)
    edge_features = torch.stack(edge_features)   # (T-h, E, 8)

    return node_features, edge_features, edge_index, all_positions[h:]
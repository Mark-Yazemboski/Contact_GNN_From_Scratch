import torch
import wall
import train_gnn
import generate_node_states
from train_gnn import GNSModel
import display_results 
BLOCK_HALF_WIDTH = 0.0524

#Sets device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Makes the floor object
Floor = wall.wall(center_position=(0,0,0), size=(2,2), normal=(0,0,1))


#Sets number of trajectories for training and testing
Num_total_trajectories = 569
Num_train_trajectories = 256
Num_test_trajectories = Num_total_trajectories - Num_train_trajectories

nodes_per_edge = 2
message_passing_layers = 3

show_meshed_cube = False
show_augmentation = False
show_rollout = True

Train = False

#Trains the GNN model
if Train:
    train_gnn.train_gnn(
        Floor, 
        num_trajectories_train=Num_train_trajectories, 
        num_trajectories_test=Num_test_trajectories,
        save_train_dataset_path="data/pytorch_datasets/gns_train_dataset.pt", 
        save_test_dataset_path="data/pytorch_datasets/gns_test_dataset.pt",
        save_model_path="models/gns_model.pt", 
        rebuild_datasets=False,
        epochs=400, 
        batch_size=64, 
        lr=1e-4,
        nodes_per_edge=nodes_per_edge,
        message_passing_layers=message_passing_layers,
    )



# Get dims from one test trajectory
node_feat, edge_feat, edge_index, true_positions = generate_node_states.get_gns_features(
    Floor,
    throw_number=0,
    nodes_per_edge=nodes_per_edge
)
node_dim = node_feat.shape[2]
edge_dim = edge_feat.shape[2]

#Load trained model
model = GNSModel(node_dim, edge_dim, latent_dim=128, num_layers=message_passing_layers)
model.load_state_dict(torch.load("models/gns_model.pt", map_location=device))
model.to(device)
model.eval()

# Load normalization stats
norm_stats = torch.load("models/gns_model_norms.pt", map_location=device)
x_mean = norm_stats["x_mean"]
x_std = norm_stats["x_std"]
e_mean = norm_stats["e_mean"]
e_std = norm_stats["e_std"]
accel_std = norm_stats["acc_std"]
accel_mean = norm_stats["acc_mean"]

nodes_body = torch.tensor(
        generate_node_states.mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )


if show_meshed_cube:
    
    display_results.display_meshed_cube(nodes_body, edge_index=edge_index)


if show_augmentation:
    display_results.animate_rotated_with_velocities_and_edges(
        Floor,
        throw_number=0,
        save_path="Gifs/Showing_Rotated_Cube.gif",
        nodes_per_edge=nodes_per_edge,
    )


if show_rollout:
    pred_positions, true_positions, edge_info = display_results.rollout_trajectory_feedback_shape_match(
        model,
        Floor,
        throw_number=568,
        nodes_per_edge=nodes_per_edge,
        rest_positions=nodes_body,
        accel_std=accel_std,
        accel_mean=accel_mean,
        x_mean=x_mean,
        x_std=x_std,
        e_mean=e_mean,
        e_std=e_std,
        do_shape_match=True,
        shape_alpha= 1.0,
        return_edge_info=True
        
    )  
    display_results.animate_cube(
        pred_positions,
        true_positions,
        edge_info=edge_info,
        save_path="Gifs/Big_Test.gif"
    )



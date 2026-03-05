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
        rebuild_datasets=True,
        epochs=100, 
        batch_size=64, 
        lr=1e-4,
        nodes_per_edge=nodes_per_edge,
        message_passing_layers=message_passing_layers,
    )



# Get dims from one test trajectory
node_feat, edge_feat, edge_index, true_positions = generate_node_states.get_gns_features(Floor, throw_number=0)
node_dim = node_feat.shape[2]
edge_dim = edge_feat.shape[2]

#Load trained model
model = GNSModel(node_dim, edge_dim, latent_dim=128, num_layers=message_passing_layers)
model.load_state_dict(torch.load("models/gns_model.pt", map_location=device))
model.to(device)
model.eval()

# Load accel normalization stats
norm_stats = torch.load("models/gns_model_accel_norm.pt", map_location=device)
accel_std = norm_stats["acc_std"]
accel_mean = norm_stats["acc_mean"]


#Runs a rollout on a test trajectory
nodes_body = torch.tensor(
        generate_node_states.mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )

display_results.display_meshed_cube(nodes_body)





# pred_positions, true_positions = display_results.rollout_trajectory(
#     model,
#     Floor,
#     throw_number=0,
#     nodes_per_edge=nodes_per_edge,
#     rest_positions=nodes_body,
#     accel_min=accel_min,
#     accel_max=accel_max
# )  

pred_positions, true_positions = display_results.rollout_trajectory_feedback_shape_match(
    model,
    Floor,
    throw_number=568,
    nodes_per_edge=nodes_per_edge,
    rest_positions=nodes_body,
    accel_std=accel_std,
    accel_mean=accel_mean,
    shape_alpha=1.0
    
)  

#Animates the results
display_results.animate_cube(pred_positions, true_positions, save_path="Gifs/pen_and_Vel_loss_func_rollout.gif")

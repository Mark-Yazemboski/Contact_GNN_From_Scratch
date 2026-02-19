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
Num_train_trajectories = 550
Num_test_trajectories = Num_total_trajectories - Num_train_trajectories

nodes_per_edge = 5

#Trains the GNN model
# train_gnn.train_gnn(
#     Floor, 
#     num_trajectories_train=Num_train_trajectories, 
#     num_trajectories_test=Num_test_trajectories,
#     save_train_dataset_path="data/pytorch_datasets/gns_train_dataset.pt", 
#     save_test_dataset_path="data/pytorch_datasets/gns_test_dataset.pt",
#     save_model_path="models/gns_model.pt", 
#     epochs=30, 
#     batch_size=100, 
#     lr=5e-4,
#     nodes_per_edge=nodes_per_edge
# )



# Get dims from one test trajectory
node_feat, edge_feat, edge_index, true_positions = generate_node_states.get_gns_features(Floor, throw_number=0)
node_dim = node_feat.shape[2]
edge_dim = edge_feat.shape[2]

#Load trained model
model = GNSModel(node_dim, edge_dim)
model.load_state_dict(torch.load("models/gns_model.pt", map_location=device))
model.to(device)
model.eval()


#Runs a rollout on a test trajectory
nodes_body = torch.tensor(
        generate_node_states.mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )
pred_positions, true_positions = display_results.rollout_trajectory(model, Floor, throw_number=0, rest_positions=nodes_body)  

#Animates the results
display_results.animate_cube(pred_positions, true_positions, save_path="Gifs/pen_loss_func_rollout.gif") 

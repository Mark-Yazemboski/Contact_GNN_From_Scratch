import torch
import wall
import train_gnn
import generate_node_states
from train_gnn import GNSModel
import display_results 

#This is the real blocks width from the paper. This is used to unnormalize the data that they provide in the trajectories,
#And also create the node positions relitive to the COM data that they provide.
BLOCK_HALF_WIDTH = 0.0524

#Sets device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Makes the floor object
Floor = wall.wall(center_position=(0,0,0), size=(2,2), normal=(0,0,1))


#Sets number of trajectories for training and testing
Num_total_trajectories = 569
Num_train_trajectories = 256
Num_test_trajectories = Num_total_trajectories - Num_train_trajectories

#Sets the number of nodes per edge. at 2 nodes per edge there is 8 nodes, each one at a corner of the cube.
nodes_per_edge = 2

#Sets the number of message passing layers in the GNN model. The paper doesn't say how many they used, 
#but 3 is a common choice for GNNs and should be sufficient for this task.
message_passing_layers = 10

#Sets which visuals you want to turn on.
#Meshed cube shows the initial node positions and edges. 
#Augmentation shows the effect of random rotations on the trajectories. 
#Rollout shows the model's predictions when rolled out over a trajectory, with shape matching to the true positions at each step.
show_meshed_cube = True
show_augmentation = False
show_rollout = True

#This turns on and off model training, so you can train the model once, and then turn it off and just 
#run the visualizations without having to retrain the model every time you run the code.
Train = True

#Trains the GNN model
if Train:
    #Trains the Gnn using parameters like the floor object, number of trajectories, 
    # where to save the datasets and model, whether to rebuild the datasets,
    # training epochs, batch size, learning rate, and GNN architecture parameters
    # like nodes per edge and message passing layers.
    train_gnn.train_gnn(
        Floor, 
        num_trajectories_train=Num_train_trajectories, 
        num_trajectories_test=Num_test_trajectories,
        save_train_dataset_path="data/pytorch_datasets/gns_train_dataset.pt", 
        save_test_dataset_path="data/pytorch_datasets/gns_test_dataset.pt",
        save_model_path="models/gns_model.pt", 
        rebuild_datasets=True,
        epochs=400, 
        batch_size=64, 
        lr=1e-4,
        nodes_per_edge=nodes_per_edge,
        message_passing_layers=message_passing_layers,
    )



#Generates the node features, edge features, edge indices, and true positions for the first trajectory in the dataset.
#This is used to get the dimensions of the node and edge features, which are needed to initialize the GNN model.
node_feat, edge_feat, edge_index, true_positions = generate_node_states.get_gns_features(
    Floor,
    throw_number=0,
    nodes_per_edge=nodes_per_edge
)
node_dim = node_feat.shape[2]
edge_dim = edge_feat.shape[2]

#This generates the initial node positions for the meshed cube, which are used for the visualizations.
nodes_body = torch.tensor(
        generate_node_states.mesh_cube_surface(BLOCK_HALF_WIDTH*2, nodes_per_edge),
        dtype=torch.float32
    )

#Loads the trained GNN model from the saved file, and sets it to evaluation mode. 
#This model will be used for the rollouts and visualizations later in the code.
model = GNSModel(node_dim, edge_dim, latent_dim=128, num_layers=message_passing_layers)
model.load_state_dict(torch.load("models/gns_model.pt", map_location=device))
model.to(device)
model.eval()

#This loads the normalization statistics that were used to normalize the data during training.
norm_stats = torch.load("models/gns_model_norms.pt", map_location=device)
x_mean = norm_stats["x_mean"]
x_std = norm_stats["x_std"]
e_mean = norm_stats["e_mean"]
e_std = norm_stats["e_std"]
accel_std = norm_stats["acc_std"]
accel_mean = norm_stats["acc_mean"]



#This will show the meshed cube with the initial node positions and edges,
#which is useful for visualizing how the nodes are arranged on the cube and how the edges connect them.
if show_meshed_cube:
    display_results.display_meshed_cube(nodes_body, edge_index=edge_index)


#This will show the effect of random rotations on the trajectories, 
#which is a common data augmentation technique used in training GNNs for physical systems.
#This is usefull in detemrining if the augmentation is working properly, 
#and also gives a visual intuition for how the trajectories change with different rotations.
if show_augmentation:
    display_results.animate_augmented_data(
        Floor,
        throw_number=0,
        save_path="Gifs/Showing_Rotated_Cube.gif",
        nodes_per_edge=nodes_per_edge,
    )


#This will show the model's predictions when rolled out over a trajectory,
#with shape matching to the true positions at each step.
if show_rollout:
    pred_positions, true_positions, edge_info = display_results.rollout_trajectory_feedback_shape_match(
        model,
        Floor,
        throw_number=0,
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
        save_path="Gifs/Big_2500_final_one.gif"
    )



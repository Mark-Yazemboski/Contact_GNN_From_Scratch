from random import random

import torch
import wall
import train_gnn
import generate_node_states
from train_gnn import GNSModel
import display_results 
import evaluate_metrics
import random
import os

#This is the real blocks width from the paper. This is used to unnormalize the data that they provide in the trajectories,
#And also create the node positions relitive to the COM data that they provide.
BLOCK_HALF_WIDTH = 0.0524

#Sets device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Makes the floor object
Floor = wall.wall(center_position=(0,0,0), size=(2,2), normal=(0,0,1))


Num_total_trajectories = 570
training_percentage = 0.5
validation_percentage = 0.3
testing_percentage = 0.2

Num_train_trajectories = int(training_percentage * Num_total_trajectories)  # 285
Num_validation_trajectories = int(validation_percentage * Num_total_trajectories)  # 171
Num_test_trajectories = int(testing_percentage * Num_total_trajectories)  # 114

# Override for experiments with smaller training sets
Used_Num_train_trajectories = 256

train_range = range(0, Used_Num_train_trajectories)
val_range   = range(Num_train_trajectories, Num_train_trajectories + Num_validation_trajectories)
test_range  = range(Num_train_trajectories + Num_validation_trajectories, Num_total_trajectories)

print(f"Using {Used_Num_train_trajectories} out of {Num_train_trajectories} training trajectories.")
print(f"Training range: {(train_range)}")
print(f"Validation range: {(val_range)}")
print(f"Test range: {(test_range)}")



#Sets the number of nodes per edge. at 2 nodes per edge there is 8 nodes, each one at a corner of the cube.
nodes_per_edge = 2

K_nearest_neighbors = 3

#ADD COMMENT -------------------------------------------------------------------------------------
message_passing_layers = 5
repeat_blocks = 1

batch_size=64
steps = 1000000
traj_timesteps = 100
epochs = int(steps / (Used_Num_train_trajectories * traj_timesteps / batch_size))

print("Training for {} epochs".format(epochs))

epoch_checkpoint_interval = 500
validation_check_interval = 20

#Sets which visuals you want to turn on.
#Meshed cube shows the initial node positions and edges. 
#Augmentation shows the effect of random rotations on the trajectories. 
#Rollout shows the model's predictions when rolled out over a trajectory, with shape matching to the true positions at each step.
display_loss_curves = True
display_stats = True
show_meshed_cube = True
show_augmentation = False
show_rollout = True

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#---------------------------------------------------------------------------------------------------------
extra_name = "" #CHANGE THIS---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#This turns on and off model training, so you can train the model once, and then turn it off and just 
#run the visualizations without having to retrain the model every time you run the code.
Train = False

#Model save/load settings.
save_model_path = f"models/{Used_Num_train_trajectories}_train{extra_name}/{Used_Num_train_trajectories}_train_gns_model.pt"

#Set False when resuming from an existing checkpoint to keep dataset and normalization consistent.
rebuild_datasets = True

#Set this to a checkpoint file (for example: models/gns_model_epoch500.pt) to resume training.
resume_training_checkpoint_path = None

#Set this to a checkpoint file or model file to load for inference.
#If None, the script will load the final model saved after training.
inference_model_path = f"models/{Used_Num_train_trajectories}_train{extra_name}/{Used_Num_train_trajectories}_train_gns_model_final.pt"

#Trains the GNN model
if Train:
    if resume_training_checkpoint_path is not None and rebuild_datasets:
        print("Resume checkpoint detected. Forcing rebuild_datasets=False for consistent continuation.")
        rebuild_datasets = False

    #Trains the Gnn using parameters like the floor object, number of trajectories, 
    # where to save the datasets and model, whether to rebuild the datasets,
    # training epochs, batch size, learning rate, and GNN architecture parameters
    # like nodes per edge and message passing layers.
    train_gnn.train_gnn(
        Floor, 
        train_range=train_range,
        val_range=val_range,
        save_train_dataset_path=f"data/pytorch_datasets/gns_train_dataset.pt",
        save_val_dataset_path=f"data/pytorch_datasets/gns_val_dataset.pt",
        save_model_path=save_model_path,
        rebuild_datasets=rebuild_datasets,
        epochs=epochs, 
        batch_size=batch_size, 
        lr=1e-4,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        message_passing_layers=message_passing_layers,
        repeat_blocks=repeat_blocks,
        resume_checkpoint_path=resume_training_checkpoint_path,
        epoch_checkpoint_interval=epoch_checkpoint_interval,
        validation_check_interval = validation_check_interval
    )



#Generates the node features, edge features, edge indices, and true positions for the first trajectory in the dataset.
#This is used to get the dimensions of the node and edge features, which are needed to initialize the GNN model.
node_feat, edge_feat, edge_index, true_positions = generate_node_states.get_gns_features(
    Floor,
    throw_number=0,
    nodes_per_edge=nodes_per_edge,
    nearest_neighbors=K_nearest_neighbors,
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
model = GNSModel(node_dim, edge_dim, latent_dim=128, L=message_passing_layers, K=repeat_blocks)

if inference_model_path is None:
    load_model_path = os.path.splitext(save_model_path)[0] + "_best_model.pt"
else:
    load_model_path = inference_model_path

print(f"Loading model from {load_model_path} for evaluation")

loaded_obj = torch.load(load_model_path, map_location=device, weights_only=False)
if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
    model.load_state_dict(loaded_obj["model_state_dict"])
else:
    model.load_state_dict(loaded_obj)

model.to(device)
model.eval()

#This loads the normalization statistics that were used to normalize the data during training.

norm_stats_path = os.path.splitext(save_model_path)[0] + "_norms.pt"
norm_stats = torch.load(norm_stats_path, map_location=device)
x_mean = norm_stats["x_mean"]
x_std = norm_stats["x_std"]
e_mean = norm_stats["e_mean"]
e_std = norm_stats["e_std"]
accel_std = norm_stats["acc_std"]
accel_mean = norm_stats["acc_mean"]

#Load saved loss history for plotting.
loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"
train_loss_epochs = []
train_loss_values = []
val_loss_epochs = []
val_loss_values = []

if os.path.exists(loss_history_path):
    loss_history = torch.load(loss_history_path, map_location="cpu", weights_only=False)
    train_loss_epochs = list(loss_history.get("train_loss_epochs", []))
    train_loss_values = list(loss_history.get("train_loss_values", []))
    val_loss_epochs = list(loss_history.get("val_loss_epochs", []))
    val_loss_values = list(loss_history.get("val_loss_values", []))
    print(f"Loaded loss history from {loss_history_path}")
else:
    print(f"Loss history file not found: {loss_history_path}")

if display_loss_curves:
    os.makedirs("Plots", exist_ok=True)
    evaluate_metrics.plot_loss_curves(
        train_loss_epochs=train_loss_epochs,
        train_loss_values=train_loss_values,
        val_loss_epochs=val_loss_epochs,
        val_loss_values=val_loss_values,
        title="Training and Validation Loss",
        save_path="Plots/loss_curve.png",
        show_plot=True
    )


if display_stats:
    evaluate_metrics.evaluate_model(model, Floor, test_range, nodes_per_edge, K_nearest_neighbors, nodes_body, 
                       accel_std, accel_mean, x_mean, x_std, e_mean, e_std)



#This will show the meshed cube with the initial node positions and edges,
#which is useful for visualizing how the nodes are arranged on the cube and how the edges connect them.
if show_meshed_cube:
    display_results.display_meshed_cube(nodes_body, edge_index=edge_index)


#This will show the effect of random rotations on the trajectories, 
#which is a common data augmentation technique used in training GNNs for physical systems.
#This is usefull in detemrining if the augmentation is working properly, 
#and also gives a visual intuition for how the trajectories change with different rotations.
if show_augmentation:
    throw_number = random.choice(test_range)
    print("Showing augmentation for trajectory number: ", throw_number)
    display_results.animate_augmented_data(
        Floor,
        throw_number=throw_number,
        save_path="Gifs/Showing_Rotated_Cube.gif",
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
    )


#This will show the model's predictions when rolled out over a trajectory,
#with shape matching to the true positions at each step.
if show_rollout:

    throw_number = random.choice(test_range)
    print("Showing rollout for trajectory number: ", throw_number)
    pred_positions, true_positions, edge_info = display_results.rollout_trajectory_feedback_shape_match(
        model,
        Floor,
        throw_number=throw_number,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
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



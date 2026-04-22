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
#This file is the same as run.py, except it has perameters specific to handel the wind data in the mojoco trajectories


#This function will take in the number of trajectories we are training on, and the 
#number of optimizer steps we want to hit, and some other perameters, and calculate
#how many epochs we need to train for.
def compute_epochs(num_trajectories, target_steps, batch_size, accumulation_steps, traj_timesteps=100, history=2):
    usable_per_traj = traj_timesteps - history - 1
    total_samples = num_trajectories * usable_per_traj
    num_batches = (total_samples + batch_size - 1) // batch_size  # ceil division
    effective_accum = min(accumulation_steps, num_batches)
    steps_per_epoch = num_batches // effective_accum
    steps_per_epoch = max(steps_per_epoch, 1)
    epochs = (target_steps + steps_per_epoch - 1) // steps_per_epoch
    return epochs

#This is the real blocks width from the paper. This is used to unnormalize the data that they provide in the trajectories,
#And also create the node positions relitive to the COM data that they provide.
BLOCK_HALF_WIDTH = 0.0524

#Sets device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Makes the floor object
Floor = wall.wall(center_position=(0,0,0), size=(2,2), normal=(0,0,1))

#This is the total number of trajectories in the dataset. 
Num_total_trajectories = 1024

#This sets the percentage of trajectories to use for training, validation, and testing.
training_percentage = 0.5
validation_percentage = 0.3
testing_percentage = 0.2

#Calculates the number of trajectories to use for training, validation, and testing based 
#on the total number and the percentages.
Num_train_trajectories = int(training_percentage * Num_total_trajectories)  
Num_validation_trajectories = int(validation_percentage * Num_total_trajectories)  
Num_test_trajectories = int(testing_percentage * Num_total_trajectories)  

#---------------------------------------------------------------------------------------------------------
#This is the number of training trajectories to actually use.
# Override for experiments with smaller training sets
Used_Num_train_trajectories = 512
#---------------------------------------------------------------------------------------------------------

#Get the directory of this script for relative path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))

#This is the folder where the trajectory data is stored. 
trajectory_folder = os.path.join(script_dir, "data/mojoco_trajectories_no_wind_wth_sliding")  # Folder where the trajectory .pt files are stored

#Calculates the ranges of trajectory indices to use for training, validation, and testing.
train_range = range(0, Used_Num_train_trajectories)
val_range   = range(Num_train_trajectories, Num_train_trajectories + Num_validation_trajectories)
test_range  = range(Num_train_trajectories + Num_validation_trajectories, Num_total_trajectories-1)

print(f"Using {Used_Num_train_trajectories} out of {Num_train_trajectories} training trajectories.")
print(f"Training range: {(train_range)}")
print(f"Validation range: {(val_range)}")
print(f"Test range: {(test_range)}")


#-----------------------------------------------------------------------------------------------------

#Sets the number of nodes per edge. at 2 nodes per edge there is 8 nodes, each one at a corner of the cube.
nodes_per_edge = 2

#Sets the number of nearest neighbors to use for creating edges in the graph.
K_nearest_neighbors = 3

#Sets the number of message passing layers in the GNN, and the number of times to repeat the 
#blocks of message passing layers.
message_passing_layers = 5
repeat_blocks = 1

#This is the batch size for training.
batch_size=128

#This is the learning rate for training the GNN.
learning_rate = 4e-4

#This is the total number of optimizer steps to train for. 
steps = 100000

#This is the number of timesteps in each trajectory. 
traj_timesteps = 200

#This is an important parameter that sets how many past positions the model can see when making
#its predictions. Basically giving the model more past positions can give it information about
#the velocity and acceleration of the nodes, which can help it make better predictions.
pos_history = 2

#The paper says it had a batch size of 64 on 8 gpus so to simulate the same effective batch size on a single GPU,
# we use gradient accumulation over 8 steps.
accumulation_steps = 8

#Data loading options
#Set to False for older PyTorch or datasets with object serialization
weights_only_load = False
#Set to False if your dataset is already unscaled, or True to apply unscale_position_velocity 
unscale_trajectory_data = False

#This will compute the number of epochs to train for based on the number of trajectories we
# are using, the target number of optimizer steps, the batch size, and the accumulation steps.
epochs = compute_epochs(Used_Num_train_trajectories, steps, batch_size, accumulation_steps, traj_timesteps=traj_timesteps, history=pos_history)

print("Training for {} epochs".format(epochs))

#Sets how often the program will save a checkpoint of the model during training,
# and how often it will check the validation loss.
epoch_checkpoint_interval = 100
validation_check_interval = 10

#Sets which visuals you want to turn on.
#Meshed cube shows the initial node positions and edges. 
#Augmentation shows the effect of random rotations on the trajectories. 
#Rollout shows the model's predictions when rolled out over a trajectory, with shape matching to the true positions at each step.
display_loss_curves = True
display_stats = True
show_meshed_cube = False
show_augmentation = False
show_rollout = True

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#---------------------------------------------------------------------------------------------------------
#This adds an extra name to the saved model and dataset files, which is useful for keeping track of different experiments when you are training multiple models with different parameters.
extra_name = "" #CHANGE THIS---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#------------------------------------------------------------------------------------------------------
#This turns on and off model training, so you can train the model once, and then turn it off and just 
#run the visualizations without having to retrain the model every time you run the code.
Train = True
#------------------------------------------------------------------------------------------------------

#Model save/load settings.
# save_model_path = os.path.join(script_dir, f"models/mojoco_{Used_Num_train_trajectories}_train{extra_name}/{Used_Num_train_trajectories}_train_gns_model.pt")
save_model_path = os.path.join(script_dir, f"models/mojoco_no_wind_sliding/{Used_Num_train_trajectories}_train_gns_model.pt")


#Set False when resuming from an existing checkpoint to keep dataset and normalization consistent.
rebuild_datasets = True

#Set this to a checkpoint file (for example: models/gns_model_epoch500.pt) to resume training.
resume_training_checkpoint_path = None

#Set this to a checkpoint file or model file to load for inference.
#If None, the script will load the final model saved after training.
inference_model_path = None
# inference_model_path = os.path.join(script_dir, f"models/mojoco_{Used_Num_train_trajectories}_train{extra_name}/{Used_Num_train_trajectories}_train_gns_model_best_model.pt")

#Trains the GNN model
if Train:

    #Makes sure if we are resuming from a checkpoint, we don't accidentally rebuild the
    # datasets, which would lead to inconsistent training since the model would be trained
    # on different data than it was originally trained on before the checkpoint.
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
        save_train_dataset_path=os.path.join(script_dir, "data/pytorch_datasets/gns_train_dataset.pt"),
        save_val_dataset_path=os.path.join(script_dir, "data/pytorch_datasets/gns_val_dataset.pt"),
        save_model_path=save_model_path,
        rebuild_datasets=rebuild_datasets,
        epochs=epochs, 
        batch_size=batch_size, 
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        trajectory_folder=trajectory_folder,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        h = pos_history,
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
    data_folder=trajectory_folder,
    weights_only=weights_only_load,
    unscale_data=unscale_trajectory_data
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

#Checks to see if we specified a model to load for inference. If not, it 
#loads the best model saved during training.
if inference_model_path is None:
    load_model_path = os.path.splitext(save_model_path)[0] + "_best_model.pt"
else:
    load_model_path = inference_model_path

print(f"Loading model from {load_model_path} for evaluation")

#loads the model state dict from the specified path. The code checks if the loaded object
#is a dictionary containing a "model_state_dict" key, which is a common format for saving 
#checkpoints that include additional information like optimizer state and training epoch. 
loaded_obj = torch.load(load_model_path, map_location=device, weights_only=False)
if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
    model.load_state_dict(loaded_obj["model_state_dict"])
else:
    model.load_state_dict(loaded_obj)

#Moves the model to the GPU for faster computations.
model.to(device)

#Sets the model to evaluation mode, which is important for certain layers 
#like dropout and batch normalization that behave differently during training and evaluation.
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

#Sets the loss history path
loss_history_path = os.path.splitext(save_model_path)[0] + "_loss_history.pt"
train_loss_epochs = []
train_loss_values = []
val_loss_epochs = []
val_loss_values = []

#Loads the loss history from the specified path. This is used to plot the training and 
#validation loss curves later in the code. 
if os.path.exists(loss_history_path):
    loss_history = torch.load(loss_history_path, map_location="cpu", weights_only=False)
    train_loss_epochs = list(loss_history.get("train_loss_epochs", []))
    train_loss_values = list(loss_history.get("train_loss_values", []))
    val_loss_epochs = list(loss_history.get("val_loss_epochs", []))
    val_loss_values = list(loss_history.get("val_loss_values", []))
    print(f"Loaded loss history from {loss_history_path}")
else:
    print(f"Loss history file not found: {loss_history_path}")

#Shows the loss curve
if display_loss_curves:
    os.makedirs(os.path.join(script_dir, "Plots"), exist_ok=True)
    display_results.plot_loss_curves(
        train_loss_epochs=train_loss_epochs,
        train_loss_values=train_loss_values,
        val_loss_epochs=val_loss_epochs,
        val_loss_values=val_loss_values,
        title="Training and Validation Loss",
        save_path=os.path.join(script_dir, "Plots/loss_curve.png"),
        show_plot=True
    )

#This evaluates the model on the test set and prints out various metrics like mean positon error, 
#mean angle error, and mean penitration error
if display_stats:
    evaluate_metrics.evaluate_model(trajectory_folder, model, Floor, test_range, nodes_per_edge, K_nearest_neighbors, nodes_body, 
                       accel_std, accel_mean, x_mean, x_std, e_mean, e_std, weights_only_load, unscale_trajectory_data)



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
        save_path=os.path.join(script_dir, "Gifs/Showing_Rotated_Cube.gif"),
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
    )


#This will show the model's predictions when rolled out over a trajectory,
#with shape matching to the true positions at each step.
if show_rollout:

    throw_number = random.choice(test_range)
    print("Showing rollout for trajectory number: ", throw_number)
    pred_positions, true_positions, edge_info = display_results.rollout_trajectory_feedback_shape_match(
        trajectory_folder=trajectory_folder,
        model=model,
        Wall=Floor,
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
        return_edge_info=True,
        weights_only_load=weights_only_load,
        unscale_trajectory_data=unscale_trajectory_data
        
    )  
    display_results.animate_cube(
        pred_positions,
        true_positions,
        edge_info=edge_info,
        save_path=os.path.join(script_dir, "Gifs/Big_2500_final_one.gif")
    )
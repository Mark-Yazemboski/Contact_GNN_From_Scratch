
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(script_dir, "models/mojoco_pure_sliding_no_wind_fixed_noise")
file_location = os.path.join(model_folder_path, "1024_train_gns_model_norms.pt")

# Load the norms file
norms = torch.load(file_location, map_location="cpu")

# See what keys are available
print("Keys:", norms.keys())

# Print the variables of interest
print("\nacc_std:")
print(norms["acc_std"])

print("\nx_std:")
print(norms["x_std"])
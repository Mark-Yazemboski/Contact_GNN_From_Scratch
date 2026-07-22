
import torch
import os
torch.set_printoptions(precision=10)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(script_dir, "models/Testing")
file_location = os.path.join(model_folder_path, "256_train_gns_model_norms.pt")

# Load the norms file
norms = torch.load(file_location, map_location="cpu")

# See what keys are available
print("Keys:", norms.keys())

# Print the variables of interest
print("\nacc_mean:")
print(norms["acc_mean"])

print("\nacc_std:")
print(norms["acc_std"])

print("\ne_mean:")
print(norms["e_mean"])

print("\ne_std:")
print(norms["e_std"])

print("\nx_mean:")
print(norms["x_mean"])

print("\nx_std:")
print(norms["x_std"])
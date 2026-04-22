import os

import mujoco
import mujoco.viewer
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
#This file is used to visualize the trajectories we collected from the MuJoCo simulations with wind.
#It loads a saved trajectory file, extracts the trajectory data and initial conditions, and then uses
#the MuJoCo viewer to step through the trajectory frame by frame, rendering the cube's motion according
#to the saved positions and orientations.

#This function takes in the path to a saved trajectory file, loads the trajectory data, 
#and visualizes it using the MuJoCo physics engine. 
def playback_trajectory(script_dir, traj_path):
    model = mujoco.MjModel.from_xml_path(os.path.join(script_dir, "cube.xml"))
    data = mujoco.MjData(model)

    save_data = torch.load(traj_path, weights_only=False)
    traj = save_data[0].numpy()      
    wind = save_data[1].numpy()      
    mass = save_data[2]             
    params = save_data[3]            

    print(f"Wind: {wind}")
    print(f"Mass: {mass}")
    print(f"Initial Position: {params['pos']}")
    print(f"Initial Quaternion: {params['quat']}")
    print(f"Initial Velocity: {params['vel']}")
    print(f"Initial Angular Velocity: {params['angvel']}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        time.sleep(12)  # wait for viewer to initialize
        for t in range(traj.shape[0]):
            data.qpos[:3] = traj[t, 0:3]    # position
            data.qpos[3:7] = traj[t, 3:7]   # quaternion
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep*50)

        input("Press Enter to close: ")

#Trajectory index to visualize.
traj = 10

script_dir = os.path.dirname(os.path.abspath(__file__))

#This is the folder where the trajectory data is stored. 
path = os.path.join(script_dir, f"data/mojoco_trajectories_no_wind_wth_sliding/{traj}.pt")  # Folder where the trajectory .pt files are stored

# #Path of the trajectory file to visualize.
# path = f"data/mojoco_trajectories/{traj}.pt"

#Visualizes the trajectory using the MuJoCo physics engine.
playback_trajectory(script_dir, path)

data = torch.load(path, weights_only=False)
pos = data[0][:, 2].numpy()  # z position over time
accel = pos[2:] - 2*pos[1:-1] + pos[:-2]
posxy = data[0][:, :2].numpy()  # x and y position over time
accelxy = posxy[2:] - 2*posxy[1:-1] + posxy[:-2]
accelxy = np.linalg.norm(accelxy, axis=1)  # magnitude of x and y acceleration
print(accel[0])
print(max(accel[100:]))
print(min(accel[100:]))
plt.plot(accel)  # after the cube has landed
plt.title("Z acceleration during 'resting'")
plt.xlabel("Time Step")
plt.ylabel("Acceleration")

plt.plot(accelxy)  # after the cube has landed
plt.title("X and Y acceleration during 'resting'")
plt.xlabel("Time Step")
plt.ylabel("Acceleration")
plt.show()
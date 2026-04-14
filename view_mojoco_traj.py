import mujoco
import mujoco.viewer
import torch
import time
#This file is used to visualize the trajectories we collected from the MuJoCo simulations with wind.
#It loads a saved trajectory file, extracts the trajectory data and initial conditions, and then uses
#the MuJoCo viewer to step through the trajectory frame by frame, rendering the cube's motion according
#to the saved positions and orientations.

#This function takes in the path to a saved trajectory file, loads the trajectory data, 
#and visualizes it using the MuJoCo physics engine. 
def playback_trajectory(traj_path):
    model = mujoco.MjModel.from_xml_path("cube.xml")
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
        
        time.sleep(3)  # wait for viewer to initialize
        for t in range(traj.shape[0]):
            data.qpos[:3] = traj[t, 0:3]    # position
            data.qpos[3:7] = traj[t, 3:7]   # quaternion
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        input("Press Enter to close: ")

#Trajectory index to visualize.
traj = 3

#Path of the trajectory file to visualize.
path = f"data/mojoco_trajectories/{traj}.pt"

#Visualizes the trajectory using the MuJoCo physics engine.
playback_trajectory(path)
import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation
import time
#This file will generate all of the training data we will use to train the contact and fluid GNN. Using mojoco's built in
# physics engine, it will simulate the trajectory of a cube under the influence of different wind vectors, initial positions,
#orientations, velocities, and angular velocities. The trajectories are saved as .pt files, which contain the trajectory data 
#as well as the initial conditions and parameters for each simulation. We will use this data to train our GNN to predict the effect
# of wind on the trajectory of the cube, and to learn the underlying physics of the system.

#Generates a random quaternion
def random_quat():
    return Rotation.random().as_quat(scalar_first=True) 


#Takes in the initial conditions and parameters for a MuJoCo simulation, runs the simulation for a specified number of steps, 
#and returns the trajectory of the cube's position and orientation over time.
def collect_trajectory(wind_vector, initial_pos, initial_quat, initial_vel,
                       initial_angvel, mass, n_steps=1000, visualize=False):
    
    #Loads the model into MuJoCo, sets the initial conditions and parameters and resets the simulation data.
    model = mujoco.MjModel.from_xml_path("cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    model.opt.wind[:] = wind_vector
    model.body_mass[1] = mass
    data.qpos[:3] = initial_pos
    data.qpos[3:7] = initial_quat
    data.qvel[:3] = initial_vel
    data.qvel[3:6] = initial_angvel

    mujoco.mj_forward(model, data)

    states = []

    #If visualize is True, it will launch the MuJoCo viewer and step through the simulation, 
    #rendering the cube's motion
    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            time.sleep(3)  # wait for viewer to initialize
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
                states.append(np.concatenate([
                    data.qpos[:3].copy(),
                    data.qpos[3:7].copy(),
                ]))
                viewer.sync()
                time.sleep(.05)
            # Keep window open after sim finishes
            input("Press Enter to close viewer...")
    
    #If not visualizing, it will just run the simulation and record the trajectory data without rendering.
    else:
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            states.append(np.concatenate([
                data.qpos[:3].copy(),
                data.qpos[3:7].copy(),
            ]))

    trajectory = np.stack(states)
    return trajectory


#This function generates random initial conditions and parameters for the MuJoCo simulation, including a random wind vector,
#initial position, orientation, velocity, and angular velocity of the cube, and returns them in a dictionary.
def generate_random_params(mass, wind_range, horizontal_pos_range, vertical_pos_range, horizontal_speed_range, vertical_speed_range, angvel_range):
     
    wind_dir = np.random.randn(3)
    wind_dir[2] = 0
    wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
    wind_vec = wind_dir * np.random.uniform(wind_range[0], wind_range[1])

    pos = np.array([
        np.random.uniform(horizontal_pos_range[0], horizontal_pos_range[1]),
        np.random.uniform(horizontal_pos_range[0], horizontal_pos_range[1]),
        np.random.uniform(vertical_pos_range[0], vertical_pos_range[1]),
    ])
    quat = random_quat()
    vel = np.array([
        np.random.uniform(horizontal_speed_range[0], horizontal_speed_range[1]),
        np.random.uniform(horizontal_speed_range[0], horizontal_speed_range[1]),
        np.random.uniform(vertical_speed_range[0], vertical_speed_range[1]),
    ])
    angvel = np.random.uniform(angvel_range[0], angvel_range[1], size=3)

    return {
        'wind': wind_vec,
        'pos': pos,
        'quat': quat,
        'vel': vel,
        'angvel': angvel,
        'mass': mass,
    }


if __name__ == "__main__":
    #Sets the directory the trajectories will be saved to.
    save_dir = "data/mojoco_trajectories"
    os.makedirs(save_dir, exist_ok=True)

    #Sets the number of trajectories we will generate aswell as how long the simulation will run for.
    n_trajectories = 512
    n_steps = 200

    #Sets whether we want to see one of the trajectories visualized.
    visualize_first = False

    #Runs for the specified number of trajectories, generating random parameters for each one,
    # running the simulation to collect the trajectory data,
    for i in range(n_trajectories):

        #Generates the random initial conditions and parameters for the simulation
        params = generate_random_params(mass = 0.37, wind_range=(0, 5), horizontal_pos_range=(-0.2, 0.2), vertical_pos_range=(0.3, 0.8), horizontal_speed_range=(-1.25, 1.25), vertical_speed_range=(-0.3, 0.3), angvel_range=(-3, 3))
        print(f"Trajectory {i}: wind={params['wind'].round(2)}, "
              f"mass={params['mass']:.2f}, "
              f"pos={params['pos'].round(2)}, "
              f"vel={params['vel']}, "
                f"angvel={params['angvel']}"
              )

        #Collects the trajectories.
        traj = collect_trajectory(
            wind_vector=params['wind'],
            initial_pos=params['pos'],
            initial_quat=params['quat'],
            initial_vel=params['vel'],
            initial_angvel=params['angvel'],
            mass=params['mass'],
            n_steps=n_steps,
            visualize=True if i == 0 and visualize_first else False
        )

        #Saves everything in a list as a .pt file, which includes the trajectory data as well as the initial conditions
        # and parameters for each simulation.
        traj_tensor = torch.tensor(traj, dtype=torch.float32)
        save_data = [
            traj_tensor,                                              # [0] trajectory - what generate_node_states expects
            torch.tensor(params['wind'], dtype=torch.float32),        # [1] wind
            params['mass'],                                           # [2] mass
            params,                                                   # [3] params
        ]

        save_path = os.path.join(save_dir, f"{i}.pt")
        torch.save(save_data, save_path)
        print(f"Saved trajectory {i}")
    

    print(f"\nSaved {n_trajectories} trajectories to {save_dir}/")
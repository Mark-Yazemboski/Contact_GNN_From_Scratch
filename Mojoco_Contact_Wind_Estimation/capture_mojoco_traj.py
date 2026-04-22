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
def collect_trajectory(model, wind_vector, initial_pos, initial_quat, initial_vel,
                       initial_angvel, mass, n_steps=1000,substeps= 10, visualize=False):
    
    #Loads the model into MuJoCo, sets the initial conditions and parameters and resets the simulation data.
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
                # Take multiple small physics steps per recorded frame
                for _ in range(substeps):
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
            for _ in range(substeps):
                    mujoco.mj_step(model, data)
            states.append(np.concatenate([
                data.qpos[:3].copy(),
                data.qpos[3:7].copy(),
            ]))

    trajectory = np.stack(states)
    return trajectory


#This function generates random initial conditions and parameters for the MuJoCo simulation, including a random wind vector,
#initial position, orientation, velocity, and angular velocity of the cube, and returns them in a dictionary.
def generate_toss_params(mass, wind_range, horizontal_pos_range, vertical_pos_range, horizontal_speed_range, vertical_speed_range, angvel_range):
     
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
        'type': 'toss',
    }

def generate_sliding_params(mass, wind_range, sliding_speed_range, angvel_z_range, half_width=0.0524):

    #Wind uses the same logic as toss trajectories
    wind_dir = np.random.randn(3)
    wind_dir[2] = 0
    wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
    wind_vec = wind_dir * np.random.uniform(wind_range[0], wind_range[1])

    #Start sitting on the floor with random x/y position
    pos = np.array([
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(-0.2, 0.2),
        half_width,
    ])

    #Random rotation around z only so the cube sits flat on the floor
    theta = np.random.uniform(0, 2 * np.pi)
    quat = Rotation.from_euler('z', theta).as_quat(scalar_first=True)

    #Horizontal velocity only — friction has to stop this
    speed = np.random.uniform(sliding_speed_range[0], sliding_speed_range[1])
    angle = np.random.uniform(0, 2 * np.pi)
    vel = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    #Only spin around z axis so the cube doesn't tip over
    angvel = np.array([0.0, 0.0, np.random.uniform(angvel_z_range[0], angvel_z_range[1])])

    return {
        'wind': wind_vec,
        'pos': pos,
        'quat': quat,
        'vel': vel,
        'angvel': angvel,
        'mass': mass,
        'type': 'sliding',
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = os.path.join(script_dir, "data", "mojoco_trajectories_no_wind_wth_sliding")
    os.makedirs(save_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(os.path.join(script_dir, "cube.xml"))

    #----- Dataset parameters -----
    n_trajectories = 1024
    n_steps = 200
    substeps = 50
    visualize_first = False

    #What percentage of the total trajectories should be sliding-only (0.0 to 1.0)
    sliding_percentage = 0.2

    #----- Shared parameters -----
    wind_range = (0,0)
    mass = 0.37

    #----- Toss-specific parameters -----
    toss_horizontal_pos_range = (-0.2, 0.2)
    toss_vertical_pos_range = (0.3, 0.8)
    toss_horizontal_speed_range = (-1.25, 1.25)
    toss_vertical_speed_range = (-0.3, 0.3)
    toss_angvel_range = (-3, 3)

    #----- Sliding-specific parameters -----
    sliding_speed_range = (0.3, 2.0)
    sliding_angvel_z_range = (-3, 3)

    #----- Build the schedule of which indices are sliding vs toss -----
    n_sliding = int(n_trajectories * sliding_percentage)
    n_toss = n_trajectories - n_sliding

    #Spread sliding trajectories evenly throughout the dataset so that 
    #training batches always contain a mix of both types.
    sliding_indices = set()
    if n_sliding > 0:
        spacing = n_trajectories / n_sliding
        for k in range(n_sliding):
            sliding_indices.add(int(k * spacing))

    print(f"Total: {n_trajectories} | Toss: {n_toss} | Sliding: {n_sliding} ({sliding_percentage*100:.0f}%)")

    for i in range(n_trajectories):

        if i in sliding_indices:
            params = generate_sliding_params(
                mass=mass,
                wind_range=wind_range,
                sliding_speed_range=sliding_speed_range,
                angvel_z_range=sliding_angvel_z_range,
            )
        else:
            params = generate_toss_params(
                mass=mass,
                wind_range=wind_range,
                horizontal_pos_range=toss_horizontal_pos_range,
                vertical_pos_range=toss_vertical_pos_range,
                horizontal_speed_range=toss_horizontal_speed_range,
                vertical_speed_range=toss_vertical_speed_range,
                angvel_range=toss_angvel_range,
            )

        print(f"Trajectory {i} [{params['type']}]: wind={params['wind'].round(2)}, "
              f"mass={params['mass']:.2f}, "
              f"pos={params['pos'].round(2)}, "
              f"vel={params['vel'].round(3)}, "
              f"angvel={params['angvel'].round(3)}")

        traj = collect_trajectory(
            model=model,
            wind_vector=params['wind'],
            initial_pos=params['pos'],
            initial_quat=params['quat'],
            initial_vel=params['vel'],
            initial_angvel=params['angvel'],
            mass=params['mass'],
            n_steps=n_steps,
            substeps=substeps,
            visualize=True if i == 0 and visualize_first else False
        )

        traj_tensor = torch.tensor(traj, dtype=torch.float32)
        save_data = [
            traj_tensor,
            torch.tensor(params['wind'], dtype=torch.float32),
            params['mass'],
            params,
        ]

        save_path = os.path.join(save_dir, f"{i}.pt")
        torch.save(save_data, save_path)
        print(f"Saved trajectory {i}")
    
    print(f"\nSaved {n_trajectories} trajectories to {save_dir}/")
    print(f"  Toss: {n_toss} | Sliding: {n_sliding}")
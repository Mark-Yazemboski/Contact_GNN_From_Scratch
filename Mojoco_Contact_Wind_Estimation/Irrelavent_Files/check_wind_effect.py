import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation
import time
#This file is used to check the effect different wind magnitudes have on the overal trajectory of the cube vs a no win baseline.
#The goal was to find the range of magnitudes that would cause 10 - 20% deviation in the final position, 
#which we thought would be a reasonable range to test our GNN's ability to predict the effect of wind on the trajectory.
#From this analysis we found that range to be 0 - 5 magnitude of horizontal wind.



#Returns the final position offset in meters and as a percent of the baseline total path length.
def calc_trajectory_difference(traj1, traj2):
    final_offset = float(np.linalg.norm(traj1[-1, :3] - traj2[-1, :3]))
    total_path = float(np.sum(np.linalg.norm(np.diff(traj1[:, :3], axis=0), axis=1)))
    pct = 100.0 * final_offset / (total_path + 1e-8)
    return final_offset, pct


# Run multiple trials with random initial conditions and wind in the given range, 
# then return average final offset and percent difference.
def find_wind_magnitude_for_target_effect(min_magnitude,max_magnitude,n_trials=100,n_steps=500):

    percents = []
    offsets = []
    magnitudes = []

    #Runs through all the trials, generating random initial conditions and wind vectors, 
    #then simulating with and without wind and comparing the trajectories.
    for _ in range(n_trials):
        params = generate_random_params(mass=0.37, wind_range=(min_magnitude, max_magnitude), horizontal_pos_range=(-0.2, 0.2), vertical_pos_range=(0.3, 0.8), horizontal_speed_range=(-1.25, 1.25), vertical_speed_range=(-0.3, 0.3), angvel_range=(-3, 3))
        shared = dict(
            initial_pos=params['pos'],
            initial_quat=params['quat'],
            initial_vel=params['vel'],
            initial_angvel=params['angvel'],
            mass=params['mass'],
            n_steps=n_steps,
            visualize=False,
        )
        wind_vector = params['wind']

        base_traj = collect_trajectory(wind_vector = np.zeros(3), **shared)

        wind_traj = collect_trajectory(wind_vector=wind_vector, **shared)

        offset, pct = calc_trajectory_difference(base_traj, wind_traj)
        offsets.append(offset)
        percents.append(pct)
        magnitudes.append(np.linalg.norm(wind_vector))

    #Returns a dictionary with the average final offset in meters, average percent difference, 
    #and standard deviation of the percent difference across all trials.
    return {
        'n_trials': n_trials,
        'min_magnitude': min_magnitude,
        'max_magnitude': max_magnitude,
        'avg_wind_magnitude': float(np.mean(magnitudes)),
        'avg_final_offset_m': float(np.mean(offsets)),
        'avg_percent_diff': float(np.mean(percents)),
        'std_percent_diff': float(np.std(percents)),
    }


#Generates a random quaternion with the scalar component first, suitable for Mujoco's qpos format.
def random_quat():
    return Rotation.random().as_quat(scalar_first=True)

#Runs the Mujoco simulation with the specified wind vector and initial conditions, 
#collecting the trajectory of the cube's position and orientation over time.
def collect_trajectory(wind_vector, initial_pos, initial_quat, initial_vel, initial_angvel, mass, n_steps=1000, visualize=False):

    #Loads the cube model
    model = mujoco.MjModel.from_xml_path("cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    #sets the initial conditions and wind vector in the model
    model.opt.wind[:] = wind_vector
    model.body_mass[1] = mass
    data.qpos[:3] = initial_pos
    data.qpos[3:7] = initial_quat
    data.qvel[:3] = initial_vel
    data.qvel[3:6] = initial_angvel

    #Runs a forward pass to compute initial forces, then steps through the simulation, collecting the trajectory.
    mujoco.mj_forward(model, data)

    states = []

    #If visualize is True, it launches a Mujoco viewer to show the simulation in real time. 
    #Otherwise, it just runs the simulation and collects the trajectory data without rendering.
    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            time.sleep(3)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
                states.append(np.concatenate([
                    data.qpos[:3].copy(),
                    data.qpos[3:7].copy(),
                    data.qvel[:3].copy(),
                    data.qvel[3:6].copy(),
                ]))
                viewer.sync()
                time.sleep(.05)
            input("Press Enter to close viewer...")
    else:
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            states.append(np.concatenate([
                data.qpos[:3].copy(),
                data.qpos[3:7].copy(),
            ]))

    #returns a stack of states, which are a list of 7D vectors of position and orientation at each time step.
    return np.stack(states)

#Generates random parameters for the simulation given certain bounds.
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

#This function will run a single comparison of trajectories with and without wind, then visualize 
#the results side by side in a Mujoco viewer.
def visualize_comparison(wind_magnitude=0.3, n_steps=250):
    
    #Generates random initial conditions and a wind vector with the specified magnitude
    params = generate_random_params(mass=0.37, wind_range=(0, 5), horizontal_pos_range=(-0.2, 0.2), vertical_pos_range=(0.3, 0.8), horizontal_speed_range=(-1.25, 1.25), vertical_speed_range=(-0.3, 0.3), angvel_range=(-3, 3))
    
    #Both the no-wind and wind trajectories will use the same initial conditions for a fair comparison.
    shared = dict(
        initial_pos=params['pos'],
        initial_quat=params['quat'],
        initial_vel=params['vel'],
        initial_angvel=params['angvel'],
        mass=params['mass'],
        n_steps=n_steps,
        visualize=False,
    )

    #Runs the simulation without wind to get the baseline trajectory
    base_traj = collect_trajectory(wind_vector=np.zeros(3), **shared)

    #Generates a random horizontal wind direction and scales it to the specified magnitude, 
    wind_dir = np.random.randn(3)
    wind_dir[2] = 0.0
    norm = np.linalg.norm(wind_dir)
    wind_dir = wind_dir / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    wind_vector = wind_dir * wind_magnitude

    #Runs the simulation with the wind vector to get the wind-affected trajectory
    wind_traj = collect_trajectory(wind_vector=wind_vector, **shared)

    #Calculates the final position offset and percent difference between the two trajectories, then prints the results.
    offset, pct = calc_trajectory_difference(base_traj, wind_traj)
    print(f"Wind: {wind_vector.round(3)}, magnitude: {wind_magnitude:.2f}")
    print(f"Final offset: {offset:.4f} m, path %: {pct:.1f}%")

    #Build a replay XML with two free cubes (blue = no wind, red = wind)
    replay_xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <asset>
        <material name="blue" rgba="0.2 0.4 0.9 0.7"/>
        <material name="red"  rgba="0.9 0.2 0.2 0.7"/>
      </asset>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.01" rgba="0.3 0.3 0.3 1"/>
        <body name="no_wind" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" material="blue"/>
        </body>
        <body name="wind" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" material="red"/>
        </body>
      </worldbody>
    </mujoco>
    """

    #Load the replay model and data, then step through the trajectories frame by frame, 
    #setting the positions of the two cubes according to the no-wind and wind trajectories, and render in the viewer.
    model = mujoco.MjModel.from_xml_string(replay_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    n_frames = min(len(base_traj), len(wind_traj))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(3)
        for i in range(n_frames):
            # Set no-wind cube pose (first 7 qpos)
            data.qpos[0:3] = base_traj[i, :3]
            data.qpos[3:7] = base_traj[i, 3:7]
            # Set wind cube pose (next 7 qpos)
            data.qpos[7:10] = wind_traj[i, :3]
            data.qpos[10:14] = wind_traj[i, 3:7]

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)

        input("Press Enter to close viewer...")


#This will execute multiple trials for different wind magnitude ranges and 
#print out the average final offset and percent difference for each range.
for lo, hi in [(0, 2), (0, 5), (0, 10), (0, 20)]:
    r = find_wind_magnitude_for_target_effect(lo, hi, n_trials=250, n_steps=250)
    print(f"Wind [{lo}-{hi}]: offset={r['avg_final_offset_m']:.4f}m, {r['avg_percent_diff']:.1f}% ± {r['std_percent_diff']:.1f}%")
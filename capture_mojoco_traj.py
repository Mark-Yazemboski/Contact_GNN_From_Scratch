import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation
import time



def random_quat():
    return Rotation.random().as_quat(scalar_first=True)  # [w, x, y, z]

def collect_trajectory(wind_vector, initial_pos, initial_quat, initial_vel,
                       initial_angvel, mass, n_steps=1000, visualize=False):
    
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

    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            time.sleep(3)  # wait for viewer to initialize
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
            # Keep window open after sim finishes
            input("Press Enter to close viewer...")
    else:
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            states.append(np.concatenate([
                data.qpos[:3].copy(),
                data.qpos[3:7].copy(),
            ]))

    trajectory = np.stack(states)
    return trajectory


def generate_random_params():
    # Wind: random direction, magnitude 0-10 m/s
    wind_dir = np.random.randn(3)
    wind_dir[2] = 0  # keep wind horizontal
    wind_dir = wind_dir / (np.linalg.norm(wind_dir) + 1e-8)
    wind_mag = np.random.uniform(0, 10)
    wind = wind_dir * wind_mag

    # Initial position: above the floor, random x/y within small range
    pos = np.array([
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(0.3, 0.8),
    ])

    # Random initial rotation
    quat = random_quat()

    # Random initial velocity (gentle toss)
    vel = np.array([
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-0.3, 0.5),
    ])

    # Random initial angular velocity (rad/s)
    angvel = np.random.uniform(-3, 3, size=3)

    # Mass close to the real cube (0.37 kg) with some variation
    mass = np.random.uniform(0.2, 0.6)

    return {
        'wind': wind,
        'pos': pos,
        'quat': quat,
        'vel': vel,
        'angvel': angvel,
        'mass': mass,
    }


if __name__ == "__main__":
    save_dir = "data/mojoco_trajectories"
    os.makedirs(save_dir, exist_ok=True)

    n_trajectories = 512
    n_steps = 200
    visualize_first = False

    for i in range(n_trajectories):
        params = generate_random_params()
        print(f"Trajectory {i}: wind={params['wind'].round(2)}, "
              f"mass={params['mass']:.2f}, "
              f"pos={params['pos'].round(2)}, "
              f"vel={params['vel']}, "
                f"angvel={params['angvel']}"
              )

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
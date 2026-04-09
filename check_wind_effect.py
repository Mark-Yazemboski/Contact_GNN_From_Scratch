import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation
import time


def calc_trajectory_difference(traj1, traj2):
    """Return final position offset (meters) and as a percent of baseline total path length."""
    final_offset = float(np.linalg.norm(traj1[-1, :3] - traj2[-1, :3]))
    total_path = float(np.sum(np.linalg.norm(np.diff(traj1[:, :3], axis=0), axis=1)))
    pct = 100.0 * final_offset / (total_path + 1e-8)
    return final_offset, pct


def find_wind_magnitude_for_target_effect(min_magnitude,
                                          max_magnitude,
                                          n_trials=100,
                                          n_steps=500):
    """Run n_trials and return average percent trajectory difference for wind in [min, max]."""
    if min_magnitude < 0.0:
        raise ValueError("min_magnitude must be non-negative")
    if max_magnitude < min_magnitude:
        raise ValueError("max_magnitude must be >= min_magnitude")

    percents = []
    offsets = []
    magnitudes = []

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

        # wind_dir = np.random.randn(3)
        # wind_dir[2] = 0.0
        # norm = np.linalg.norm(wind_dir)
        # wind_dir = wind_dir / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])

        # wind_mag = np.random.uniform(min_magnitude, max_magnitude)
        wind_traj = collect_trajectory(wind_vector=wind_vector, **shared)

        offset, pct = calc_trajectory_difference(base_traj, wind_traj)
        offsets.append(offset)
        percents.append(pct)
        magnitudes.append(np.linalg.norm(wind_vector))

    return {
        'n_trials': n_trials,
        'min_magnitude': min_magnitude,
        'max_magnitude': max_magnitude,
        'avg_wind_magnitude': float(np.mean(magnitudes)),
        'avg_final_offset_m': float(np.mean(offsets)),
        'avg_percent_diff': float(np.mean(percents)),
        'std_percent_diff': float(np.std(percents)),
    }


def random_quat():
    return Rotation.random().as_quat(scalar_first=True)


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

    return np.stack(states)


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


def visualize_comparison(wind_magnitude=0.3, n_steps=250):
    """Simulate with and without wind, then replay both trajectories side by side."""
    params = generate_random_params(mass=0.37, wind_range=(0, 5), horizontal_pos_range=(-0.2, 0.2), vertical_pos_range=(0.3, 0.8), horizontal_speed_range=(-1.25, 1.25), vertical_speed_range=(-0.3, 0.3), angvel_range=(-3, 3))
    shared = dict(
        initial_pos=params['pos'],
        initial_quat=params['quat'],
        initial_vel=params['vel'],
        initial_angvel=params['angvel'],
        mass=params['mass'],
        n_steps=n_steps,
        visualize=False,
    )

    base_traj = collect_trajectory(wind_vector=np.zeros(3), **shared)

    wind_dir = np.random.randn(3)
    wind_dir[2] = 0.0
    norm = np.linalg.norm(wind_dir)
    wind_dir = wind_dir / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    wind_vector = wind_dir * wind_magnitude

    wind_traj = collect_trajectory(wind_vector=wind_vector, **shared)

    offset, pct = calc_trajectory_difference(base_traj, wind_traj)
    print(f"Wind: {wind_vector.round(3)}, magnitude: {wind_magnitude:.2f}")
    print(f"Final offset: {offset:.4f} m, path %: {pct:.1f}%")

    # Build a replay XML with two free cubes (blue = no wind, red = wind)
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


if __name__ == "__main__":
    # Quick comparison visualization
    # visualize_comparison(wind_magnitude=5.0, n_steps=250)

    # Uncomment to run the sweep instead:
    for lo, hi in [(0, 2), (0, 5), (0, 10), (0, 20)]:
        r = find_wind_magnitude_for_target_effect(lo, hi, n_trials=250, n_steps=250)
        print(f"Wind [{lo}-{hi}]: offset={r['avg_final_offset_m']:.4f}m, {r['avg_percent_diff']:.1f}% ± {r['std_percent_diff']:.1f}%")
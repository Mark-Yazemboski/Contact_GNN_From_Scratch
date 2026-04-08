import mujoco
import mujoco.viewer
import torch
import time

def playback_trajectory(traj_path):
    model = mujoco.MjModel.from_xml_path("cube.xml")
    data = mujoco.MjData(model)

    save_data = torch.load(traj_path, weights_only=False)
    traj = save_data['trajectory'].numpy()
    print(f"Wind: {save_data['wind'].numpy()}")
    print(f"Mass: {save_data['mass']}")
    print(f"Initial Position: {save_data['params']['pos']}")
    print(f"Initial Quaternion: {save_data['params']['quat']}")
    print(f"Initial Velocity: {save_data['params']['vel']}")
    print(f"Initial Angular Velocity: {save_data['params']['angvel']}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        time.sleep(3)  # wait for viewer to initialize
        for t in range(traj.shape[0]):
            data.qpos[:3] = traj[t, 0:3]    # position
            data.qpos[3:7] = traj[t, 3:7]   # quaternion
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        input("Press Enter to close: ")

traj = 3
path = f"data/wind_trajectories/{traj}.pt"
playback_trajectory(path)
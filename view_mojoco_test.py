import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("cube_wind.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
"""Render frames around the grasp of an episode, from a close-up camera on the target object."""

import sys

sys.path.insert(0, ".")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import SETTLE_FRAMES, prepare  # noqa: E402

cfg = GenerationConfig()
seed, out = int(sys.argv[1]), sys.argv[2]
ep = prepare(seed, cfg, {})
m, arm, plan = ep.model, ep.arm, ep.plan
d = mujoco.MjData(m)
sub = round(1 / (cfg.fps * m.opt.timestep))
d.qpos[arm.qadr] = arm.to_ctrl(ep.start)
d.ctrl[arm.actuators] = arm.to_ctrl(ep.start)
mujoco.mj_forward(m, d)
mujoco.mj_step(m, d, nstep=SETTLE_FRAMES * sub)
body = ep.info.object_bodies[ep.instance.roles["object"]]
r = mujoco.Renderer(m, 240, 320)
cam = mujoco.MjvCamera()
cam.lookat[:] = d.xpos[body]
cam.distance, cam.azimuth, cam.elevation = (
    0.25,
    np.rad2deg(np.arctan2(d.xpos[body][1], d.xpos[body][0])) + 90,
    -15,
)
g = plan.marks["grasped"]
frames = sorted({0, g - 40, g - 25, g - 12, g - 4, g, plan.marks["lifted"] - 10, plan.marks["lifted"]})
shots = []
for t, cmd in enumerate(plan.commands):
    if t in frames:
        r.update_scene(d, camera=cam)
        shots.append(r.render())
        r.update_scene(d, camera="front")
        shots.append(r.render())
    if t >= plan.marks["lifted"]:
        break
    d.ctrl[arm.actuators] = arm.to_ctrl(cmd)
    mujoco.mj_step(m, d, nstep=sub)
rows = [np.concatenate(shots[i : i + 4], 1) for i in range(0, len(shots) - 3, 4)]
Image.fromarray(np.concatenate(rows, 0)).save(out)
print(ep.instance.instruction, [(o.name, o.label) for o in ep.scene.objects], frames)

"""Inspect grasps that fail the lift check: object, grasp parameters and jaw state at lift time."""

import sys

sys.path.insert(0, ".")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import SETTLE_FRAMES, EpisodeError, prepare  # noqa: E402
from examples.so101_datagen.tasks import TASKS, find, touches_gripper  # noqa: E402

cfg = GenerationConfig()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for seed in range(n):
    try:
        ep = prepare(seed, cfg, {})
    except EpisodeError:
        continue
    m, arm, plan = ep.model, ep.arm, ep.plan
    d = mujoco.MjData(m)
    sub = round(1 / (cfg.fps * m.opt.timestep))
    d.qpos[arm.qadr] = arm.to_ctrl(ep.start)
    d.ctrl[arm.actuators] = arm.to_ctrl(ep.start)
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=SETTLE_FRAMES * sub)
    name = ep.instance.roles["object"]
    obj = find(ep.scene, name)
    body = ep.info.object_bodies[name]
    log = {}
    for t, cmd in enumerate(plan.commands):
        if t == plan.marks["grasped"]:
            log["at_grasp"] = (
                arm.from_qpos(d.qpos)[5].round(1),
                cmd[5].round(1),
                touches_gripper(name, m, d, ep.info),
            )
            log["tcp_obj"] = (d.site_xpos[arm.tcp] - d.xpos[body]).round(3)
        if t == plan.marks["lifted"]:
            ok = TASKS[ep.task].check("lifted", ep.instance, m, d, ep.info)
            log["lift_ok"] = ok
            log["obj_z"] = d.xpos[body][2].round(3)
            break
        d.ctrl[arm.actuators] = arm.to_ctrl(cmd)
        mujoco.mj_step(m, d, nstep=sub)
    if not log.get("lift_ok", True):
        print(
            seed,
            obj.shape,
            np.round(obj.half_size, 3),
            "mass",
            round(obj.mass, 3),
            "fr",
            round(obj.friction, 2),
            log,
        )

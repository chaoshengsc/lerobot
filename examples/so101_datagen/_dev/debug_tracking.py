"""Tracking error and target displacement before the grasp, for failing seeds."""

import sys

sys.path.insert(0, ".")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import SETTLE_FRAMES, prepare  # noqa: E402

cfg = GenerationConfig()
for seed in map(int, sys.argv[1].split(",")):
    ep = prepare(seed, cfg, {})
    m, arm, plan = ep.model, ep.arm, ep.plan
    d = mujoco.MjData(m)
    sub = round(1 / (cfg.fps * m.opt.timestep))
    d.qpos[arm.qadr] = arm.to_ctrl(ep.start)
    d.ctrl[arm.actuators] = arm.to_ctrl(ep.start)
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=SETTLE_FRAMES * sub)
    body = ep.info.object_bodies[ep.instance.roles["object"]]
    p0 = d.xpos[body].copy()
    worst = np.zeros(6)
    moved_at = None
    for t, cmd in enumerate(plan.commands):
        if t == plan.marks["grasped"]:
            break
        err = np.abs(arm.from_qpos(d.qpos) - cmd)
        worst = np.maximum(worst, err)
        if moved_at is None and np.linalg.norm(d.xpos[body] - p0) > 0.005:
            moved_at = t
            contacts = [
                (m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)
                for c in d.contact[: d.ncon]
            ]
        d.ctrl[arm.actuators] = arm.to_ctrl(cmd)
        mujoco.mj_step(m, d, nstep=sub)
    planned_tcp, _ = arm.tcp_pose(plan.commands[plan.marks["grasped"] - 1])
    print(
        seed,
        ep.start_mode,
        "worst tracking deg",
        worst.round(1),
        "moved_at",
        moved_at,
        "of",
        plan.marks["grasped"],
        "tcp actual-planned",
        (d.site_xpos[arm.tcp] - planned_tcp).round(3),
    )
    if moved_at is not None:
        print("   contacts when moved:", [c for c in contacts if "object" in c[0] + c[1]])

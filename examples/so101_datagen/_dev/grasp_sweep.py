"""Grasp success of a single box/cylinder vs width, height, fixed-jaw clearance and opening margin."""

import itertools
import sys

sys.path.insert(0, ".")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import SETTLE_FRAMES  # noqa: E402
from examples.so101_datagen.motion import Planner, PlanningError  # noqa: E402
from examples.so101_datagen.randomization import sample_scene  # noqa: E402
from examples.so101_datagen.scene import build_scene  # noqa: E402
from examples.so101_datagen.so101 import REST, Arm  # noqa: E402
from examples.so101_datagen.tasks import is_lifted  # noqa: E402

cfg = GenerationConfig()
offsets = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.002, 0.006]
margins = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.02, 0.04]
for offset, margin in itertools.product(offsets, margins):
    ok = total = 0
    fails = []
    for shape, width, height, yaw, r in itertools.product(
        ["box", "cylinder"], [0.022, 0.033, 0.045], [0.02, 0.04, 0.06], [0.3], [0.16, 0.24]
    ):
        rng = np.random.default_rng(0)
        scene = sample_scene(rng, cfg, {}, 0, 1)
        obj = scene.objects[0]
        scene.objects = [obj]
        obj.shape = shape
        obj.half_size = (
            [width / 2, height / 2] if shape == "cylinder" else [width / 2, width / 2 * 1.2, height / 2]
        )
        obj.pos = [r * np.cos(0.4), r * np.sin(0.4)]
        obj.yaw = yaw
        obj.mass, obj.friction = 0.05, 1.0
        model, info = build_scene(scene, {}, 64, 64)
        sub = max(1, round(1 / (cfg.fps * model.opt.timestep)))
        model.opt.timestep = 1 / (cfg.fps * sub)
        arm = Arm(model)
        cfg.motion.grasp_offset = 0.0
        cfg.motion.open_margin = (margin, margin)
        planner = Planner(arm, REST, cfg.fps, cfg.motion, np.random.default_rng(1))
        arm.fixed_jaw_offset += offset
        try:
            planner.subtask("pick")
            planner.pick(obj)
        except PlanningError:
            continue
        arm.fixed_jaw_offset -= offset
        plan = planner.plan()
        d = mujoco.MjData(model)
        d.qpos[arm.qadr] = arm.to_ctrl(REST)
        d.ctrl[arm.actuators] = arm.to_ctrl(REST)
        mujoco.mj_forward(model, d)
        mujoco.mj_step(model, d, nstep=SETTLE_FRAMES * sub)
        for cmd in plan.commands:
            d.ctrl[arm.actuators] = arm.to_ctrl(cmd)
            mujoco.mj_step(model, d, nstep=sub)
        mujoco.mj_step(model, d, nstep=10 * sub)
        total += 1
        if is_lifted(obj.name, model, d, info):
            ok += 1
        else:
            fails.append((shape, width, height, r))
    print(f"offset={offset} margin={margin}: {ok}/{total}", fails)

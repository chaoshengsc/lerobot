import sys
import time

sys.path.insert(0, ".")
import mujoco
import numpy as np
from PIL import Image

from examples.so101_datagen.config import GenerationConfig
from examples.so101_datagen.randomization import sample_scene
from examples.so101_datagen.scene import build_scene
from examples.so101_datagen.so101 import Arm

cfg = GenerationConfig()
rows = []
for seed in range(6):
    rng = np.random.default_rng(seed)
    p = sample_scene(rng, cfg, {}, containers=1, min_objects=2)
    t0 = time.perf_counter()
    m, info = build_scene(p, {}, 240, 320)
    t1 = time.perf_counter()
    d = mujoco.MjData(m)
    arm = Arm(m)
    d.qpos[arm.qadr] = arm.to_ctrl(np.array([0, -60, 60, 35, 0, 30.0]))
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 240, 320)
    ims = []
    for cam in ["front", "wrist_cam"]:
        r.update_scene(d, camera=cam)
        ims.append(r.render())
    r.close()
    rows.append(np.concatenate(ims, 1))
    print(
        seed,
        f"build {1000 * (t1 - t0):.0f}ms",
        [o.label for o in p.objects],
        [c.label for c in p.containers],
        "ncon",
        d.ncon,
    )
Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[1])

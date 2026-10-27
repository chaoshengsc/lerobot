"""Physics-only success rate of the scripted episodes, per task and start mode (milestone 1)."""

import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import EpisodeError, prepare, simulate  # noqa: E402
from examples.so101_datagen.tasks import TASKS  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
tasks = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TASKS)
cfg = GenerationConfig()
cfg.tasks.weights = dict.fromkeys(tasks, 1.0)
results = defaultdict(Counter)
times = defaultdict(list)
for seed in range(n):
    t0 = time.perf_counter()
    try:
        ep = prepare(seed, cfg, {})
    except EpisodeError as e:
        results["?"][e.reason] += 1
        continue
    t1 = time.perf_counter()
    key = f"{ep.task}/{ep.start_mode}"
    try:
        simulate(ep, TASKS[ep.task], cfg.fps)
        results[key]["ok"] += 1
    except EpisodeError as e:
        results[key][e.reason] += 1
    t2 = time.perf_counter()
    times["prepare"].append(t1 - t0)
    times["simulate"].append(t2 - t1)
    times["frames"].append(len(ep.plan.commands))
for key in sorted(results):
    c = results[key]
    total = sum(c.values())
    print(f"{key:28s} {c.get('ok', 0)}/{total} {dict(c)}")
print({k: round(float(np.mean(v)), 3) for k, v in times.items()})

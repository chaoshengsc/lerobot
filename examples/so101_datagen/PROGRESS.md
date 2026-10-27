# SO-101 synthetic data generator: plan and progress

> Working notes, to delete before opening a PR (along with `_dev/`).

## Goal

A fast, parallel generator of SO-101 pick and place demonstrations, built on PR #5109 (branch `feat/sim-foundation`). Its output is a LeRobotDataset that is already annotated in `lerobot-annotate`'s format (subtasks plus VQA boxes), with heavy domain randomization. Modularity matters more than completeness.

**Decisions already taken:**

- MuJoCo runs in-process.
- Objects are procedural primitives plus a small curated graspable subset of Google Scanned Objects.
- Runs are local only for now.
- Each worker process writes its own dataset shard, and `aggregate_datasets` merges them. We don't touch LeRobotDataset's single-writer limit.

## How it works

Each episode runs in two passes, success first. The expensive work only happens for episodes that already succeeded.

1. **Prepare** (`episode.py: prepare`) builds everything from the seed:
   - task and scene params (`randomization.py`);
   - the MjSpec model (`scene.py`);
   - the start pose;
   - the scripted plan (`motion.py` + `tasks/`).
2. **Pass 1** (`simulate`) runs physics only. Commands go through the teleop-side processors (drift and jerk). It early-aborts on a failed grasp and rejects episodes where non-task objects get disturbed.
3. **Pass 2** (`worker.py: record_episode`) replays the recorded qpos:
   - renders front and wrist (`render.py`);
   - runs segmentation only on VQA frames;
   - maps state and action through the robot-side calibration processors;
   - calls `add_frame`.
4. **After `finalize()`** (`annotations.py`), the rows are stamped with the parquet's own float32 timestamps, then staged, validated and written with lerobot's own `EpisodeStaging`, `StagingValidator` and `LanguageColumnsWriter`.

**Schema:**

- `observation.state` and `action` hold 6 values (`shoulder_pan.pos …`), in degrees with the gripper as 0-100.
- `observation.images.{front,wrist}` are 480×640.
- fps is 30, and `robot_type` is `so_follower`.

**Randomized per episode:**

- objects: shape, size, named color, texture, mass, friction, layout, with unique labels;
- container;
- background: table, floor, walls and sky, mostly muted HSV colors with some vivid ones;
- lights;
- front camera: look-at sampling, roll and fovy;
- wrist camera: ±3 mm, ±2°, fovy 57–65;
- actuator kp, damping and friction loss;
- arm color;
- start pose: rest, random collision-free, or mid-task;
- motion: speed, via points, tilt, dwell, approach and lift heights;
- calibration;
- command noise.

**Calibration:** `processors.VirtualCalibration` samples the sweep ends and the homing pose in encoder ticks. Its output matches `MotorsBus._normalize` exactly; this was checked against the real bus code over 200 random calibrations with zero difference.

## Status

**Milestone 1 (scene, IK, tasks, physics only): done.**

- Success rate is about 95–97% of simulated episodes, for all three tasks and all three start modes.
- About 10% of seeds are rejected cheaply at the planning stage.
- Fixes that mattered:
  - **Tilt search:** the SO-101 can't point straight down near the base, so pick and place search over approach tilts.
  - **Adaptive heights:** approach and lift heights step down until the pose is reachable.
  - **Multi-start IK.**
  - **Tilt clearance:** jaws get extra clearance when the approach is tilted.
  - **Raised REST pose:** the old one dragged the gripper over the table.
  - **Slower joint speeds:** the old ones exceeded what the servo model can follow.
  - **Minimum object distance of 17 cm,** as the user asked.

**Milestone 2 (rendering, dataset, shards, merge): working, still being tuned.**

- A smoke run with 2 shards × 3 episodes works end to end.
  - The merged dataset loads, and the video frame count equals the parquet rows.
  - All shipped recipes (`subtask`, `subtask_mem`, `subtask_mem_vqa_speech`) render every frame.
  - Boxes were checked visually on decoded frames.
- Fixed:
  - **Merge drops `info.json` tools:** `aggregate_datasets` drops the `tools` entry, so we re-apply it after merging.
  - **Stray segmentation pixels:** multisampling made edge pixels decode to other geoms' ids, so segmentation now uses a context without multisampling.
  - **Slow enum compare:** numpy vs a pybind enum compare cost 100 ms, fixed with `int()`.
  - **EGL leak:** a new `mujoco.Renderer` per episode leaked memory through EGL contexts and caused an OOM at 14 workers. It now uses one GL context per process.
- **Last benchmark, before the renderer rewrite:**
  - 6 workers: 98.5 frames/s total, about 1,290 episodes/h.
  - 1 worker: 68 frames/s.
  - Worker time split: render 35%, encoder wait 32%, save 16%, IK planning 8%.
- **Memory:**
  - About 1.3 GB per worker comes from imports alone (torch, `lerobot.datasets`, processors).
  - Peak PSS was about 2.1 GB per worker, but that was with the leak.
- **Codecs on one worker:**

| Codec                        | frames/s | Size  |
| ---------------------------- | -------- | ----- |
| svtav1 at its default preset | 68       | 10 MB |
| h264 veryfast                | 69       | 9 MB  |

## Next steps

1. **Re-benchmark with the new renderer.** `_dev/bench.sh <out> <workers> <shards> <eps> --encoding.encoder_threads=2` reports throughput and peak worker PSS. Find the best worker count for 16 cores and 31 GB.
2. **Throughput:**
   - Encoder threads vs workers.
   - Try `h264_nvenc`. The `preset` field must be a string; draccus parsed `p1` wrong through an `int | str` union.
   - Smaller `video_files_size_in_mb`.
   - Possibly batch `save_episode`.
3. **Milestone 3, quick:** a recipe render test and validator check inside a tiny e2e test (64×64, 1 episode).
4. **Milestone 4:**
   - Fill `gso_subset.json`: pick graspable GSO models by AABB, about 40–80 of them with labels and scale ranges.
   - Verify the mesh offset and centering in `assets.MeshAsset.add_to`.
   - Check that the curated subset grasps reliably.
5. **Calibration parity with a real SO-101:** compare state quantiles against a real so101 dataset. Sim zero/sign parity is still unverified.
6. **Milestone 5:**
   - README.
   - Unit test of calibration vs `MotorsBus._normalize`.
   - Final benchmark.
7. **Follow-ups:**
   - Synthetic interjection and speech rows.
   - Re-grasp and recovery episodes.
   - Stacking and "all X in Y" tasks.
   - Upstream fixes: a blocking streaming encoder (we currently read its private queues in `worker.wait_for_encoder`), and `aggregate_datasets` dropping `tools`.

## Commands

The venv is `uv venv -p 3.12 && uv pip install -e ".[dataset,serial-motors]" mujoco`.

```bash
# full run (resumable: finished shards are skipped)
python -m examples.so101_datagen.generate --output_dir=outputs/so101_datagen \
  --num_shards=8 --episodes_per_shard=10 --num_workers=6
```

**Tools in `_dev/`** (run from the repo root):

| Tool                                                                    | What it does                                  |
| ----------------------------------------------------------------------- | --------------------------------------------- |
| `physics_success.py N`                                                  | Success per task and start mode               |
| `debug_grasp.py`, `debug_tracking.py`, `render_failure.py SEED OUT.png` | Grasp failures                                |
| `grasp_sweep.py`                                                        | Grasp success vs clearance and opening margin |
| `draw_annotations.py ROOT OUT.png`                                      | Box and point overlays on decoded frames      |
| `render_scenes.py OUT.png`                                              | Scene contact sheet                           |
| `bench.sh`                                                              | Throughput and memory benchmark               |

**Caveat:** never run 14 workers on this laptop. It OOM-killed VSCode.

#!/usr/bin/env python3
"""Convert an SO101 LeRobot **v3.0** recording into the PiStar LeRobot schema.

Why this script exists
----------------------
* The user's SO101 teleop datasets (e.g. ``meow/so101_cube_into_plate_v2``) are stored in
  LeRobot **codebase_version v3.0** (packed parquet + packed mp4, ``meta/episodes/*.parquet``).
* The PiStar training stack pins an **old** lerobot (``lerobot.common.datasets``,
  ``CODEBASE_VERSION = v2.1``) that *cannot read* v3.0 datasets.
* So we read the v3.0 source with plain ``pandas`` (parquet) + ``pyav`` (video) — no lerobot on
  the read side — and **write** the output through PiStar's own old lerobot so the result is
  directly trainable by ``scripts/train.py`` / ``scripts/compute_norm_stats.py``.

Run it **inside the PiStar venv** (which has ``av`` 14.x, ``cv2``, ``pandas``, ``pyarrow``).

What it fills (demo / teleop semantics — every SO101 teleop frame is expert positive)
------------------------------------------------------------------------------------
Following ``examples/libero/pistar_rlds_demo_processing.py`` exactly:
* ``intervention``  = 1 for every frame (human teleop)
* ``adv_ind``       = "positive" for every frame
* ``value_label``   = -(T-1-t)/T   (last frame 0, first frame ~ -1)         -> in [-1, 0]
* ``reward``        = 0 except last frame = 1
* ``reward_label``  = -1/T except last frame = 0

Field mapping (SO101 v3.0 -> PiStar)
------------------------------------
* observation.images.fixed -> image        (main / 3rd-person view, resized to a square)
* observation.images.wrist -> wrist_image  (wrist view, resized to a square)
* observation.state (6,)   -> state        (5 joints + gripper, raw calibrated units)
* action (6,)              -> actions       (5 joints + gripper, raw calibrated units)
  (raw units are fine: PiStar computes its own norm stats via compute_norm_stats.py.)

Usage
-----
    source /data/users/szk/pistar/.venv/bin/activate
    python scripts/convert_so101_v3_to_pistar.py \
        --source ~/.cache/huggingface/lerobot/meow/so101_cube_into_plate_v2 \
        --repo_name meow/so101_cube_into_plate_v2_pistar \
        --output_dir ~/.cache/huggingface/lerobot   # optional; default = HF_LEROBOT_HOME

Then point the training config (pi05_star_so101) ``repo_id`` at the produced dataset, run
``scripts/compute_norm_stats.py --config-name pi05_star_so101`` and ``scripts/train.py``.

NOTE on real-robot deploy (out of scope here): the reused ``LiberoOutputs`` slices model
actions to ``[:, :7]``; for SO101's 6-DoF action the deploy side should slice ``[:, :6]``.
This does NOT affect training / norm stats (actions are padded to action_dim=32 internally).
"""

from __future__ import annotations

import os

# Keep this data-prep script off the GPU (image_tools.resize_with_pad is jax.jit'd). Set
# before importing jax/openpi. Override by exporting JAX_PLATFORMS=cuda if you want GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib
import shutil
from typing import Any

import av
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tyro

# Use openpi's OWN resize so stored frames are pixel-identical to what the training/inference
# pipeline produces (transforms.ResizeImages -> image_tools.resize_with_pad(224,224)).
from openpi.shared import image_tools

# PiStar pins old lerobot (lerobot.common.datasets); fall back to new API just in case.
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - only if run under a newer lerobot
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


# --------------------------------------------------------------------------------------
# PiStar label schedules (success/fail rules from examples/libero/main.py
# LiberoRolloutLeRobotWriter). For demo mode every episode is a successful expert
# trajectory (success=True), matching examples/libero/pistar_rlds_demo_processing.py.
# --------------------------------------------------------------------------------------
def compute_value_labels(episode_length: int, success: bool = True, penalty_value: float = -1.0) -> np.ndarray:
    if success:
        t = np.arange(episode_length, dtype=np.float32)
        return (-(episode_length - 1 - t) / float(episode_length)).astype(np.float32)
    return np.full((episode_length,), penalty_value, dtype=np.float32)


def compute_rewards(episode_length: int, success: bool = True) -> np.ndarray:
    rewards = np.zeros((episode_length,), dtype=np.float32)
    if success:
        rewards[-1] = 1.0
    return rewards


def compute_reward_labels(episode_length: int, success: bool = True) -> np.ndarray:
    reward_labels = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
    reward_labels[-1] = 0.0 if success else -1.0
    return reward_labels


# --------------------------------------------------------------------------------------
# v3.0 readers (no lerobot dependency) — ported from HIL-RL convert_so101_v3_to_hilrl_*.py
# --------------------------------------------------------------------------------------
def load_episode_table(source: pathlib.Path) -> pd.DataFrame:
    files = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3 episode metadata under {source / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True).sort_values("episode_index")


def load_default_task(source: pathlib.Path) -> str:
    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return "perform the task"
    tasks = pd.read_parquet(tasks_path)
    if tasks.index.size:
        return str(tasks.index[0])
    if "task" in tasks.columns and len(tasks):
        return str(tasks["task"].iloc[0])
    return "perform the task"


def episode_task(row: pd.Series, default_task: str) -> str:
    tasks = row.get("tasks", None)
    if isinstance(tasks, str):
        return tasks
    if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks):
        return str(tasks[0])
    return default_task


def source_video_path(source: pathlib.Path, video_key: str, chunk_index: int, file_index: int) -> pathlib.Path:
    return source / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def stack_column(df: pd.DataFrame, key: str, dtype: Any) -> np.ndarray:
    return np.stack(df[key].to_numpy()).astype(dtype, copy=False)


def read_scalar_column(df: pd.DataFrame, key: str) -> np.ndarray:
    """Read a per-frame scalar column, tolerating either plain scalars or (1,)-shaped arrays."""
    return np.array([np.asarray(v).reshape(-1)[0] for v in df[key].to_numpy()])


def resize_with_pad_batch(frames: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize of a (N, h, w, 3) uint8 batch to (N, size, size, 3) uint8 using
    openpi's own ``image_tools.resize_with_pad`` — the EXACT function PiStar applies at train and
    inference time (transforms.ResizeImages -> resize_with_pad(224, 224)). Producing the stored
    frame with the same function guarantees train/inference are pixel-identical (no aspect
    distortion; black padding on the short side). The pipeline's later resize_with_pad on an
    already-`size`x`size` image is an identity. See docs §12.
    """
    out = image_tools.resize_with_pad(jnp.asarray(frames, dtype=jnp.uint8), size, size)
    return np.asarray(out, dtype=np.uint8)


def decode_frames_pyav(
    video_path: pathlib.Path,
    abs_timestamps: np.ndarray,
    *,
    tolerance_s: float,
    image_size: int,
) -> list[np.ndarray]:
    """Decode the frames nearest to ``abs_timestamps`` from a (packed) mp4 using pyav.

    Returns a list of uint8 HWC RGB arrays resized to (image_size, image_size).
    Episode frames in a v3 packed video are contiguous, so we decode the keyframe-aligned
    range covering [min_ts, max_ts] once and nearest-match each requested timestamp.
    """
    targets = np.asarray(abs_timestamps, dtype=np.float64)
    first_ts, last_ts = float(targets.min()), float(targets.max())

    dec_ts: list[float] = []
    dec_frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # seek backward to the keyframe at/just before first_ts (seek unit = stream.time_base)
        seek_target = int(max(first_ts, 0.0) / stream.time_base)
        container.seek(seek_target, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < first_ts - tolerance_s:
                continue
            dec_ts.append(t)
            dec_frames.append(frame.to_ndarray(format="rgb24"))
            if t >= last_ts - 1e-9:
                break

    if not dec_frames:
        raise RuntimeError(f"No frames decoded from {video_path} for ts range [{first_ts}, {last_ts}]")

    dec_ts_arr = np.asarray(dec_ts)
    matched: list[np.ndarray] = []
    for ts in targets:
        j = int(np.argmin(np.abs(dec_ts_arr - ts)))
        if abs(dec_ts_arr[j] - ts) > tolerance_s:
            raise RuntimeError(
                f"{video_path.name}: no frame within {tolerance_s}s of t={ts:.4f} "
                f"(closest {dec_ts_arr[j]:.4f}). Increase --tolerance_s."
            )
        matched.append(dec_frames[j])  # native-resolution uint8 HWC

    # Batch resize_with_pad (one jax call per camera per chunk; all frames here share a shape).
    resized = resize_with_pad_batch(np.stack(matched), image_size)
    return [np.ascontiguousarray(resized[i], dtype=np.uint8) for i in range(resized.shape[0])]


def parse_index_spec(spec: str) -> set[int]:
    """Parse a comma/range episode-index spec into a set of ints, e.g. "5-9,20,25" -> {5,6,7,8,9,20,25}."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"Bad range '{part}' (hi < lo)")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


# --------------------------------------------------------------------------------------
def main(
    source: str,
    *,
    repo_name: str = "meow/so101_pistar",
    output_dir: str | None = None,
    overwrite: bool = True,
    image_size: int = 224,
    fps: int | None = None,
    tolerance_s: float = 0.05,
    fixed_key: str = "observation.images.fixed",
    wrist_key: str = "observation.images.wrist",
    right_wrist_key: str | None = None,
    state_key: str = "observation.state",
    action_key: str = "action",
    decode_batch_size: int = 64,
    max_episodes: int | None = None,
    episode_indices: str | None = None,
    exclude_indices: str | None = None,
    mode: str = "demo",
    intervention_key: str = "intervention",
    success_key: str = "success",
    penalty_value: float = -1.0,
    push_to_hub: bool = False,
) -> None:
    """Convert SO101 v3.0 -> PiStar schema. Produces the SAME 9-field v2.1 schema in both modes
    so demo + rollout sets merge cleanly with scripts/merge_datasets.py.

    --mode demo (default): teleop expert demos. Every frame intervention=1, adv_ind="positive",
      and value_label/reward/reward_label by the SUCCESS rule (each demo treated as successful).
    --mode rollout: autonomous/DAgger rollouts recorded by the SO101 openpi client. The source v3.0
      dataset MUST carry two extra columns: `intervention` (per-frame int 0/1) and `success`
      (per-frame constant int 0/1, set at episode end). Then per episode (T frames, success bool):
        value_label  = success ? -(T-1-t)/T (last 0) : penalty_value (all)
        reward       = success ? (last 1, else 0)    : 0 (all)
        reward_label = -1/T all; last = success ? 0 : -1
        intervention = from source column (per frame)
        adv_ind      = "none"  (placeholder; scripts/label_advantage_from_vlm.py overwrites it)
      Images/state/actions are read the SAME way as demo mode (shared reader).

    Episode subset selection (mutually exclusive; for stratified train/val holdout):
      --episode_indices "5,15,25"  -> convert ONLY these source episode_index (e.g. val set)
      --exclude_indices  "5,15,25" -> convert ALL EXCEPT these (e.g. train set)
    Both accept comma lists and ranges ("5-9,20"). Neither given -> convert all (unchanged).
    Filtering is applied to source episode_index; --max_episodes (if set) caps the result after.
    Output episode_index is reassigned contiguously from 0 by LeRobot save_episode() call order.
    """
    src = pathlib.Path(source).expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset (no meta/info.json): {src}")

    out_path = (pathlib.Path(output_dir).expanduser() / repo_name) if output_dir else (HF_LEROBOT_HOME / repo_name)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_path}")
        shutil.rmtree(out_path)

    import json

    info = json.loads((src / "meta" / "info.json").read_text())
    src_fps = int(fps or info.get("fps", 30))
    state_dim = int(info["features"][state_key]["shape"][0])
    action_dim = int(info["features"][action_key]["shape"][0])
    print(f"[info] source={src}")
    print(f"[info] codebase_version={info.get('codebase_version')} fps={src_fps} "
          f"state_dim={state_dim} action_dim={action_dim} image_size={image_size} mode={mode}")

    mode = mode.lower()
    if mode not in ("demo", "rollout"):
        raise ValueError(f"--mode must be 'demo' or 'rollout', got {mode!r}")

    episode_table = load_episode_table(src)
    default_task = load_default_task(src)
    print(f"[info] source episodes={len(episode_table)} default_task={default_task!r}")

    # Episode subset selection (stratified train/val holdout). Filter on SOURCE episode_index.
    if episode_indices and exclude_indices:
        raise ValueError("--episode_indices and --exclude_indices are mutually exclusive")
    if episode_indices:
        keep = parse_index_spec(episode_indices)
        avail = set(int(x) for x in episode_table["episode_index"].tolist())
        missing = keep - avail
        if missing:
            raise ValueError(f"--episode_indices not present in source: {sorted(missing)}")
        episode_table = episode_table[episode_table["episode_index"].isin(keep)]
        print(f"[info] --episode_indices: keeping {len(episode_table)} episodes {sorted(keep)}")
    elif exclude_indices:
        drop = parse_index_spec(exclude_indices)
        before = len(episode_table)
        episode_table = episode_table[~episode_table["episode_index"].isin(drop)]
        print(f"[info] --exclude_indices: dropped {before - len(episode_table)}, keeping {len(episode_table)} "
              f"(excluded {sorted(drop)})")
    if len(episode_table) == 0:
        raise ValueError("No episodes left after subset selection.")

    features = {
        "image": {"dtype": "image", "shape": (image_size, image_size, 3),
                  "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (image_size, image_size, 3),
                        "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
        "intervention": {"dtype": "int64", "shape": (1,), "names": ["intervention_flag"]},
        "value_label": {"dtype": "float32", "shape": (1,), "names": ["value_label"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "reward_label": {"dtype": "float32", "shape": (1,), "names": ["reward_label"]},
        "adv_ind": {"dtype": "string", "shape": (1,), "names": ["adv_ind"]},
    }
    # Optional 3rd camera (e.g. observation.images.fixed_1) -> right_wrist_image.
    if right_wrist_key:
        features["right_wrist_image"] = {
            "dtype": "image", "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=out_path,
        robot_type="so101",
        fps=src_fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    total_episodes = 0
    total_frames = 0

    for _, row in episode_table.iterrows():
        if max_episodes is not None and total_episodes >= max_episodes:
            print(f"[info] reached --max_episodes={max_episodes}, stopping")
            break
        ep_idx = int(row["episode_index"])
        data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_key not in data_cache:
            data_cache[data_key] = pd.read_parquet(
                src / "data" / f"chunk-{data_key[0]:03d}" / f"file-{data_key[1]:03d}.parquet"
            )
        ep_df = data_cache[data_key]
        ep_df = ep_df[ep_df["episode_index"] == ep_idx].reset_index(drop=True)
        T = len(ep_df)
        if T == 0:
            print(f"[warn] episode {ep_idx} empty, skipped")
            continue

        states = stack_column(ep_df, state_key, np.float32)
        actions = stack_column(ep_df, action_key, np.float32)
        task = episode_task(row, default_task)

        # Per-frame intervention + episode-level success drive the PiStar labels.
        if mode == "rollout":
            for col in (intervention_key, success_key):
                if col not in ep_df.columns:
                    raise KeyError(
                        f"--mode rollout requires column '{col}' in the source dataset "
                        f"(episode {ep_idx}). Columns present: {list(ep_df.columns)}"
                    )
            interventions = read_scalar_column(ep_df, intervention_key).astype(np.int64)
            success = bool(int(round(float(read_scalar_column(ep_df, success_key)[0]))))
            adv_ind_value = "none"  # placeholder; label_advantage_from_vlm.py overwrites it
        else:  # demo: every frame is expert/positive, every episode treated as successful
            interventions = np.ones(T, dtype=np.int64)
            success = True
            adv_ind_value = "positive"

        value_labels = compute_value_labels(T, success, penalty_value)
        rewards = compute_rewards(T, success)
        reward_labels = compute_reward_labels(T, success)

        fixed_video = source_video_path(
            src, fixed_key, int(row[f"videos/{fixed_key}/chunk_index"]), int(row[f"videos/{fixed_key}/file_index"])
        )
        wrist_video = source_video_path(
            src, wrist_key, int(row[f"videos/{wrist_key}/chunk_index"]), int(row[f"videos/{wrist_key}/file_index"])
        )
        fixed_base = float(row[f"videos/{fixed_key}/from_timestamp"])
        wrist_base = float(row[f"videos/{wrist_key}/from_timestamp"])
        if right_wrist_key:
            right_wrist_video = source_video_path(
                src, right_wrist_key,
                int(row[f"videos/{right_wrist_key}/chunk_index"]),
                int(row[f"videos/{right_wrist_key}/file_index"]),
            )
            right_wrist_base = float(row[f"videos/{right_wrist_key}/from_timestamp"])
        ep_ts = ep_df["timestamp"].to_numpy(dtype=np.float64)

        for start in range(0, T, decode_batch_size):
            end = min(start + decode_batch_size, T)
            sl = slice(start, end)
            fixed_frames = decode_frames_pyav(
                fixed_video, fixed_base + ep_ts[sl], tolerance_s=tolerance_s, image_size=image_size
            )
            wrist_frames = decode_frames_pyav(
                wrist_video, wrist_base + ep_ts[sl], tolerance_s=tolerance_s, image_size=image_size
            )
            right_wrist_frames = None
            if right_wrist_key:
                right_wrist_frames = decode_frames_pyav(
                    right_wrist_video, right_wrist_base + ep_ts[sl], tolerance_s=tolerance_s, image_size=image_size
                )
            for k in range(end - start):
                i = start + k
                frame = {
                        "image": fixed_frames[k],
                        "wrist_image": wrist_frames[k],
                        "state": states[i],
                        "actions": actions[i],
                        "intervention": np.asarray([int(interventions[i])], dtype=np.int64),
                        "value_label": np.asarray([value_labels[i]], dtype=np.float32),
                        "reward": np.asarray([rewards[i]], dtype=np.float32),
                        "reward_label": np.asarray([reward_labels[i]], dtype=np.float32),
                        "adv_ind": adv_ind_value,
                        "task": task,
                    }
                if right_wrist_frames is not None:
                    frame["right_wrist_image"] = right_wrist_frames[k]
                dataset.add_frame(frame)
                total_frames += 1

        dataset.save_episode()
        total_episodes += 1
        if mode == "rollout":
            n_interv = int(interventions.sum())
            print(f"[ok] episode {ep_idx}: {T} frames | success={success} "
                  f"intervention={n_interv}/{T} | task={task!r}")
        else:
            print(f"[ok] episode {ep_idx}: {T} frames | task={task!r}")

    print(f"[done] wrote {total_episodes} episodes / {total_frames} frames -> {out_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["so101", "pistar"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)

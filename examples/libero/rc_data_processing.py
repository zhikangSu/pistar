"""Convert LIBERO RLDS datasets to LeRobot schema for RC pipeline.

Target LeRobot schema:
    - image: main RGB image, shape (256, 256, 3)
    - wrist_image: wrist RGB image, shape (256, 256, 3)
    - reward: original RLDS reward, shape (1,)

Notes:
    - Reward is read from RLDS step['reward'] directly (same as pistar_data_processing_optimized.py).
    - This script intentionally does NOT include state, joint_state, actions, abs_pose.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def main(
    data_dir: str,
    repo_id: str = "ybpy/libero_pistar_rc",
    raw_dataset_names: tuple[str, ...] = (
        "libero_10",
    ),
    output_root: str | None = None,
    overwrite: bool = False,
):
    output_path = (Path(output_root) / repo_id) if output_root else (HF_LEROBOT_HOME / repo_id)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite true to recreate.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_count = 0
    for raw_name in raw_dataset_names:
        print(f"[INFO] Loading RLDS split: {raw_name}")
        raw_dataset = tfds.load(raw_name, data_dir=data_dir, split="train")

        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                task = _to_str(step["language_instruction"])
                reward = float(step["reward"])
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "reward": np.array([reward], dtype=np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"[INFO] Saved {episode_count} episodes")

    print(f"[DONE] LeRobot dataset saved at: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

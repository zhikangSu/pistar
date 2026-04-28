"""
Convert LIBERO RLDS demonstration data to LeRobot format with PI* labels.

This is based on convert_libero_data_to_lerobot.py, with the extra fields used
by main.py when saving LeRobot rollouts:
intervention, value_label, reward, reward_label, and adv_ind.

For demo data, every episode is treated as a successful positive trajectory:
- intervention is 1 for every frame
- value_label follows main.py's successful-episode schedule
- reward is 1 on the final frame and 0 otherwise
- reward_label is -1 / episode_length except the final frame, which is 0
- adv_ind is "positive" for every frame

Usage:
python examples/libero/pistar_rlds_demo_processing.py --data_dir /path/to/modified_libero_rlds
"""

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "ybpy/libero_pistar"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def compute_value_labels(episode_length: int) -> np.ndarray:
    t = np.arange(episode_length, dtype=np.float32)
    value_labels = -(episode_length - 1 - t) / float(episode_length)
    return value_labels.astype(np.float32)


def compute_rewards(episode_length: int) -> np.ndarray:
    rewards = np.zeros((episode_length,), dtype=np.float32)
    rewards[-1] = 1.0
    return rewards


def compute_reward_labels(episode_length: int) -> np.ndarray:
    reward_labels = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
    reward_labels[-1] = 0.0
    return reward_labels


def main(
    data_dir: str,
    *,
    repo_name: str = REPO_NAME,
    output_dir: str | None = None,
    overwrite: bool = True,
    push_to_hub: bool = False,
):
    if output_dir:
        output_path = pathlib.Path(output_dir).expanduser() / repo_name
    else:
        output_path = HF_LEROBOT_HOME / repo_name

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="panda",
        fps=10,
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
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "intervention": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["intervention_flag"],
            },
            "value_label": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["value_label"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "reward_label": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward_label"],
            },
            "adv_ind": {
                "dtype": "string",
                "shape": (1,),
                "names": ["adv_ind"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    total_frames = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"Processing {raw_dataset_name}")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        for episode in raw_dataset:
            steps = list(episode["steps"].as_numpy_iterator())
            episode_length = len(steps)
            if episode_length == 0:
                continue

            value_labels = compute_value_labels(episode_length)
            rewards = compute_rewards(episode_length)
            reward_labels = compute_reward_labels(episode_length)

            for idx, step in enumerate(steps):
                language_instruction = step["language_instruction"]
                task = (
                    language_instruction.decode()
                    if isinstance(language_instruction, bytes)
                    else str(language_instruction)
                )

                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": np.asarray(step["observation"]["state"], dtype=np.float32),
                        "actions": np.asarray(step["action"], dtype=np.float32),
                        "intervention": np.asarray([1], dtype=np.int64),
                        "value_label": np.asarray([value_labels[idx]], dtype=np.float32),
                        "reward": np.asarray([rewards[idx]], dtype=np.float32),
                        "reward_label": np.asarray([reward_labels[idx]], dtype=np.float32),
                        "adv_ind": "positive",
                        "task": task,
                    }
                )
                total_frames += 1

            dataset.save_episode()
            total_episodes += 1
            if total_episodes % 50 == 0:
                print(f"  Written {total_episodes} episodes, {total_frames} frames")

    print(f"Done. Wrote {total_episodes} episodes and {total_frames} frames to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "pistar"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

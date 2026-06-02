"""
Memory-efficient PiStar conversion for Aloha-style LeRobot datasets.

This script keeps the native Aloha LeRobot schema such as:
- observation.images.cam_high
- observation.images.cam_left_wrist
- observation.images.cam_right_wrist
- observation.state
- action

It only adds PiStar-related columns:
- reward
- value
- adv
- epsilon
- adv_ind
- intervention (optional, only if requested and missing in the input)

Example:
python examples/aloha_real/pistar_lerobot_data_processing.py \
    --input-repo-id physical-intelligence/aloha_pen_uncap_diverse \
    --output-repo-id physical-intelligence/aloha_pen_uncap_diverse_pistar \
    --default-adv-ind positive \
    --default-intervention 1
"""

from __future__ import annotations

import copy
import io
import shutil
from collections import defaultdict
from typing import Any

from datasets.features.features import _FEATURE_TYPES, Sequence
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro


if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = Sequence


def _get_raw_hf_dataset(dataset: LeRobotDataset):
    """Return the underlying HF dataset without the default tensor transform."""
    if dataset.hf_dataset is None:
        raise ValueError("LeRobotDataset.hf_dataset is not initialized.")
    dataset.hf_dataset.reset_format()
    return dataset.hf_dataset


def _to_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(value, dtype=np.float32)


def _to_numpy_int64(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        pass
    return np.asarray(value, dtype=np.int64)


def _to_image_array(value: Any) -> np.ndarray:
    """Convert PIL/np/tensor/arrow-image values to uint8 HWC."""
    if isinstance(value, np.ndarray):
        img = value
    elif isinstance(value, Image.Image):
        img = np.asarray(value.convert("RGB"))
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            img = np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
        elif value.get("path"):
            img = np.asarray(Image.open(value["path"]).convert("RGB"))
        else:
            raise ValueError(f"Unsupported image dict keys: {list(value.keys())}")
    else:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                img = value.detach().cpu().numpy()
            else:
                img = np.asarray(value)
        except Exception:
            img = np.asarray(value)

    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and np.nanmax(img) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.nan_to_num(img).clip(0, 255).astype(np.uint8)
    return img


def _image_layout_from_feature(feature: dict[str, Any]) -> str:
    shape = tuple(feature.get("shape") or ())
    names = feature.get("names") or ()
    if len(shape) != 3:
        return "hwc"
    if names and len(names) == 3:
        if names[0] in {"channel", "channels"}:
            return "chw"
        if names[-1] in {"channel", "channels"}:
            return "hwc"
    if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
        return "chw"
    return "hwc"


def _format_image_for_feature(value: Any, feature: dict[str, Any]) -> np.ndarray:
    img = _to_image_array(value)
    if _image_layout_from_feature(feature) == "chw":
        return np.transpose(img, (2, 0, 1))
    return img


def _normalize_adv_ind(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value_str = str(value).strip().lower()
    if value_str in {"positive", "negative"}:
        return value_str
    return None


def _as_scalar_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    value_array = np.asarray(value).reshape(-1)
    if value_array.size == 0:
        raise ValueError("Cannot convert empty value to float.")
    return float(value_array[0])


def transform_reward(original_reward: float, is_terminal: bool, is_last: bool, episode_length: int) -> float:
    if is_terminal or is_last:
        return 0.0 if original_reward == 1.0 else -1.0
    return -1.0 / episode_length


def compute_value_placeholder(step_data: dict[str, Any]) -> float:
    del step_data
    return 0.0


def compute_advantage(rewards: np.ndarray, values: np.ndarray, n_steps: int, gamma: float = 1.0) -> np.ndarray:
    episode_length = len(rewards)
    advantages = np.zeros(episode_length, dtype=np.float32)

    for t in range(episode_length):
        n_step_return = 0.0
        actual_steps = min(n_steps, episode_length - t)

        for i in range(actual_steps):
            n_step_return += (gamma**i) * rewards[t + i]

        if t + n_steps < episode_length:
            n_step_return += (gamma**n_steps) * values[t + n_steps]
        else:
            n_step_return += (gamma**actual_steps) * values[episode_length - 1]

        advantages[t] = n_step_return - values[t]

    return advantages


def _resolve_task(dataset: LeRobotDataset, frame: dict[str, Any], episode_idx: int) -> str:
    if "task" in frame and frame["task"] is not None:
        return str(frame["task"])
    if "prompt" in frame and frame["prompt"] is not None:
        return str(frame["prompt"])
    task_mapping = getattr(dataset.meta, "tasks", None) or {}
    if "task_index" in frame:
        task_index = int(np.asarray(frame["task_index"]).reshape(-1)[0])
        if task_index in task_mapping:
            return str(task_mapping[task_index])
    return f"episode_{episode_idx}"


def _prepare_frame(
    frame: dict[str, Any],
    features: dict[str, dict[str, Any]],
    *,
    default_intervention: int | None,
) -> dict[str, Any]:
    system_fields = {"index", "task_index", "episode_index", "frame_index", "timestamp"}
    new_frame = {k: v for k, v in frame.items() if k not in system_fields}

    for key, feature in features.items():
        if key not in new_frame:
            continue

        dtype = feature.get("dtype")
        if dtype in {"image", "video"}:
            new_frame[key] = _format_image_for_feature(new_frame[key], feature)
        elif dtype and dtype.startswith("float"):
            new_frame[key] = _to_numpy_float32(new_frame[key])
        elif dtype and dtype.startswith("int"):
            new_frame[key] = _to_numpy_int64(new_frame[key])

    if "intervention" not in new_frame and default_intervention is not None:
        new_frame["intervention"] = np.asarray([default_intervention], dtype=np.int64)

    return new_frame


def main(
    input_repo_id: str,
    output_repo_id: str,
    *,
    original_reward_key: str | None = None,
    n_steps: int = 10,
    value_model_path: str | None = None,
    default_value: float = 0.0,
    default_adv_ind: str | None = None,
    preserve_existing_adv_ind: bool = True,
    default_intervention: int | None = None,
    epsilon_percentile: float = 70.0,
    push_to_hub: bool = False,
):
    print("=" * 80)
    print("PiStar Aloha LeRobot processing")
    print("=" * 80)
    print(f"Input repo: {input_repo_id}")
    print(f"Output repo: {output_repo_id}")
    print(f"N-step window: {n_steps}")
    print(f"Default value: {default_value}")
    if default_adv_ind:
        print(f"Default adv_ind: {default_adv_ind} (skip epsilon scan)")
        print(f"Preserve existing adv_ind: {preserve_existing_adv_ind}")
    else:
        print(f"Epsilon percentile: {epsilon_percentile}%")
    if default_intervention is not None:
        print(f"Default intervention: {default_intervention}")
    if value_model_path:
        print("Value model loading is not implemented; using placeholder values.")

    task_epsilon: dict[str, float] = {}

    if not default_adv_ind:
        print("\n" + "=" * 80)
        print("Pass 1: lightweight scan for epsilon")
        print("=" * 80)

        input_dataset = LeRobotDataset(input_repo_id)
        raw_hf_dataset = _get_raw_hf_dataset(input_dataset)

        print(f"Total frames: {len(input_dataset)}")
        print(f"Total episodes: {input_dataset.num_episodes}")
        print(f"Features: {list(input_dataset.features.keys())}")

        task_advantages: dict[str, list[float]] = defaultdict(list)

        for episode_idx in range(input_dataset.num_episodes):
            episode_start = input_dataset.episode_data_index["from"][episode_idx].item()
            next_episode_start = (
                input_dataset.episode_data_index["from"][episode_idx + 1].item()
                if episode_idx + 1 < input_dataset.num_episodes
                else len(input_dataset)
            )
            episode_length = next_episode_start - episode_start

            rewards = []
            values = []
            task = None

            for step_idx in range(episode_length):
                frame_idx = episode_start + step_idx
                frame = raw_hf_dataset[frame_idx]

                if task is None:
                    task = _resolve_task(input_dataset, frame, episode_idx)

                if original_reward_key and original_reward_key in frame:
                    original_reward = _as_scalar_float(frame[original_reward_key])
                else:
                    original_reward = 1.0 if step_idx == episode_length - 1 else 0.0

                is_last = step_idx == episode_length - 1
                is_terminal = (original_reward == 1.0) and is_last
                rewards.append(transform_reward(original_reward, is_terminal, is_last, episode_length))
                values.append(compute_value_placeholder(frame) if value_model_path else default_value)

            advantages = compute_advantage(
                np.asarray(rewards, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
                n_steps,
            )
            task_advantages[task].extend(advantages.tolist())

            if (episode_idx + 1) % 50 == 0:
                print(f"Scanned {episode_idx + 1}/{input_dataset.num_episodes} episodes")

        print(f"\nComputing epsilon for {len(task_advantages)} tasks")
        for task, advantages in task_advantages.items():
            epsilon = float(np.percentile(advantages, epsilon_percentile))
            task_epsilon[task] = epsilon
            print(f"Task: {task[:60]}")
            print(f"  Advantages: {len(advantages)}")
            print(f"  Epsilon: {epsilon:.4f}")
    else:
        print("\n" + "=" * 80)
        print(f"Pass 1 skipped: using default adv_ind={default_adv_ind}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Pass 2: stream write output dataset")
    print("=" * 80)

    input_dataset = LeRobotDataset(input_repo_id)
    raw_hf_dataset = _get_raw_hf_dataset(input_dataset)
    output_path = HF_LEROBOT_HOME / output_repo_id

    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    new_features = copy.deepcopy(dict(input_dataset.features))
    if default_intervention is not None and "intervention" not in new_features:
        new_features["intervention"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["intervention_flag"],
        }
    new_features.update(
        {
            "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
            "value": {"dtype": "float32", "shape": (1,), "names": ["value"]},
            "adv": {"dtype": "float32", "shape": (1,), "names": ["adv"]},
            "epsilon": {"dtype": "float32", "shape": (1,), "names": ["epsilon"]},
            "adv_ind": {"dtype": "string", "shape": (1,), "names": ["adv_ind"]},
        }
    )

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type=input_dataset.meta.robot_type,
        fps=input_dataset.meta.fps,
        features=new_features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_steps = 0

    for episode_idx in range(input_dataset.num_episodes):
        episode_start = input_dataset.episode_data_index["from"][episode_idx].item()
        next_episode_start = (
            input_dataset.episode_data_index["from"][episode_idx + 1].item()
            if episode_idx + 1 < input_dataset.num_episodes
            else len(input_dataset)
        )
        episode_length = next_episode_start - episode_start

        rewards = []
        values = []
        task = None

        for step_idx in range(episode_length):
            frame_idx = episode_start + step_idx
            frame = raw_hf_dataset[frame_idx]

            if task is None:
                task = _resolve_task(input_dataset, frame, episode_idx)

            if original_reward_key and original_reward_key in frame:
                original_reward = _as_scalar_float(frame[original_reward_key])
            else:
                original_reward = 1.0 if step_idx == episode_length - 1 else 0.0

            is_last = step_idx == episode_length - 1
            is_terminal = (original_reward == 1.0) and is_last
            rewards.append(transform_reward(original_reward, is_terminal, is_last, episode_length))
            values.append(compute_value_placeholder(frame) if value_model_path else default_value)

        rewards_array = np.asarray(rewards, dtype=np.float32)
        values_array = np.asarray(values, dtype=np.float32)
        advantages = (
            np.zeros(episode_length, dtype=np.float32)
            if default_adv_ind
            else compute_advantage(rewards_array, values_array, n_steps)
        )
        epsilon = 0.0 if default_adv_ind else task_epsilon[task]

        for step_idx in range(episode_length):
            frame_idx = episode_start + step_idx
            frame = raw_hf_dataset[frame_idx]
            new_frame = _prepare_frame(frame, input_dataset.features, default_intervention=default_intervention)

            if default_adv_ind:
                existing_adv_ind = _normalize_adv_ind(frame.get("adv_ind"))
                adv_ind = existing_adv_ind if preserve_existing_adv_ind and existing_adv_ind is not None else default_adv_ind
            else:
                adv_ind = "positive" if advantages[step_idx] > epsilon else "negative"

            new_frame.update(
                {
                    "task": task,
                    "reward": np.asarray([rewards_array[step_idx]], dtype=np.float32),
                    "value": np.asarray([values_array[step_idx]], dtype=np.float32),
                    "adv": np.asarray([advantages[step_idx]], dtype=np.float32),
                    "epsilon": np.asarray([epsilon], dtype=np.float32),
                    "adv_ind": adv_ind,
                }
            )

            output_dataset.add_frame(new_frame)
            total_steps += 1

        output_dataset.save_episode()

        if (episode_idx + 1) % 50 == 0:
            print(f"Written {episode_idx + 1}/{input_dataset.num_episodes} episodes")

    print("\nConversion finished")
    print(f"Total episodes: {input_dataset.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Output path: {output_path}")

    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub")
        output_dataset.push_to_hub(
            tags=["pistar", "value", "advantage"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

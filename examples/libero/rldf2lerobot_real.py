#!/usr/bin/env python3
"""Real-data repair conversion: rewrite an old real LeRobot dataset into a strict training-ready LeRobot dataset.

中文说明:
- 这是“真机数据转换脚本”，不是“合并脚本”。
- 主要用于把老的真机数据集重新写成标准 LeRobot 数据集，并按保底机制补齐缺失字段。
- 标签计算方式参考 `examples/libero/pistar_rlds_demo_processing.py` 和 `examples/libero/main.py`，
  但这里处理的是老的真机数据，因此只借鉴“按 episode 先计算、再逐帧写入”的处理方式。
- 输出数据集只保留这 9 个业务键:
  - `image` (`uint8` image)
  - `wrist_image` (`uint8` image)
  - `state` (`float32` array)
  - `actions` (`float32` array)
  - `intervention` (`int64` array, shape `(1,)`)
  - `value_label` (`float32` array, shape `(1,)`)
  - `reward` (`float32` array, shape `(1,)`)
  - `reward_label` (`float32` array, shape `(1,)`)
  - `adv_ind` (`string`)
- 其他列都会被过滤掉；`task` 会保留为 episode 元数据，不作为普通业务列保留。
- 兼容性说明:
  - 参考 `examples/libero/pistar_rlds_demo_processing.py` / `examples/libero/main.py` 的写法，
    脚本会优先把 `task` 一起传给 `add_frame()`。
  - `save_episode()` 是否接受 `task=` 参数在不同 LeRobot 版本里并不一致，因此脚本会自动兼容。
  - 业务语义上，`task` 仍然按“元数据字段”看待，不参与 reward / adv_ind 等保底逻辑。

保底机制:
- 已有合法值则保留。
- 缺失/空值才补默认值。
- `value_lable` 会自动修正为 `value_label`。
- 如果缺失 `wrist_image`，会退化为复制 `image`。
- 如果缺失 `image` 但有 `wrist_image`，会退化为复制 `wrist_image`。
- `value_label` 不会按仿真脚本规则重算，而是沿用输入数据；rollout 的成功/失败判断也只看最后一帧 `value_label`。

Episode 规则:
- 不需要手动区分 demo / rollout，也不需要传 `--dataset-kind`。
- 对每个 episode，只看最后一帧 `value_label`:
  - 最后一帧 `value_label == -1` 视为失败 episode
  - 最后一帧 `value_label == 0` 视为成功 episode
- `value_label`: 保留输入值，不重算
- `reward`:
  - 成功: 最后一帧 `1`，其余帧 `0`
  - 失败: 全部 `0`
- `reward_label`:
  - 非最后一帧 `-1 / T`
  - 成功最后一帧 `0`
  - 失败最后一帧 `-1`
- `adv_ind`:
  - 已有合法值则保留
  - 缺失时: 成功 episode 补 `"positive"`，失败 episode 补 `"none"`

数据转换示意:
    +--------------------------------------------------------------+
    | Input old real dataset                                       |
    | keys may contain:                                            |
    |   image, wrist_image, state, actions/action, intervention,   |
    |   value_label/value_lable, reward, reward_label, adv_ind, ...|
    +--------------------------------------------------------------+
                         |
                         |-----> 保留/重编码: image, wrist_image, state, actions
                         |-----> 保底补齐: intervention, reward, reward_label, adv_ind
                         |-----> 原样保留: value_label
                         |- - -> 重命名: value_lable -> value_label
                         |- - -> 过滤: index, frame_index, episode_index, timestamp, 其他额外字段
                         v
    +--------------------------------------------------------------+
    | Output LeRobot dataset                                       |
    | keys:                                                        |
    |   image, wrist_image, state, actions, intervention,          |
    |   value_label, reward, reward_label, adv_ind                 |
    | metadata: task                                               |
    +--------------------------------------------------------------+

Usage:
    sudo HF_LEROBOT_HOME=/public/home/wangsenbao_it/litianheng/lerobot_datasets \
    PYTHONPATH=/public/home/wangsenbao_it/litianheng/pistar:/public/home/wangsenbao_it/litianheng/pistar/src:/public/home/wangsenbao_it/litianheng/pistar/packages/openpi-client/src \
    /.venv/bin/python examples/libero/rldf2lerobot_real.py \
      --input-repo-id old_real_demo \
      --output-repo-id old_real_demo_fixed \
      --overwrite
"""

from __future__ import annotations

import io
import math
import shutil
from typing import Any, Literal

from datasets.features.features import _FEATURE_TYPES, Sequence
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro


if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = Sequence


OUTPUT_KEYS = (
    "image",
    "wrist_image",
    "state",
    "actions",
    "intervention",
    "value_label",
    "reward",
    "reward_label",
    "adv_ind",
)

FIELD_CANDIDATES = {
    "image": ("image",),
    "wrist_image": ("wrist_image", "wristimage"),
    "state": ("state", "observation.state"),
    "actions": ("actions", "action"),
    "intervention": ("intervention", "intervene"),
    "value_label": ("value_label", "value_lable"),
    "reward": ("reward",),
    "reward_label": ("reward_label", "reward_lable"),
    "adv_ind": ("adv_ind", "advind"),
    "task": ("task", "prompt", "language_instruction"),
}


def _get_raw_hf_dataset(dataset: LeRobotDataset):
    if dataset.hf_dataset is None:
        raise ValueError("LeRobotDataset.hf_dataset is not initialized.")
    dataset.hf_dataset.reset_format()
    return dataset.hf_dataset


def _first_present(container: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in container and container[name] is not None:
            return container[name]
    return None


def _first_feature(dataset: LeRobotDataset, names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        if name in dataset.features:
            return dict(dataset.features[name])
    return None


def _require_present(container: dict[str, Any], names: tuple[str, ...], field_name: str) -> Any:
    value = _first_present(container, names)
    if not _has_value(value):
        raise KeyError(f"Missing required field `{field_name}` in input frame.")
    return value


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


def _coerce_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return arr
    if arr.shape == shape:
        return arr
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return arr


def _extract_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        value = float(value)
        if math.isnan(value):
            return None
        return value
    try:
        arr = np.asarray(value).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    scalar = float(arr[0])
    if math.isnan(scalar):
        return None
    return scalar


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    return True


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
    text = str(value).strip().lower()
    if text in {"positive", "negative", "none"}:
        return text
    return None


def _to_image_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        img = value
    elif isinstance(value, Image.Image):
        img = np.asarray(value.convert("RGB"))
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image_obj:
                img = np.asarray(image_obj.convert("RGB"))
        elif value.get("path"):
            with Image.open(value["path"]) as image_obj:
                img = np.asarray(image_obj.convert("RGB"))
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

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={img.shape}")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and np.nanmax(img) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.nan_to_num(img).clip(0, 255).astype(np.uint8)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] >= 4:
        img = img[..., :3]
    return img


def _resize_image(img: np.ndarray, *, height: int, width: int, channels: int) -> np.ndarray:
    if img.shape[:2] != (height, width):
        img = np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))
    if channels == 1:
        img = np.asarray(Image.fromarray(img).convert("L"))[..., None]
    elif channels == 3 and img.shape[-1] != 3:
        img = np.asarray(Image.fromarray(img).convert("RGB"))
    return img


def _image_layout_from_feature(feature: dict[str, Any] | None, fallback_img: np.ndarray) -> str:
    if feature is not None:
        shape = tuple(feature.get("shape") or ())
        names = feature.get("names") or ()
        if len(shape) == 3 and len(names) == 3:
            if names[0] in {"channel", "channels"}:
                return "chw"
            if names[-1] in {"channel", "channels"}:
                return "hwc"
        if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            return "chw"
    if fallback_img.ndim == 3 and fallback_img.shape[0] in (1, 3, 4) and fallback_img.shape[-1] not in (1, 3, 4):
        return "chw"
    return "hwc"


def _format_image_for_output(
    value: Any,
    *,
    height: int,
    width: int,
    channels: int,
    layout: Literal["chw", "hwc"],
) -> np.ndarray:
    img = _to_image_hwc_uint8(value)
    img = _resize_image(img, height=height, width=width, channels=channels)
    if layout == "chw":
        return np.transpose(img, (2, 0, 1))
    return img


def _resolve_task(dataset: LeRobotDataset, frame: dict[str, Any], episode_idx: int) -> str:
    task_value = _first_present(frame, FIELD_CANDIDATES["task"])
    if isinstance(task_value, bytes):
        return task_value.decode("utf-8", errors="ignore")
    if task_value is not None:
        return str(task_value)

    task_mapping = getattr(dataset.meta, "tasks", None) or {}
    task_index = _extract_scalar(frame.get("task_index"))
    if task_index is not None and int(task_index) in task_mapping:
        return str(task_mapping[int(task_index)])

    return f"episode_{episode_idx}"


def _infer_episode_success(last_frame: dict[str, Any]) -> bool:
    final_value = _extract_scalar(_first_present(last_frame, FIELD_CANDIDATES["value_label"]))
    if final_value is None:
        raise ValueError("Episode is missing final `value_label`; cannot infer success/failure.")
    if abs(final_value + 1.0) < 1e-6:
        return False
    if abs(final_value) < 1e-6:
        return True
    raise ValueError(
        f"Unexpected final `value_label`: {final_value}. Expected 0 or -1."
    )


def _compute_rewards(episode_len: int, success: bool) -> np.ndarray:
    rewards = np.zeros((episode_len,), dtype=np.float32)
    if success:
        rewards[-1] = 1.0
    return rewards


def _compute_reward_labels(episode_len: int, success: bool) -> np.ndarray:
    reward_labels = np.full((episode_len,), -1.0 / float(episode_len), dtype=np.float32)
    reward_labels[-1] = 0.0 if success else -1.0
    return reward_labels


def _derive_adv_ind(success: bool) -> str:
    return "positive" if success else "none"


def _normalize_scalar_feature(value: Any, *, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "float32":
        return _coerce_shape(_to_numpy_float32(value), shape)
    if dtype == "int64":
        return _coerce_shape(_to_numpy_int64(value), shape)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _infer_array_shape(dataset: LeRobotDataset, sample_frame: dict[str, Any], key: str) -> tuple[int, ...]:
    feature = _first_feature(dataset, FIELD_CANDIDATES[key])
    if feature is not None and feature.get("shape"):
        return tuple(feature["shape"])
    value = _first_present(sample_frame, FIELD_CANDIDATES[key])
    if value is None:
        raise KeyError(f"Missing required field: {key}")
    return tuple(np.asarray(value).shape)


def _read_first_image_sample(dataset: LeRobotDataset, sample_frame: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
    image_value = _first_present(sample_frame, FIELD_CANDIDATES["image"])
    image_feature = _first_feature(dataset, FIELD_CANDIDATES["image"])
    if image_value is not None:
        return image_value, image_feature

    wrist_value = _first_present(sample_frame, FIELD_CANDIDATES["wrist_image"])
    wrist_feature = _first_feature(dataset, FIELD_CANDIDATES["wrist_image"])
    if wrist_value is not None:
        return wrist_value, wrist_feature

    raise KeyError("Missing both `image` and `wrist_image` in the input dataset.")


def _build_output_features(
    dataset: LeRobotDataset,
    sample_frame: dict[str, Any],
    *,
    image_height: int | None,
    image_width: int | None,
    image_channels: int | None,
    image_layout: Literal["chw", "hwc"] | None,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    image_value, image_feature = _read_first_image_sample(dataset, sample_frame)
    image_sample = _to_image_hwc_uint8(image_value)

    resolved_height = image_height or int(image_sample.shape[0])
    resolved_width = image_width or int(image_sample.shape[1])
    resolved_channels = image_channels or int(image_sample.shape[2])
    resolved_layout = image_layout or _image_layout_from_feature(image_feature, image_sample)

    if resolved_layout == "chw":
        image_shape = (resolved_channels, resolved_height, resolved_width)
        image_names = ["channels", "height", "width"]
    else:
        image_shape = (resolved_height, resolved_width, resolved_channels)
        image_names = ["height", "width", "channel"]

    features = {
        "image": {
            "dtype": "image",
            "shape": image_shape,
            "names": image_names,
        },
        "wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": image_names,
        },
        "state": {
            "dtype": "float32",
            "shape": _infer_array_shape(dataset, sample_frame, "state"),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": _infer_array_shape(dataset, sample_frame, "actions"),
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
    }
    image_spec = {
        "height": resolved_height,
        "width": resolved_width,
        "channels": resolved_channels,
        "layout": resolved_layout,
    }
    return features, image_spec


def _prepare_frame(
    raw_frame: dict[str, Any],
    *,
    step_idx: int,
    features: dict[str, Any],
    image_spec: dict[str, int | str],
    default_intervention: int,
    success: bool,
    rewards: np.ndarray,
    reward_labels: np.ndarray,
) -> dict[str, Any]:
    image_value = _first_present(raw_frame, FIELD_CANDIDATES["image"])
    wrist_image_value = _first_present(raw_frame, FIELD_CANDIDATES["wrist_image"])
    if image_value is None and wrist_image_value is None:
        raise KeyError("Each frame must contain at least one of `image` or `wrist_image`.")
    if image_value is None:
        image_value = wrist_image_value
    if wrist_image_value is None:
        wrist_image_value = image_value

    frame = {
        "image": _format_image_for_output(
            image_value,
            height=int(image_spec["height"]),
            width=int(image_spec["width"]),
            channels=int(image_spec["channels"]),
            layout=str(image_spec["layout"]),
        ),
        "wrist_image": _format_image_for_output(
            wrist_image_value,
            height=int(image_spec["height"]),
            width=int(image_spec["width"]),
            channels=int(image_spec["channels"]),
            layout=str(image_spec["layout"]),
        ),
        "state": _normalize_scalar_feature(
            _require_present(raw_frame, FIELD_CANDIDATES["state"], "state"),
            dtype="float32",
            shape=tuple(features["state"]["shape"]),
        ),
        "actions": _normalize_scalar_feature(
            _require_present(raw_frame, FIELD_CANDIDATES["actions"], "actions"),
            dtype="float32",
            shape=tuple(features["actions"]["shape"]),
        ),
    }

    intervention_value = _first_present(raw_frame, FIELD_CANDIDATES["intervention"])
    if _has_value(intervention_value):
        frame["intervention"] = _normalize_scalar_feature(intervention_value, dtype="int64", shape=(1,))
    else:
        frame["intervention"] = np.asarray([default_intervention], dtype=np.int64)

    frame["value_label"] = _normalize_scalar_feature(
        _require_present(raw_frame, FIELD_CANDIDATES["value_label"], "value_label"),
        dtype="float32",
        shape=(1,),
    )

    reward_value = _first_present(raw_frame, FIELD_CANDIDATES["reward"])
    if _has_value(reward_value):
        frame["reward"] = _normalize_scalar_feature(reward_value, dtype="float32", shape=(1,))
    else:
        frame["reward"] = np.asarray([rewards[step_idx]], dtype=np.float32)

    reward_label_value = _first_present(raw_frame, FIELD_CANDIDATES["reward_label"])
    if _has_value(reward_label_value):
        frame["reward_label"] = _normalize_scalar_feature(reward_label_value, dtype="float32", shape=(1,))
    else:
        frame["reward_label"] = np.asarray([reward_labels[step_idx]], dtype=np.float32)

    existing_adv_ind = _normalize_adv_ind(_first_present(raw_frame, FIELD_CANDIDATES["adv_ind"]))
    frame["adv_ind"] = existing_adv_ind if existing_adv_ind is not None else _derive_adv_ind(success)

    return frame


def _add_frame(dataset: LeRobotDataset, frame: dict[str, Any], task: str) -> None:
    frame_with_task = dict(frame)
    frame_with_task["task"] = task
    try:
        dataset.add_frame(frame_with_task)
        return
    except TypeError:
        dataset.add_frame(frame)


def _save_episode(dataset: LeRobotDataset, task: str) -> None:
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()



def main(
    input_repo_id: str,
    output_repo_id: str,
    *,
    default_intervention: int = 1,
    image_height: int | None = None,
    image_width: int | None = None,
    image_channels: int | None = None,
    image_layout: Literal["chw", "hwc"] | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> None:
    print("=" * 80)
    print("Real data repair conversion to standard LeRobot format")
    print("=" * 80)
    print(f"Input repo: {input_repo_id}")
    print(f"Output repo: {output_repo_id}")
    print(f"Default intervention: {default_intervention}")

    input_dataset = LeRobotDataset(input_repo_id)
    raw_hf_dataset = _get_raw_hf_dataset(input_dataset)
    if len(raw_hf_dataset) == 0:
        raise ValueError("Input dataset is empty.")

    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
        shutil.rmtree(output_path)

    sample_frame = raw_hf_dataset[0]
    features, image_spec = _build_output_features(
        input_dataset,
        sample_frame,
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        image_layout=image_layout,
    )

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type=input_dataset.meta.robot_type,
        fps=input_dataset.meta.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_steps = 0
    for episode_idx in range(input_dataset.num_episodes):
        start = int(input_dataset.episode_data_index["from"][episode_idx].item())
        end = (
            int(input_dataset.episode_data_index["from"][episode_idx + 1].item())
            if episode_idx + 1 < input_dataset.num_episodes
            else len(input_dataset)
        )
        episode_len = end - start
        if episode_len <= 0:
            continue

        first_frame = raw_hf_dataset[start]
        last_frame = raw_hf_dataset[end - 1]
        task = _resolve_task(input_dataset, first_frame, episode_idx)
        success = _infer_episode_success(last_frame)
        rewards = _compute_rewards(episode_len, success)
        reward_labels = _compute_reward_labels(episode_len, success)

        for step_idx in range(episode_len):
            raw_frame = raw_hf_dataset[start + step_idx]
            frame = _prepare_frame(
                raw_frame,
                step_idx=step_idx,
                features=features,
                image_spec=image_spec,
                default_intervention=default_intervention,
                success=success,
                rewards=rewards,
                reward_labels=reward_labels,
            )
            _add_frame(output_dataset, frame, task)
            total_steps += 1

        _save_episode(output_dataset, task)
        if (episode_idx + 1) % 50 == 0:
            print(f"Processed {episode_idx + 1}/{input_dataset.num_episodes} episodes")

    print(f"Done. episodes={input_dataset.num_episodes}, steps={total_steps}")
    print(f"Output path: {output_path}")
    print(f"Output keys: {list(OUTPUT_KEYS)}")
    print(
        "Image spec: "
        f"layout={image_spec['layout']}, channels={image_spec['channels']}, "
        f"height={image_spec['height']}, width={image_spec['width']}"
    )

    if push_to_hub:
        output_dataset.push_to_hub(
            tags=["real", "lerobot", "repair", "adv_ind", "reward_label"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

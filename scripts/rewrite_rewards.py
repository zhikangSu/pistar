"""重写 LeRobot 数据集的 reward_label 列。

规则：
0. 若 parquet 缺少 reward_label 列，则创建 reward_label 列
1. 对每个 episode，除最后一帧外，reward_label 全部设置为 -1 / T
2. 查看该 episode 最后一帧的 value_label
3. 若最后一帧 value_label 为 -1，则最后一帧 reward_label 设为 -1
4. 若最后一帧 value_label 为 0，则最后一帧 reward_label 设为 0

其中 T 为当前 episode 的总帧数。

用法:
    python scripts/rewrite_rewards.py --data_dir /path/to/lerobot_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.shared import console
from openpi.shared import progress


REWARD_COLUMN = "reward_label"
VALUE_LABEL_COLUMN = "value_label"
LEGACY_VALUE_LABEL_COLUMN = "value_lable"


def _episode_row_slices(df: pd.DataFrame) -> list[slice]:
    if "episode_index" not in df.columns:
        return [slice(0, len(df))]

    episode_ids = df["episode_index"].to_numpy()
    if len(episode_ids) == 0:
        return []

    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(df)]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _rewrite_episode_rewards(value_labels: np.ndarray) -> np.ndarray:
    episode_length = len(value_labels)
    if episode_length == 0:
        return value_labels.astype(np.float32)

    updated = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
    last_value_label = float(value_labels[-1])
    if np.isclose(last_value_label, -1.0):
        updated[-1] = -1.0
    elif np.isclose(last_value_label, 0.0):
        updated[-1] = 0.0
    else:
        raise ValueError(
            f"episode 最后一帧的 {VALUE_LABEL_COLUMN} 必须是 -1 或 0，实际为 {last_value_label}"
        )
    return updated


def _get_value_labels(df: pd.DataFrame, parquet_path: Path) -> tuple[np.ndarray, str]:
    for column_name in (VALUE_LABEL_COLUMN, LEGACY_VALUE_LABEL_COLUMN):
        if column_name in df.columns:
            return df[column_name].to_numpy(dtype=np.float32, copy=True), column_name
    raise ValueError(
        f"{parquet_path} 缺少价值标签列，尝试过: {(VALUE_LABEL_COLUMN, LEGACY_VALUE_LABEL_COLUMN)}"
    )


def rewrite_rewards_in_parquet(parquet_path: Path) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    value_labels, source_column = _get_value_labels(df, parquet_path)
    rewards = np.empty((len(df),), dtype=np.float32)
    episode_count = 0

    for row_slice in _episode_row_slices(df):
        rewards[row_slice] = _rewrite_episode_rewards(value_labels[row_slice])
        episode_count += 1

    if REWARD_COLUMN not in df.columns:
        print(console.warn(f"{parquet_path} 缺少 {REWARD_COLUMN} 列，已按 {source_column} 新建"))
    df[REWARD_COLUMN] = rewards
    df.to_parquet(parquet_path, index=False)
    return len(df), episode_count


def update_info_json(info_path: Path) -> None:
    with open(info_path) as f:
        info = json.load(f)

    if "features" not in info:
        info["features"] = {}

    reward_feature = info["features"].setdefault(
        REWARD_COLUMN,
        {
            "dtype": "float32",
            "shape": [1],
        },
    )
    reward_feature["description"] = (
        "Reward rewritten per episode: non-terminal frames are -1/T; "
        "last frame is set from terminal value_label (-1->-1, 0->0)."
    )

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="重写 LeRobot 数据集的 reward_label 列")
    parser.add_argument("--data_dir", type=str, required=True, help="LeRobot 数据集路径")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")

    parquet_dir = data_dir / "data"
    if not parquet_dir.exists():
        raise ValueError(f"找不到 data 目录: {parquet_dir}")

    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

    print(console.info(f"找到 {len(parquet_files)} 个 parquet 文件"))

    total_frames = 0
    total_episodes = 0
    pbar = tqdm(
        parquet_files,
        desc="重写 reward_label",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    for parquet_path in pbar:
        progress.sync_pbar_color(pbar)
        frames, episodes = rewrite_rewards_in_parquet(parquet_path)
        total_frames += frames
        total_episodes += episodes

    info_path = data_dir / "meta" / "info.json"
    if info_path.exists():
        update_info_json(info_path)

    print(console.ok(f"\n完成! 共处理 {len(parquet_files)} 个 parquet, {total_episodes} 个 episodes, {total_frames} 帧"))


if __name__ == "__main__":
    main()

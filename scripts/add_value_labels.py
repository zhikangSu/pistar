"""为 LeRobot 数据集添加归一化价值标签。

价值计算公式：value_label = -(T - t) / T
- t: 当前帧索引
- T: episode 总帧数
- value_label 范围: [-1, 0]

用法:
    python scripts/add_value_labels.py --data_dir /path/to/lerobot_dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.shared import console
from openpi.shared import progress


VALUE_LABEL_COLUMN = "value_label"
LEGACY_VALUE_LABEL_COLUMN = "value_lable"


def compute_value_labels(episode_length: int) -> np.ndarray:
    """计算一个 episode 的归一化价值标签。

    Args:
        episode_length: 总帧数 T

    Returns:
        value_labels: [T], 范围 [-1, 0]
    """
    T = episode_length
    t = np.arange(T)
    value_normalized = -(T - t) / T
    return value_normalized.astype(np.float32)


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


def add_value_to_parquet(parquet_path: Path) -> int:
    """为单个 parquet 文件添加 value_label 列。"""
    df = pd.read_parquet(parquet_path)

    if VALUE_LABEL_COLUMN in df.columns:
        print(console.warn(f"覆盖 {parquet_path.name}: 重写已有 {VALUE_LABEL_COLUMN} 列"))
    if LEGACY_VALUE_LABEL_COLUMN in df.columns:
        print(console.warn(f"迁移 {parquet_path.name}: 删除旧列 {LEGACY_VALUE_LABEL_COLUMN}"))
        df = df.drop(columns=[LEGACY_VALUE_LABEL_COLUMN])

    value_labels = np.empty((len(df),), dtype=np.float32)
    for row_slice in _episode_row_slices(df):
        episode_length = row_slice.stop - row_slice.start
        value_labels[row_slice] = compute_value_labels(episode_length)

    df[VALUE_LABEL_COLUMN] = value_labels
    df.to_parquet(parquet_path, index=False)
    return len(df)


def update_info_json(info_path: Path):
    """更新 info.json，添加 value_label 字段信息。"""
    with open(info_path) as f:
        info = json.load(f)

    if "features" not in info:
        info["features"] = {}

    if LEGACY_VALUE_LABEL_COLUMN in info["features"]:
        print(console.warn(f"info.json 删除旧字段 {LEGACY_VALUE_LABEL_COLUMN}"))
        info["features"].pop(LEGACY_VALUE_LABEL_COLUMN, None)

    info["features"][VALUE_LABEL_COLUMN] = {
        "dtype": "float32",
        "shape": [1],
        "description": "Normalized value label: -(T-t)/T, range [-1, 0]",
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(console.ok(f"已更新 {info_path}"))


def main():
    parser = argparse.ArgumentParser(description="为 LeRobot 数据集添加价值标签")
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
    pbar = tqdm(
        parquet_files,
        desc="处理 episodes",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    progress.sync_pbar_color(pbar)
    for parquet_path in pbar:
        progress.sync_pbar_color(pbar)
        frames = add_value_to_parquet(parquet_path)
        total_frames += frames

    info_path = data_dir / "meta" / "info.json"
    if info_path.exists():
        update_info_json(info_path)

    print(console.ok(f"\n完成! 共处理 {len(parquet_files)} 个 episodes, {total_frames} 帧"))
    print(console.ok(f"每帧添加了 {VALUE_LABEL_COLUMN} 字段，范围 [-1, 0]"))


if __name__ == "__main__":
    main()

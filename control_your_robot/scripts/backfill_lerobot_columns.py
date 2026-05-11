#!/usr/bin/env python
"""
Backfill derived columns for a local LeRobot dataset.

This script adds the following columns when they are missing:
- adv_ind: constant "positive"
- value_label: -(T - t) / T with the last frame forced to 0.0
- reward: all zeros except the final frame set to 1.0
- reward_label: -1 / T for all but the last frame, last frame is 0.0
- intervention: constant 1

It also updates:
- meta/info.json features
- meta/episodes_stats.jsonl stats for numeric columns

Existing columns are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TARGET_FEATURES = {
    "adv_ind": {
        "dtype": "string",
        "shape": [1],
        "names": ["adv_ind"],
    },
    "value_label": {
        "dtype": "float32",
        "shape": [1],
        "names": ["value_label"],
    },
    "reward": {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward"],
    },
    "reward_label": {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward_label"],
    },
    "intervention": {
        "dtype": "int64",
        "shape": [1],
        "names": ["intervention_flag"],
    },
}

COLUMN_WRITE_ORDER = ("adv_ind", "value_label", "reward", "reward_label", "intervention")
NUMERIC_TARGETS = ("value_label", "reward", "reward_label", "intervention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing columns for a local LeRobot dataset")
    parser.add_argument("--dataset-root", required=True, help="Path to the dataset root")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_files(dataset_root: Path, parquet_paths_to_backup: list[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep backups outside the dataset root so LeRobot's recursive parquet count
    # does not mistake backup files for episode parquet files.
    backup_root = dataset_root.parent / f"{dataset_root.name}__backups"
    backup_dir = backup_root / f"backfill_columns_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    for meta_name in ("info.json", "episodes_stats.jsonl"):
        src = dataset_root / "meta" / meta_name
        if src.exists():
            shutil.copy2(src, backup_dir / meta_name)

    for parquet_path in parquet_paths_to_backup:
        shutil.copy2(parquet_path, backup_dir / parquet_path.name)

    return backup_dir


def write_table_atomically(table: pa.Table, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmpwrite")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(table, tmp_path)
    tmp_path.replace(path)


def compute_missing_column_values(num_rows: int, missing_columns: set[str]) -> dict[str, np.ndarray | list[str]]:
    values: dict[str, np.ndarray | list[str]] = {}
    frame_idx = np.arange(num_rows, dtype=np.float32)

    if "adv_ind" in missing_columns:
        values["adv_ind"] = ["positive"] * num_rows

    if "value_label" in missing_columns:
        value_label = -((num_rows - frame_idx) / float(num_rows))
        value_label[-1] = 0.0
        values["value_label"] = value_label.astype(np.float32)

    if "reward" in missing_columns:
        reward = np.zeros(num_rows, dtype=np.float32)
        reward[-1] = 1.0
        values["reward"] = reward

    if "reward_label" in missing_columns:
        reward_label = np.full(num_rows, -1.0 / float(num_rows), dtype=np.float32)
        reward_label[-1] = 0.0
        values["reward_label"] = reward_label

    if "intervention" in missing_columns:
        values["intervention"] = np.ones(num_rows, dtype=np.int64)

    return values


def append_missing_columns(table: pa.Table, missing_values: dict[str, np.ndarray | list[str]]) -> pa.Table:
    updated = table
    for column_name in COLUMN_WRITE_ORDER:
        if column_name not in missing_values:
            continue

        if column_name == "adv_ind":
            array = pa.array(missing_values[column_name], type=pa.string())
        elif column_name == "intervention":
            array = pa.array(missing_values[column_name], type=pa.int64())
        else:
            array = pa.array(missing_values[column_name], type=pa.float32())

        updated = updated.append_column(column_name, array)

    return updated


def compute_numeric_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(len(values))],
    }


def get_numeric_column_values(table: pa.Table, column_name: str) -> np.ndarray:
    if column_name == "intervention":
        return np.asarray(table.column(column_name).to_pylist(), dtype=np.int64)
    return np.asarray(table.column(column_name).to_pylist(), dtype=np.float32)


def get_episode_index_from_path(parquet_path: Path) -> int:
    stem = parquet_path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected parquet filename format: {parquet_path.name}")
    return int(stem[len(prefix):])


def get_parquet_schema_info(parquet_path: Path) -> tuple[list[str], int]:
    parquet_file = pq.ParquetFile(parquet_path)
    return parquet_file.schema_arrow.names, parquet_file.metadata.num_rows


def load_columns_for_stats(parquet_path: Path, column_names: list[str]) -> dict[str, np.ndarray]:
    if not column_names:
        return {}

    table = pq.read_table(parquet_path, columns=column_names)
    return {column_name: get_numeric_column_values(table, column_name) for column_name in column_names}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    data_root = dataset_root / "data"

    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    episode_stats_rows = load_jsonl(stats_path)
    episode_stats_by_index = {row["episode_index"]: row for row in episode_stats_rows}
    parquet_paths = sorted(data_root.rglob("episode_*.parquet"))

    if len(parquet_paths) != len(episodes):
        raise ValueError(
            f"Parquet file count ({len(parquet_paths)}) does not match episodes.jsonl row count ({len(episodes)})"
        )

    features = info.setdefault("features", {})
    missing_info_features = [name for name in TARGET_FEATURES if name not in features]
    parquet_rewrites: list[tuple[Path, int, set[str]]] = []
    missing_stats_counts = {name: 0 for name in NUMERIC_TARGETS}

    print(f"dataset_root: {dataset_root}", flush=True)
    print(f"episodes: {len(episodes)}", flush=True)
    print(f"missing info features: {missing_info_features if missing_info_features else 'none'}", flush=True)

    for parquet_path in parquet_paths:
        episode_index = get_episode_index_from_path(parquet_path)

        if episode_index not in episode_stats_by_index:
            raise ValueError(f"episode_index={episode_index} missing from episodes_stats.jsonl")

        column_names, num_rows = get_parquet_schema_info(parquet_path)
        existing_columns = set(column_names)
        missing_columns = {name for name in TARGET_FEATURES if name not in existing_columns}
        if missing_columns:
            parquet_rewrites.append((parquet_path, num_rows, missing_columns))

        stats_row = episode_stats_by_index[episode_index]
        stats = stats_row.setdefault("stats", {})
        generated_values = compute_missing_column_values(num_rows, missing_columns)
        existing_columns_for_stats = [
            column_name
            for column_name in NUMERIC_TARGETS
            if column_name not in stats and column_name not in generated_values
        ]
        loaded_stats_columns = load_columns_for_stats(parquet_path, existing_columns_for_stats)

        for column_name in NUMERIC_TARGETS:
            if column_name in stats:
                continue
            if column_name in generated_values:
                values = np.asarray(generated_values[column_name])
            else:
                values = loaded_stats_columns[column_name]
            stats[column_name] = compute_numeric_stats(values)
            missing_stats_counts[column_name] += 1

    if args.dry_run:
        print(f"parquet files to rewrite: {len(parquet_rewrites)}", flush=True)
        if parquet_rewrites:
            preview = parquet_rewrites[:5]
            for parquet_path, _, missing_columns in preview:
                print(f"  {parquet_path.name}: add {sorted(missing_columns)}", flush=True)
        print(f"episodes_stats additions: {missing_stats_counts}", flush=True)
        return

    parquet_paths_to_backup = [item[0] for item in parquet_rewrites]
    if parquet_paths_to_backup or missing_info_features or any(missing_stats_counts.values()):
        backup_dir = backup_files(dataset_root, parquet_paths_to_backup)
        print(f"backup_dir: {backup_dir}", flush=True)
    else:
        print("No changes needed")
        return

    for parquet_path, num_rows, missing_columns in parquet_rewrites:
        print(f"rewrite {parquet_path.name}: add {sorted(missing_columns)}", flush=True)
        table = pq.read_table(parquet_path)
        updated_table = append_missing_columns(
            table,
            compute_missing_column_values(num_rows, missing_columns),
        )
        write_table_atomically(updated_table, parquet_path)

    for feature_name in missing_info_features:
        features[feature_name] = TARGET_FEATURES[feature_name]

    save_json(info_path, info)
    save_jsonl(stats_path, episode_stats_rows)

    print(f"rewritten parquet files: {len(parquet_rewrites)}", flush=True)
    print(f"episodes_stats additions: {missing_stats_counts}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Filter successful episodes from a local LeRobot dataset.

中文说明:
- 这是通用的 LeRobot episode 过滤脚本，主要用于仿真 rollout 数据。
- 输入数据应已经完成标签补齐，例如包含:
  - reward
  - reward_label
  - value_label
  - adv_ind
- 脚本只负责“剔除失败 episode”，不会重新计算 reward / reward_label / adv_ind。
- 默认按 `reward` 判断成功:
  - 一个 episode 内任意一帧 reward > threshold，视为成功
  - 全部 reward <= threshold，视为失败并丢弃
- 也支持按 `value_label` 判断:
  - 最后一帧 value_label == 0，视为成功
  - 最后一帧 value_label == -1，视为失败
- 输出会重写:
  - episode_index
  - frame_index
  - index
  - task_index
  - meta/info.json
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/episodes_stats.jsonl
- 输出 parquet 会保留输入 episode 的全部列，不额外增删字段。

典型用途:
1. 对 rollout 数据剔除失败轨迹，只保留成功 rollout:
   sudo HF_LEROBOT_HOME=/public/home/wangsenbao_it/litianheng/lerobot_datasets \
     /.venv/bin/python scripts/filter_success_episodes.py \
       --input-repo-id libero_10_task8_rollout_positive \
       --output-repo-id libero_10_task8_rollout_success_only \
       --overwrite

2. 如果 rollout 没有 reward，但有 value_label:
   sudo HF_LEROBOT_HOME=/public/home/wangsenbao_it/litianheng/lerobot_datasets \
     /.venv/bin/python scripts/filter_success_episodes.py \
       --input-repo-id rollout_with_value_label \
       --output-repo-id rollout_success_only \
       --criterion value_label \
       --overwrite

3. 使用绝对路径:
   sudo /.venv/bin/python scripts/filter_success_episodes.py \
       --input-root /public/home/user/lerobot_datasets/rollout \
       --output-root /public/home/user/lerobot_datasets/rollout_success_only \
       --overwrite
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")


def _hf_lerobot_home() -> Path:
    env_value = os.environ.get("HF_LEROBOT_HOME")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface" / "lerobot").resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_root(root: str | None, repo_id: str | None, *, arg_name: str) -> Path:
    if (root is None) == (repo_id is None):
        raise ValueError(f"Exactly one of --{arg_name}-root or --{arg_name}-repo-id must be provided.")
    if root is not None:
        return Path(root).expanduser().resolve()
    assert repo_id is not None
    return (_hf_lerobot_home() / repo_id).resolve()


def _discover_episode_files(root: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for fpath in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        match = PARQUET_RE.match(fpath.name)
        if match:
            files[int(match.group(1))] = fpath
    return files


def _extract_scalar(value: Any) -> float:
    if isinstance(value, dict):
        # Image-like dicts should never be passed here, but this makes failures explicit.
        raise TypeError(f"Expected scalar/list numeric value, got dict keys={list(value.keys())}")
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty scalar/list value encountered.")
    return float(arr[0])


def _column_values(table: pa.Table, column_name: str) -> list[Any]:
    if table.schema.get_field_index(column_name) < 0:
        raise KeyError(f"Missing required column `{column_name}`")
    return table.column(column_name).combine_chunks().to_pylist()


def _is_success_by_reward(table: pa.Table, threshold: float) -> bool:
    rewards = [_extract_scalar(v) for v in _column_values(table, "reward")]
    return bool(rewards) and max(rewards) > threshold


def _is_success_by_value_label(table: pa.Table, threshold: float) -> bool:
    values = _column_values(table, "value_label")
    if not values:
        return False
    final_value = _extract_scalar(values[-1])
    return abs(final_value) <= threshold


def _is_success(table: pa.Table, criterion: str, threshold: float) -> bool:
    if criterion == "reward":
        return _is_success_by_reward(table, threshold)
    if criterion == "value_label":
        return _is_success_by_value_label(table, threshold)
    if criterion == "auto":
        if table.schema.get_field_index("reward") >= 0:
            return _is_success_by_reward(table, threshold)
        return _is_success_by_value_label(table, threshold)
    raise ValueError(f"Unknown criterion: {criterion}")


def _task_map(tasks_rows: list[dict[str, Any]]) -> dict[int, str]:
    return {int(row["task_index"]): str(row["task"]) for row in tasks_rows}


def _infer_episode_tasks(table: pa.Table, tasks_map: dict[int, str], episode_row: dict[str, Any]) -> list[str]:
    if episode_row.get("tasks"):
        return [str(t) for t in episode_row["tasks"]]
    if table.schema.get_field_index("task_index") < 0:
        return []
    old_task_indices = sorted(set(np.asarray(table.column("task_index").combine_chunks()).astype(np.int64).tolist()))
    return [tasks_map.get(int(idx), f"task_{idx}") for idx in old_task_indices]


def _remap_task_index(table: pa.Table, tasks_map: dict[int, str], global_task_to_index: dict[str, int]) -> tuple[pa.Table, list[str]]:
    if table.schema.get_field_index("task_index") < 0:
        return table, []

    old_task_idx = np.asarray(table.column("task_index").combine_chunks()).astype(np.int64)
    mapped = np.zeros(table.num_rows, dtype=np.int64)
    used_tasks: list[str] = []

    for i, old_idx in enumerate(old_task_idx.tolist()):
        task = tasks_map.get(int(old_idx), f"task_{old_idx}")
        if task not in global_task_to_index:
            global_task_to_index[task] = len(global_task_to_index)
        mapped[i] = global_task_to_index[task]
        used_tasks.append(task)

    arr = pa.array(mapped, type=pa.int64())
    return _set_or_add_column(table, "task_index", arr), sorted(set(used_tasks))


def _set_or_add_column(table: pa.Table, name: str, arr: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table.append_column(name, arr)
    return table.set_column(idx, name, arr)


def _rewrite_indices(table: pa.Table, episode_index: int, global_frame_offset: int) -> pa.Table:
    n = table.num_rows
    table = _set_or_add_column(table, "episode_index", pa.array(np.full(n, episode_index, dtype=np.int64)))
    table = _set_or_add_column(table, "frame_index", pa.array(np.arange(n, dtype=np.int64)))
    table = _set_or_add_column(table, "index", pa.array(np.arange(global_frame_offset, global_frame_offset + n, dtype=np.int64)))
    return table


def _available_cpu_threads() -> int:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except Exception:
        cpu_count = os.cpu_count() or 1
    try:
        load1, _, _ = os.getloadavg()
        idle_estimate = cpu_count - int(math.floor(load1))
    except OSError:
        idle_estimate = cpu_count
    return max(1, min(cpu_count, idle_estimate))


def _resolve_num_workers(user_value: int | None, total_episodes: int) -> int:
    if total_episodes <= 0:
        return 1
    if user_value is not None:
        return max(1, min(int(user_value), total_episodes))
    return max(1, min(_available_cpu_threads(), total_episodes))


def _read_and_score(item: tuple[int, Path, dict[str, Any], str, float]) -> tuple[int, Path, dict[str, Any], pa.Table, bool]:
    episode_index, episode_file, episode_row, criterion, threshold = item
    table = pq.read_table(episode_file)
    success = _is_success(table, criterion, threshold)
    return episode_index, episode_file, episode_row, table, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter successful episodes from a LeRobot dataset.")
    parser.add_argument("--input-root", default=None, help="Input dataset root path.")
    parser.add_argument("--input-repo-id", default=None, help="Input repo id under HF_LEROBOT_HOME.")
    parser.add_argument("--output-root", default=None, help="Output dataset root path.")
    parser.add_argument("--output-repo-id", default=None, help="Output repo id under HF_LEROBOT_HOME.")
    parser.add_argument("--criterion", choices=["reward", "value_label", "auto"], default="reward")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Success threshold for reward or abs(value_label).")
    parser.add_argument("--chunks-size", type=int, default=None, help="Output chunk size. Defaults to input chunks_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel episode readers. Defaults to auto.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = _resolve_root(args.input_root, args.input_repo_id, arg_name="input")
    output_root = _resolve_root(args.output_root, args.output_repo_id, arg_name="output")

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")
    if output_root == input_root:
        raise ValueError("Input and output must be different. This script does not edit in place.")
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")

    info = _load_json(input_root / "meta" / "info.json")
    tasks_rows = _load_jsonl(input_root / "meta" / "tasks.jsonl")
    episodes_rows = _load_jsonl(input_root / "meta" / "episodes.jsonl")
    episode_stats_rows = _load_jsonl(input_root / "meta" / "episodes_stats.jsonl")
    episode_stats_map = {int(row["episode_index"]): row for row in episode_stats_rows if "episode_index" in row}
    tasks_map = _task_map(tasks_rows)
    episode_files = _discover_episode_files(input_root)

    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under: {input_root / 'data'}")

    if episodes_rows:
        source_episode_indices = [int(row["episode_index"]) for row in sorted(episodes_rows, key=lambda r: int(r["episode_index"]))]
    else:
        source_episode_indices = sorted(episode_files)

    missing = [idx for idx in source_episode_indices if idx not in episode_files]
    if missing:
        raise FileNotFoundError(f"Missing parquet files for episode indices: {missing[:20]}")

    chunks_size = int(info.get("chunks_size", 1000) if args.chunks_size is None else args.chunks_size)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    episode_row_map = {int(row["episode_index"]): row for row in episodes_rows}
    work_items = [
        (idx, episode_files[idx], episode_row_map.get(idx, {}), args.criterion, args.threshold)
        for idx in source_episode_indices
    ]
    num_workers = _resolve_num_workers(args.num_workers, len(work_items))

    global_task_to_index: dict[str, int] = {}
    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []
    kept = 0
    dropped = 0
    global_frame_offset = 0

    print(f"Input: {input_root}", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Criterion: {args.criterion}, threshold={args.threshold}", flush=True)
    print(f"Using worker threads: {num_workers}", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(_read_and_score, work_items)
        progress = tqdm.tqdm(iterator, total=len(work_items), desc="Filtering episodes", unit="ep")
        for old_idx, episode_file, episode_row, table, success in progress:
            if not success:
                dropped += 1
                progress.set_postfix_str(f"drop old_ep={old_idx}")
                continue

            new_idx = kept
            table, used_tasks = _remap_task_index(table, tasks_map, global_task_to_index)
            if not used_tasks:
                used_tasks = _infer_episode_tasks(table, tasks_map, episode_row)
                for task in used_tasks:
                    global_task_to_index.setdefault(task, len(global_task_to_index))
            table = _rewrite_indices(table, new_idx, global_frame_offset)

            out_chunk = new_idx // chunks_size
            out_file = output_root / "data" / f"chunk-{out_chunk:03d}" / f"episode_{new_idx:06d}.parquet"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_file)

            out_episodes.append(
                {
                    "episode_index": new_idx,
                    "tasks": used_tasks,
                    "length": int(table.num_rows),
                }
            )
            old_stats = dict(episode_stats_map.get(old_idx, {"stats": {}}))
            old_stats["episode_index"] = new_idx
            old_stats.setdefault("stats", {})
            out_episode_stats.append(old_stats)

            kept += 1
            global_frame_offset += table.num_rows
            progress.set_postfix_str(f"keep old_ep={old_idx}")

    out_tasks = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(global_task_to_index.items(), key=lambda item: item[1])
    ]

    out_info = dict(info)
    out_info["total_episodes"] = int(kept)
    out_info["total_frames"] = int(global_frame_offset)
    out_info["total_tasks"] = int(len(out_tasks))
    out_info["total_chunks"] = int(math.ceil(kept / chunks_size)) if kept > 0 else 0
    out_info["chunks_size"] = chunks_size
    out_info["splits"] = {"train": f"0:{kept}"}
    out_info["data_path"] = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    _write_json(output_root / "meta" / "info.json", out_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", out_tasks)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", out_episode_stats)

    print("Done.", flush=True)
    print(f"Kept successful episodes: {kept}", flush=True)
    print(f"Dropped failed episodes: {dropped}", flush=True)
    print(f"Output frames: {global_frame_offset}", flush=True)
    if kept == 0:
        print("Warning: no successful episodes were kept.", flush=True)


if __name__ == "__main__":
    main()

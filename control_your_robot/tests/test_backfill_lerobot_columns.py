import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts import backfill_lerobot_columns as backfill


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_episode_table(episode_index: int, num_rows: int) -> pa.Table:
    return pa.table(
        {
            "state": [np.asarray([float(i)], dtype=np.float32) for i in range(num_rows)],
            "actions": [np.asarray([float(i + 1)], dtype=np.float32) for i in range(num_rows)],
            "timestamp": np.asarray([i * 0.1 for i in range(num_rows)], dtype=np.float32),
            "frame_index": np.arange(num_rows, dtype=np.int64),
            "episode_index": np.full(num_rows, episode_index, dtype=np.int64),
            "index": np.arange(num_rows, dtype=np.int64),
            "task_index": np.zeros(num_rows, dtype=np.int64),
            "intervention": np.ones(num_rows, dtype=np.int64),
            "value_label": np.asarray(
                [-(num_rows - i) / num_rows for i in range(num_rows - 1)] + [0.0],
                dtype=np.float32,
            ),
        }
    )


def test_backfill_adds_reward_columns_and_meta(tmp_path, monkeypatch):
    dataset_root = tmp_path / "white_plug"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()

    pq.write_table(build_episode_table(0, 3), data_dir / "episode_000000.parquet")
    pq.write_table(build_episode_table(1, 2), data_dir / "episode_000001.parquet")

    info = {
        "fps": 10,
        "features": {
            "state": {"dtype": "float32", "shape": [1], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [1], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
            "intervention": backfill.TARGET_FEATURES["intervention"],
            "value_label": backfill.TARGET_FEATURES["value_label"],
        },
    }
    write_json(meta_dir / "info.json", info)
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["task"], "length": 3},
            {"episode_index": 1, "tasks": ["task"], "length": 2},
        ],
    )
    write_jsonl(
        meta_dir / "episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {}},
            {"episode_index": 1, "stats": {}},
        ],
    )

    monkeypatch.setattr(sys, "argv", ["backfill", "--dataset-root", str(dataset_root)])
    backfill.main()

    backup_roots = sorted(tmp_path.glob("white_plug__backups/backfill_columns_*"))
    assert len(backup_roots) == 1
    assert backup_roots[0].is_dir()

    updated_info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    assert "reward" in updated_info["features"]
    assert "reward_label" in updated_info["features"]
    assert "adv_ind" in updated_info["features"]

    first_table = pq.read_table(data_dir / "episode_000000.parquet")
    second_table = pq.read_table(data_dir / "episode_000001.parquet")
    assert {"reward", "reward_label", "adv_ind"}.issubset(first_table.column_names)
    assert {"reward", "reward_label", "adv_ind"}.issubset(second_table.column_names)
    assert first_table.column("reward").to_pylist() == [0.0, 0.0, 1.0]
    assert second_table.column("reward").to_pylist() == [0.0, 1.0]
    assert np.allclose(first_table.column("reward_label").to_pylist(), [-1.0 / 3.0, -1.0 / 3.0, 0.0])
    assert np.allclose(second_table.column("reward_label").to_pylist(), [-0.5, 0.0])
    assert first_table.column("adv_ind").to_pylist() == ["positive"] * 3

    stats_rows = [json.loads(line) for line in (meta_dir / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()]
    for row, expected_count in zip(stats_rows, [3, 2], strict=True):
        assert "reward" in row["stats"]
        assert "reward_label" in row["stats"]
        assert row["stats"]["reward"]["count"] == [expected_count]

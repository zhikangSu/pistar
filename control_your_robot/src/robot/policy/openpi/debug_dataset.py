#!/usr/bin/env python
"""Debug script to identify the List type error in the dataset."""

import sys
import traceback

# Set environment variables
import os
os.environ["HF_LEROBOT_HOME"] = "/public/home/wangsenbao_it/litianheng/lerobot_datasets"

sys.path.insert(0, '/public/home/wangsenbao_it/litianheng/pistar/src')

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    print("=" * 80)
    print("Loading dataset metadata...")
    print("=" * 80)

    meta = lerobot_dataset.LeRobotDatasetMetadata('realman_remote_teleop')
    print("Successfully loaded metadata")
    print(f"Info keys: {meta.info.keys()}")

    print("\n" + "=" * 80)
    print("Creating dataset...")
    print("=" * 80)

    dataset = lerobot_dataset.LeRobotDataset(
        'realman_remote_teleop',
        delta_timestamps={
            'actions': [t / meta.fps for t in range(10)]
        }
    )
    print("Successfully created dataset")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset columns: {dataset.hf_dataset.column_names}")

    print("\n" + "=" * 80)
    print("First sample...")
    print("=" * 80)

    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: {type(value)} - {value if not isinstance(value, (dict, bytes)) else type(value).__name__}")

except Exception as e:
    print("\n" + "=" * 80)
    print("ERROR!")
    print("=" * 80)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

    # Try to get more info about where the error comes from
    print("\n" + "=" * 80)
    print("Investigating error source...")
    print("=" * 80)

    import re
    error_str = str(e)
    if "Feature type 'List'" in error_str:
        print("The error is about a 'List' feature type.")
        print("This typically means a field is incorrectly typed as Python 'list'")
        print("instead of a HuggingFace supported type like 'Sequence'.")

    # Try to check if there's a schema issue
    try:
        from datasets import Dataset, DatasetDict, Features, Value, Sequence, Array2D
        print("\nAttempting to manually create features...")

        features = {
            "state": Sequence(Value("float32"), length=14),
            "actions": Sequence(Value("float32"), length=14),
            "image": None,  # Image type
            "left_wrist_image": None,
            "right_wrist_image": None,
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
        print("Features definition looks OK")
    except Exception as e2:
        print(f"Error creating features: {e2}")

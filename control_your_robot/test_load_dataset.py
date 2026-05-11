import sys
sys.path.append("./")
import os
from src.robot.data.collect_lerobot import CollectLeRobot

repo_id = "piper_plug_task"
output_dir = "/home/chaihoa/project_wang/dataset/lerobot"

print(f"Testing loading dataset from {os.path.join(output_dir, repo_id)}")

if not os.path.exists(os.path.join(output_dir, repo_id)):
    print("Dataset directory not found!")
else:
    try:
        collector = CollectLeRobot(
            repo_id=repo_id,
            output_dir=output_dir,
            task_name="test",
        )
        # This triggers _create_dataset which loads the dataset
        collector._create_dataset()
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()

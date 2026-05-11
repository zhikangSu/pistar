import inspect
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print("="*20 + " __init__ " + "="*20)
try:
    print(inspect.getsource(LeRobotDataset.__init__))
except Exception as e:
    print(e)

print("\n" + "="*20 + " load_hf_dataset " + "="*20)
try:
    print(inspect.getsource(LeRobotDataset.load_hf_dataset))
except Exception as e:
    print(e)

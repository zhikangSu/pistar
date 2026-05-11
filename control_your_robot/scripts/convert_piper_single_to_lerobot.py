"""
直接将 Piper 单臂 HDF5 数据转换为 LeRobot 格式
支持单臂 7 维数据（6关节 + 1夹爪）

使用方法:
python scripts/convert_piper_single_to_lerobot.py \
    --raw-dir "./datasets/Plug the black plug into the three-hole socket/" \
    --repo-id piper_plug_task \
    --fps 10
"""

import sys
sys.path.append("./")

import shutil
from pathlib import Path
import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import json
import os
import fnmatch
import argparse
import cv2

def get_hdf5_files(raw_dir: Path) -> list[Path]:
    """获取所有 HDF5 文件"""
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))
    # 按文件名排序
    hdf5_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    return hdf5_files

def load_instructions(raw_dir: Path, task_name: str = None) -> list[str]:
    """加载任务指令"""
    # 首先尝试从 config.json 读取 task_name
    config_path = raw_dir / "config.json"
    if config_path.exists() and task_name is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
            task_name = config.get('task_name', 'default_task')

    # 查找指令文件
    inst_path = Path(f"task_instructions/{task_name}.json")
    if not inst_path.exists():
        print(f"⚠ 警告: 未找到指令文件 {inst_path}, 使用默认指令")
        return [task_name if task_name else "complete the task"]

    with open(inst_path, 'r') as f:
        instruction_dict = json.load(f)
        return instruction_dict['instructions']

def decode_image(img_data):
    """解码图像数据"""
    if isinstance(img_data, np.ndarray) and img_data.ndim == 3:
        # 已经是解码后的图像 (H, W, 3)
        return img_data

    if isinstance(img_data, (bytes, bytearray)):
        # JPEG 编码的数据
        data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise ValueError(f"无法解码图像数据，类型: {type(img_data)}")

def load_raw_episode_data(ep_path: Path):
    """加载原始 episode 数据"""
    with h5py.File(ep_path, "r") as ep:
        # 读取关节和夹爪数据
        joint = np.array(ep["left_arm"]["joint"][:])  # [T, 6]
        gripper = np.array(ep["left_arm"]["gripper"][:])  # [T, 1] 或 [T]

        # 确保 gripper 是 2D
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)

        # 合并为状态 [T, 7]
        state = np.concatenate([joint, gripper], axis=1).astype(np.float32)

        # Action 就是下一个时间步的状态
        action = np.zeros_like(state)
        action[:-1] = state[1:]  # 将下一帧作为当前的 action
        action[-1] = state[-1]   # 最后一帧保持不变

        # 读取图像
        imgs_per_cam = {}

        # cam_head
        if "cam_head" in ep:
            cam_head_data = ep["cam_head"]["color"][:]
            imgs_per_cam["cam_head"] = np.array([decode_image(img) for img in cam_head_data])

        # cam_wrist
        if "cam_wrist" in ep:
            cam_wrist_data = ep["cam_wrist"]["color"][:]
            imgs_per_cam["cam_wrist"] = np.array([decode_image(img) for img in cam_wrist_data])

        return imgs_per_cam, torch.from_numpy(state), torch.from_numpy(action)

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    instructions: list[str],
    episodes: list[int] | None = None,
):
    """填充数据集"""
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes, desc="转换 episodes"):
        ep_path = hdf5_files[ep_idx]

        print(f"\n处理: {ep_path}")
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # 随机选择一个任务指令
        instruction = np.random.choice(instructions)

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            # 添加图像
            if "cam_head" in imgs_per_cam:
                frame["observation.images.cam_high"] = imgs_per_cam["cam_head"][i]

            if "cam_wrist" in imgs_per_cam:
                frame["observation.images.cam_wrist"] = imgs_per_cam["cam_wrist"][i]

            dataset.add_frame(frame)

        dataset.save_episode(task=instruction)
        print(f"✓ Episode {ep_idx} 完成 ({num_frames} 帧)")

    return dataset

def convert_piper_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    fps: int = 10,
    task_name: str = None,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
):
    """主转换函数"""
    print("=" * 60)
    print("Piper 单臂数据转换为 LeRobot 格式")
    print("=" * 60)
    print(f"原始数据目录: {raw_dir}")
    print(f"LeRobot repo_id: {repo_id}")
    print(f"采集频率: {fps} Hz")
    print("=" * 60)

    # 清理已有数据集
    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"⚠ 清理已有数据集: {output_path}")
        shutil.rmtree(output_path)

    # 获取 HDF5 文件
    hdf5_files = get_hdf5_files(raw_dir)
    print(f"\n找到 {len(hdf5_files)} 个 HDF5 文件:")
    for f in hdf5_files[:5]:  # 只显示前5个
        print(f"  - {f}")
    if len(hdf5_files) > 5:
        print(f"  ... 还有 {len(hdf5_files) - 5} 个文件")

    if not hdf5_files:
        raise ValueError(f"未在 {raw_dir} 中找到 HDF5 文件！")

    # 加载指令
    instructions = load_instructions(raw_dir, task_name)
    print(f"\n任务指令 ({len(instructions)} 条):")
    for inst in instructions:
        print(f"  - {inst}")

    # 创建 LeRobot 数据集
    print("\n创建 LeRobot 数据集...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="piper",
        fps=fps,
        features={
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 填充数据
    print("\n开始转换数据...")
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        instructions,
        episodes=episodes,
    )

    # 合并数据集
    print("\n合并数据集...")
    dataset.consolidate()

    print("\n✅ 转换完成！")
    print(f"数据集保存在: {output_path}")

    # 推送到 Hub（可选）
    if push_to_hub:
        print("\n上传到 Hugging Face Hub...")
        dataset.push_to_hub()
        print("✓ 上传完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 Piper 单臂 HDF5 数据转换为 LeRobot 格式')
    parser.add_argument('--raw-dir', type=str, required=True,
                        help='原始 HDF5 数据目录')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='LeRobot 数据集 repo ID')
    parser.add_argument('--fps', type=int, default=10,
                        help='数据采集频率 (默认: 10 Hz)')
    parser.add_argument('--task-name', type=str, default=None,
                        help='任务名称（从 task_instructions/{task_name}.json 读取）')
    parser.add_argument('--push-to-hub', action='store_true',
                        help='是否上传到 Hugging Face Hub')

    args = parser.parse_args()

    convert_piper_to_lerobot(
        raw_dir=Path(args.raw_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        task_name=args.task_name,
        push_to_hub=args.push_to_hub,
    )

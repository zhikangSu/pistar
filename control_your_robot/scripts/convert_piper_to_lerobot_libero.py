"""
将 Piper 单臂 HDF5 数据转换为 LeRobot Libero 格式（单臂7维）
避免14维填充，直接使用原生单臂格式

使用方法:
python scripts/convert_piper_to_lerobot_libero.py \
    --raw-dir "./datasets/Plug the black plug into the three-hole socket/" \
    --repo-id piper_plug_libero \
    --task "Plug the black plug into the three-hole socket" \
    --output-dir "./datasets/lerobot_datasets"
"""

import sys
sys.path.append("./")

import shutil
from pathlib import Path
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import json
import os
import fnmatch
import tyro
import cv2

def get_hdf5_files(raw_dir: Path) -> list[Path]:
    """获取所有 HDF5 文件"""
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))
    hdf5_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    return hdf5_files

def decode_image(img_data):
    """解码图像数据"""
    if isinstance(img_data, np.ndarray) and img_data.ndim == 3:
        return img_data

    if isinstance(img_data, (bytes, bytearray)):
        data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise ValueError(f"无法解码图像数据")

def load_episode_data(ep_path: Path):
    """加载单个 episode 数据"""
    with h5py.File(ep_path, "r") as ep:
        # 读取关节和夹爪数据
        joint = np.array(ep["left_arm"]["joint"][:])  # [T, 6]
        gripper = np.array(ep["left_arm"]["gripper"][:])  # [T, 1] or [T]

        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)

        # 单臂格式：7维 [6关节 + 1夹爪]
        state = np.concatenate([joint, gripper], axis=1).astype(np.float32)

        # Action = 下一帧的状态
        action = np.zeros_like(state)
        action[:-1] = state[1:]
        action[-1] = state[-1]

        # 读取图像（只读取实际存在的）
        imgs = {}

        if "cam_head" in ep:
            cam_head_data = ep["cam_head"]["color"][:]
            imgs["cam_high"] = np.array([decode_image(img) for img in cam_head_data])

        if "cam_wrist" in ep:
            cam_wrist_data = ep["cam_wrist"]["color"][:]
            imgs["wrist_image"] = np.array([decode_image(img) for img in cam_wrist_data])

        return imgs, torch.from_numpy(state), torch.from_numpy(action)

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
):
    """填充数据集"""
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes, desc="转换 episodes"):
        ep_path = hdf5_files[ep_idx]

        print(f"\n处理: {ep_path}")
        imgs, state, action = load_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "state": state[i],
                "actions": action[i],
            }

            # 添加图像（使用 Libero 的字段名）
            if "cam_high" in imgs:
                frame["image"] = imgs["cam_high"][i]

            if "wrist_image" in imgs:
                frame["wrist_image"] = imgs["wrist_image"][i]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()
        print(f"✓ Episode {ep_idx} 完成 ({num_frames} 帧)")

    return dataset

def convert_piper_to_libero(
    raw_dir: Path,
    repo_id: str,
    task: str = "complete the task",
    fps: int = 10,
    output_dir: str | None = None,  # 添加自定义输出目录参数
    push_to_hub: bool = False,
):
    """主转换函数"""
    print("=" * 60)
    print("Piper 单臂数据转换为 LeRobot Libero 格式（7维）")
    print("=" * 60)
    print(f"原始数据目录: {raw_dir}")
    print(f"LeRobot repo_id: {repo_id}")
    print(f"采集频率: {fps} Hz")
    print("=" * 60)

    # 确定输出路径
    if output_dir:
        output_path = Path(output_dir) / repo_id
    else:
        output_path = LEROBOT_HOME / repo_id

    print(f"输出目录: {output_path}")

    # 清理已有数据集
    if output_path.exists():
        print(f"⚠ 清理已有数据集: {output_path}")
        shutil.rmtree(output_path)

    # 获取 HDF5 文件
    hdf5_files = get_hdf5_files(raw_dir)
    print(f"\n找到 {len(hdf5_files)} 个 HDF5 文件")

    if not hdf5_files:
        raise ValueError(f"未在 {raw_dir} 中找到 HDF5 文件！")

    # 创建 LeRobot 数据集（Libero 格式：7维单臂）
    print("\n创建 LeRobot 数据集（Libero 单臂格式）...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path if output_dir else None,  # 指定自定义路径
        robot_type="piper",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # 🎯 单臂7维！
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 🎯 单臂7维！
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 填充数据
    print("\n开始转换数据...")
    dataset = populate_dataset(dataset, hdf5_files, task)

    print("\n✅ 转换完成！")
    print(f"数据集保存在: {output_path}")

    if push_to_hub:
        print("\n上传到 Hugging Face Hub...")
        dataset.push_to_hub()
        print("✓ 上传完成")

if __name__ == "__main__":
    tyro.cli(convert_piper_to_libero)

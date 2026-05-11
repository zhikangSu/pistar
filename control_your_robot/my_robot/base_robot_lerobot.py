"""
机器人基类 - 直接生成 LeRobot 格式
"""
import sys
sys.path.append("./")

import os
from typing import Dict, Any, List
import numpy as np
import time
import json
from pathlib import Path

from robot.data.collect_lerobot import CollectLeRobot
from robot.utils.base.data_handler import debug_print

import cv2

# add your controller/sensor type here
ALLOW_TYPES = ["arm", "mobile", "image", "tactile", "teleop"]


class RobotLeRobot:
    """使用 LeRobot 格式的机器人基类"""

    def __init__(
        self,
        repo_id: str,
        output_dir: str,
        task_name: str,
        fps: int = 10,
        robot_type: str = "piper",
        state_dim: int = 7,
        action_dim: int = 7,
        image_size: tuple = (480, 640),
        camera_keys: dict = None,
        move_check: bool = True,
        tolerance: float = 0.0001,
    ) -> None:
        """
        Args:
            repo_id: LeRobot 数据集 ID
            output_dir: 输出目录
            task_name: 任务名称
            fps: 采集频率
            robot_type: 机器人类型
            state_dim: 状态维度（7=单臂，14=双臂）
            action_dim: 动作维度
            image_size: 图像尺寸 (height, width)
            camera_keys: 相机映射，例如 {"cam_head": "image", "cam_wrist": "wrist_image"}
            move_check: 是否检查机器人移动
            tolerance: 移动检测容差
        """
        self.name = "robot_lerobot"
        self.controllers = {}
        self.sensors = {}

        # 创建 LeRobot 收集器
        self.collection = CollectLeRobot(
            repo_id=repo_id,
            output_dir=output_dir,
            task_name=task_name,
            fps=fps,
            robot_type=robot_type,
            state_dim=state_dim,
            action_dim=action_dim,
            image_size=image_size,
            camera_keys=camera_keys or {},
            move_check=move_check,
            tolerance=tolerance,
        )

    def set_up(self):
        """初始化机器人（子类需要实现具体逻辑）"""
        for controller_type in self.controllers.keys():
            if controller_type not in ALLOW_TYPES:
                debug_print(
                    self.name,
                    f"建议将控制器类型设置为标准格式。\n当前类型: {controller_type}\n允许类型: {ALLOW_TYPES}",
                    "WARNING",
                )

        for sensor_type in self.sensors.keys():
            if sensor_type not in ALLOW_TYPES:
                debug_print(
                    self.name,
                    f"建议将传感器类型设置为标准格式。\n当前类型: {sensor_type}\n允许类型: {ALLOW_TYPES}",
                    "WARNING",
                )

    def set_collect_type(self, INFO_NAMES: Dict[str, Any]):
        """设置需要收集的数据类型"""
        for key, value in INFO_NAMES.items():
            if key in self.controllers:
                for controller in self.controllers[key].values():
                    controller.set_collect_info(value)
            if key in self.sensors:
                for sensor in self.sensors[key].values():
                    sensor.set_collect_info(value)

    def get(self):
        """获取当前机器人状态"""
        controller_data = {}
        sensor_data = {}

        if self.controllers is not None:
            for type_name, controller_type in self.controllers.items():
                for controller_name, controller in controller_type.items():
                    controller_data[controller_name] = controller.get()

        if self.sensors is not None:
            for type_name, sensor_type in self.sensors.items():
                for sensor_name, sensor in sensor_type.items():
                    sensor_data[sensor_name] = sensor.get()

        return [controller_data, sensor_data]

    def collect(self, data):
        """收集一帧数据"""
        self.collection.collect(data[0], data[1])

    def finish(self):
        """保存当前 episode"""
        self.collection.save_episode()
        debug_print(self.name, f"Episode 保存成功！路径: {self.collection.get_dataset_path()}", "INFO")

    def move(self, move_data, key_banned=None):
        """移动机器人"""
        if move_data is None:
            return
        for controller_type_name, controller_type in move_data.items():
            for controller_name, controller_action in controller_type.items():
                if key_banned is None:
                    self.controllers[controller_type_name][controller_name].move(
                        controller_action, is_delta=False
                    )
                else:
                    controller_action = remove_duplicate_keys(controller_action, key_banned)
                    self.controllers[controller_type_name][controller_name].move(
                        controller_action, is_delta=False
                    )

    def is_start(self):
        """判断是否开始（子类可重写）"""
        debug_print(self.name, "使用默认 is_start()，始终返回 True", "DEBUG")
        return True

    def reset(self):
        """重置机器人（子类需要实现）"""
        debug_print(self.name, "使用默认 reset()，不执行任何操作", "DEBUG")
        return True


def remove_duplicate_keys(source_dict, keys_to_remove):
    """移除字典中的指定键"""
    return {k: v for k, v in source_dict.items() if k not in keys_to_remove}

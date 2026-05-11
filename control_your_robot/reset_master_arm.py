#!/usr/bin/env python3
"""
主臂复位脚本

用途：将主臂移动到起始位置
"""
import sys
sys.path.append("./")

from robot.controller.Piper_controller import PiperController
import numpy as np
import time

# 主臂起始位置（弧度）
START_POSITION_ANGLE_MASTER_ARM = [
    0,        # Joint 1
    -0.4208,  # Joint 2
    0.0324,   # Joint 3
    0.0780,   # Joint 4
    0.3558,   # Joint 5
    0.0078,   # Joint 6
]

print("=" * 60)
print("主臂复位脚本")
print("=" * 60)

# 初始化主臂
print("\n[1/4] 初始化主臂 (can0)...")
master_arm = PiperController("master_arm")
master_arm.set_up("can0", use_id_offset=False, force_motion_output=True)
print("✓ 主臂初始化完成")

# 确保退出MIT模式
print("\n[2/4] 确保退出MIT模式...")
master_arm.disable_drag_teach_mode()
time.sleep(0.5)  # 等待模式切换和机械臂激活完成
print("✓ 已切换到位置控制模式")

# 复位到起始位置
print("\n[3/4] 移动到起始位置...")
print(f"  目标位置: {START_POSITION_ANGLE_MASTER_ARM}")
master_arm.reset(np.array(START_POSITION_ANGLE_MASTER_ARM), speed=20)
print("  正在移动...")
time.sleep(3)  # 等待移动完成

# 验证位置
print("\n[4/4] 验证位置...")
current_state = master_arm.get_state()
current_joint = current_state["joint"]
print(f"  当前位置: {[f'{j:.4f}' for j in current_joint]}")

# 计算误差
error = np.abs(current_joint - np.array(START_POSITION_ANGLE_MASTER_ARM))
max_error = np.max(error)
print(f"  最大误差: {max_error:.4f} 弧度 ({np.degrees(max_error):.2f}°)")

if max_error < 0.01:  # 误差小于0.01弧度（约0.57度）
    print("✓ 复位成功！")
else:
    print("⚠ 复位完成，但误差较大")

print("\n" + "=" * 60)
print("完成！主臂已复位到起始位置")
print("=" * 60)

#!/usr/bin/env python3
"""
测试主臂拖动示教模式

用途：验证拖动示教模式是否正确工作
"""
import sys
sys.path.append("./")

from robot.controller.Piper_controller import PiperController
import time
import numpy as np

print("=" * 60)
print("拖动示教模式测试")
print("=" * 60)

# 初始化主臂
print("\n[1/4] 初始化主臂 (can0)...")
master_arm = PiperController("master_arm")
master_arm.set_up("can0", use_id_offset=False)
print("✓ 主臂初始化完成")

# 移动到一个位置
print("\n[2/4] 移动到测试位置...")
test_position = np.array([0.0, -0.4, 0.0, 0.0, 0.3, 0.0])
master_arm.set_joint(test_position)
time.sleep(2)
print("✓ 到达测试位置")

# 启用拖动示教模式
print("\n[3/4] 启用拖动示教模式...")
print("⚠️  请准备手动移动主臂！")
time.sleep(1)

master_arm.enable_drag_teach_mode()
print("✓ 拖动示教模式已启用")
print("\n" + "="*60)
print("现在请尝试手动移动主臂（测试10秒）")
print("如果能自由移动（无阻力），说明拖动示教模式正常工作")
print("如果仍然有阻力，说明拖动示教模式未生效")
print("="*60)

# 等待10秒，让用户测试
for i in range(10, 0, -1):
    print(f"剩余 {i} 秒...")
    time.sleep(1)

# 禁用拖动示教模式
print("\n[4/4] 禁用拖动示教模式，恢复位置控制...")
master_arm.disable_drag_teach_mode()
time.sleep(0.5)
print("✓ 已恢复位置控制")

# 验证：移动回原位
print("\n验证位置控制是否恢复...")
master_arm.set_joint(test_position)
time.sleep(2)
print("✓ 位置控制正常")

print("\n" + "="*60)
print("测试完成！")
print("\n结果解读：")
print("  - 如果在拖动示教模式下能自由移动 → 功能正常")
print("  - 如果仍然有阻力 → 可能的原因：")
print("    1. MotionCtrl_1 命令发送失败")
print("    2. 机械臂固件不支持此功能")
print("    3. 需要额外的配置步骤")
print("="*60)

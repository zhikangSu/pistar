#!/usr/bin/env python3
"""
DAgger 基础测试脚本 - 诊断机械臂控制问题
"""
import sys
sys.path.append("./")

from my_robot.piper_dagger import PiperDAgger
import time
import numpy as np

print("=" * 60)
print("DAgger 基础测试")
print("=" * 60)

# 初始化机器人
print("\n[1/5] 初始化机器人...")
robot = PiperDAgger()
robot.set_up()
print("✓ 初始化完成")

# 测试1：读取当前状态
print("\n[2/5] 测试读取状态...")
try:
    data = robot.get()
    print("✓ 成功读取数据")
    print(f"  从臂关节: {data[0]['left_arm']['joint']}")
    print(f"  主臂关节: {data[0]['right_arm']['joint']}")
except Exception as e:
    print(f"✗ 读取失败: {e}")
    exit(1)

# 测试2：测试从臂移动
print("\n[3/5] 测试从臂移动...")
print("  从臂将移动关节1 (+0.1弧度)...")
try:
    current_state = robot.controllers["arm"]["left_arm"].get_state()
    current_joint = current_state["joint"]
    print(f"  当前位置: {current_joint}")

    # 小幅度移动
    target_joint = current_joint.copy()
    target_joint[0] += 0.1

    move_data = {
        "joint": target_joint.tolist(),
        "gripper": current_state["gripper"]
    }

    robot.move_follower(move_data)
    print("  已发送移动指令，等待3秒...")
    time.sleep(3)

    # 验证是否移动
    new_state = robot.controllers["arm"]["left_arm"].get_state()
    new_joint = new_state["joint"]
    movement = abs(new_joint[0] - current_joint[0])

    print(f"  移动后位置: {new_joint}")
    print(f"  实际移动量: {movement:.4f} 弧度 ({np.degrees(movement):.2f}°)")

    if movement > 0.05:
        print("✓ 从臂移动正常")
    else:
        print("✗ 从臂未移动或移动量过小")

except Exception as e:
    print(f"✗ 从臂移动失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3：测试主臂移动
print("\n[4/5] 测试主臂移动...")
print("  主臂将移动关节1 (+0.1弧度)...")
try:
    current_state = robot.controllers["arm"]["right_arm"].get_state()
    current_joint = current_state["joint"]
    print(f"  当前位置: {current_joint}")

    # 小幅度移动
    target_joint = current_joint.copy()
    target_joint[0] += 0.1

    robot.controllers["arm"]["right_arm"].set_joint(target_joint)
    robot.controllers["arm"]["right_arm"].set_gripper(current_state["gripper"])
    print("  已发送移动指令，等待3秒...")
    time.sleep(3)

    # 验证是否移动
    new_state = robot.controllers["arm"]["right_arm"].get_state()
    new_joint = new_state["joint"]
    movement = abs(new_joint[0] - current_joint[0])

    print(f"  移动后位置: {new_joint}")
    print(f"  实际移动量: {movement:.4f} 弧度 ({np.degrees(movement):.2f}°)")

    if movement > 0.05:
        print("✓ 主臂移动正常")
    else:
        print("✗ 主臂未移动或移动量过小")

except Exception as e:
    print(f"✗ 主臂移动失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4：测试拖动示教模式
print("\n[5/5] 测试拖动示教模式...")
print("  启用主臂拖动示教模式...")
try:
    robot.enable_master_drag_mode()
    print("✓ 已启用拖动示教模式")
    print("  请尝试手动移动主臂（5秒）...")
    time.sleep(5)

    print("  退出拖动示教模式...")
    robot.disable_master_drag_mode()
    print("✓ 已退出拖动示教模式")

except Exception as e:
    print(f"✗ 拖动示教模式测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

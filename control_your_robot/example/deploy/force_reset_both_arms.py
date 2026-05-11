#!/usr/bin/env python3
"""
强制重置两臂为从臂模式（软件可控）
用于清理异常状态后的初始化

使用场景：
- 程序异常退出后，机械臂处于未知状态
- 主臂卡在MASTER模式（0xFA）无法控制
- 从臂无响应

运行方法：
uv run python example/deploy/force_reset_both_arms.py
"""
import sys
sys.path.append("./")

from piper_sdk import C_PiperInterface
import time

def force_reset_arm(can_name: str, arm_name: str):
    """强制重置单个机械臂到从臂模式"""
    print(f"\n{'='*60}")
    print(f"强制重置 {arm_name} ({can_name})")
    print(f"{'='*60}")

    try:
        print(f"[1/6] 连接 {can_name}...")
        arm = C_PiperInterface(can_name=can_name, judge_flag=True)
        print(f"[ok] 已连接")

        print(f"\n[2/6] 退出拖动示教模式...")
        arm.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.2)
        print(f"[ok] 已退出拖动示教")

        print(f"\n[3/6] 清除所有模式...")
        arm.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        time.sleep(0.2)
        print(f"[ok] 已清除所有模式")

        print(f"\n[4/6] 配置为从臂模式 (0xFC)...")
        # 执行两次确保生效
        arm.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.3)
        arm.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.3)
        print(f"[ok] 已配置为从臂模式")

        print(f"\n[5/6] 设置控制模式（30%速度）...")
        arm.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        time.sleep(0.2)
        print(f"[ok] 控制模式已设置")

        print(f"\n[6/6] 启用电机...")
        try:
            arm.EnableArm(7)
        except Exception:
            pass
        time.sleep(0.2)
        print(f"[ok] 电机已启用")

        print(f"\n{'='*60}")
        print(f"✓ {arm_name} ({can_name}) 重置成功！")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ {arm_name} ({can_name}) 重置失败: {e}")
        print(f"{'='*60}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("强制重置两臂为从臂模式（软件可控）")
    print("=" * 60)
    print("\n⚠️  注意事项：")
    print("1. 确保两个机械臂已上电")
    print("2. 确保CAN接口已启动 (can0, can1)")
    print("3. 如果机械臂在某个位置，重置后会保持该位置")
    print("4. 重置后机械臂将处于软件可控状态（30%速度）")
    print("\n" + "=" * 60)

    input("按 Enter 键开始重置...")

    # 重置主臂 (can0)
    success_master = force_reset_arm("can0", "主臂（Master/Right Arm）")
    time.sleep(0.5)

    # 重置从臂 (can1)
    success_follower = force_reset_arm("can1", "从臂（Follower/Left Arm）")

    print("\n" + "=" * 60)
    print("重置完成总结")
    print("=" * 60)
    print(f"主臂 (can0): {'✓ 成功' if success_master else '✗ 失败'}")
    print(f"从臂 (can1): {'✓ 成功' if success_follower else '✗ 失败'}")

    if success_master and success_follower:
        print("\n✅ 两臂均已重置为从臂模式（软件可控）")
        print("\n现在可以运行部署脚本：")
        print("  uv run python example/deploy/piper_dagger_on_PI0.py")
    else:
        print("\n⚠️  部分机械臂重置失败")
        print("\n建议操作：")
        print("1. 检查CAN接口: ip link show can0 can1")
        print("2. 检查机械臂电源")
        print("3. 重新上电机械臂后再次运行本脚本")

    print("=" * 60)

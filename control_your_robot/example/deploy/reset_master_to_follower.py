"""
强制将主臂从主臂模式（0xFA）切换回从臂模式（0xFC）

使用方法:
uv run python example/deploy/reset_master_to_follower.py
"""
import sys
sys.path.append("./")

from piper_sdk import C_PiperInterface_V2
import time

if __name__ == "__main__":
    print("=" * 60)
    print("强制重置主臂为从臂模式")
    print("=" * 60)

    # 连接主臂 (can0)
    print("\n[1/4] 连接主臂 (can0)...")
    master = C_PiperInterface_V2("can0")
    master.ConnectPort()
    time.sleep(0.2)
    print("[ok] 已连接")

    # 退出拖动示教模式
    print("\n[2/4] 退出拖动示教模式...")
    for i in range(3):
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # 退出拖动示教
        time.sleep(0.1)
    print("[ok] 已退出拖动示教模式")

    # 配置为从臂模式
    print("\n[3/4] 配置为从臂模式 (0xFC)...")
    for i in range(3):
        master.MasterSlaveConfig(0xFC, 0, 0, 0)  # 从臂模式
        time.sleep(0.2)
    print("[ok] 已配置为从臂模式")

    # 启用电机
    print("\n[4/4] 启用电机...")
    try:
        master.EnableArm(7)
        time.sleep(0.5)
        print("[ok] 电机已启用")
    except Exception as e:
        print(f"[warn] 电机启用失败: {e}")
        print("       如果还是失败，可能需要重新上电")

    print("\n" + "=" * 60)
    print("重置完成！")
    print("现在可以运行遥操作脚本了")
    print("=" * 60)

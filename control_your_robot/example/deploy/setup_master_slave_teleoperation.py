#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
设置主从遥操作模式（反向配置）
- can0 (你想拖动的臂): 0xFA (主动臂，可拖动)
- can1 (应该跟随的臂): 0xFC (从动臂，跟随主臂)

使用方法:
1. 运行此脚本设置主从模式
2. 按照提示重新上电
3. 然后运行数据采集脚本

uv run python example/deploy/setup_master_slave_teleoperation.py
"""
import sys
sys.path.append("./")

from piper_sdk import C_PiperInterface
import time

if __name__ == "__main__":
    print("=" * 60)
    print("设置主从遥操作模式（反向配置）")
    print("=" * 60)

    # 设置 can0 为主动臂（可拖动）
    print("\n[1/2] 设置 can0 为主动臂 (0xFA) - 可拖动...")
    try:
        master = C_PiperInterface(can_name='can0', judge_flag=True)
        master.MasterSlaveConfig(0xFA, 0, 0, 0)
        time.sleep(0.5)
        print("[ok] can0 已设置为主动臂 (0xFA)")
    except Exception as e:
        print(f"[error] can0 设置失败: {e}")

    # 设置 can1 为从动臂（跟随）
    print("\n[2/2] 设置 can1 为从动臂 (0xFC) - 跟随...")
    try:
        slave = C_PiperInterface(can_name='can1', judge_flag=True)
        slave.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.5)
        print("[ok] can1 已设置为从动臂 (0xFC)")
    except Exception as e:
        print(f"[error] can1 设置失败: {e}")

    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)
    print("\n重要提示：")
    print("1. 断开两个机械臂的电源")
    print("2. 先给从动臂 (can1) 上电")
    print("3. 再给主动臂 (can0) 上电")
    print("4. 等待几秒钟")
    print("5. 现在拖动 can0，can1 会跟随")
    print("\n如果力反馈太大，可以调整主从联动参数")
    print("=" * 60)

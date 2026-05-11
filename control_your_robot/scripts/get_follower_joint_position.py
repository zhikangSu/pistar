import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _add_local_sdk_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sdk_root = repo_root / "src" / "robot" / "piper_sdk"
    for path in (repo_root, sdk_root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


try:
    from piper_sdk import C_PiperInterface_V2
except ModuleNotFoundError:
    _add_local_sdk_to_path()
    from piper_sdk import C_PiperInterface_V2


def format_joint_deg(joint_rad: np.ndarray) -> list[float]:
    return np.degrees(joint_rad).round(3).tolist()


def read_once(can_name: str, timeout: float, judge_flag: bool) -> None:
    arm = C_PiperInterface_V2(can_name=can_name, judge_flag=judge_flag)
    arm.ConnectPort()

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            joint_msg = arm.GetArmJointMsgs()
            if joint_msg.Hz > 0:
                joint_deg = np.array(
                    [
                        joint_msg.joint_state.joint_1,
                        joint_msg.joint_state.joint_2,
                        joint_msg.joint_state.joint_3,
                        joint_msg.joint_state.joint_4,
                        joint_msg.joint_state.joint_5,
                        joint_msg.joint_state.joint_6,
                    ],
                    dtype=float,
                ) * 0.001
                joint_rad = np.deg2rad(joint_deg)

                print(f"CAN: {can_name}")
                print(f"joint_hz: {round(float(joint_msg.Hz), 3)}")
                print(f"joint_rad: {joint_rad.round(6).tolist()}")
                print(f"joint_deg: {joint_deg.round(3).tolist()}")

                gripper_msg = arm.GetArmGripperMsgs()
                if gripper_msg.Hz > 0:
                    gripper = gripper_msg.gripper_state.grippers_angle * 0.001 / 70
                    print(f"gripper: {round(float(gripper), 6)}")
                return
            time.sleep(0.02)
    finally:
        arm.DisconnectPort()

    raise TimeoutError(f"在 {timeout:.1f}s 内未收到 {can_name} 的关节反馈，请检查 CAN 和机械臂上电状态")


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 Piper SDK 读取从臂当前关节角度")
    parser.add_argument("--can", default="can1", help="从臂对应的 CAN 口，默认 can1")
    parser.add_argument("--timeout", type=float, default=3.0, help="等待反馈超时时间，默认 3 秒")
    parser.add_argument(
        "--no-judge-flag",
        dest="judge_flag",
        action="store_false",
        help="如果使用的是非官方 CAN 模块（例如 PCIe/串口转 CAN），请加这个参数",
    )
    parser.set_defaults(judge_flag=True)
    args = parser.parse_args()
    try:
        read_once(args.can, args.timeout, args.judge_flag)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

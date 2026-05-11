import sys
sys.path.append("./")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from my_robot.agilex_piper_single_base import PiperSingle

OPENPI_ROOT = Path(__file__).resolve().parents[2] / "src" / "robot" / "policy" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from robot.utils.base.data_handler import is_enter_pressed


JOINT_LIMITS_RAD = [
    (math.radians(-150), math.radians(150)),
    (math.radians(0), math.radians(180)),
    (math.radians(-170), math.radians(0)),
    (math.radians(-100), math.radians(100)),
    (math.radians(-70), math.radians(70)),
    (math.radians(-120), math.radians(120)),
]
GRIPPER_LIMIT = (0.0, 1.0)


def _load_task_instructions(task_name):
    root_dir = Path(__file__).resolve().parents[2]
    possible_paths = [
        root_dir / "task_instructions" / f"{task_name}.json",
        root_dir / "datasets" / "instructions" / f"{task_name}.json",
        Path("task_instructions") / f"{task_name}.json",
    ]

    for path in possible_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file_obj:
            instruction_dict = json.load(file_obj)
        instructions = instruction_dict.get("instructions", [])
        if instructions:
            return instructions

    print(f"Warning: instruction file not found for task '{task_name}', using task name as prompt.")
    return [task_name]


def _choose_instruction(task_name, fixed_instruction=None):
    if fixed_instruction:
        return fixed_instruction
    instructions = _load_task_instructions(task_name)
    return str(np.random.choice(instructions))


def _get_color_image(sensors, cam_keys):
    for cam_key in cam_keys:
        cam_data = sensors.get(cam_key)
        if cam_data is not None and "color" in cam_data:
            return cam_data["color"]
    return None


def _preprocess_image(image):
    resized = image_tools.resize_with_pad(np.asarray(image), 224, 224)
    return image_tools.convert_to_uint8(np.asarray(resized))


def input_transform(data, instruction, adv_ind=None):
    state_7d = np.concatenate(
        [
            np.asarray(data[0]["left_arm"]["joint"]).reshape(-1),
            np.asarray(data[0]["left_arm"]["gripper"]).reshape(-1),
        ]
    ).astype(np.float32)

    sensors = data[1]
    img_wrist = _get_color_image(sensors, ["cam_wrist", "wrist_image"])
    if img_wrist is None:
        raise KeyError(
            f"未找到 wrist 相机图像，当前可用键: {list(sensors.keys())}。"
            "期望其中之一: ['cam_wrist', 'wrist_image']"
        )

    img_head = _get_color_image(sensors, ["cam_head", "image"])
    if img_head is None:
        img_head = np.zeros_like(img_wrist)

    observation = {
        "observation/state": state_7d,
        "observation/image": _preprocess_image(img_head),
        "observation/wrist_image": _preprocess_image(img_wrist),
        "prompt": instruction,
    }
    if adv_ind is not None:
        observation["adv_ind"] = adv_ind
    return observation


def output_transform(action):
    action_7d = np.asarray(action, dtype=np.float32)[:7]

    joints = [
        max(limit_min, min(float(action_7d[index]), limit_max))
        for index, (limit_min, limit_max) in enumerate(JOINT_LIMITS_RAD)
    ]
    gripper = max(GRIPPER_LIMIT[0], min(float(action_7d[6]), GRIPPER_LIMIT[1]))

    return {
        "arm": {
            "left_arm": {
                "joint": joints,
                "gripper": gripper,
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description="PI0 单臂 websocket 远程部署脚本（默认 pi05，传 adv_ind 时兼容 PiStar）")
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="远端 websocket 推理服务器地址")
    parser.add_argument("--server-port", type=int, default=8000, help="远端 websocket 推理服务器端口")
    #parser.add_argument("--task-name", type=str, default="Put these toys into the box", help="任务名称")
    parser.add_argument("--task-name", type=str, default="put the white plug into the two-hole socket", help="任务名称")
    parser.add_argument("--instruction", type=str, default=None, help="显式指定 prompt；不传则从任务文件中随机采样")
    parser.add_argument("--adv-ind", type=str, default=None, help="可选。传入时会把请求按 PiStar 方式附带 adv_ind，例如 positive/negative")
    parser.add_argument("--chunk-size", type=int, default=10, help="每次仅执行动作块前多少步")
    parser.add_argument("--control-freq", type=float, default=10.0, help="本地 CAN 控制频率")
    parser.add_argument("--max-step", type=int, default=200, help="单个 episode 最大步数")
    parser.add_argument("--num-episode", type=int, default=10, help="episode 数量")
    args = parser.parse_args()

    print("=" * 50)
    print("PI0 单臂 websocket 远程部署脚本")
    print("=" * 50)
    print(f"服务器地址: ws://{args.server_host}:{args.server_port}")
    print(f"任务名称: {args.task_name}")
    print(f"adv_ind: {args.adv_ind if args.adv_ind is not None else 'None (pi05 default)'}")
    print("=" * 50)

    print("\n[1/3] 初始化机器人...")
    robot = PiperSingle()
    robot.set_up()
    print("✓ 机器人初始化完成")

    print("\n[2/3] 连接 websocket 推理服务器...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.server_host,
        port=args.server_port,
    )
    server_metadata = policy_client.get_server_metadata()
    print(f"✓ 服务器连接成功，metadata: {server_metadata}")

    requires_adv_ind = bool(server_metadata.get("requires_adv_ind", False))
    deploy_mode = server_metadata.get("deploy_mode", "unknown")
    if requires_adv_ind and not args.adv_ind:
        parser.error(
            "connected server requires adv_ind (PiStar), but --adv-ind was not provided."
        )
    if not requires_adv_ind and args.adv_ind:
        print("Warning: 当前服务端不要求 adv_ind；该字段将随请求发送，但通常会被普通 pi05 配置忽略。")
    print(f"服务端模式: {deploy_mode}")

    print(f"\n[3/3] 准备执行 {args.num_episode} 个 episode")
    print("-" * 50)

    for episode_idx in range(args.num_episode):
        step = 0
        instruction = _choose_instruction(args.task_name, args.instruction)

        print(f"\n{'=' * 20} Episode {episode_idx + 1}/{args.num_episode} {'=' * 20}")
        print(f"Prompt: {instruction}")

        robot.reset()

        print("\n按 Enter 键开始推理...")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("✓ 开始执行任务...")
            else:
                time.sleep(0.1)

        while step < args.max_step:
            observation = input_transform(robot.get(), instruction, args.adv_ind)
            response = policy_client.infer(observation)
            action_chunk = np.asarray(response["actions"], dtype=np.float32)
            action_chunk = action_chunk[: args.chunk_size]

            for action in action_chunk:
                robot.move(output_transform(action))
                step += 1
                time.sleep(1.0 / args.control_freq)

                if is_enter_pressed():
                    print("\n⚠ 用户中断执行")
                    is_start = False
                    break

                if step >= args.max_step:
                    break

            if not is_start:
                break

        print(f"✓ Episode {episode_idx + 1} 完成 (总步数: {step})")

    print("\n" + "=" * 50)
    print("全部 episode 执行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

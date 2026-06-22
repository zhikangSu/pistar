#!/usr/bin/env python3
"""SO101 real-robot client for the openpi PiStar policy server.

Runs in the **conda `lerobot` env** on the laptop (has SO101 Feetech driver + cameras +
`lerobot_camera_crop` plugin). Needs the lightweight openpi client:
    pip install -e /path/to/pistar/packages/openpi-client   # no jax

Architecture (openpi, NOT lerobot async_inference):
    [this client] --websocket(openpi-client)--> [serve_policy.py server, has the policy]

What it does each control tick
------------------------------
1. Read SO101 follower observation via lerobot `SO101Follower.get_observation()`:
   - 6 joint positions (degrees) -> state, in the dataset's joint order.
   - `fixed` camera = ROTATE_270 + crop_top=135 -> 505x480 (IDENTICAL to record_pretty.py /
     the training data), `wrist` = 640x480 raw. Cameras are built from the SAME config as
     record_pretty.py so train/deploy images match (this is critical — mismatch = crash/garbage).
2. Build the openpi observation (same shape eval_so101_val_sanity.py used; the SERVER's
   ResizeImages does resize_with_pad to 224, so we send the cropped native resolution):
       {observation/image, observation/wrist_image, observation/state, prompt, adv_ind}
3. `client.infer(obs)` -> action chunk (action_horizon, 7). Slice **[:, :6]** (SO101 is 6-DoF;
   the reused LiberoOutputs pads to 7). Send each row as ABSOLUTE joint position targets
   (`{joint}.pos`) to the follower — not delta, not EE, direct position control.
4. Execute `--exec-horizon` steps of the chunk at `--fps`, then re-infer.

Safety: requires explicit confirmation, `--max-steps` cap, and `--max-relative-target` (per-step
joint jump clamp via lerobot's ensure_safe_goal_position) — STRONGLY recommended on first runs.

Usage
-----
    # logic self-check (no robot, no server) — verify imports/obs format/[:6]/joint order
    python scripts/so101_openpi_robot_client.py --self-check

    # real run (robot WILL move):
    python scripts/so101_openpi_robot_client.py \
        --server-host 127.0.0.1 --port 8000 \
        --max-steps 300 --max-relative-target 15
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Joint order MUST match the training dataset's observation.state / action names.
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Camera + robot hardware config — kept in lockstep with /home/meow/SO101/scripts/record_pretty.py
# (CAMERAS / ROBOT_ARGS). fixed: physical 640x480 -> ROTATE_270 -> 480x640 -> crop top 135 -> 480x505.
FIXED_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
    "width": 480, "height": 505, "fps": 30, "fourcc": "MJPG",
    "rotation": "ROTATE_270", "crop_top": 135,
}
WRIST_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
    # MJPG (not YUYV): with 3 cameras sharing USB bandwidth, YUYV starves fixed_1. Matches
    # record_pretty.py's 3-cam config so deploy images == training images.
    "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG",
}
# 3rd camera (fixed_1) -> observation/right_wrist_image. Only sent when --no-third-cam is NOT set.
# Plain opencv 640x480 MJPG, identical to record_pretty.py CAMERAS["fixed_1"].
FIXED1_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-04014008_P040200_SN0002_720P_USB_Camera_04014008_P040200_SN0002-video-index0",
    "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG",
}
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128258-if00"
DEFAULT_ROBOT_ID = "so101_follower"
DEFAULT_PROMPT = "Pick up the cube and place it into the blue plate"


def build_robot(robot_port: str, robot_id: str, max_relative_target, third_cam: bool = True):
    """Construct the lerobot SO101Follower with cameras identical to record_pretty.py.

    third_cam=True (default) adds the fixed_1 camera for 3-camera models (e.g.
    pi05_star_so101_3cam). Pass third_cam=False for 2-camera models (demoA/recap) —
    sending a real 3rd image to a 2-cam policy (trained with that slot zero-masked) garbles it.
    """
    from lerobot.cameras import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot_camera_crop.cropped_camera import CroppedOpenCVCameraConfig

    cameras = {
        "fixed": CroppedOpenCVCameraConfig(
            index_or_path=FIXED_CAM["index_or_path"],
            width=FIXED_CAM["width"], height=FIXED_CAM["height"], fps=FIXED_CAM["fps"],
            fourcc=FIXED_CAM["fourcc"],
            rotation=Cv2Rotation[FIXED_CAM["rotation"]],
            crop_top=FIXED_CAM["crop_top"],
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST_CAM["index_or_path"],
            width=WRIST_CAM["width"], height=WRIST_CAM["height"], fps=WRIST_CAM["fps"],
            fourcc=WRIST_CAM["fourcc"],
        ),
    }
    if third_cam:
        cameras["fixed_1"] = OpenCVCameraConfig(
            index_or_path=FIXED1_CAM["index_or_path"],
            width=FIXED1_CAM["width"], height=FIXED1_CAM["height"], fps=FIXED1_CAM["fps"],
            fourcc=FIXED1_CAM["fourcc"],
        )
    config = SO101FollowerConfig(
        port=robot_port, id=robot_id, cameras=cameras, max_relative_target=max_relative_target
    )
    return SO101Follower(config)


def build_openpi_obs(robot_obs: dict, prompt: str, adv_ind: str, third_cam: bool = True) -> dict:
    """Map lerobot observation -> openpi observation (same format as eval_so101_val_sanity.py).

    third_cam=True adds observation/right_wrist_image (fixed_1) for 3-camera models. The
    server's LiberoInputs uses it (mask=True) when present, else zero-pads + masks it off.
    """
    state = np.array([float(robot_obs[f"{j}.pos"]) for j in JOINT_ORDER], dtype=np.float32)
    obs = {
        "observation/image": np.asarray(robot_obs["fixed"]),        # 505x480x3 uint8 (rotated+cropped)
        "observation/wrist_image": np.asarray(robot_obs["wrist"]),  # 480x640x3 uint8
        "observation/state": state,                                 # (6,) degrees, dataset joint order
        "prompt": prompt,
        "adv_ind": adv_ind,
    }
    if third_cam:
        obs["observation/right_wrist_image"] = np.asarray(robot_obs["fixed_1"])  # 480x640x3 uint8
    return obs


def actions_to_joint_command(action_row: np.ndarray) -> dict:
    """A (>=6,) policy action row -> SO101 absolute joint targets (drop padding past dim 6)."""
    a = np.asarray(action_row).reshape(-1)[:6]
    return {f"{j}.pos": float(a[k]) for k, j in enumerate(JOINT_ORDER)}


def self_check() -> int:
    """Validate the data plumbing without a robot or server."""
    print("=== self-check: imports / obs format / [:6] slice / joint order ===")
    ok = True

    # 1) lerobot robot + camera config construction (no hardware connect)
    try:
        robot = build_robot(DEFAULT_ROBOT_PORT, DEFAULT_ROBOT_ID, max_relative_target=15)
        cams = robot.config.cameras
        fx = cams["fixed"]
        assert fx.width == 480 and fx.height == 505 and fx.crop_top == 135, "fixed crop mismatch"
        assert str(fx.rotation).endswith("ROTATE_270"), f"fixed rotation mismatch: {fx.rotation}"
        assert cams["wrist"].width == 640 and cams["wrist"].height == 480, "wrist size mismatch"
        assert "fixed_1" in cams and cams["fixed_1"].width == 640 and cams["fixed_1"].height == 480, \
            "fixed_1 (3rd cam) missing/size mismatch"
        print(f"[ok] robot+cameras built | fixed=480x505 ROTATE_270 crop_top=135 | wrist=640x480 "
              f"| fixed_1=640x480 (3rd cam) | obs_features keys sample={list(robot.observation_features)[:3]}...")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] robot/camera construction: {e}")
        ok = False

    # 2) obs construction from a fake robot observation
    fake_robot_obs = {f"{j}.pos": 1.0 * i for i, j in enumerate(JOINT_ORDER)}
    fake_robot_obs["fixed"] = np.zeros((505, 480, 3), dtype=np.uint8)
    fake_robot_obs["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_robot_obs["fixed_1"] = np.zeros((480, 640, 3), dtype=np.uint8)
    # 3-camera (default): obs carries the 3rd image
    obs = build_openpi_obs(fake_robot_obs, DEFAULT_PROMPT, "positive", third_cam=True)
    assert set(obs) == {"observation/image", "observation/wrist_image", "observation/right_wrist_image",
                        "observation/state", "prompt", "adv_ind"}, f"3cam obs keys wrong: {sorted(obs)}"
    assert obs["observation/image"].shape == (505, 480, 3)
    assert obs["observation/wrist_image"].shape == (480, 640, 3)
    assert obs["observation/right_wrist_image"].shape == (480, 640, 3)
    assert obs["observation/state"].shape == (6,)
    assert list(obs["observation/state"]) == [0, 1, 2, 3, 4, 5], "state joint order wrong"
    # 2-camera (--no-third-cam): obs must NOT carry the 3rd image (back-compat for demoA/recap)
    obs2 = build_openpi_obs(fake_robot_obs, DEFAULT_PROMPT, "positive", third_cam=False)
    assert "observation/right_wrist_image" not in obs2, "2cam mode must omit right_wrist_image"
    print(f"[ok] openpi obs (3cam) keys={sorted(obs)} | right_wrist{obs['observation/right_wrist_image'].shape} "
          f"| 2cam mode omits 3rd cam OK")

    # 3) action chunk [:, :6] slice + joint command mapping
    fake_chunk = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)  # server returns (H, 7)
    cmd = actions_to_joint_command(fake_chunk[0])
    assert list(cmd) == [f"{j}.pos" for j in JOINT_ORDER], "command keys/order wrong"
    assert list(cmd.values()) == [0, 1, 2, 3, 4, 5], "should use first 6 dims (drop padding dim 6)"
    print(f"[ok] action (10,7) -> row[:6] -> {cmd}")

    # 4) openpi_client import (may be absent until `pip install -e packages/openpi-client`)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401
        print("[ok] openpi_client import OK")
    except ImportError:
        print("[warn] openpi_client not installed yet — run `pip install -e packages/openpi-client` "
              "in the conda lerobot env before the real run (not needed for self-check)")

    print("=== self-check", "PASSED" if ok else "FAILED", "===")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="SO101 real-robot client for the openpi PiStar server")
    ap.add_argument("--self-check", action="store_true", help="validate logic without robot/server, then exit")
    ap.add_argument("--server-host", default="127.0.0.1", help="policy server host/IP")
    ap.add_argument("--port", type=int, default=8000, help="policy server websocket port")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"])
    ap.add_argument("--fps", type=float, default=15.0, help="control loop frequency")
    ap.add_argument("--exec-horizon", type=int, default=10,
                    help="how many steps of each predicted chunk to execute before re-inferring (<=action_horizon)")
    ap.add_argument("--max-steps", type=int, default=3000, help="hard cap on total action steps (safety)")
    ap.add_argument("--max-relative-target", type=float, default=None,
                    help="clamp per-step joint jump in degrees (safety; recommend ~15 on first runs)")
    ap.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    ap.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--no-confirm", action="store_true", help="skip the startup confirmation prompt")
    ap.add_argument("--no-third-cam", action="store_true",
                    help="omit the 3rd camera (fixed_1 -> right_wrist_image); use for 2-camera "
                         "models like pi05_star_so101_demoA/recap. Default sends 3 cameras (3cam models).")
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    third_cam = not args.no_third_cam
    robot = build_robot(args.robot_port, args.robot_id, args.max_relative_target, third_cam=third_cam)

    print("=" * 64)
    print("⚠️  REAL ROBOT — the SO101 arm WILL move under policy control.")
    print(f"    server   : ws://{args.server_host}:{args.port}")
    print(f"    prompt   : {args.prompt!r}   adv_ind={args.adv_ind}")
    print(f"    cameras  : {'3 (fixed+wrist+fixed_1)' if third_cam else '2 (fixed+wrist) — --no-third-cam'}")
    print(f"    fps={args.fps}  exec_horizon={args.exec_horizon}  max_steps={args.max_steps}")
    print(f"    max_relative_target={args.max_relative_target}"
          f"{'  (UNSET — no per-step clamp; consider --max-relative-target 15)' if args.max_relative_target is None else ''}")
    print("    Clear the workspace. Keep a hand on the e-stop / power.")
    print("=" * 64)
    if not args.no_confirm:
        if input("Type 'GO' to start (anything else aborts): ").strip() != "GO":
            print("aborted.")
            return 1

    print(f"[client] connecting to ws://{args.server_host}:{args.port} ...")
    client = WebsocketClientPolicy(host=args.server_host, port=args.port)
    print(f"[client] server metadata: {client.get_server_metadata()}")

    print("[robot] connecting ...")
    robot.connect()
    period = 1.0 / args.fps
    exec_h = max(1, args.exec_horizon)
    step = 0
    try:
        while step < args.max_steps:
            robot_obs = robot.get_observation()
            obs = build_openpi_obs(robot_obs, args.prompt, args.adv_ind, third_cam=third_cam)
            t0 = time.perf_counter()
            result = client.infer(obs)
            chunk = np.asarray(result["actions"])  # (H, 7)
            infer_ms = (time.perf_counter() - t0) * 1e3
            n = min(exec_h, chunk.shape[0], args.max_steps - step)
            print(f"[step {step:4d}] infer={infer_ms:6.1f}ms chunk={chunk.shape} exec {n} steps "
                  f"| state0={np.round(obs['observation/state'], 1)}")
            for h in range(n):
                tick = time.perf_counter()
                robot.send_action(actions_to_joint_command(chunk[h]))
                step += 1
                dt = time.perf_counter() - tick
                if period > dt:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[client] interrupted by user (Ctrl-C)")
    finally:
        print("[robot] disconnecting ...")
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] disconnect error: {e}")
    print(f"[client] done — executed {step} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())

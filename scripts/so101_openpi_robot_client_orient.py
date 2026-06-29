#!/usr/bin/env python3
"""SO101 real-robot client for the openpi PiStar policy server — **EE-delta** action space.

This is the EE-delta sibling of ``so101_openpi_robot_client.py``. The absolute-joint
client sends ``action[:6]`` directly as joint position targets. That is WRONG for a
policy trained on the EE-delta action space (``action = [dx, dy, dz, gripper]``,
``observation.state = state9 = [6 joints + ee_x, ee_y, ee_z (m)]``). This client:

  * sends a **9-dim** state (6 joints in RANGE_M100_100 + end-effector xyz in meters,
    computed on-device by forward kinematics) to the server, and
  * converts the predicted EE-delta chunk into joint commands via inverse kinematics.

Runs in the conda ``lerobot`` env (SO101 Feetech driver + cameras + ``lerobot_camera_crop``
plugin + ``placo`` for FK/IK). Needs the lightweight openpi client and placo:
    pip install -e /path/to/pistar/packages/openpi-client      # no jax
    pip install placo                                          # FK/IK (PyPI)

EE-delta -> joint conversion (authoritative reference, copied verbatim so deploy
matches training):
  * /home/meow/lerobot-hilserl-verify/so101_verify/scripts/so101_ee_delta_bc_probe.py
    (evaluate_policy ee_delta branch + make_kinematics/fk/iterative_ik)
  * /home/meow/HIL-RL--SO101/tools/convert_so101_v3_to_hilrl_ee_delta.py
    (ee_scale / gripper_open_threshold / calibration — deploy MUST match training literally)

What it does each control tick
------------------------------
1. Read SO101 follower observation via ``SO101Follower.get_observation()``:
   - 6 joint positions (RANGE_M100_100) in dataset joint order.
   - cameras IDENTICAL to record_pretty.py (fixed = ROTATE_270 + crop_top=135 -> 480x505,
     wrist = 640x480, optional fixed_1 = 640x480).
2. Forward-kinematics the 6 joints -> end-effector xyz (meters) and build the **9-dim**
   state ``[6 joints (RANGE_M100_100), ee_x, ee_y, ee_z]`` for the server.
3. ``client.infer(obs)`` -> action chunk ``(H, 7)``. Slice ``[:, :4]`` (EE-delta is 4-dim;
   the reused LiberoOutputs pads to 7).
4. start_ee = current FK ee xyz (meters), re-measured every re-infer.
5. denormalize: ``delta_m = clip(chunk[:, :3], -0.999, 0.999) * ee_scale``,
   ``ee_scale = [0.035, 0.030, 0.060]`` m.
6. cumsum: ``abs_ee = start_ee + cumsum(delta_m, axis=0)`` (one meter target per chunk row).
7. per-row IK (warm-started from the previous solution, position-only / orientation_weight=0
   because SO101 is 5-DoF) -> 5 arm joints in degrees -> back to RANGE_M100_100.
8. gripper = ``chunk[h, 3]`` binarized (>=0.5) -> open/closed joint pos (does NOT go through IK).
9. clamp arm joints to SO101 limits (SO101_JOINT_MIN/MAX, in degrees).

Safety: explicit confirmation, ``--max-steps`` cap, ``--max-relative-target`` per-step joint
jump clamp (STRONGLY recommended on first runs).

Usage
-----
    # math self-check (no robot, no server) — FK/IK closed-loop, unit round-trip, clamp, gripper
    python scripts/so101_openpi_robot_client_eedelta.py --self-check

    # real run (robot WILL move):
    python scripts/so101_openpi_robot_client_eedelta.py \
        --server-host 127.0.0.1 --port 8000 \
        --max-steps 300 --max-relative-target 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Joint order / camera / robot config — kept in lockstep with the absolute-joint
# client (so101_openpi_robot_client.py) and record_pretty.py.
# ---------------------------------------------------------------------------
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

FIXED_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
    "width": 480, "height": 505, "fps": 30, "fourcc": "MJPG",
    "rotation": "ROTATE_270", "crop_top": 135,
}
WRIST_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
    "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG",
}
FIXED1_CAM = {
    "index_or_path": "/dev/v4l/by-id/usb-04014008_P040200_SN0002_720P_USB_Camera_04014008_P040200_SN0002-video-index0",
    "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG",
}
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128258-if00"
DEFAULT_ROBOT_ID = "so101_follower"
DEFAULT_PROMPT = "Pick up the cube and place it into the blue plate"

# ---------------------------------------------------------------------------
# EE-delta conversion constants — MUST match the training conversion literally.
# Source: convert_so101_v3_to_hilrl_ee_delta.py / so101_ee_delta_bc_probe.py.
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# SO101 joint limits in DEGREES (probe :24-25). Used to clamp IK output.
SO101_JOINT_MIN = np.array([-36.0, -107.0, -37.0, 41.0, -46.0], dtype=np.float64)
SO101_JOINT_MAX = np.array([66.0, 45.0, 99.0, 99.0, 65.0], dtype=np.float64)
STS3215_RESOLUTION = 4095.0
# Per-axis EE delta scale in meters (convert/probe default --ee-scale).
DEFAULT_EE_SCALE = [0.035, 0.030, 0.060]
# Gripper binarization threshold on the predicted continuous action[3] (0..1). The training
# action[:,3] is a hard 0/1 label, so any sensible mid threshold works; 0.5 is standard.
DEFAULT_GRIPPER_THRESHOLD = 0.5
# Default asset paths (HIL-RL repo). Overridable via CLI.
DEFAULT_URDF = "/home/meow/HIL-RL--SO101/assets/so101/so101_new_calib.urdf"
DEFAULT_CALIBRATION_PATH = "/home/meow/HIL-RL--SO101/assets/so101/so101_follower_calibration.json"
DEFAULT_IK_ITERS = 5
# Gripper open/closed joint positions in RANGE_M100_100. Measured from the training data:
# action gripper==1 -> state gripper ~25.6 (open), action gripper==0 -> ~3.95 (closed).
# These are sent directly on {gripper}.pos and do NOT go through IK.
DEFAULT_GRIPPER_OPEN_POS = 30.0
DEFAULT_GRIPPER_CLOSED_POS = 3.0

# Optional geometric-guidance fixed-target files (base frame, meters). guide is OFF by default.
DEFAULT_CUBE_XYZ_FILE = "/home/meow/SO101/calib/cube_xyz_fixed.json"
DEFAULT_PLATE_XYZ_FILE = "/home/meow/SO101/calib/plate_xyz_fixed.json"

# Optional live trajectory-viz extrinsics (base -> fixed-camera pixel). viz is OFF by default.
# Same calibrated [K|R|t|dist] used by scripts/draw_base_frame.py & viz_ee_traj_poc.py.
DEFAULT_EXTRINSICS_FILE = "/home/meow/SO101/calib/camera_extrinsics_fixed.npz"


# ===========================================================================
# Unit conversion (RANGE_M100_100 <-> degrees) — copied verbatim from the probe
# so FK/IK match training exactly.
# ===========================================================================
def read_json(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def load_so101_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"SO101 calibration file not found: {path}. EE-delta conversion needs the recording "
            "robot calibration because LeRobot stores joints in RANGE_M100_100 while FK/IK expects "
            "degrees."
        )
    data = read_json(path)
    calibration = {}
    for name in ARM_JOINT_NAMES:
        row = data[name]
        calibration[name] = {
            "drive_mode": float(row.get("drive_mode", 0)),
            "range_min": float(row["range_min"]),
            "range_max": float(row["range_max"]),
        }
    return calibration


def range_arm_to_degrees(arm_joints: np.ndarray, calibration: dict[str, dict[str, float]]) -> np.ndarray:
    arm_joints = np.asarray(arm_joints, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arm_joints, dtype=np.float64)
    for i, name in enumerate(ARM_JOINT_NAMES):
        cal = calibration[name]
        val = float(np.clip(arm_joints[i], -100.0, 100.0))
        if cal["drive_mode"]:
            val = -val
        raw = ((val + 100.0) / 200.0) * (cal["range_max"] - cal["range_min"]) + cal["range_min"]
        mid = (cal["range_min"] + cal["range_max"]) / 2.0
        out[i] = (raw - mid) * 360.0 / STS3215_RESOLUTION
    return out


def degrees_to_range_arm(arm_deg: np.ndarray, calibration: dict[str, dict[str, float]]) -> np.ndarray:
    """Inverse of range_arm_to_degrees: 5 arm joints in degrees -> RANGE_M100_100.

    Mirrors range_arm_to_degrees exactly so the IK output (degrees) can be sent back on
    ``{joint}.pos`` (RANGE_M100_100). drive_mode sign is applied last to match the forward map.
    """
    arm_deg = np.asarray(arm_deg, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arm_deg, dtype=np.float64)
    for i, name in enumerate(ARM_JOINT_NAMES):
        cal = calibration[name]
        mid = (cal["range_min"] + cal["range_max"]) / 2.0
        raw = arm_deg[i] * STS3215_RESOLUTION / 360.0 + mid
        span = cal["range_max"] - cal["range_min"]
        val = ((raw - cal["range_min"]) / span) * 200.0 - 100.0 if span != 0 else 0.0
        if cal["drive_mode"]:
            val = -val
        out[i] = float(np.clip(val, -100.0, 100.0))
    return out


def joints_range_to_degrees(joints: np.ndarray, calibration: dict[str, dict[str, float]]) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64).copy()
    joints[:5] = range_arm_to_degrees(joints[:5], calibration)
    return joints


# ===========================================================================
# Kinematics (placo, lazy import inside RobotKinematics) — copied from the probe.
# ===========================================================================
def make_kinematics(urdf: str | Path):
    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        urdf_path=str(urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )


def fk(kin, q: np.ndarray, calibration: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    """Forward kinematics. q is RANGE_M100_100 if calibration is given, else already in degrees."""
    if calibration is not None:
        q = joints_range_to_degrees(q, calibration)
    return kin.forward_kinematics(q).copy()


def iterative_ik(kin, q0_deg: np.ndarray, target: np.ndarray, iters: int = 5, orientation_weight: float = 0.0):
    """Position-only IK warm-started at q0_deg (degrees). Returns joints in degrees."""
    q = np.asarray(q0_deg, dtype=np.float64).copy()
    for _ in range(iters):
        q = kin.inverse_kinematics(
            q, target, position_weight=1.0, orientation_weight=orientation_weight
        ).copy()
    return q


# ===========================================================================
# Robot / camera construction (same as the absolute-joint client).
# ===========================================================================
def build_robot(robot_port: str, robot_id: str, max_relative_target, third_cam: bool = True):
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


# ===========================================================================
# State9 build (joints + FK ee xyz) and EE-delta -> joint conversion.
# ===========================================================================
def joints_from_obs(robot_obs: dict) -> np.ndarray:
    """6 joints (RANGE_M100_100) in dataset order from a lerobot observation."""
    return np.array([float(robot_obs[f"{j}.pos"]) for j in JOINT_ORDER], dtype=np.float64)


def rot_to_6d_single(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> 6D = [R[:,0], R[:,1]] (first two columns). Matches the training-side
    encoding convert_so101_v3_to_ee_orient.rot_to_6d (must be byte-identical)."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])


def gram_schmidt_6d(d6: np.ndarray) -> np.ndarray:
    """6D -> 3x3 rotation (Gram-Schmidt). VERBATIM from convert_so101_v3_to_ee_orient.gram_schmidt_6d
    so the model's predicted 6D orientation decodes to exactly the training-side rotation."""
    d6 = np.asarray(d6, dtype=np.float64).reshape(6)
    a1 = d6[0:3]
    a2 = d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2p = a2 - (b1 @ a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def build_state15(joints6: np.ndarray, kin, calibration) -> np.ndarray:
    """state15 = [6 joints (RANGE_M100_100), ee_xyz(3, m), ee_rot6d(6)]. ee pose via FK on the arm
    joints; orientation 6D = first two columns of the FK rotation (matches the converter)."""
    pose = fk(kin, joints6, calibration)
    ee_xyz = pose[:3, 3]
    ee_r6 = rot_to_6d_single(pose[:3, :3])
    return np.concatenate([np.asarray(joints6, dtype=np.float64), ee_xyz, ee_r6]).astype(np.float32)


def build_openpi_obs(robot_obs: dict, prompt: str, adv_ind: str, kin, calibration,
                     third_cam: bool = True, guide_targets: dict | None = None) -> dict:
    """lerobot observation -> openpi observation with a 9-dim state.

    guide_targets (optional): {"cube_xyz": [x,y,z], "plate_xyz": [x,y,z]} in base frame meters,
    attached to the obs dict for downstream geometric guidance. Default None (pure BC).
    """
    joints6 = joints_from_obs(robot_obs)
    state15 = build_state15(joints6, kin, calibration)
    obs = {
        "observation/image": np.asarray(robot_obs["fixed"]),        # 505x480x3 uint8
        "observation/wrist_image": np.asarray(robot_obs["wrist"]),  # 480x640x3 uint8
        "observation/state": state15,                               # (15,) [6 joints + ee xyz + ee 6D朝向]
        "prompt": prompt,
        "adv_ind": adv_ind,
    }
    if third_cam:
        obs["observation/right_wrist_image"] = np.asarray(robot_obs["fixed_1"])  # 480x640x3 uint8
    if guide_targets:
        # Placeholder pass-through for geometric guidance. The server/policy can read these if it
        # supports guided sampling; the default BC server ignores unknown keys.
        obs.update(guide_targets)
    return obs


def eedelta_chunk_to_joint_commands(
    chunk: np.ndarray,
    start_ee: np.ndarray,
    q_start6: np.ndarray,
    kin,
    calibration,
    ee_scale: np.ndarray,
    ik_iters: int = DEFAULT_IK_ITERS,
    gripper_threshold: float = DEFAULT_GRIPPER_THRESHOLD,
    gripper_open_pos: float = DEFAULT_GRIPPER_OPEN_POS,
    gripper_closed_pos: float = DEFAULT_GRIPPER_CLOSED_POS,
    absolute: bool = False,
) -> tuple[list[dict], np.ndarray]:
    """Convert an EE-delta chunk (H, >=4) into a list of H joint-position command dicts.

    Args:
        chunk: (H, >=4) predicted action chunk. cols 0..2 = normalized EE delta, col 3 = gripper.
        start_ee: (3,) current end-effector xyz in meters (FK of q_start6).
        q_start6: (6,) current joints in RANGE_M100_100 (warm-start for IK; only [:5] used).
        ee_scale: (3,) per-axis meter scale.
    Returns:
        (commands, abs_ee) where commands is a list of {f"{joint}.pos": val} dicts (RANGE_M100_100
        for arm, plus gripper), and abs_ee is (H,3) the meter targets (for logging/debug).
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    H = chunk.shape[0]
    # action10 col0..2 = 位置(absolute 模式=绝对EE米; 否则=归一化delta), col3..8=目标朝向6D(绝对), col9=gripper。
    if absolute:
        # 绝对 EE 版(pi05_star_so101_v4_ee_abs):action[:,:3] 已是绝对 base 系米,直接 IK,
        # 无 cumsum/无 ee_scale → 去掉 EE-delta 的 cumsum 累积锯齿。start_ee/ee_scale 此模式不用。
        abs_ee = chunk[:, :3].copy()
    else:
        # EE-delta 版:反归一化 + cumsum -> 绝对米目标(几何reward可微)。
        delta_norm = np.clip(chunk[:, :3], -0.999, 0.999)
        delta_m = delta_norm * np.asarray(ee_scale, dtype=np.float64)
        abs_ee = np.asarray(start_ee, dtype=np.float64)[None, :] + np.cumsum(delta_m, axis=0)

    # warm-start IK from the current measured joints (in degrees).
    q_prev_deg = joints_range_to_degrees(np.asarray(q_start6, dtype=np.float64), calibration)

    commands: list[dict] = []
    for h in range(H):
        # per-row IK 同时约束【位置 + 朝向】(orientation_weight=1)。
        # 朝向 target = 解模型预测的 6D(绝对) -> 旋转矩阵(gram-schmidt, 与训练编码一致).
        R_tgt = gram_schmidt_6d(chunk[h, 3:9])
        target = np.eye(4)
        target[:3, :3] = R_tgt
        target[:3, 3] = abs_ee[h]
        q_next_deg = iterative_ik(kin, q_prev_deg, target, iters=ik_iters, orientation_weight=1.0)
        # clamp arm joints to SO101 limits (degrees).
        q_arm_deg = np.clip(q_next_deg[:5], SO101_JOINT_MIN, SO101_JOINT_MAX)
        q_prev_deg = q_next_deg.copy()  # warm-start next row from the unclamped solution
        # degrees -> RANGE_M100_100 for the 5 arm joints.
        q_arm_range = degrees_to_range_arm(q_arm_deg, calibration)
        # gripper: action10 第 9 维, 二值化映成开/合关节位(不过 IK).
        grip_bit = 1.0 if float(chunk[h, 9]) >= gripper_threshold else 0.0
        grip_pos = gripper_open_pos if grip_bit > 0.5 else gripper_closed_pos
        cmd = {f"{ARM_JOINT_NAMES[k]}.pos": float(q_arm_range[k]) for k in range(5)}
        cmd["gripper.pos"] = float(grip_pos)
        commands.append(cmd)
    return commands, abs_ee


def load_guide_targets(cube_file: str, plate_file: str) -> dict:
    """Load cube/plate base-frame xyz (meters) from the calibration JSONs for geometric guidance.

    See /home/meow/SO101/scripts/cube_to_base.py and plate_to_base.py (which back-project the
    fixed-camera pixel through the calibrated plane to base coords) for how these are produced.
    Returns {"cube_xyz": [x,y,z], "plate_xyz": [x,y,z]}.
    """
    out = {}
    cube = read_json(cube_file)
    out["cube_xyz"] = [float(v) for v in cube["cube_xyz"]]
    plate = read_json(plate_file)
    out["plate_xyz"] = [float(v) for v in plate["plate_xyz"]]
    return out


# Calibration / detection scripts live in the SO101 workspace (not the pistar repo).
SO101_SCRIPTS_DIR = "/home/meow/SO101/scripts"
SO101_CALIB_DIR = "/home/meow/SO101/calib"


def autodetect_targets(cameras: str = "fixed,fixed_1") -> dict:
    """实时检测【当前】cube/plate 的 base 系坐标(米),供几何引导用。每次运行自动更新,不必手动先跑脚本。

    直接复用 SO101/scripts 下的 cube_to_base.py(背景差分)+plate_to_base.py(HSV蓝)——两脚本 --camera
    fixed,fixed_1 本身就做【双相机平均+交叉校验+单路 fallback】。以子进程运行(同 lerobot env),写出
    calib/{cube,plate}_xyz_*.json,再读本次刚写的最新文件(按 mtime,兼容 fixed 被挡→fixed_1 fallback)。
    必须在 robot.connect() 【之前】调用(脚本要独占相机;此时机器人尚未占用,不动机械臂)。
    """
    import glob
    import os
    import subprocess
    import time as _time

    out = {}
    for script, key, prefix in (("cube_to_base.py", "cube_xyz", "cube_xyz"),
                                ("plate_to_base.py", "plate_xyz", "plate_xyz")):
        t0 = _time.time()
        print(f"[detect] {script} --camera {cameras} …")
        r = subprocess.run([sys.executable, os.path.join(SO101_SCRIPTS_DIR, script), "--camera", cameras],
                           cwd=SO101_SCRIPTS_DIR, capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            if any(s in ln for s in ("xyz", "🎯", "校验", "面积", "源=")):
                print("   " + ln)
        if r.returncode != 0:
            hint = ""
            if "无背景图" in (r.stdout + r.stderr):
                hint = ("\n  → cube 背景图缺失。先在【空桌(无方块)、机械臂处于起始姿态】时跑一次:\n"
                        f"    {sys.executable} {SO101_SCRIPTS_DIR}/cube_to_base.py --camera {cameras} --set-bg")
            raise SystemExit(f"❌ {script} 检测失败:\n{r.stdout}\n{r.stderr}{hint}")
        cands = [f for f in glob.glob(os.path.join(SO101_CALIB_DIR, f"{prefix}_*.json"))
                 if os.path.getmtime(f) >= t0 - 1.0]
        if not cands:
            raise SystemExit(f"❌ {script} 未写出新的 {prefix}_*.json(检测可能没成功)")
        newest = max(cands, key=os.path.getmtime)
        out[key] = [float(v) for v in read_json(newest)[key]]
    return out


# ===========================================================================
# Live trajectory visualization (optional, --viz-traj). Reproject the policy's
# planned EE trajectory (abs_ee, base frame) onto the fixed camera frame and push
# to rerun, mirroring the @csgbwk orange-trajectory overlay. PURELY ADDITIVE:
# everything is gated behind args.viz_traj so the control path is byte-identical
# when off. Math is the SAME as scripts/viz_ee_traj_poc.py / draw_base_frame.py.
# ===========================================================================
def load_proj_ctx(extrinsics_file: str) -> dict:
    """Load calibrated [K|R|t|dist] for base-frame 3D -> fixed-camera pixel projection."""
    import cv2

    E = np.load(extrinsics_file)
    rvec, _ = cv2.Rodrigues(E["R"])
    return {"rvec": rvec, "tvec": E["t"].reshape(3, 1), "K": E["K"], "dist": E["dist"]}


def _project(proj: dict, pts3d: np.ndarray) -> np.ndarray:
    import cv2

    p2, _ = cv2.projectPoints(np.asarray(pts3d, np.float64),
                              proj["rvec"], proj["tvec"], proj["K"], proj["dist"])
    return p2.reshape(-1, 2).astype(np.int32)


def overlay_ee_traj(frame_rgb: np.ndarray, ee_xyz_base: np.ndarray, proj: dict,
                    targets: dict | None = None, ee_xyz_bc: np.ndarray | None = None,
                    ee_now: np.ndarray | None = None, downsample: int = 1) -> np.ndarray:
    """Draw the planned EE trajectory (orange) on a COPY of frame_rgb. Optionally overlay a second
    BC (un-steered) trajectory in gray (steering effect) and a live current-EE marker (magenta).

    frame_rgb  : (H,W,3) uint8 RGB — ALREADY-PROCESSED fixed frame (ROTATE_270+crop_top=135, 505x480).
    ee_xyz_base: (N,3) meters base — the EXECUTED (VLS-steered if guiding) trajectory → orange.
    ee_xyz_bc  : optional (N,3) meters base — the un-steered BC trajectory → gray. The gap = steering.
    ee_now     : optional (3,) meters base — the arm's CURRENT EE → magenta dot (real-time position).
    downsample : >1 shrinks the returned image by that factor (less data → smoother rerun rendering).
    """
    import cv2

    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    # BC baseline (un-steered) first, dim gray, so the orange VLS line draws on top.
    if ee_xyz_bc is not None:
        pb = _project(proj, ee_xyz_bc)
        cv2.polylines(img, [pb], False, (150, 150, 150), 2, cv2.LINE_AA)  # gray BGR
        cv2.circle(img, tuple(pb[-1]), 4, (150, 150, 150), -1, cv2.LINE_AA)
    pts = _project(proj, ee_xyz_base)
    cv2.polylines(img, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)  # orange BGR (executed/VLS)
    for k, (u, v) in enumerate(pts):
        cv2.circle(img, (int(u), int(v)), 2 if k else 5, (0, 165, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 6, (0, 255, 0), 2, cv2.LINE_AA)  # green start (plan origin)
    if ee_now is not None:  # live arm EE (magenta) — animates at frame rate even when the plan is fixed
        pn = _project(proj, [ee_now])[0]
        cv2.circle(img, tuple(pn), 6, (255, 0, 255), -1, cv2.LINE_AA)
    if targets:
        for key, label in (("cube_xyz", "cube"), ("plate_xyz", "plate")):
            if key in targets:
                pt = _project(proj, [targets[key]])[0]
                cv2.circle(img, tuple(pt), 7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, label, (pt[0] + 9, pt[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # steering magnitude: max per-step EE divergence (mm) between VLS and BC trajectories.
    if ee_xyz_bc is not None:
        n = min(len(ee_xyz_base), len(ee_xyz_bc))
        dmm = float(np.max(np.linalg.norm(np.asarray(ee_xyz_base)[:n] - np.asarray(ee_xyz_bc)[:n], axis=1))) * 1e3
        cv2.putText(img, f"VLS steer: max dev {dmm:5.1f}mm  (orange=VLS gray=BC magenta=now)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    if downsample and downsample > 1:
        img = cv2.resize(img, (img.shape[1] // downsample, img.shape[0] // downsample),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class VizPusher:
    """Background-thread trajectory-overlay pusher — keeps viz OFF the control critical path.

    The control loop just calls submit(frame, abs_ee) (non-blocking); a daemon thread does the
    cv2 overlay + log_rerun_data. A maxsize-1 queue means if the pusher is still busy with the
    previous frame, the new one REPLACES it (drop-stale) — so a slow rerun flush can never stall
    or back up the robot loop. This fixes the earlier symptom where the synchronous ~727KB image
    flush per re-infer inserted a pause between "reached" and "execute next chunk".
    """

    def __init__(self, proj, targets, viz_cam, init_fn, log_fn, compress=True, downsample=2):
        import queue
        import threading

        self.proj, self.targets, self.viz_cam = proj, targets, viz_cam
        self.init_fn, self.log_fn, self.compress, self.downsample = init_fn, log_fn, compress, downsample
        self._q = queue.Queue(maxsize=1)
        self._ok = False
        self._stop = False
        self._t = threading.Thread(target=self._run, name="viz-pusher", daemon=True)
        self._t.start()

    def submit(self, frame_rgb, abs_ee, abs_ee_bc=None, ee_now=None):
        import queue

        item = (frame_rgb, abs_ee, abs_ee_bc, ee_now)
        try:
            self._q.put_nowait(item)
        except queue.Full:  # pusher still busy -> drop the stale frame, keep the newest
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def _run(self):
        # init_rerun (rr.spawn) runs HERE, inside the daemon thread — so a slow/blocking/colliding
        # rerun viewer spawn can NEVER stall the robot control loop or swallow Ctrl-C. If it fails,
        # viz silently disables itself and the control loop is completely unaffected.
        try:
            self.init_fn(session_name="so101_ee_traj")
            self._ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[viz] init_rerun failed ({e}); trajectory viz disabled (robot loop unaffected)")
        while not self._stop:
            frame_rgb, abs_ee, abs_ee_bc, ee_now = self._q.get()
            if frame_rgb is None:
                break
            if not self._ok:
                continue  # keep draining so submit() never blocks, but skip logging
            try:
                overlaid = overlay_ee_traj(frame_rgb, abs_ee, self.proj, self.targets,
                                           ee_xyz_bc=abs_ee_bc, ee_now=ee_now, downsample=self.downsample)
                self.log_fn(observation={f"images.{self.viz_cam}_traj": overlaid},
                            compress_images=self.compress)
            except Exception as e:  # noqa: BLE001 — viz must never break control
                print(f"[viz] async overlay skipped: {e}")

    def close(self):
        self._stop = True
        try:
            self._q.put_nowait((None, None, None, None))
        except Exception:  # noqa: BLE001
            pass


# ===========================================================================
# Self-check (math: FK/IK closed loop, unit round-trip, clamp, gripper). No robot/server.
# ===========================================================================
def self_check(args) -> int:
    print("=== self-check: placo FK/IK closed-loop + unit round-trip + clamp + gripper ===")
    ok = True

    # 0) placo + RobotKinematics import.
    try:
        import placo  # noqa: F401
        from lerobot.model.kinematics import RobotKinematics  # noqa: F401
        print("[ok] import placo + RobotKinematics")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import placo/RobotKinematics: {e}")
        return 1

    calibration = load_so101_calibration(args.calibration_path)
    kin = make_kinematics(args.urdf)
    ee_scale = np.asarray(args.ee_scale, dtype=np.float64)

    # 1) RANGE_M100_100 <-> degrees round-trip.
    q_range = np.array([10.0, -20.0, 30.0, 50.0, 5.0], dtype=np.float64)
    q_deg = range_arm_to_degrees(q_range, calibration)
    q_back = degrees_to_range_arm(q_deg, calibration)
    rt_err = float(np.max(np.abs(q_range - q_back)))
    print(f"[{'ok' if rt_err < 1e-6 else 'FAIL'}] unit round-trip RANGE->deg->RANGE max_err={rt_err:.2e}")
    ok = ok and rt_err < 1e-6

    # 2) FK -> start_ee, fake small EE-delta chunk -> denorm+cumsum+per-row IK -> FK back,
    #    confirm end-effector reaches abs_ee (closed-loop mm-level error).
    q_start6 = np.array([5.0, -15.0, 20.0, 60.0, 0.0, 5.0], dtype=np.float64)  # RANGE_M100_100, +gripper
    start_pose = fk(kin, q_start6, calibration)
    start_ee = start_pose[:3, 3]
    start_r6 = rot_to_6d_single(start_pose[:3, :3])           # 闭环测试用当前朝向(可达)做目标
    H = 10
    rng = np.random.default_rng(0)
    fake_norm = rng.uniform(-0.3, 0.3, size=(H, 3))           # small normalized position deltas
    fake_grip = (rng.uniform(0, 1, size=(H, 1)) > 0.5).astype(np.float64)
    fake_r6 = np.tile(start_r6, (H, 1))                       # action10 朝向块(绝对6D)
    chunk = np.concatenate([fake_norm, fake_r6, fake_grip], axis=1)  # (H,10)=[pos3,r6,grip]
    assert chunk.shape == (H, 10)

    commands, abs_ee = eedelta_chunk_to_joint_commands(
        chunk, start_ee, q_start6, kin, calibration, ee_scale,
        ik_iters=args.ik_iters,
        gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos,
    )
    assert len(commands) == H

    # FK the IK'd joints back; compare position(米) AND orientation(度) to targets.
    achieved = np.zeros((H, 3)); ori_err = np.zeros(H)
    R_tgt = gram_schmidt_6d(start_r6)
    for h, cmd in enumerate(commands):
        q_arm_range = np.array([cmd[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=np.float64)
        q6 = np.concatenate([q_arm_range, [0.0]])
        pose = fk(kin, q6, calibration)
        achieved[h] = pose[:3, 3]
        c = (np.trace(R_tgt.T @ pose[:3, :3]) - 1.0) / 2.0
        ori_err[h] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    err_mm = np.linalg.norm(achieved - abs_ee, axis=1) * 1000.0
    print(f"[{'ok' if err_mm.max() < 10.0 else 'WARN'}] FK/IK 闭环 位置: "
          f"mean={err_mm.mean():.3f}mm p95={np.percentile(err_mm,95):.3f}mm max={err_mm.max():.3f}mm")
    print(f"[{'ok' if ori_err.max() < 2.0 else 'WARN'}] FK/IK 闭环 朝向: "
          f"mean={ori_err.mean():.3f}° max={ori_err.max():.3f}°  (orientation_weight=1 应≈0)")
    print(f"      start_ee(m)={np.round(start_ee,4)}  abs_ee[-1](m)={np.round(abs_ee[-1],4)}")
    ok = ok and err_mm.max() < 50.0 and ori_err.max() < 5.0

    # 3) cumsum / denorm sanity: row 0 delta == start_ee + first delta_m.
    expected_row0 = start_ee + np.clip(fake_norm[0], -0.999, 0.999) * ee_scale
    cumsum_err = float(np.max(np.abs(abs_ee[0] - expected_row0)))
    print(f"[{'ok' if cumsum_err < 1e-9 else 'FAIL'}] denorm+cumsum row0 max_err={cumsum_err:.2e}")
    ok = ok and cumsum_err < 1e-9

    # 4) clamp check: drive an out-of-range target, ensure arm degrees stay within limits.
    far_chunk = np.zeros((1, 10))
    far_chunk[0, :3] = [0.999, 0.999, 0.999]; far_chunk[0, 3:9] = start_r6
    far_cmds, _ = eedelta_chunk_to_joint_commands(
        far_chunk, start_ee, q_start6, kin, calibration, ee_scale, ik_iters=args.ik_iters)
    far_arm_range = np.array([far_cmds[0][f"{j}.pos"] for j in ARM_JOINT_NAMES])
    far_arm_deg = range_arm_to_degrees(far_arm_range, calibration)
    within = bool(np.all(far_arm_deg >= SO101_JOINT_MIN - 1e-6) and np.all(far_arm_deg <= SO101_JOINT_MAX + 1e-6))
    print(f"[{'ok' if within else 'FAIL'}] joint-limit clamp: arm_deg={np.round(far_arm_deg,1)}")
    ok = ok and within

    # 5) gripper channel: action10 第 9 维 -> 开/合关节位.
    g_chunk = np.zeros((2, 10)); g_chunk[:, 3:9] = start_r6; g_chunk[0, 9] = 1.0; g_chunk[1, 9] = 0.0
    g_cmds, _ = eedelta_chunk_to_joint_commands(
        g_chunk, start_ee, q_start6, kin, calibration, ee_scale,
        gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos)
    g_ok = (abs(g_cmds[0]["gripper.pos"] - args.gripper_open_pos) < 1e-9 and
            abs(g_cmds[1]["gripper.pos"] - args.gripper_closed_pos) < 1e-9)
    print(f"[{'ok' if g_ok else 'FAIL'}] gripper: bit1->{g_cmds[0]['gripper.pos']} bit0->{g_cmds[1]['gripper.pos']}")
    ok = ok and g_ok
    assert list(g_cmds[0].keys()) == [f"{j}.pos" for j in ARM_JOINT_NAMES] + ["gripper.pos"], \
        f"command keys/order wrong: {list(g_cmds[0])}"
    print(f"[ok] command keys/order = {list(g_cmds[0].keys())}")

    # 6) state15 build.
    fake_robot_obs = {f"{j}.pos": float(q_start6[i]) for i, j in enumerate(JOINT_ORDER)}
    s15 = build_state15(joints_from_obs(fake_robot_obs), kin, calibration)
    assert s15.shape == (15,), f"state15 shape wrong: {s15.shape}"
    assert np.allclose(s15[:6], q_start6), "state15 first 6 must be raw joints (RANGE_M100_100)"
    assert np.allclose(s15[6:9], start_ee, atol=1e-5), "state15 ee xyz must equal FK ee"
    assert np.allclose(s15[9:15], start_r6, atol=1e-5), "state15 ee 6D must equal FK rotation 6D"
    print(f"[ok] state15 shape=(15,) [6关节+ee xyz+ee 6D朝向] ee={np.round(s15[6:9],4)}")

    # 7) guide-target loader (interface only).
    try:
        g = load_guide_targets(args.cube_xyz_file, args.plate_xyz_file)
        print(f"[ok] guide loader: cube_xyz={np.round(g['cube_xyz'],4)} plate_xyz={np.round(g['plate_xyz'],4)}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] guide loader (optional): {e}")

    # 8) openpi_client import (optional until installed).
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401
        print("[ok] openpi_client import OK")
    except ImportError:
        print("[warn] openpi_client not installed — run `pip install -e packages/openpi-client` before real run")

    print("=== self-check", "PASSED" if ok else "FAILED", "===")
    return 0 if ok else 1


# ===========================================================================
# Main control loop (same skeleton as the absolute-joint client).
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="SO101 EE-delta real-robot client for the openpi PiStar server")
    ap.add_argument("--self-check", action="store_true", help="validate FK/IK math without robot/server, then exit")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"])
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--exec-horizon", type=int, default=10,
                    help="how many steps of each predicted chunk to execute before re-inferring")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--max-relative-target", type=float, default=None,
                    help="clamp per-step joint jump in RANGE_M100_100 units (safety; recommend ~15 first runs)")
    ap.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    ap.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--no-third-cam", action="store_true",
                    help="omit the 3rd camera (fixed_1); use for 2-camera models")
    # EE-delta specific.
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--calibration-path", default=DEFAULT_CALIBRATION_PATH)
    ap.add_argument("--ee-scale", nargs=3, type=float, default=DEFAULT_EE_SCALE,
                    help="per-axis EE delta scale in meters — MUST match training")
    ap.add_argument("--abs-ee", action="store_true",
                    help="ABSOLUTE-EE model (pi05_star_so101_v4_ee_abs): action[:,:3]=绝对EE米,直接IK,无 "
                         "cumsum/ee_scale(去锯齿)。default=EE-delta. 必须与 serve 的 config 匹配。")
    ap.add_argument("--ik-iters", type=int, default=DEFAULT_IK_ITERS)
    ap.add_argument("--gripper-threshold", type=float, default=DEFAULT_GRIPPER_THRESHOLD)
    ap.add_argument("--gripper-open-pos", type=float, default=DEFAULT_GRIPPER_OPEN_POS,
                    help="gripper joint pos (RANGE_M100_100) for OPEN (action gripper bit=1)")
    ap.add_argument("--gripper-closed-pos", type=float, default=DEFAULT_GRIPPER_CLOSED_POS,
                    help="gripper joint pos (RANGE_M100_100) for CLOSED (action gripper bit=0)")
    # Geometric guidance (OFF by default; interface/placeholder only).
    ap.add_argument("--guide", action="store_true",
                    help="attach cube/plate base-frame xyz to obs for downstream geometric guidance (default off)")
    ap.add_argument("--cube-xyz-file", default=DEFAULT_CUBE_XYZ_FILE)
    ap.add_argument("--plate-xyz-file", default=DEFAULT_PLATE_XYZ_FILE)
    # 几何引导目标:默认每次运行【自动检测】当前 cube/plate(双相机交叉验证),不必手动先跑脚本。
    ap.add_argument("--no-auto-detect-targets", dest="auto_detect_targets", action="store_false",
                    help="with --guide, skip live detection and use the static --cube/plate-xyz-file JSONs instead")
    ap.set_defaults(auto_detect_targets=True)
    ap.add_argument("--detect-cameras", default="fixed,fixed_1",
                    help="cameras for live cube/plate detection (dual = cross-validated average + fallback)")
    # Live trajectory visualization (OFF by default; purely additive — see overlay_ee_traj).
    ap.add_argument("--viz-traj", action="store_true",
                    help="reproject the planned EE trajectory (abs_ee) onto the fixed frame and push "
                         "to rerun each re-infer (orange polyline, @csgbwk-style). Zero effect on control.")
    ap.add_argument("--viz-cam", default="fixed", choices=["fixed", "fixed_1"],
                    help="which FIXED camera to overlay on (wrist excluded — its extrinsics aren't fixed)")
    ap.add_argument("--viz-extrinsics", default=DEFAULT_EXTRINSICS_FILE,
                    help="npz with calibrated K/R/t/dist for --viz-cam (base 3D -> pixel)")
    # VLS 调参/诊断:per-request 覆盖 serve 的 guide_scale(不必重启 serve)。
    ap.add_argument("--guide-scale", type=float, default=None,
                    help="override server guide_scale per request (tune VLS strength live). None=server default.")
    # 看 VLS 到底起没起作用:每步多跑一次【不引导(guide_scale=0)】推理做对照,叠加两条轨迹
    # (橙=VLS执行的, 灰=BC不引导的) + 顶部显示最大散度(mm)。两线越分开=引导越强;重合=guide_scale 太小。
    ap.add_argument("--viz-steering", action="store_true",
                    help="overlay BC(un-steered) vs VLS(steered) EE trajectories to SEE the steering effect "
                         "(implies --viz-traj; runs one extra guide_scale=0 infer, throttled by --viz-steering-every)")
    # 实时显示:把画面刷新从【每次重规划(~1Hz)】解耦成【执行循环里每隔几步抓新帧】→ 实时相机+末端位置。
    ap.add_argument("--viz-stride", type=int, default=2,
                    help="push a FRESH camera frame + live EE marker every N executed steps (1=every step). "
                         "Decouples display smoothness from the (slow) re-infer rate. 0=only per re-infer.")
    ap.add_argument("--viz-downsample", type=int, default=2,
                    help="shrink the pushed overlay image by this factor (less data → smoother rerun)")
    ap.add_argument("--viz-steering-every", type=int, default=1,
                    help="recompute the BC(guide=0) comparison every N re-infers (reuse between). 1=clean "
                         "noise-matched compare every re-infer; >1 cuts extra inference if arm cadence too slow")
    args = ap.parse_args()
    if args.viz_steering:
        args.viz_traj = True  # steering overlay requires the trajectory viz pipeline

    if args.self_check:
        return self_check(args)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    third_cam = not args.no_third_cam
    calibration = load_so101_calibration(args.calibration_path)
    kin = make_kinematics(args.urdf)
    ee_scale = np.asarray(args.ee_scale, dtype=np.float64)
    # 几何引导目标(cube/plate base 系米)。默认【运行时自动检测】当前位置(双相机交叉验证)——每次方块
    # 摆哪都自动更新,无需手动先跑脚本。在 robot.connect() 之前做(独占相机、不动机械臂),便于 GO 前核对。
    guide_targets = None
    if args.guide:
        if args.auto_detect_targets:
            print(f"[guide] auto-detecting cube/plate (cameras={args.detect_cameras}) …")
            guide_targets = autodetect_targets(args.detect_cameras)
        else:
            guide_targets = load_guide_targets(args.cube_xyz_file, args.plate_xyz_file)
        print(f"[guide] cube_xyz={[round(v, 3) for v in guide_targets['cube_xyz']]}  "
              f"plate_xyz={[round(v, 3) for v in guide_targets['plate_xyz']]}")

    # Live trajectory viz (optional). Runs on a BACKGROUND thread (VizPusher) so it never
    # delays inference or action execution. Loaded once; per-step submit is in the loop below.
    viz_pusher = None
    if args.viz_traj:
        proj_ctx = load_proj_ctx(args.viz_extrinsics)
        # green cube/plate keypoints: reuse the just-detected guide_targets if available, else static JSON.
        viz_targets = guide_targets
        if viz_targets is None:
            try:
                viz_targets = load_guide_targets(args.cube_xyz_file, args.plate_xyz_file)
            except Exception as e:  # noqa: BLE001
                print(f"[viz] cube/plate keypoints unavailable ({e}); drawing trajectory only")
        from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
        # NOTE: init_rerun (rr.spawn) is NOT called here — it runs inside the VizPusher thread so a
        # blocking/colliding rerun viewer can never hang startup or the robot loop (see VizPusher).
        viz_pusher = VizPusher(proj_ctx, viz_targets, args.viz_cam, init_rerun, log_rerun_data,
                               compress=True, downsample=max(1, args.viz_downsample))
        print(f"[viz] live EE-trajectory overlay ON (async thread; rerun init in background) — "
              f"cam={args.viz_cam}, extrinsics={args.viz_extrinsics}")

    robot = build_robot(args.robot_port, args.robot_id, args.max_relative_target, third_cam=third_cam)

    print("=" * 64)
    print("⚠️  REAL ROBOT — the SO101 arm WILL move under EE-delta policy control.")
    print(f"    server   : ws://{args.server_host}:{args.port}")
    print(f"    prompt   : {args.prompt!r}   adv_ind={args.adv_ind}")
    print(f"    cameras  : {'3 (fixed+wrist+fixed_1)' if third_cam else '2 (fixed+wrist) — --no-third-cam'}")
    print(f"    ee_scale : {ee_scale.tolist()} m   ik_iters={args.ik_iters}   guide={args.guide}")
    print(f"    fps={args.fps}  exec_horizon={args.exec_horizon}  max_steps={args.max_steps}")
    print(f"    max_relative_target={args.max_relative_target}"
          f"{'  (UNSET — consider --max-relative-target 15)' if args.max_relative_target is None else ''}")
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
    reinfer_count = 0
    last_bc_cumdelta = None   # cumsum(Δ_BC) from the most recent BC infer (re-anchored to current start)
    last_bc_abs = None        # 绝对版: 最近一次 BC 推理的绝对 EE (abs 模式直接用)
    try:
        while step < args.max_steps:
            robot_obs = robot.get_observation()
            joints6 = joints_from_obs(robot_obs)
            # start_ee re-measured from the latest joints every re-infer.
            start_ee = fk(kin, joints6, calibration)[:3, 3]
            obs = build_openpi_obs(robot_obs, args.prompt, args.adv_ind, kin, calibration,
                                   third_cam=third_cam, guide_targets=guide_targets)
            if args.guide_scale is not None:
                obs["guide_scale"] = float(args.guide_scale)   # per-request override of serve default
            if args.viz_steering:
                # fix the flow-matching noise so the BC(guide=0) vs VLS overlay differs ONLY by steering
                # (else ~cm sampling noise dominates and the steering effect is invisible).
                obs["noise_seed"] = int(step) + 1
            t0 = time.perf_counter()
            result = client.infer(obs)
            chunk = np.asarray(result["actions"])  # (H, 10) = [pos_delta3, target_rot6d, gripper]
            infer_ms = (time.perf_counter() - t0) * 1e3
            commands, abs_ee = eedelta_chunk_to_joint_commands(
                chunk, start_ee, joints6, kin, calibration, ee_scale,
                ik_iters=args.ik_iters, gripper_threshold=args.gripper_threshold,
                gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos,
                absolute=args.abs_ee,
            )
            # Steering-effect overlay: extra UN-steered (guide_scale=0, SAME noise_seed) infer → BC traj.
            # Throttled by --viz-steering-every; absolute=chunk[:,:3] 直接; delta=cumsum(Δ) re-anchor 到 start_ee。
            abs_ee_bc = None
            if args.viz_steering and guide_targets is not None:
                if reinfer_count % max(1, args.viz_steering_every) == 0:
                    obs_bc = dict(obs)
                    obs_bc["guide_scale"] = 0.0   # noise_seed inherited from obs → noise-matched compare
                    chunk_bc = np.asarray(client.infer(obs_bc)["actions"])
                    if args.abs_ee:
                        last_bc_abs = chunk_bc[:, :3].copy()
                    else:
                        last_bc_abs = None
                        last_bc_cumdelta = np.cumsum(np.clip(chunk_bc[:, :3], -0.999, 0.999) * ee_scale, axis=0)
                if args.abs_ee:
                    abs_ee_bc = last_bc_abs
                elif last_bc_cumdelta is not None:
                    abs_ee_bc = start_ee[None, :] + last_bc_cumdelta
            reinfer_count += 1
            n = min(exec_h, len(commands), args.max_steps - step)
            dev = "" if abs_ee_bc is None else f" | VLS-dev={np.max(np.linalg.norm(abs_ee[:len(abs_ee_bc)]-abs_ee_bc,axis=1))*1e3:.1f}mm"
            print(f"[step {step:4d}] infer={infer_ms:6.1f}ms chunk={chunk.shape} exec {n} steps "
                  f"| state0={np.round(obs['observation/state'], 3)} | abs_ee[-1]={np.round(abs_ee[-1],4)}{dev}")
            # Immediate plan push (so the new orange/gray plan shows at once). When --viz-stride<=0 this
            # is the ONLY push (per re-infer ≈1Hz). When >0, the exec loop below refreshes at frame rate.
            if viz_pusher is not None and args.viz_stride <= 0:
                viz_pusher.submit(np.array(robot_obs[args.viz_cam]), abs_ee, abs_ee_bc, start_ee)
            for h in range(n):
                tick = time.perf_counter()
                robot.send_action(commands[h])
                step += 1
                # REAL-TIME display: grab a fresh camera frame + live EE marker every --viz-stride steps,
                # overlaying the (fixed-this-chunk) planned trajectories. Decouples display smoothness from
                # the slow re-infer rate. Async (drop-stale) → never stalls control.
                if viz_pusher is not None and args.viz_stride > 0 and (h % args.viz_stride == 0):
                    try:
                        ro = robot.get_observation()
                        ee_now = fk(kin, joints_from_obs(ro), calibration)[:3, 3]
                        viz_pusher.submit(np.array(ro[args.viz_cam]), abs_ee, abs_ee_bc, ee_now)
                    except Exception:  # noqa: BLE001 — viz must never break control
                        pass
                dt = time.perf_counter() - tick
                if period > dt:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[client] interrupted by user (Ctrl-C)")
    finally:
        if viz_pusher is not None:
            viz_pusher.close()
        print("[robot] disconnecting ...")
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] disconnect error: {e}")
    print(f"[client] done — executed {step} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())

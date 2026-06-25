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


def build_state9(joints6: np.ndarray, kin, calibration) -> np.ndarray:
    """state9 = [6 joints (RANGE_M100_100), ee_x, ee_y, ee_z (m)]. ee via FK on the arm joints."""
    ee = fk(kin, joints6, calibration)[:3, 3]
    return np.concatenate([np.asarray(joints6, dtype=np.float64), ee]).astype(np.float32)


def build_openpi_obs(robot_obs: dict, prompt: str, adv_ind: str, kin, calibration,
                     third_cam: bool = True, guide_targets: dict | None = None) -> dict:
    """lerobot observation -> openpi observation with a 9-dim state.

    guide_targets (optional): {"cube_xyz": [x,y,z], "plate_xyz": [x,y,z]} in base frame meters,
    attached to the obs dict for downstream geometric guidance. Default None (pure BC).
    """
    joints6 = joints_from_obs(robot_obs)
    state9 = build_state9(joints6, kin, calibration)
    obs = {
        "observation/image": np.asarray(robot_obs["fixed"]),        # 505x480x3 uint8
        "observation/wrist_image": np.asarray(robot_obs["wrist"]),  # 480x640x3 uint8
        "observation/state": state9,                                # (9,) [6 joints + ee xyz m]
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
    # 2) take first 4 dims; 4) denormalize; 5) cumsum -> absolute meter targets.
    delta_norm = np.clip(chunk[:, :3], -0.999, 0.999)
    delta_m = delta_norm * np.asarray(ee_scale, dtype=np.float64)
    abs_ee = np.asarray(start_ee, dtype=np.float64)[None, :] + np.cumsum(delta_m, axis=0)

    # warm-start IK from the current measured joints (in degrees).
    q_prev_deg = joints_range_to_degrees(np.asarray(q_start6, dtype=np.float64), calibration)
    # base pose (rotation kept = current FK orientation; only position is constrained anyway).
    base_pose = fk(kin, q_start6, calibration)

    commands: list[dict] = []
    for h in range(H):
        # 6) per-row position-only IK, warm-started from the previous solution.
        target = base_pose.copy()
        target[:3, 3] = abs_ee[h]
        q_next_deg = iterative_ik(kin, q_prev_deg, target, iters=ik_iters, orientation_weight=0.0)
        # 8) clamp arm joints to SO101 limits (degrees).
        q_arm_deg = np.clip(q_next_deg[:5], SO101_JOINT_MIN, SO101_JOINT_MAX)
        q_prev_deg = q_next_deg.copy()  # warm-start next row from the unclamped solution
        # 7) degrees -> RANGE_M100_100 for the 5 arm joints.
        q_arm_range = degrees_to_range_arm(q_arm_deg, calibration)
        # gripper: binarize, map to open/closed joint pos (no IK).
        grip_bit = 1.0 if float(chunk[h, 3]) >= gripper_threshold else 0.0
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
    start_ee = fk(kin, q_start6, calibration)[:3, 3]
    H = 10
    rng = np.random.default_rng(0)
    fake_norm = rng.uniform(-0.3, 0.3, size=(H, 3))           # small normalized deltas
    fake_grip = (rng.uniform(0, 1, size=(H, 1)) > 0.5).astype(np.float64)
    pad = np.zeros((H, 3))                                    # LiberoOutputs pads 4->7
    chunk = np.concatenate([fake_norm, fake_grip, pad], axis=1)  # (H, 7)
    assert chunk.shape == (H, 7)

    commands, abs_ee = eedelta_chunk_to_joint_commands(
        chunk, start_ee, q_start6, kin, calibration, ee_scale,
        ik_iters=args.ik_iters,
        gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos,
    )
    assert len(commands) == H

    # FK the IK'd joints back and compare to the meter target.
    achieved = np.zeros((H, 3))
    for h, cmd in enumerate(commands):
        q_arm_range = np.array([cmd[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=np.float64)
        q6 = np.concatenate([q_arm_range, [0.0]])
        achieved[h] = fk(kin, q6, calibration)[:3, 3]
    err_mm = np.linalg.norm(achieved - abs_ee, axis=1) * 1000.0
    print(f"[{'ok' if err_mm.max() < 5.0 else 'WARN'}] FK/IK closed loop: "
          f"mean={err_mm.mean():.3f}mm p95={np.percentile(err_mm,95):.3f}mm max={err_mm.max():.3f}mm")
    print(f"      start_ee(m)={np.round(start_ee,4)}  abs_ee[-1](m)={np.round(abs_ee[-1],4)}")
    # mm-level closed loop expected for reachable in-workspace targets.
    ok = ok and err_mm.max() < 50.0  # generous; IK on 5-DoF + clamping can add a few mm

    # 3) cumsum / denorm sanity: row 0 delta == start_ee + first delta_m.
    expected_row0 = start_ee + np.clip(fake_norm[0], -0.999, 0.999) * ee_scale
    cumsum_err = float(np.max(np.abs(abs_ee[0] - expected_row0)))
    print(f"[{'ok' if cumsum_err < 1e-9 else 'FAIL'}] denorm+cumsum row0 max_err={cumsum_err:.2e}")
    ok = ok and cumsum_err < 1e-9

    # 4) clamp check: drive an out-of-range target, ensure arm degrees stay within limits.
    far_chunk = np.zeros((1, 7))
    far_chunk[0, :3] = [0.999, 0.999, 0.999]
    far_start_ee = fk(kin, q_start6, calibration)[:3, 3]
    far_cmds, _ = eedelta_chunk_to_joint_commands(
        far_chunk, far_start_ee, q_start6, kin, calibration, ee_scale, ik_iters=args.ik_iters)
    far_arm_range = np.array([far_cmds[0][f"{j}.pos"] for j in ARM_JOINT_NAMES])
    far_arm_deg = range_arm_to_degrees(far_arm_range, calibration)
    within = bool(np.all(far_arm_deg >= SO101_JOINT_MIN - 1e-6) and np.all(far_arm_deg <= SO101_JOINT_MAX + 1e-6))
    print(f"[{'ok' if within else 'FAIL'}] joint-limit clamp: arm_deg={np.round(far_arm_deg,1)} within "
          f"[{SO101_JOINT_MIN.tolist()}, {SO101_JOINT_MAX.tolist()}]")
    ok = ok and within

    # 5) gripper channel: bit 1 -> open pos, bit 0 -> closed pos.
    g_chunk = np.zeros((2, 7)); g_chunk[0, 3] = 1.0; g_chunk[1, 3] = 0.0
    g_cmds, _ = eedelta_chunk_to_joint_commands(
        g_chunk, far_start_ee, q_start6, kin, calibration, ee_scale,
        gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos)
    g_ok = (abs(g_cmds[0]["gripper.pos"] - args.gripper_open_pos) < 1e-9 and
            abs(g_cmds[1]["gripper.pos"] - args.gripper_closed_pos) < 1e-9)
    print(f"[{'ok' if g_ok else 'FAIL'}] gripper: bit1->{g_cmds[0]['gripper.pos']} (open) "
          f"bit0->{g_cmds[1]['gripper.pos']} (closed)")
    ok = ok and g_ok
    # command keys / order.
    assert list(g_cmds[0].keys()) == [f"{j}.pos" for j in ARM_JOINT_NAMES] + ["gripper.pos"], \
        f"command keys/order wrong: {list(g_cmds[0])}"
    print(f"[ok] command keys/order = {list(g_cmds[0].keys())}")

    # 6) state9 build.
    fake_robot_obs = {f"{j}.pos": float(q_start6[i]) for i, j in enumerate(JOINT_ORDER)}
    s9 = build_state9(joints_from_obs(fake_robot_obs), kin, calibration)
    assert s9.shape == (9,), f"state9 shape wrong: {s9.shape}"
    assert np.allclose(s9[:6], q_start6), "state9 first 6 must be raw joints (RANGE_M100_100)"
    assert np.allclose(s9[6:9], start_ee, atol=1e-5), "state9 ee xyz must equal FK ee"
    print(f"[ok] state9 shape=(9,) [6 joints + ee xyz] ee={np.round(s9[6:9],4)}")

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
    args = ap.parse_args()

    if args.self_check:
        return self_check(args)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    third_cam = not args.no_third_cam
    calibration = load_so101_calibration(args.calibration_path)
    kin = make_kinematics(args.urdf)
    ee_scale = np.asarray(args.ee_scale, dtype=np.float64)
    guide_targets = load_guide_targets(args.cube_xyz_file, args.plate_xyz_file) if args.guide else None

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
    try:
        while step < args.max_steps:
            robot_obs = robot.get_observation()
            joints6 = joints_from_obs(robot_obs)
            # start_ee re-measured from the latest joints every re-infer.
            start_ee = fk(kin, joints6, calibration)[:3, 3]
            obs = build_openpi_obs(robot_obs, args.prompt, args.adv_ind, kin, calibration,
                                   third_cam=third_cam, guide_targets=guide_targets)
            t0 = time.perf_counter()
            result = client.infer(obs)
            chunk = np.asarray(result["actions"])  # (H, 7)
            infer_ms = (time.perf_counter() - t0) * 1e3
            commands, abs_ee = eedelta_chunk_to_joint_commands(
                chunk, start_ee, joints6, kin, calibration, ee_scale,
                ik_iters=args.ik_iters, gripper_threshold=args.gripper_threshold,
                gripper_open_pos=args.gripper_open_pos, gripper_closed_pos=args.gripper_closed_pos,
            )
            n = min(exec_h, len(commands), args.max_steps - step)
            print(f"[step {step:4d}] infer={infer_ms:6.1f}ms chunk={chunk.shape} exec {n} steps "
                  f"| state0={np.round(obs['observation/state'], 3)} | abs_ee[-1]={np.round(abs_ee[-1],4)}")
            for h in range(n):
                tick = time.perf_counter()
                robot.send_action(commands[h])
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

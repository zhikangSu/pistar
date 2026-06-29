#!/usr/bin/env python
"""一次性诊断:把【当前夹爪末端 gripper_frame_link】+【基座原点/XYZ轴】投到 fixed 相机当前帧,
存图供核对。用于回答:EE 标记是否落在夹爪末端? 相机外参是否仍准?(机械臂不动,只读关节+相机)

跑法(lerobot env;机械臂先摆到初始位置,相机别被别的进程占用):
  /home/meow/miniconda3/envs/lerobot/bin/python pistar/scripts/check_ee_marker.py [--no-third-cam]

判读:
  - 品红点 = 我们认为的夹爪末端(gripper_frame_link 投影)。应落在钳口区。
  - 黄点 = 基座原点(0,0,0)投影;红/绿/蓝短线 = X/Y/Z 轴(各5cm)。黄点应落在机械臂底座中心。
  - 若黄点不在底座 → 外参失准(相机被挪过)→ 需重标外参。
  - 若黄点在底座但品红点不在夹爪 → FK/TCP 问题(而非外参)。
  - 若两者都对 → 定位没问题,抓取问题在别处(EE-delta 部署)。
"""
import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.join(ROOT, "so101_openpi_robot_client_orient.py")
_spec = importlib.util.spec_from_file_location("_cli_orient", CLIENT)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-third-cam", action="store_true", help="不连 fixed_1(只 fixed+wrist)")
    ap.add_argument("--cam", default="fixed", choices=["fixed", "fixed_1"])
    ap.add_argument("--out", default="/tmp/ee_marker_check.png")
    a = ap.parse_args()

    calib = C.load_so101_calibration(C.DEFAULT_CALIBRATION_PATH)
    kin = C.make_kinematics(C.DEFAULT_URDF)
    proj = C.load_proj_ctx(f"/home/meow/SO101/calib/camera_extrinsics_{a.cam}.npz")
    robot = C.build_robot(C.DEFAULT_ROBOT_PORT, C.DEFAULT_ROBOT_ID, None, third_cam=not a.no_third_cam)

    print("[robot] connecting (不会移动机械臂,只读取)…")
    robot.connect()
    try:
        ro = robot.get_observation()
        joints6 = C.joints_from_obs(ro)
        ee = C.fk(kin, joints6, calib)[:3, 3]
        img = cv2.cvtColor(np.asarray(ro[a.cam]), cv2.COLOR_RGB2BGR).copy()

        def P(p):
            return C._project(proj, [p])[0]

        o = P([0.0, 0.0, 0.0])
        for vec, col in (([0.05, 0, 0], (0, 0, 255)), ([0, 0.05, 0], (0, 255, 0)), ([0, 0, 0.05], (255, 0, 0))):
            cv2.line(img, tuple(o), tuple(P(vec)), col, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(o), 5, (0, 255, 255), -1, cv2.LINE_AA)        # base origin (yellow)
        g = P(ee.tolist())
        cv2.circle(img, tuple(g), 8, (255, 0, 255), 2, cv2.LINE_AA)         # gripper EE (magenta)
        cv2.circle(img, tuple(g), 2, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, f"magenta=EE  yellow=base0  axes:X red Y grn Z blu", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(a.out, img)

        print(f"[joints RANGE_M100_100] {np.round(joints6, 1)}")
        print(f"[EE gripper_frame_link] base(m)= {np.round(ee, 4)}  pixel= {g.tolist()}")
        print(f"[base origin] pixel= {o.tolist()}   图尺寸= {img.shape[1]}x{img.shape[0]} (应 480x505)")
        print(f"🖼️  存图 → {a.out}")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())

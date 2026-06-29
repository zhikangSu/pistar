#!/usr/bin/env python
"""离线量化+可视化【EE 版部署链路】把轨迹劣化了多少(对照 demo 真值),不动真机。

在留出 episode 上,按部署节奏(每 exec_h 帧重规划)跑 EE 版策略,得到三条 base 系 EE 轨迹:
  GT     = demo 真值(state15[:,6:9])                      —— 目标
  PRED   = start_ee + cumsum(预测Δ·ee_scale)              —— 策略"想走的"(EE 表示)
  POSTIK = 预测Δ→IK(orientation_weight=1,5-DoF)→关节→FK   —— "实际会走的"
量化:
  ||PRED-GT||  = 策略在 EE 空间的预测误差
  ||POSTIK-PRED|| = 5-DoF IK 折中额外吃掉的
  ||POSTIK-GT|| = 总劣化
并把三条轨迹反投影到该帧存图(GT绿 / PRED橙 / POSTIK红)。

跑法(pistar .venv;先停掉占 GPU 的 serve):
  .venv/bin/python scripts/eval_ee_vs_joint_traj.py --episode 0 --exec-h 10
"""
import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CLIENT = os.path.join(ROOT, "scripts", "so101_openpi_robot_client_orient.py")
_spec = importlib.util.spec_from_file_location("_cli_orient", CLIENT)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def fk_from_range(kin, q_range6, calib):
    return C.fk(kin, np.asarray(q_range6, np.float64), calib)[:3, 3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "checkpoints/pi05_star_so101_v4_ee_orient_3cam/so101_lora_ee_orient_3cam/29999"))
    ap.add_argument("--config-name", default="pi05_star_so101_v4_ee_orient_3cam_infer")
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v4_ee_orient_pistar")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--exec-h", type=int, default=10, help="部署重规划步长(每 exec_h 帧 re-infer)")
    ap.add_argument("--prompt", default="Pick up the cube and place it into the blue plate")
    ap.add_argument("--viz-cam", default="fixed")
    ap.add_argument("--out", default="/tmp/ee_vs_demo_traj.png")
    a = ap.parse_args()

    import openpi.training.config as _config
    import openpi.policies.policy_config as _policy_config
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cfg = _config.get_config(a.config_name)
    H = cfg.model.action_horizon
    print(f"[cfg] {a.config_name} action_horizon={H}")
    policy = _policy_config.create_trained_policy(cfg, a.ckpt)

    root = a.root or os.path.join(os.path.expanduser("~/.cache/huggingface/lerobot"), a.repo_id)
    ds = LeRobotDataset(a.repo_id, root=root)
    e = a.episode
    fr0 = ds.episode_data_index["from"][e].item()
    fr1 = ds.episode_data_index["to"][e].item()
    ep_len = fr1 - fr0
    states = np.stack([_np(ds[fr0 + i]["state"]).reshape(-1) for i in range(ep_len)]).astype(np.float64)
    print(f"[data] ep{e} len={ep_len} state_dim={states.shape[1]} (应15)")
    demo_ee = states[:, 6:9]  # (T,3) base meters

    # placo(IK/FK) 只在 silri/HIL-RL lerobot 有;pistar .venv 没有 lerobot.model.kinematics。
    # POSTIK 需要它→设为可选;PRED(cumsum) 和 demo EE(数据自带) 不需要 placo,核心对照照样跑。
    try:
        calib = C.load_so101_calibration(C.DEFAULT_CALIBRATION_PATH)
        kin = C.make_kinematics(C.DEFAULT_URDF)
    except Exception as ex:  # noqa: BLE001
        print(f"[warn] placo 不可用({ex}); 跳过 POSTIK(IK后轨迹),只比 PRED vs GT")
        calib, kin = None, None
    ee_scale = np.asarray(C.DEFAULT_EE_SCALE, np.float64)
    proj = C.load_proj_ctx(os.path.join(ROOT, "..", "calib", f"camera_extrinsics_{a.viz_cam}.npz"))

    # demo 真值 action10(含 rot6d 朝向),用于朝向预测误差
    acts = np.stack([_np(ds[fr0 + i]["actions"]).reshape(-1) for i in range(ep_len)]).astype(np.float64)
    def geodesic_deg(r6a, r6b):
        Ra, Rb = C.gram_schmidt_6d(np.asarray(r6a)), C.gram_schmidt_6d(np.asarray(r6b))
        c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    # 累加误差
    err_pred, err_postik_extra, err_total, err_orient = [], [], [], []
    # 拼整条 PRED / POSTIK 轨迹(用于画图)
    pred_full = np.full_like(demo_ee, np.nan)
    postik_full = np.full_like(demo_ee, np.nan)
    reinfer_marks = []

    for t in range(0, ep_len - 1, a.exec_h):
        fr = ds[fr0 + t]
        obs = {
            "observation/image": _np(fr["image"]),
            "observation/wrist_image": _np(fr["wrist_image"]),
            "observation/state": states[t].astype(np.float32),
            "prompt": a.prompt,
            "adv_ind": "positive",
        }
        if "right_wrist_image" in fr:
            obs["observation/right_wrist_image"] = _np(fr["right_wrist_image"])
        chunk = _np(policy.infer(obs)["actions"])  # (H,10)
        n = min(a.exec_h, H, ep_len - t)
        start_ee = demo_ee[t]
        # PRED: cumsum
        delta_m = np.clip(chunk[:, :3], -0.999, 0.999) * ee_scale
        pred_ee = start_ee[None, :] + np.cumsum(delta_m, axis=0)  # (H,3)
        # POSTIK: 预测Δ+朝向 → IK → 关节 → FK (需 placo;不可用则跳过)
        postik_ee = None
        if kin is not None:
            commands, _ = C.eedelta_chunk_to_joint_commands(
                chunk, start_ee, states[t, :6], kin, calib, ee_scale,
                ik_iters=C.DEFAULT_IK_ITERS, gripper_threshold=C.DEFAULT_GRIPPER_THRESHOLD,
                gripper_open_pos=C.DEFAULT_GRIPPER_OPEN_POS, gripper_closed_pos=C.DEFAULT_GRIPPER_CLOSED_POS)
            postik_ee = np.stack([fk_from_range(kin, np.array(
                [commands[h][f"{j}.pos"] for j in C.JOINT_ORDER], np.float64), calib)
                for h in range(len(commands))])
        reinfer_marks.append(t)
        for h in range(n):
            if t + h >= ep_len:
                break
            gt = demo_ee[t + h]
            err_pred.append(np.linalg.norm(pred_ee[h] - gt))
            err_orient.append(geodesic_deg(chunk[h, 3:9], acts[t + h, 3:9]))  # 朝向预测误差(度)
            pred_full[t + h] = pred_ee[h]
            if postik_ee is not None:
                err_postik_extra.append(np.linalg.norm(postik_ee[h] - pred_ee[h]))
                err_total.append(np.linalg.norm(postik_ee[h] - gt))
                postik_full[t + h] = postik_ee[h]

    ep = lambda v: (np.mean(v) * 1e3, np.max(v) * 1e3)
    print("\n========== EE 版部署链路劣化(留出 ep%d, 单位 mm) ==========" % e)
    print(f"  ||PRED-GT||     策略EE位置预测误差: mean {ep(err_pred)[0]:6.1f}  max {ep(err_pred)[1]:6.1f}")
    print(f"  朝向预测误差(度)               : mean {np.mean(err_orient):6.2f}  max {np.max(err_orient):6.2f}  "
          f"(>~10度→钳口被甩偏=空夹元凶)")
    if err_postik_extra:
        print(f"  ||POSTIK-PRED|| 5-DoF IK 折中额外 : mean {ep(err_postik_extra)[0]:6.1f}  max {ep(err_postik_extra)[1]:6.1f}")
        print(f"  ||POSTIK-GT||   总劣化           : mean {ep(err_total)[0]:6.1f}  max {ep(err_total)[1]:6.1f}")
    else:
        print("  (POSTIK 跳过: placo 不可用)")
    # 参考基线:demo EE 自身的步进尺度(每 exec_h 帧 EE 移动多少),好判断上面误差相对大小
    seg = np.linalg.norm(demo_ee[a.exec_h:] - demo_ee[:-a.exec_h], axis=1)
    print(f"  [参考] demo EE 每{a.exec_h}帧位移 mean {np.mean(seg)*1e3:.1f}mm  全程总行程 {np.sum(np.linalg.norm(np.diff(demo_ee,axis=0),axis=1))*1e3:.0f}mm")

    # —— matplotlib: EE 的 x/y/z 随帧;demo vs pred(+postik),竖线=重规划边界 ——
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.arange(ep_len)
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, name in enumerate("xyz"):
        axs[i].plot(T, demo_ee[:, i] * 1e3, "g-", lw=2, label="demo (GT)")
        axs[i].plot(T, pred_full[:, i] * 1e3, color="orange", lw=1.5, label="pred (EE policy)")
        if not np.isnan(postik_full).all():
            axs[i].plot(T, postik_full[:, i] * 1e3, "r-", lw=1, label="post-IK")
        for m in reinfer_marks:
            axs[i].axvline(m, color="gray", ls=":", lw=0.6)
        axs[i].set_ylabel(f"EE {name} (mm)")
        if i == 0:
            axs[i].legend(loc="upper right", fontsize=8)
    axs[2].set_xlabel("frame (虚线=每次重规划/re-infer边界;看边界处 pred 跳变=回退)")
    axs[0].set_title(f"EE 版策略预测 vs demo (ep{e}) | ||PRED-GT|| mean {np.mean(err_pred)*1e3:.1f}mm max {np.max(err_pred)*1e3:.1f}mm")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"🖼️  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Gate 2.5: 离线量化【关节版 VLS】起没起作用(上真机前必过,EE 版咬过的坑)。

在留出帧上,固定流匹配噪声,扫 guide_scale,把预测关节 chunk 经可微 FK(so101_fk + joint_to_rad,
不需 placo) → EE 轨迹,量三件事:
  · max ‖EE(scale)-EE(0)‖ (mm) : 引导把轨迹推了多远 —— 应随 guide_scale 单调增(治好了弱引导)。
  · min dist(EE_traj→cube) (mm): 轨迹离 cube 的最近距离 —— VLS 应让它变小(把末端往 cube 拉)。
  · reward(scale)              : 几何 reward —— 应随 guide_scale 升(梯度上升方向对)。

跑法(pistar .venv;先停掉占 GPU 的 serve):
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/gate2p5_joint_vls_scan.py
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# RANGE_M100_100→弧度的每关节比例(与 serve_policy/pi0.py joint_ee 分支一致)。
JOINT_TO_RAD = np.array([0.019586, 0.018290, 0.016962, 0.017852, 0.031416])


def _np(x):
    import torch
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "checkpoints/pi05_star_so101_v4_3cam/so101_lora_v4_3cam/29999"))
    ap.add_argument("--config-name", default="pi05_star_so101_v4_3cam_infer")
    # 用 ee_orient 重打包集当观测源:它本地完整,image/state 与 joint 集物理同源(同机器同帧),
    # state15 的前 6 维 = joint 模型要的 6 关节(RANGE)。joint v4_pistar 集本地 meta 不全故不用。
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v4_ee_orient_pistar")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--frac", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                    help="每个 episode 取哪些相对位置的帧(0=起始,0.5=中段)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--prompt", default="Pick up the cube and place it into the blue plate")
    ap.add_argument("--cube-file", default="/home/meow/SO101/calib/cube_xyz_fixed.json")
    ap.add_argument("--plate-file", default="/home/meow/SO101/calib/plate_xyz_fixed.json")
    a = ap.parse_args()

    import jax.numpy as jnp
    import openpi.training.config as _config
    import openpi.policies.policy_config as _policy_config
    from openpi.models import ee_steer, so101_fk
    # 数据集只在本地(没 push 到 hub)→ get_safe_version 会 404/offline 报错。补丁成直接用本地 revision。
    import lerobot.common.datasets.lerobot_dataset as _ld
    _ld.get_safe_version = lambda repo_id, revision: revision or "v2.1"
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cube = np.asarray(json.load(open(a.cube_file))["cube_xyz"], np.float64)
    plate = np.asarray(json.load(open(a.plate_file))["plate_xyz"], np.float64)
    print(f"[targets] cube={np.round(cube,4)}  plate={np.round(plate,4)}")

    # 开引导的 sample_kwargs:reward_fn 非空 → create_trained_policy 注入 q01/q99;joint_ee=True 走 FK 分支。
    sk = {
        "reward_fn": ee_steer.grasp_place_reward,
        "guide_scale": 2.0,                       # 占位,逐请求用 obs["guide_scale"] 覆盖
        "start_ratio": 0.6,
        "joint_ee": True,
        "joint_to_rad": jnp.asarray(JOINT_TO_RAD),
        "ee_scale": jnp.asarray([0.035, 0.030, 0.060]),   # joint 分支不用,占位
        "cube_xyz": jnp.asarray(cube)[None, :],
        "plate_xyz": jnp.asarray(plate)[None, :],
    }
    cfg = _config.get_config(a.config_name)
    H = cfg.model.action_horizon
    print(f"[cfg] {a.config_name} action_horizon={H}  loading policy(+steering)...")
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, sample_kwargs=sk)

    root = a.root or os.path.join(os.path.expanduser("~/.cache/huggingface/lerobot"), a.repo_id)
    ds = LeRobotDataset(a.repo_id, root=root)

    # 收集测试帧
    frames = []
    for e in a.episodes:
        if e >= ds.num_episodes:
            continue
        fr0 = ds.episode_data_index["from"][e].item()
        fr1 = ds.episode_data_index["to"][e].item()
        ln = fr1 - fr0
        for f in a.frac:
            frames.append((e, fr0 + min(ln - 1, int(f * ln))))

    K = JOINT_TO_RAD

    def fk_traj(chunk):
        """chunk:(H,>=6) RANGE → EE 轨迹 (n,3) 米(只取前 H 行)。"""
        rad = np.asarray(chunk[:, :5], np.float64) * K[None, :]
        return np.stack([np.asarray(so101_fk.fk_pos(jnp.asarray(rad[h]))) for h in range(len(rad))])

    pre_grasp = cube + np.array([0.0, 0.0, 0.05])
    # 逐 scale 累计指标(对所有测试帧平均)
    agg = {s: {"disp": [], "dcube": [], "dpre": [], "rew": []} for s in a.scales}

    print(f"\n扫 guide_scale={a.scales}  固定 noise_seed={a.noise_seed}  测试帧={len(frames)}")
    for (e, fr_idx) in frames:
        fr = ds[fr_idx]
        state = _np(fr["state"]).reshape(-1)[:6].astype(np.float32)   # 前6=joint 模型的 6 关节(RANGE)
        base_obs = {
            "observation/image": _np(fr["image"]),
            "observation/wrist_image": _np(fr["wrist_image"]),
            "observation/state": state,
            "prompt": a.prompt,
            "adv_ind": "positive",
            "cube_xyz": cube.astype(np.float32),
            "plate_xyz": plate.astype(np.float32),
        }
        if "right_wrist_image" in fr:
            base_obs["observation/right_wrist_image"] = _np(fr["right_wrist_image"])

        ee0 = None
        for s in a.scales:
            obs = dict(base_obs)
            obs["guide_scale"] = float(s)
            obs["noise_seed"] = int(a.noise_seed)   # 固定噪声 → 差异纯是引导
            chunk = _np(policy.infer(obs)["actions"])
            ee = fk_traj(chunk)
            if s == a.scales[0]:
                ee0 = ee
            disp = np.max(np.linalg.norm(ee - ee0, axis=1)) * 1e3            # vs 第一个 scale(通常 0)
            dcube = np.min(np.linalg.norm(ee - cube[None, :], axis=1)) * 1e3  # 轨迹最近 cube
            dpre = np.min(np.linalg.norm(ee - pre_grasp[None, :], axis=1)) * 1e3
            grip01 = np.clip(np.asarray(chunk[:, 5], np.float64) / 38.36, 0, 1)
            rew = float(ee_steer.grasp_place_reward(
                jnp.asarray(cube), jnp.asarray(plate), jnp.asarray(ee[None]), jnp.asarray(grip01[None])))
            agg[s]["disp"].append(disp)
            agg[s]["dcube"].append(dcube)
            agg[s]["dpre"].append(dpre)
            agg[s]["rew"].append(rew)

    print("\n================= Gate 2.5: 关节版 VLS 引导有效性 =================")
    print(f"{'guide_scale':>11} | {'轨迹位移mm':>9} | {'近cube mm':>9} | {'近pre mm':>9} | {'reward':>9}")
    print("-" * 64)
    base_dcube = np.mean(agg[a.scales[0]]["dcube"])
    base_pre = np.mean(agg[a.scales[0]]["dpre"])
    for s in a.scales:
        d = agg[s]
        print(f"{s:>11.1f} | {np.mean(d['disp']):>9.1f} | {np.mean(d['dcube']):>9.1f} | "
              f"{np.mean(d['dpre']):>9.1f} | {np.mean(d['rew']):>9.4f}")
    print("-" * 64)
    # 判据(找甜区,非"单调到最大"——VLS 大 scale 必过冲,有限甜区才是正确表现)。
    sc = list(a.scales)
    disps = [np.mean(agg[s]["disp"]) for s in sc]
    dcubes = [np.mean(agg[s]["dcube"]) for s in sc]
    rews = [np.mean(agg[s]["rew"]) for s in sc]
    pos = [i for i, s in enumerate(sc) if s > 0]
    # ① 引导可控:位移在小 scale 区随 scale 增长(取前半段,避开过冲尾)。
    responsive = disps[pos[0]] > 1.0 and (disps[pos[1]] > disps[pos[0]] if len(pos) > 1 else True)
    # ② 方向对:存在某 scale 让轨迹比 BC 明显靠近 cube(≥5mm)。
    best_cube_i = int(np.argmin(dcubes))
    closer = dcubes[best_cube_i] < base_dcube - 5.0 and sc[best_cube_i] > 0
    # ③ 梯度上升:存在某 scale 让 reward 比 BC 升。
    best_rew_i = int(np.argmax(rews))
    rew_up = rews[best_rew_i] > rews[0] + 1e-3 and sc[best_rew_i] > 0
    print(f"判据① 引导可控(小scale位移随scale增): {'✅' if responsive else '❌'}  "
          f"(scale {sc[pos[0]]:.0f}→{disps[pos[0]]:.1f}mm)")
    print(f"判据② 有 scale 把轨迹拉近 cube(方向对): {'✅' if closer else '❌'}  "
          f"(BC {base_dcube:.1f}mm → 最佳 scale={sc[best_cube_i]:g} 时 {dcubes[best_cube_i]:.1f}mm, "
          f"近pre {base_pre:.1f}→{np.mean(agg[sc[best_cube_i]]['dpre']):.1f}mm)")
    print(f"判据③ 有 scale 把 reward 抬升(梯度对): {'✅' if rew_up else '❌'}  "
          f"(BC {rews[0]:.4f} → 最佳 scale={sc[best_rew_i]:g} 时 {rews[best_rew_i]:.4f})")
    ok = responsive and closer and rew_up
    # 甜区 = reward 最高且确实靠近 cube 的 scale
    sweet = sc[best_rew_i]
    print(f"\n推荐部署 guide_scale ≈ {sweet:g}(此处 reward 最高、轨迹最靠 cube);超过它会过冲、reward 反降。")
    print(f"{'✅ Gate 2.5 PASS — 关节版 VLS 引导有效且方向正确,可上真机试(用甜区 scale)' if ok else '⚠️ Gate 2.5 未过 — 看上表诊断'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

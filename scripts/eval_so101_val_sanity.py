#!/usr/bin/env python3
"""Offline action-prediction sanity check for the SO101 LoRA PiStar policy.

Goal: quantitatively answer "did the cold-start LoRA policy actually learn?" by comparing the
policy's predicted action chunk against ground-truth teleop actions on a held-out val set —
no robot, no simulator, read-only.

What it does
------------
* Loads the trained policy from a checkpoint with config ``pi05_star_so101_infer``
  (adv_ind_dropout=False) via ``policy_config.create_trained_policy``.
* For each val episode, samples frames t and feeds the observation (image/wrist_image/state +
  prompt + adv_ind="positive") to ``policy.infer``; compares the predicted action chunk
  (horizon H) against GT ``actions[t : t+H]``. Both are in the SAME raw joint space
  (policy.infer un-normalizes; GT stored raw), so they are directly comparable.
* Reports: overall MSE, per-joint MAE, gripper(dim5) error, first-step MSE, GT action std/scale,
  and a naive "hold current state" baseline for context (a learned policy should beat it clearly).
* Optionally probes adv_ind positive-vs-negative separability (expected ~0 for a pure-positive
  cold start — that is NORMAL, not a failure).

Run (server, GPU2, read-only):
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python scripts/eval_so101_val_sanity.py \
        --ckpt /data/users/szk/pistar/checkpoints/pi05_star_so101/so101_lora_v1/16000 \
        --val-root ~/.cache/huggingface/lerobot/meow/so101_cube_into_plate_v2_pistar_val
"""

from __future__ import annotations

import argparse
import os

# Set before importing jax. Keep the policy on GPU2; do not preallocate the whole card.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pathlib

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _dataset_task(ds, fallback: str) -> str:
    """Best-effort extraction of the (single) task string from a LeRobot dataset."""
    try:
        tasks = ds.meta.tasks
        if isinstance(tasks, dict):
            # could be {idx: task} or {task: idx}
            for k, v in tasks.items():
                return v if isinstance(v, str) else k
        if hasattr(tasks, "iloc"):  # pandas
            if "task" in getattr(tasks, "columns", []):
                return str(tasks["task"].iloc[0])
            # index may hold the task strings
            return str(tasks.index[0])
        if isinstance(tasks, (list, tuple)) and tasks:
            return str(tasks[0])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read task from dataset meta ({e}); using fallback prompt")
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="SO101 PiStar val offline action-prediction sanity")
    ap.add_argument("--ckpt", required=True, help="checkpoint step dir (contains params/ assets/)")
    ap.add_argument("--val-root", required=True, help="root of the val LeRobot dataset")
    ap.add_argument("--config-name", default="pi05_star_so101_infer")
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v2_pistar_val")
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"])
    ap.add_argument("--frames-per-episode", type=int, default=15, help="evenly sampled query frames per episode")
    ap.add_argument("--max-episodes", type=int, default=None, help="limit #episodes (smoke test)")
    ap.add_argument("--action-dim", type=int, default=6, help="real SO101 action dims to compare (drop padding)")
    ap.add_argument("--prompt", default="Pick up the cube and place it into the blue plate",
                    help="fallback prompt if task not readable from dataset")
    ap.add_argument("--probe-adv", action="store_true", help="also probe positive vs negative separability")
    args = ap.parse_args()

    # Imports that pull in jax/openpi happen after env vars are set.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config_name)
    horizon = cfg.model.action_horizon
    print(f"[cfg] {args.config_name} | action_horizon={horizon} | repo_id(in cfg)={cfg.data.repo_id}")

    print(f"[load] policy from {args.ckpt}")
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)

    val_root = pathlib.Path(args.val_root).expanduser()
    ds = LeRobotDataset(args.repo_id, root=val_root)
    task = _dataset_task(ds, args.prompt)
    print(f"[data] {val_root} | episodes={ds.num_episodes} frames={ds.num_frames} | task={task!r}")

    D = args.action_dim
    all_pred, all_gt = [], []          # stacked (N, H, D)
    baseline_pred = []                 # hold-current-state baseline (N, H, D)
    probe_diffs = []

    n_eps = ds.num_episodes if args.max_episodes is None else min(args.max_episodes, ds.num_episodes)
    for e in range(n_eps):
        ep_from = ds.episode_data_index["from"][e].item()
        ep_to = ds.episode_data_index["to"][e].item()
        ep_len = ep_to - ep_from
        # cache the episode's actions + states once
        acts = np.stack([_to_np(ds[ep_from + i]["actions"]).reshape(-1) for i in range(ep_len)]).astype(np.float32)
        states = np.stack([_to_np(ds[ep_from + i]["state"]).reshape(-1) for i in range(ep_len)]).astype(np.float32)

        valid_last = ep_len - horizon  # need t+horizon <= ep_len
        if valid_last <= 0:
            print(f"[ep {e}] len={ep_len} < horizon+1, skipped")
            continue
        ts = np.linspace(0, valid_last - 1, num=min(args.frames_per_episode, valid_last), dtype=int)
        ts = sorted(set(int(t) for t in ts))

        ep_mse = []
        for t in ts:
            fr = ds[ep_from + t]
            obs = {
                "observation/image": _to_np(fr["image"]),
                "observation/wrist_image": _to_np(fr["wrist_image"]),
                "observation/state": _to_np(fr["state"]).reshape(-1).astype(np.float32),
                "prompt": task,
                "adv_ind": args.adv_ind,
            }
            pred = _to_np(policy.infer(obs)["actions"])[:, :D]  # (H, D)
            gt = acts[t : t + horizon, :D]                      # (H, D)
            all_pred.append(pred)
            all_gt.append(gt)
            baseline_pred.append(np.broadcast_to(states[t, :D], (horizon, D)).copy())
            ep_mse.append(float(np.mean((pred - gt) ** 2)))

            if args.probe_adv:
                neg = _to_np(policy.infer({**obs, "adv_ind": "negative"})["actions"])[:, :D]
                probe_diffs.append(float(np.max(np.abs(pred - neg))))

        print(f"[ep {e}] len={ep_len} queried {len(ts)} frames | mean action MSE={np.mean(ep_mse):.5f}")

    if not all_pred:
        print("[result] no frames evaluated (episodes too short?)")
        return

    P = np.stack(all_pred)   # (N, H, D)
    G = np.stack(all_gt)
    B = np.stack(baseline_pred)

    mse = float(np.mean((P - G) ** 2))
    base_mse = float(np.mean((B - G) ** 2))
    per_joint_mae = np.mean(np.abs(P - G), axis=(0, 1))     # (D,)
    base_per_joint_mae = np.mean(np.abs(B - G), axis=(0, 1))
    gt_std = np.std(G, axis=(0, 1))                          # (D,)
    # near-term (h0) vs far-term (last) — far is harder over a 10-step open-loop chunk
    mse_h0 = float(np.mean((P[:, 0] - G[:, 0]) ** 2))
    base_mse_h0 = float(np.mean((B[:, 0] - G[:, 0]) ** 2))
    mse_hlast = float(np.mean((P[:, -1] - G[:, -1]) ** 2))
    base_mse_hlast = float(np.mean((B[:, -1] - G[:, -1]) ** 2))
    # normalized MSE (per joint): MSE_joint / Var_joint; <1 means better than predicting the mean
    var_joint = np.var(G, axis=(0, 1))
    mse_joint = np.mean((P - G) ** 2, axis=(0, 1))
    nmse_joint = mse_joint / np.clip(var_joint, 1e-8, None)
    base_nmse_joint = np.mean((B - G) ** 2, axis=(0, 1)) / np.clip(var_joint, 1e-8, None)

    jn = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"][:D]
    print("\n========== SO101 val offline sanity ==========")
    print(f"frames evaluated : {P.shape[0]}  (horizon={P.shape[1]}, dims={D}, adv_ind={args.adv_ind})")
    print("--- whole 10-step chunk ---")
    print(f"overall action MSE          : {mse:8.4f}   baseline(hold-state): {base_mse:8.4f}   "
          f"improvement: {(1 - mse / base_mse) * 100:5.1f}%")
    print("--- near vs far step (open-loop chunk degrades with horizon) ---")
    print(f"first-step (h0)  MSE        : {mse_h0:8.4f}   baseline: {base_mse_h0:8.4f}   "
          f"improvement: {(1 - mse_h0 / base_mse_h0) * 100:5.1f}%")
    print(f"last-step  (h9)  MSE        : {mse_hlast:8.4f}   baseline: {base_mse_hlast:8.4f}   "
          f"improvement: {(1 - mse_hlast / base_mse_hlast) * 100:5.1f}%")
    print(f"\n{'joint':<14}{'GT_std':>9}{'pred_MAE':>10}{'base_MAE':>10}{'nMSE':>8}{'base_nMSE':>11}")
    print("  (nMSE = MSE/Var; <1 beats predicting-the-mean; lower than base_nMSE = policy beats hold-state)")
    for i, name in enumerate(jn):
        print(f"{name:<14}{gt_std[i]:>9.2f}{per_joint_mae[i]:>10.3f}{base_per_joint_mae[i]:>10.3f}"
              f"{nmse_joint[i]:>8.3f}{base_nmse_joint[i]:>11.3f}")
    if args.probe_adv and probe_diffs:
        print(f"\nadv pos-vs-neg max|Δaction| : mean={np.mean(probe_diffs):.5f} max={np.max(probe_diffs):.5f}")
        print("  (≈0 expected for pure-positive cold start — separability only emerges after RECAP negatives)")
    print("==============================================")


if __name__ == "__main__":
    main()

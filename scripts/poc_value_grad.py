#!/usr/bin/env python3
"""PoC: 验证 "用 VLM value model 当推理时 reward 引导 pistar 去噪" 这条路的**门槛**。

核心问题(设计文档 13_vls_idea_adaptation_design.md §4.1 / §4.4):
  我们的 value model 是 **状态价值 V(s)、不吃 action**(value_model_config.py:67-90 的
  inputs_spec 里没有 action 字段)。要让它对 action 可微,只能走 surrogate S1:
  用 action 预测的"下一关节状态" s' 喂 value model,使 ∂V(s')/∂action != 0。
  这恰好可能撞上 memory 里记录的 **"critic flat-in-action"** 老根因
  (critic 会评状态、不会随 action 变,∂Q/∂a ≈ 0)。

  本脚本就是来**证伪/证实**这一点:逐帧测 ∂V/∂action 是不是 flat
  (norm≈0 / 方向乱 / 真值 action 的 V 并不比随机 action 高)。

严格按设计文档 §4.4 的判据:
  (a) ‖∂V/∂a‖ 是否显著非零(对比 random action 的梯度 norm)?
  (b) 沿 +grad 走一小步 V 是否上升、-grad 是否下降(方向有效性)?
  (c) 真值 action 的 V 是否高于随机 action 的 V(value 有没有判别力)?
  (d) 结论:∂V/∂a 是 flat 还是 informative?

action 语义(关键,见 convert_so101_v3_to_pistar.py:29-31):
  SO101 的 state(6,) 与 action(6,) 都是 **5 关节 + 夹爪的原始标定单位**,是
  **绝对位控**——action 直接就是"目标关节位置",不是增量。又因 value config 的
  action_horizon=1(value_model_config.py:37),每帧只有一个 action 向量。
  所以:
    - absolute 模式(默认): s' = action 本身(去掉 padding 的前 6 维)≈ 下一关节状态
    - delta 模式(对照):     s' = state + Σaction(若动作真是增量才对;此处用于排查)
  默认用 absolute,因为数据确凿是绝对位控、horizon=1,Σ 退化为单个绝对动作向量。

仅写脚本 + 语法自检,不在本机启动(value model + jax 在服务器)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):  # SRC_ROOT->openpi; REPO_ROOT->import scripts.train_value
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.value_model_config import ValueModelConfig


# --------------------------------------------------------------------------------------
# 1. 加载 value model ckpt
#    复用 train_value.py 的加载逻辑:config.create(rng) 建模 -> orbax 还原 step_xxxxxxxx,
#    优先用 ema_params(train_value.py:840 验证时也优先用 ema)。
# --------------------------------------------------------------------------------------
def load_value_model(value_ckpt: str, *, seed: int = 0):
    import flax.nnx as nnx
    import orbax.checkpoint as ocp

    config = ValueModelConfig()
    model = config.create(jax.random.key(seed))
    params = nnx.state(model)
    model_def = nnx.graphdef(model)

    ckpt_path = Path(value_ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"value ckpt 不存在: {ckpt_path}")
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(str(ckpt_path))
    # 优先 ema_params(推理用 EMA,与 train_value.py:840 一致),没有就用 params
    pure = restored.get("ema_params") or restored["params"]
    params.replace_by_pure_dict(pure)
    model = nnx.merge(model_def, params)
    model.eval()
    print(f"[ok] value model 已加载: {ckpt_path}  "
          f"(用 {'ema_params' if 'ema_params' in restored else 'params'}, step={restored.get('step')})")
    return model


# --------------------------------------------------------------------------------------
# 2. 从 v4_pistar(converted PiStar schema)取若干帧:Observation(已 resize/tokenize) + 真值 action
#    走和 train_value.py 完全相同的 value transform 管线,保证与训练时一致。
# --------------------------------------------------------------------------------------
def load_frames(data_root: str, repo_id: str, n_frames: int, *, tokenizer_path: str | None):
    import openpi.transforms as _transforms
    import openpi.policies.value_policy as value_policy
    from openpi.training import data_loader as _data_loader
    from openpi.training import config as _config
    import scripts.train_value as tv  # 复用 GemmaValueTokenizer / RemapValueLabelKey

    cfg = ValueModelConfig()
    local_dir = str(Path(data_root).expanduser() / repo_id)

    resolved_tok = tv._resolve_local_gemma_tokenizer_path(tokenizer_path)
    if resolved_tok is not None:
        print(f"[info] Gemma tokenizer: {resolved_tok}")

    # 与 train_value.build_value_data_config 等价,但保留 actions(value 管线本身会透传 actions)。
    data_config = _config.DataConfig(
        local_data_dir=local_dir,
        prompt_from_task=True,
        data_transforms=_transforms.Group(
            inputs=[tv.RemapValueLabelKey(), value_policy.ValueInputs()],
        ),
        model_transforms=_transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    tv.GemmaValueTokenizer(cfg.max_token_len, tokenizer_path=resolved_tok)
                ),
                _transforms.PadStatesAndActions(cfg.action_dim),
            ]
        ),
    )

    # action_horizon=1 -> 每帧 actions 形如 (1, action_dim);transform_dataset 应用全部 value 变换。
    ds = _data_loader.create_torch_dataset(data_config, cfg.action_horizon, cfg)
    ds = _data_loader.transform_dataset(ds, data_config, skip_norm_stats=True)

    frames = []
    n = min(n_frames, len(ds))
    # 在数据集里散开取(避开全是同一 episode 的相邻帧),拿"中段"帧(value 既非 ~0 也非 ~-1)。
    idxs = np.linspace(0, len(ds) - 1, num=n, dtype=int)
    for i in idxs:
        item = ds[int(i)]
        # item 是 dict:含 image/image_mask/state/tokenized_prompt(_mask)/actions/value
        raw = {k: item[k] for k in
               ("image", "image_mask", "state",
                "tokenized_prompt", "tokenized_prompt_mask")}
        # image_mask 的值可能是 Python bool / 0-d numpy(transform 输出),Observation.from_dict 的
        # jaxtyping 要求 Bool[Array]。用 jnp.asarray(jax array,不会被 beartype 解引用成标量;
        # 0-d numpy array 会被当 numpy.bool_ 拒绝)。
        if isinstance(raw.get("image_mask"), dict):
            raw["image_mask"] = {kk: jnp.asarray(bool(np.asarray(vv).reshape(-1)[0]))
                                 for kk, vv in raw["image_mask"].items()}
        obs = _model.Observation.from_dict(raw)
        actions = np.asarray(item["actions"], dtype=np.float32)  # (action_horizon, action_dim)
        value_label = float(np.asarray(item.get("value", np.nan)).reshape(-1)[0])
        frames.append((int(i), obs, actions, value_label))
    print(f"[ok] 取到 {len(frames)} 帧(数据集共 {len(ds)} 帧)")
    return frames, cfg


# --------------------------------------------------------------------------------------
# 3. surrogate s'(action) + reward(action) = V(images_当前, s'(action), prompt)
# --------------------------------------------------------------------------------------
def make_reward_fn(model, obs: _model.Observation, base_state: jnp.ndarray,
                   action_dim: int, action_mode: str):
    """返回 reward(a)->标量(可微);a 形状 = (action_horizon, action_dim)。"""

    def predicted_state(a: jnp.ndarray) -> jnp.ndarray:
        # a: (ah, ad)。surrogate S1。
        if action_mode == "absolute":
            # 绝对位控:执行 action chunk 后的"下一关节状态"≈最后一帧目标位置。
            # horizon=1 时就是这唯一的 action 向量。
            return a[-1]
        elif action_mode == "delta":
            # 增量假设(对照排查):s' = state + Σa。
            return base_state + jnp.sum(a, axis=0)
        raise ValueError(action_mode)

    def reward(a: jnp.ndarray) -> jnp.ndarray:
        s_next = predicted_state(a)[None, :]                       # (1, ad)
        obs2 = obs.replace(state=s_next)                          # flax struct.replace;图像/prompt 不变,只换 state
        v = model.compute_value(None, obs2)                       # (1,) 标量 value(越接近 0 越好)
        return jnp.squeeze(v)

    return reward


def add_batch(obs: _model.Observation) -> _model.Observation:
    """单帧 -> batch=1(value model 期望 leading batch 维)。"""
    return jax.tree.map(lambda x: jnp.asarray(x)[None] if x is not None else x, obs)


# --------------------------------------------------------------------------------------
# 4 + 5. 逐帧诊断 + 汇总判据
# --------------------------------------------------------------------------------------
def diagnose_frame(model, obs_b, true_action, cfg, action_mode, step_eps, rng):
    base_state = obs_b.state[0]                                   # (ad,)
    reward = make_reward_fn(model, obs_b, base_state, cfg.action_dim, action_mode)
    grad_fn = jax.value_and_grad(reward)

    a_true = jnp.asarray(true_action, dtype=jnp.float32)         # (ah, ad)
    v_true, g_true = grad_fn(a_true)
    g_true_norm = float(jnp.linalg.norm(g_true))

    # 随机 action(同尺度):用真值 action 的逐维 std 做扰动尺度,避免量纲偏差。
    scale = jnp.maximum(jnp.std(a_true), 1e-3)
    a_rand = a_true + scale * jax.random.normal(rng, a_true.shape)
    v_rand, g_rand = grad_fn(a_rand)
    g_rand_norm = float(jnp.linalg.norm(g_rand))

    # 方向有效性:沿 +g / -g 走一小步看 V 升降(g 是 ∂V/∂a 的上升方向)。
    g_unit = g_true / (jnp.linalg.norm(g_true) + 1e-12)
    v_plus = float(reward(a_true + step_eps * g_unit))
    v_minus = float(reward(a_true - step_eps * g_unit))

    return {
        "v_true": float(v_true),
        "v_rand": float(v_rand),
        "g_true_norm": g_true_norm,
        "g_rand_norm": g_rand_norm,
        "v_plus": v_plus,
        "v_minus": v_minus,
        "dir_ok": v_plus > float(v_true) > v_minus,   # 严格单调即方向有效
        "discrim_ok": float(v_true) > float(v_rand),  # 真值 V 高于随机 V
    }


def main():
    ap = argparse.ArgumentParser(description="PoC: ∂V/∂action 是否 flat(critic-flat 门槛诊断)")
    ap.add_argument("--value-ckpt", required=True,
                    help="value ckpt 目录(orbax step_xxxxxxxx),如 "
                         ".../pi05_star_so101/value_model/so101_recap_r1/step_00030000")
    ap.add_argument("--data-root", default=str(Path("~/.cache/huggingface/lerobot").expanduser()),
                    help="lerobot 数据根目录(repo-id 之上一级)")
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v4_pistar",
                    help="converted PiStar-schema 数据集名")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                    help="absolute: s'=action(SO101 绝对位控,默认); delta: s'=state+Σa(增量假设对照)")
    ap.add_argument("--step-eps", type=float, default=0.05, help="方向有效性试探步长")
    ap.add_argument("--tokenizer-path", default=None, help="Gemma3 tokenizer.model 本地路径(可选)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[cfg] action_mode={args.action_mode}  n_frames={args.n_frames}  "
          f"repo_id={args.repo_id}")
    if args.action_mode == "delta":
        print("[warn] delta 模式假设 action 是增量;SO101 实为绝对位控,delta 仅用于对照排查。")

    model = load_value_model(args.value_ckpt, seed=args.seed)
    frames, cfg = load_frames(args.data_root, args.repo_id, args.n_frames,
                              tokenizer_path=args.tokenizer_path)

    rng = jax.random.key(args.seed)
    rows = []
    print("\n=== 逐帧诊断 ===")
    print(f"{'idx':>6} {'v_label':>8} {'V(true)':>9} {'V(rand)':>9} "
          f"{'‖g_true‖':>10} {'‖g_rand‖':>10} {'V(+g)':>8} {'V(-g)':>8} {'dir':>4} {'disc':>5}")
    for fidx, obs, true_action, v_label in frames:
        rng, sub = jax.random.split(rng)
        obs_b = add_batch(obs)
        d = diagnose_frame(model, obs_b, true_action, cfg, args.action_mode, args.step_eps, sub)
        rows.append(d)
        print(f"{fidx:6d} {v_label:8.3f} {d['v_true']:9.4f} {d['v_rand']:9.4f} "
              f"{d['g_true_norm']:10.2e} {d['g_rand_norm']:10.2e} "
              f"{d['v_plus']:8.4f} {d['v_minus']:8.4f} "
              f"{'Y' if d['dir_ok'] else 'n':>4} {'Y' if d['discrim_ok'] else 'n':>5}")

    # ---- 汇总判据(设计文档 §4.4 a/b/c/d) ----
    g_true = np.array([r["g_true_norm"] for r in rows])
    g_rand = np.array([r["g_rand_norm"] for r in rows])
    dir_ok = np.mean([r["dir_ok"] for r in rows])
    disc_ok = np.mean([r["discrim_ok"] for r in rows])
    v_gap = np.mean([r["v_true"] - r["v_rand"] for r in rows])

    print("\n=== 汇总(§4.4 判据)===")
    print(f"(a) ‖∂V/∂a‖   true: mean={g_true.mean():.3e} median={np.median(g_true):.3e} "
          f"| rand: mean={g_rand.mean():.3e}")
    print(f"(b) 方向有效率(+g 升/-g 降): {dir_ok*100:.0f}%")
    print(f"(c) 判别力(V_true>V_rand 比例): {disc_ok*100:.0f}%  (平均 V_true-V_rand = {v_gap:+.4f})")

    # (d) 结论判定(经验阈值,可据实情调整):
    #   - flat 证据: ‖g‖ 量级极小(如 <1e-4)或方向有效率 ~随机(<60%)或判别力 ~50%。
    flat_grad = g_true.mean() < 1e-4
    weak_dir = dir_ok < 0.6
    weak_disc = disc_ok < 0.6
    print("\n(d) 结论:")
    if flat_grad:
        print("  -> ∂V/∂a 量级极小 => FLAT(确证 critic-flat 在推理时仍成立)。S1 引导无力,建议退 S3 零阶或放弃 A 路。")
    elif weak_dir or weak_disc:
        print("  -> 梯度非零但方向/判别力弱 => 半 flat,引导信号不可靠。需谨慎,可能仍需 S3。")
    else:
        print("  -> ∂V/∂a 非零且方向有效 + 真值 V 更高 => INFORMATIVE。S1 引导有戏,可推进 MVP 集成。")
    print("\n[note] value 越接近 0 越好(target=-(T-1-t)/T∈[-1,0])。方向有效性以 V 单调上升为准。")


if __name__ == "__main__":
    main()

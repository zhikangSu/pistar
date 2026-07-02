import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import so101_fk as _so101_fk
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# 关节版 VLS 用：可微 FK，对 (b,H,5) 弧度批量求 EE 位置 (b,H,3)。fk_pos 取单组 (5,)。
_vmap_fk_pos = jax.vmap(jax.vmap(_so101_fk.fk_pos))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pistar = config.pistar
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        if not self.pistar:
            return jnp.mean(jnp.square(v_t - u_t))
        else:
            # Compute per-timestep loss: (b, ah)
            per_timestep_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            # Compute per-sample loss: (b,)
            per_sample_loss = jnp.mean(per_timestep_loss, axis=-1)
            # Apply time-based weighting: weight = 0.5 * exp(-0.5*(1-time))
            # This emphasizes samples with more noise (larger time values)
            weight = 0.5 * jnp.exp(-0.5 * (1 - time))  # (b,)
            weighted_loss = per_sample_loss * weight  # (b,)
            # Return simple average of weighted losses (non-normalized, following original paper)
            return jnp.mean(weighted_loss)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # ---- 几何 reward 去噪引导（EE-delta 部署专用，零回归：缺省值下 Python if 短路）----
        reward_fn=None,          # callable(cube_xyz, plate_xyz, traj, grip)->scalar，None=不引导
        cube_xyz=None,           # (b,3) jnp，cube 在 base 系坐标（米），每 episode 变
        plate_xyz=None,          # (b,3) jnp，plate 在 base 系坐标（米）
        guide_scale: float = 0.0,  # 引导强度，0.0=不引导（默认，零回归）
        start_ratio: float = 0.6,  # 仅在 time<=start_ratio 的去噪后段注入引导
        # ---- 反归一化常量：去噪在【分位数归一化空间】([q01,q99]->[-1,1])，reward 需 base 系【米】----
        # raw=(xn+1)/2*(q99-q01)+q01；位置 delta 还要 ×ee_scale 得米（同 client）。缺省 None=不引导。
        state_q01=None, state_q99=None,   # (b,Sdim) state 分位数；取 [6:9]=ee_xyz
        act_q01=None, act_q99=None,       # (b,Adim) action 分位数；取 [:3]=pos, [9]=gripper
        ee_scale=None,                    # (3,) 每轴 EE delta 米尺度（仅 delta 模式用）
        absolute_ee: bool = False,        # True=绝对EE模型(action[:3]反归一化即绝对米,traj 直接=它,不 start_ee+cumsum)
        joint_ee: bool = False,           # True=关节绝对模型(action[:5]反归一化=RANGE关节值,×joint_to_rad→弧度→可微FK→EE米)
        joint_to_rad=None,                # (5,) RANGE→弧度的每关节比例(纯缩放)；仅 joint_ee 用
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # ---- 几何 reward 去噪引导（EE-delta 部署专用，零回归）----
            # reward_fn / guide_scale 是闭包捕获的【静态 Python 对象/常量】(非 tracer)，可用 Python if。
            # 缺省 reward_fn=None + guide_scale=0.0 → if 短路不进，代码路径与原 :282-284 逐字节相同。
            # 还要求 cube_xyz/plate_xyz 都在(都是闭包静态值):serve 即使 guide_scale>0,client 不带
            # --guide 时 cube/plate 缺省=None → 跳过引导=纯 BC,不崩。带 --guide 才注入坐标→引导。
            if reward_fn is not None and guide_scale and cube_xyz is not None and plate_xyz is not None:
                # 分位数反归一化到原始单位：raw = (xn+1)/2*(q99-q01)+q01。
                def _denorm(xn, q01, q99):
                    return (xn + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01

                def reward_of_x(x):
                    if joint_ee:
                        # 关节绝对模型: action 布局 [5 转动关节(RANGE), gripper, ...padding]。
                        # 反归一化→RANGE 关节值 → ×joint_to_rad 得弧度 → 可微 FK → EE 米。
                        qr = _denorm(x[:, :, :5], act_q01[:, None, :5], act_q99[:, None, :5])  # (b,H,5) RANGE
                        rad = qr * joint_to_rad[None, None, :]                                  # (b,H,5) 弧度
                        traj = _vmap_fk_pos(rad)                                                # (b,H,3) base 系米
                        # 夹爪取 [0,1] 归一化(=(xn+1)/2),与 EE 版 grip 语义一致(EE q01=0/q99=1 的 denorm 恰是它)。
                        # 关节版夹爪 norm_stats 是 RANGE(q99≈38),若直接 denorm 会把 r_grip_close 项放大~38× 扭曲梯度。
                        grip = (x[:, :, 5] + 1.0) * 0.5                                         # (b,H) 夹爪[0,1]
                        return reward_fn(cube_xyz, plate_xyz, traj, grip)
                    # action10 布局: [pos(3), 目标朝向6D, gripper]。x_t 在分位数归一化空间 → 反归一化回【米】。
                    pos_raw = _denorm(x[:, :, :3], act_q01[:, None, :3], act_q99[:, None, :3])  # (b,H,3)
                    grip = _denorm(x[:, :, 9], act_q01[:, 9], act_q99[:, 9])   # (b,H) 夹爪(~0/1)
                    if absolute_ee:
                        # 绝对EE模型: action[:3] 反归一化即绝对 base 系米 → traj 直接=它(无 start_ee/cumsum/ee_scale)。
                        traj = pos_raw
                    else:
                        # EE-delta 模型: pos 是 delta/ee_scale → ×ee_scale 得米增量,再 start_ee + cumsum。
                        ee_m = _denorm(observation.state[:, 6:9], state_q01[:, 6:9], state_q99[:, 6:9])
                        traj = ee_m[:, None, :] + jnp.cumsum(pos_raw * ee_scale[None, None, :], axis=1)
                    return reward_fn(cube_xyz, plate_xyz, traj, grip)

                g = jax.grad(reward_of_x)(x_t)
                # 梯度归一化到单位范数(同原版 VLS pi05_steer.py:394 grad/(‖grad‖+1e-8))。
                # 关键:几何 reward 是米制小数→原始 ∂R/∂x 量级很小,不归一化 guide_scale 要硬调到 20-50;
                # 归一化后 guide_scale 就是【动作空间步长】、可解释,原版用 ~80。
                g = g / (jnp.linalg.norm(g) + 1e-8)
                sc = guide_scale * jax.nn.sigmoid(12.0 * (start_ratio - time))
                sc = jnp.where(time <= start_ratio, sc, 0.0)
                # 符号：Euler 是 x_t + dt*v_t 且 dt<0；要 x_t 朝 +∇R（梯度上升），需 v_t -= sc*g。
                v_t = v_t - sc * g

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

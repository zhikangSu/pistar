import dataclasses
import enum
import logging
import socket

import tyro

from openpi.models import ee_steer as _ee_steer
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ---- 几何 reward 去噪引导（EE-delta 部署专用）----
    # guide_scale=0.0（默认）→ 零回归：sample_actions 内 Python if 短路，不注入引导。
    # guide_scale>0 → 启用 ee_steer.grasp_place_reward 引导。每-episode 的 cube_xyz/plate_xyz
    # 优先由【推理请求 obs】带进来（client 每帧附 "cube_xyz"/"plate_xyz"）；下方 CLI 仅作启动固定退路。
    guide_scale: float = 0.0
    # 仅在去噪后段 time<=start_ratio 注入引导。
    start_ratio: float = 0.6
    # 启动固定 cube/plate 坐标（base 系，米）。client 若在 obs 带 cube_xyz/plate_xyz 会逐帧覆盖之。
    cube_xyz: tuple[float, float, float] | None = None
    plate_xyz: tuple[float, float, float] | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _build_steer_sample_kwargs(args: Args) -> dict | None:
    """构造几何引导的 sample_kwargs。guide_scale<=0 返回 None（零回归，不传任何 guide kwarg）。"""
    if not args.guide_scale:
        return None
    import jax.numpy as jnp  # 局部 import，避免无引导时多余依赖

    sk: dict = {
        "reward_fn": _ee_steer.grasp_place_reward,
        "guide_scale": float(args.guide_scale),
        "start_ratio": float(args.start_ratio),
    }
    # 启动固定坐标（退路）；client 在 obs 带 cube_xyz/plate_xyz 时会在 Policy.infer 逐帧覆盖。
    if args.cube_xyz is not None:
        sk["cube_xyz"] = jnp.asarray(args.cube_xyz)[None, ...]
    if args.plate_xyz is not None:
        sk["plate_xyz"] = jnp.asarray(args.plate_xyz)[None, ...]
    logging.info("Geometric steering ENABLED: guide_scale=%s start_ratio=%s", args.guide_scale, args.start_ratio)
    return sk


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = _build_steer_sample_kwargs(args)
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

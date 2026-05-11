import sys
sys.path.append("./")

import argparse
from pathlib import Path

OPENPI_ROOT = Path(__file__).resolve().parents[1] / "src" / "robot" / "policy" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

from openpi.inference_model import _normalize_checkpoint_dir
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


DEFAULT_TRAIN_CONFIG = "pi05_piper"


def main():
    parser = argparse.ArgumentParser(description="Piper 单臂 websocket 推理服务（默认 pi05，兼容 PiStar）")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="checkpoint 根目录，应包含 params/ 或 model.safetensors")
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG, help="openpi 训练配置名")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--default-prompt", type=str, default=None, help="请求中缺省 prompt 时使用")
    args = parser.parse_args()

    train_config = _config.get_config(args.train_config)
    is_pistar = bool(getattr(getattr(train_config, "model", None), "pistar", False))

    checkpoint_dir = _normalize_checkpoint_dir(args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    metadata = dict(policy.metadata or {})
    metadata.update(
        {
            "deploy_mode": "pi05star" if is_pistar else "pi05",
            "train_config": args.train_config,
            "requires_adv_ind": is_pistar,
        }
    )

    print("=" * 50)
    print("Piper 单臂 websocket 推理服务")
    print("=" * 50)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"train_config: {args.train_config}")
    print(f"deploy_mode: {'pi05star' if is_pistar else 'pi05'}")
    print(f"listen: ws://{args.host}:{args.port}")
    print("=" * 50)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

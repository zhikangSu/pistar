#!/usr/bin/env bash
# Launch the openpi websocket policy server for the SO101 PiStar LoRA checkpoint.
#
# Runs on the machine that holds the checkpoint (laptop or server). The real robot
# client (scripts/so101_openpi_robot_client.py) connects to it over websocket.
#
# Env in the PiStar venv (has jax). Default loads ckpt 16000 with pi05_star_so101_infer
# (adv_ind_dropout=False). The infer config's weight_loader is irrelevant here — params
# are read from --policy.dir.
#
# Usage:
#   bash scripts/serve_so101_pistar.sh                         # defaults: ckpt 16000, port 8000, GPU 0
#   bash scripts/serve_so101_pistar.sh --ckpt /path/to/step   # different checkpoint step dir
#   CKPT=... CONFIG=... PORT=... GPU=... bash scripts/serve_so101_pistar.sh
#   bash scripts/serve_so101_pistar.sh --port 8000 --gpu 2 --ckpt .../16000
set -euo pipefail

CKPT="${CKPT:-/data/users/szk/pistar/checkpoints/pi05_star_so101/so101_lora_v1/16000}"
CONFIG="${CONFIG:-pi05_star_so101_infer}"
PORT="${PORT:-8000}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)   CKPT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --gpu)    GPU="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--ckpt DIR] [--config NAME] [--port N] [--gpu ID]"
      echo "  --ckpt   checkpoint step dir (contains params/ assets/)  [default: $CKPT]"
      echo "  --config train config name                               [default: $CONFIG]"
      echo "  --port   websocket port                                  [default: $PORT]"
      echo "  --gpu    CUDA_VISIBLE_DEVICES                             [default: $GPU]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$CKPT/params" ]]; then
  echo "❌ checkpoint params not found: $CKPT/params" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "════════════════════════════════════════════════════════════════"
echo "🚀  serving SO101 PiStar policy"
echo "    config : $CONFIG   (adv_ind_dropout=False; inference feeds adv_ind=positive)"
echo "    ckpt   : $CKPT"
echo "    port   : $PORT      host: 0.0.0.0   GPU: $GPU"
echo "    client : python scripts/so101_openpi_robot_client.py --server-host <this-ip> --port $PORT"
echo "════════════════════════════════════════════════════════════════"

cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$REPO_DIR/.venv/bin/python" scripts/serve_policy.py \
    --port="$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT"

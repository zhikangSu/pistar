#!/usr/bin/env python3
"""Visual-attention focus: where does the pi0.5* (pistar) VLM concentrate its visual attention?

Unlike viz_cross_attention.py (text query -> image key, "what does each word look at"), this script
answers "which image patches are the focus of the model's visual attention" by treating each image
patch as an attention *key* and measuring how much attention it *receives*.

Two methods (both produced):
  (1) Key-wise saliency (main, directly answers "where is the visual info concentrated"):
        For mean-over-heads, mean-over-layers attention A[q, s] (q=query, s=key),
        saliency(s) = mean over a group of VALID queries q of A[q, s].
        Query groups shown per camera:
          - ALL   : all valid prefix queries (image + valid text)
          - TEXT  : valid language tokens only (language-grounded visual focus)
          - IMAGE : all image tokens (visual self-importance / which patches other patches rely on)
        Only valid queries are summed (padding text rows produce uniform attention = noise).
  (2) Self-attention rollout (supplementary, classic ViT explainability):
        Per camera, take that camera's 256x256 image self-attention block per layer, row-normalize,
        Ahat_l = 0.5*A_l + 0.5*I (renormalized), roll out R = Ahat_{L-1} @ ... @ Ahat_0, then
        patch importance = mean over query rows of R. Highlights "hub" patches.

Each saliency map is the 16x16 SigLIP patch grid (per camera), bilinearly upsampled and overlaid on
the model's input image. Per-map min-max normalized for display.

Attention is captured exactly as in viz_cross_attention.py: gemma.CAPTURE_ATTN=True makes
Attention.sow the post-softmax probs (gemma.py: probs = softmax(masked_logits)), the scan maps the
"intermediates" collection over layers, and ToNNX stores them on the module -> read via nnx.state.
Captured shape: [depth, B, K(kv_heads), G(query_heads/kv), T(query), S(key)].

Prefix token layout (pi0.py embed_prefix): images first (obs.images order), then text. 3-cam SO101:
    base_0_rgb (fixed)       keys [0:256]
    left_wrist_0_rgb (wrist) keys [256:512]
    right_wrist_0_rgb(fixed_1)keys[512:768]
    text                     keys/queries [768:768+max_token_len]
NOTE: this is a prefix-only forward, so queries are image+text (no action-expert tokens).

Run (server, free GPU, avoid GPU2):
    CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /data/users/szk/pistar/.venv/bin/python scripts/viz_visual_attention.py \
        --ckpt /data/users/szk/pistar/checkpoints/pi05_star_so101_3cam/so101_lora_3cam/29999 \
        --val-root /data/users/szk/.cache/huggingface/lerobot/meow/so101_cube_into_plate_v3_pistar \
        --repo-id meow/so101_cube_into_plate_v3_pistar --out-dir /data/users/szk/pistar/outputs
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _dataset_task(ds, fallback: str) -> str:
    try:
        tasks = ds.meta.tasks
        if isinstance(tasks, dict):
            for k, v in tasks.items():
                return v if isinstance(v, str) else k
        if hasattr(tasks, "iloc"):
            if "task" in getattr(tasks, "columns", []):
                return str(tasks["task"].iloc[0])
            return str(tasks.index[0])
        if isinstance(tasks, (list, tuple)) and tasks:
            return str(tasks[0])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read task ({e}); using fallback prompt")
    return fallback


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="pi0.5* VLM visual-attention focus viz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v3_pistar")
    ap.add_argument("--config-name", default="pi05_star_so101_3cam_infer")
    ap.add_argument("--out-dir", default="/data/users/szk/pistar/outputs")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=None, help="frame index within episode (default: middle)")
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"])
    ap.add_argument("--prompt", default="Pick up the cube and place it into the blue plate")
    ap.add_argument("--layer", default="mean", help="'mean' (avg over all layers) or an int layer index")
    args = ap.parse_args()

    import openpi.models.gemma as gemma
    gemma.CAPTURE_ATTN = True

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0
    from openpi.models import tokenizer as _tok
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config_name)
    print(f"[cfg] {args.config_name} | max_token_len={cfg.model.max_token_len}")
    print(f"[load] policy from {args.ckpt}")
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    model = policy._model

    ds = LeRobotDataset(args.repo_id, root=pathlib.Path(args.val_root).expanduser())
    task = _dataset_task(ds, args.prompt)
    ep_from = ds.episode_data_index["from"][args.episode].item()
    ep_to = ds.episode_data_index["to"][args.episode].item()
    ep_len = ep_to - ep_from
    fidx = args.frame if args.frame is not None else ep_len // 2
    fidx = max(0, min(fidx, ep_len - 1))
    fr = ds[ep_from + fidx]
    print(f"[data] episode={args.episode} frame={fidx}/{ep_len} task={task!r}")

    obs_dict = {
        "observation/image": _to_np(fr["image"]),
        "observation/wrist_image": _to_np(fr["wrist_image"]),
        "observation/right_wrist_image": _to_np(fr["right_wrist_image"]),
        "observation/state": _to_np(fr["state"]).reshape(-1).astype(np.float32),
        "prompt": task,
        "adv_ind": args.adv_ind,
    }
    inputs = policy._input_transform(obs_dict)
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched)
    observation = _model.preprocess_observation(None, observation, train=False)
    cam_order = list(observation.images.keys())
    print(f"[obs] cameras (token order) = {cam_order}")

    # ---- prefix forward with attention capture ----
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions, mutable=["intermediates"])

    try:
        leaves = jtu.tree_leaves(nnx.state(model.PaliGemma.llm, nnx.Intermediate))
    except Exception:  # noqa: BLE001
        leaves = []
    probs_leaf = next((lf for lf in leaves if getattr(lf, "ndim", 0) == 6), None)
    if probs_leaf is None:
        probs_leaf = next((lf for lf in jtu.tree_leaves(nnx.state(model.PaliGemma.llm))
                           if getattr(lf, "ndim", 0) == 6), None)
    if probs_leaf is None:
        raise RuntimeError("could not find 6-dim attention probs in intermediates")
    probs = np.asarray(jax.device_get(probs_leaf)).astype(np.float32)
    depth, B, K, G, T, S = probs.shape
    print(f"[attn] probs shape (depth,B,K,G,T,S) = {probs.shape}")

    # mean over heads -> [depth, T, S]
    attn_layers = probs[:, 0].mean(axis=(1, 2))  # [depth, T, S]
    if args.layer == "mean":
        attn = attn_layers.mean(axis=0)
        layer_tag = "meanlayers"
    else:
        attn = attn_layers[int(args.layer)]
        layer_tag = f"layer{int(args.layer)}"

    # ---- spans ----
    max_tok = int(observation.tokenized_prompt.shape[1])
    n_img_tokens = T - max_tok
    n_cam = len(cam_order)
    patches_per_cam = n_img_tokens // n_cam
    grid = int(round(patches_per_cam ** 0.5))
    assert grid * grid == patches_per_cam
    text_start = n_img_tokens
    tok_mask = np.asarray(jax.device_get(observation.tokenized_prompt_mask[0])).astype(bool)
    valid_text_q = [text_start + i for i in range(max_tok) if tok_mask[i]]
    image_q = list(range(0, n_img_tokens))
    all_q = image_q + valid_text_q
    print(f"[layout] T={T} img_tokens={n_img_tokens} per_cam={patches_per_cam} grid={grid} "
          f"valid_text={len(valid_text_q)}")

    def keywise(query_idx, c0, c1):
        # mean over the given queries of attention received by keys c0:c1
        return attn[np.ix_(query_idx, list(range(c0, c1)))].mean(axis=0).reshape(grid, grid)

    def rollout(c0, c1):
        # per-cam image self-attention rollout across layers
        eye = np.eye(patches_per_cam, dtype=np.float32)
        R = None
        for l in range(depth):
            A = attn_layers[l, c0:c1, c0:c1].astype(np.float32)
            A = A / np.clip(A.sum(axis=-1, keepdims=True), 1e-9, None)
            Ah = 0.5 * A + 0.5 * eye
            Ah = Ah / np.clip(Ah.sum(axis=-1, keepdims=True), 1e-9, None)
            R = Ah if R is None else Ah @ R
        return R.mean(axis=0).reshape(grid, grid)  # importance per key patch

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    panels = [
        ("saliency: ALL queries", lambda c0, c1: keywise(all_q, c0, c1)),
        ("saliency: TEXT queries", lambda c0, c1: keywise(valid_text_q, c0, c1)),
        ("saliency: IMAGE queries", lambda c0, c1: keywise(image_q, c0, c1)),
        ("self-attn rollout", lambda c0, c1: rollout(c0, c1)),
    ]

    for ci, cam in enumerate(cam_order):
        c0, c1 = ci * patches_per_cam, (ci + 1) * patches_per_cam
        img = np.asarray(jax.device_get(observation.images[cam][0]))
        img01 = np.clip((img + 1.0) / 2.0, 0.0, 1.0)

        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.0 * (len(panels) + 1), 3.4))
        axes[0].imshow(img01)
        axes[0].set_title("input image", fontsize=10)
        axes[0].axis("off")
        for k, (title, fn) in enumerate(panels):
            ax = axes[k + 1]
            heat = _norm01(fn(c0, c1))
            ax.imshow(img01, extent=[0, grid, grid, 0])
            ax.imshow(heat, cmap="jet", alpha=0.55, extent=[0, grid, grid, 0],
                      interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Visual-attention focus (where visual info concentrates)  "
                     f"cam={cam}  ({layer_tag}, head-mean)  ep{args.episode}/f{fidx}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = out_dir / f"visual_attn_{cam}_{layer_tag}_ep{args.episode}_f{fidx}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(fname))
        print(f"[saved] {fname}")

    print("\n[done] PNGs:")
    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()

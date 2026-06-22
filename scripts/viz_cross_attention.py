#!/usr/bin/env python3
"""Cross-attention visualization: text token -> image patch, for the pi0.5* (pistar) VLM prefix.

For one sample frame, runs a single forward pass of the PaliGemma VLM *prefix* (image tokens +
language tokens, bidirectional prefix-LM attention) and extracts the post-softmax attention
weights from text-token *queries* onto image-patch *keys*. For every text token in the prompt we
draw one heatmap (16x16 SigLIP patch grid, bilinearly upsampled) overlaid on the input image,
laid out as a token-by-token grid (like a PaliGemma cross-attention figure).

How the attention is captured (no faking, real softmax weights)
---------------------------------------------------------------
`openpi.models.gemma.Attention` computes `probs = softmax(qk^T/sqrt(d))` at gemma.py:~233. We set
`gemma.CAPTURE_ATTN = True` *before* building the model, which (a) makes that Attention `self.sow`
the probs into the linen "intermediates" collection, and (b) adds "intermediates" to the
`nn.scan` `variable_axes` so per-layer probs get stacked along the layer axis. We then call the
LLM with `mutable=["intermediates"]` (passed straight through `nnx_bridge.ToNNX.__call__`) and pull
the stacked weights of shape [depth, B, K, G, T, S].

Prefix token layout (see pi0.py embed_prefix): images first (in obs.images insertion order), then
text. For the 3-cam SO101 model each SigLIP So400m/14@224 view = 16x16 = 256 tokens:
    base_0_rgb (fixed)      : keys [0:256]
    left_wrist_0_rgb (wrist): keys [256:512]
    right_wrist_0_rgb(fixed_1):keys[512:768]
    text prompt             : keys/queries [768 : 768+max_token_len]

Run (server, pick a free GPU; avoid GPU2):
    CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /data/users/szk/pistar/.venv/bin/python scripts/viz_cross_attention.py \
        --ckpt /data/users/szk/pistar/checkpoints/pi05_star_so101_3cam/so101_lora_3cam/29999 \
        --val-root /data/users/szk/.cache/huggingface/lerobot/meow/so101_cube_into_plate_v3_pistar \
        --repo-id meow/so101_cube_into_plate_v3_pistar \
        --out-dir /data/users/szk/pistar/outputs
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="pi0.5* VLM text->image cross-attention viz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--repo-id", default="meow/so101_cube_into_plate_v3_pistar")
    ap.add_argument("--config-name", default="pi05_star_so101_3cam_infer")
    ap.add_argument("--out-dir", default="/data/users/szk/pistar/outputs")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=None, help="frame index within episode (default: middle)")
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"])
    ap.add_argument("--prompt", default="Pick up the cube and place it into the blue plate")
    ap.add_argument("--layer", default="mean",
                    help="'mean' (avg over all layers), or an int layer index 0..depth-1")
    args = ap.parse_args()

    # gemma must learn to capture *before* the model graph is built.
    import openpi.models.gemma as gemma
    gemma.CAPTURE_ATTN = True

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
    model = policy._model  # nnx Pi0

    # ---- sample one frame ----
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

    # Same input transforms the policy uses at inference (LiberoInputs + TokenizePrompt + normalize).
    inputs = policy._input_transform(obs_dict)
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched)
    observation = _model.preprocess_observation(None, observation, train=False)

    cam_order = list(observation.images.keys())  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    print(f"[obs] cameras (token order) = {cam_order}")

    # ---- prefix forward with attention capture ----
    import flax.nnx as nnx

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    # ToNNX stores the sowed "intermediates" onto the module (it does NOT return them);
    # the call itself returns the model output ([prefix_out, None], kv_cache).
    model.PaliGemma.llm(
        [prefix_tokens, None], mask=attn_mask, positions=positions, mutable=["intermediates"]
    )

    # Pull the stacked attention probs leaf: [depth, B, K, G, T, S] (6-dim) from nnx Intermediate state.
    try:
        inter_state = nnx.state(model.PaliGemma.llm, nnx.Intermediate)
        leaves = jtu.tree_leaves(inter_state)
    except Exception:  # noqa: BLE001
        leaves = []
    probs_leaf = None
    for lf in leaves:
        if hasattr(lf, "ndim") and lf.ndim == 6:
            probs_leaf = lf
            break
    if probs_leaf is None:
        # Fallback: scan the full module state for the unique 6-dim attention leaf.
        leaves = jtu.tree_leaves(nnx.state(model.PaliGemma.llm))
        for lf in leaves:
            if hasattr(lf, "ndim") and lf.ndim == 6:
                probs_leaf = lf
                break
    if probs_leaf is None:
        raise RuntimeError(f"could not find attention probs in intermediates; leaf shapes={[getattr(l,'shape',None) for l in leaves]}")
    probs = np.asarray(jax.device_get(probs_leaf)).astype(np.float32)
    depth, B, K, G, T, S = probs.shape
    print(f"[attn] captured probs shape (depth,B,K,G,T,S) = {probs.shape}")

    # aggregate heads: mean over kv-heads*groups -> [depth, T, S]
    probs_heads_mean = probs[:, 0].mean(axis=(1, 2))  # B=0, mean over K,G
    if args.layer == "mean":
        attn = probs_heads_mean.mean(axis=0)  # [T, S]
        layer_tag = "meanlayers"
    else:
        li = int(args.layer)
        attn = probs_heads_mean[li]
        layer_tag = f"layer{li}"

    # ---- locate image / text spans ----
    # All cams use SigLIP So400m/14@224 -> 256 tokens each; text follows.
    max_tok = int(observation.tokenized_prompt.shape[1])
    n_img_tokens = T - max_tok
    n_cam = len(cam_order)
    patches_per_cam = n_img_tokens // n_cam
    grid = int(round(patches_per_cam ** 0.5))
    assert grid * grid == patches_per_cam, f"non-square patch grid: {patches_per_cam}"
    print(f"[layout] T={T} text_tokens={max_tok} img_tokens={n_img_tokens} per_cam={patches_per_cam} grid={grid}x{grid}")

    text_start = n_img_tokens
    tok_ids = np.asarray(jax.device_get(observation.tokenized_prompt[0]))
    tok_mask = np.asarray(jax.device_get(observation.tokenized_prompt_mask[0])).astype(bool)

    # token label strings via the paligemma sentencepiece model
    pt = _tok.PaligemmaTokenizer(max_len=max_tok)
    sp = pt._tokenizer

    def tok_label(i: int) -> str:
        tid = int(tok_ids[i])
        try:
            piece = sp.id_to_piece(tid)
        except Exception:  # noqa: BLE001
            piece = f"<{tid}>"
        piece = piece.replace("▁", "_")  # sentencepiece space marker
        if tid == sp.bos_id():
            return "<bos>"
        if tid == sp.eos_id():
            return "<eos>"
        if tid == 0:
            return "<pad>"
        return piece if piece else f"<{tid}>"

    valid_text_idx = [i for i in range(max_tok) if tok_mask[i]]
    print(f"[tokens] {len(valid_text_idx)} valid text tokens: "
          + " ".join(tok_label(i) for i in valid_text_idx))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for ci, cam in enumerate(cam_order):
        c0 = ci * patches_per_cam
        c1 = c0 + patches_per_cam
        img = np.asarray(jax.device_get(observation.images[cam][0]))  # (H,W,3) in [-1,1]
        img01 = np.clip((img + 1.0) / 2.0, 0.0, 1.0)

        n = len(valid_text_idx)
        ncols = min(8, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")

        for j, ti in enumerate(valid_text_idx):
            ax = axes[j]
            q = text_start + ti
            heat = attn[q, c0:c1].reshape(grid, grid)
            hmax = heat.max()
            heat_n = heat / hmax if hmax > 0 else heat
            ax.imshow(img01, extent=[0, grid, grid, 0])
            ax.imshow(heat_n, cmap="jet", alpha=0.5, extent=[0, grid, grid, 0],
                      interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax.set_title(tok_label(ti), fontsize=8)
            ax.axis("off")

        fig.suptitle(f"Cross-Attention: Text token -> Image patch\n"
                     f"cam={cam}  ({layer_tag}, head-mean)  ep{args.episode}/f{fidx}  "
                     f"prompt={task!r}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = out_dir / f"cross_attn_{cam}_{layer_tag}_ep{args.episode}_f{fidx}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(fname))
        print(f"[saved] {fname}")

    print("\n[done] PNGs:")
    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()

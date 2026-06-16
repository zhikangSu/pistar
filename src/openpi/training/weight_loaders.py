import dataclasses
import logging
import os
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

# Upstream (ybpy/pistar) Gemma checkpoint converter: turns the Google base Gemma
# orbax tree (FLAT ``transformer/...``) into the nested ``{embedder, layer_*,
# final_norm}`` structure and unwraps einsum ``/w`` via gemma's param_remapper.
try:
    from gemma.gm.ckpts import _checkpoint as _gm_ckpt  # type: ignore
except Exception:  # pragma: no cover - optional / env-dependent
    _gm_ckpt = None


def _maybe_convert_gemma_ckpt_tree(tree):
    """Convert a restored Gemma checkpoint tree to nested format (upstream helper).

    Returns the nested tree on success, or ``None`` if the gemma library converter
    is unavailable / fails (caller then falls back to a direct prefix mapping).
    """
    if _gm_ckpt is None:
        return None
    try:
        return _gm_ckpt._CheckpointTree(tree=tree).nested_tree
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Gemma ckpt convert via gemma lib failed (%s); using fallback mapping", exc)
        return None


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _repo_root() -> pathlib.Path:
    # weight_loaders.py lives at <repo>/src/openpi/training/weight_loaders.py
    return pathlib.Path(__file__).resolve().parents[3]


@dataclasses.dataclass(frozen=True)
class ValueModelWeightLoader(WeightLoader):
    """Loads pretrained VLM backbone weights into a ``ValueModel`` parameter tree.

    The ``ValueModel`` (see ``openpi.models.value_model``) combines a SigLIP vision
    tower (top-level ``img`` subtree) and a Gemma3-270M language model (top-level
    ``llm`` subtree) with value-function specific layers (``img_projection``,
    ``cross_attention``, ``cross_attn_norm``, ``value_head``). With the default
    ``freeze_mode=all_backbones`` the ``img`` and ``llm`` backbones are frozen, so
    they MUST be initialised from pretrained weights -- otherwise the frozen
    backbone stays random and the value model cannot learn anything useful.

    This loader replaces ONLY the two backbones and leaves every other parameter
    (i.e. the value-function specific layers) at its random init value:

      * ``img``  <- big_vision SigLIP npz, keys ``params/img/*``
                    (``<vlm_ckpt>/siglip2-so400m-patch14-224-jax/siglip2_so400m14_224.npz``)
      * ``llm``  <- Gemma3-270M orbax checkpoint, ``params/llm/*`` subtree
                    (``<vlm_ckpt>/gemma-3-270m/step_*``)

    The checkpoint root defaults to ``<repo>/assets/vlm_ckpt`` and can be overridden
    via the ``vlm_ckpt_dir`` argument or the ``OPENPI_VLM_CKPT_DIR`` environment
    variable.

    Shapes are validated per-leaf; a shape mismatch or a backbone leaf that cannot
    be filled from the source raises an error (no silent mis-loading). Source keys
    that do not correspond to a model parameter (e.g. the SigLIP ``MAPHead`` /
    text-tower, or the extra ValueModel layers stored in the Gemma checkpoint) are
    reported as a warning.
    """

    vlm_ckpt_dir: str | None = None
    siglip_npz_relpath: str = "siglip2-so400m-patch14-224-jax/siglip2_so400m14_224.npz"
    gemma_ckpt_relpath: str = "gemma-3-270m"

    def _resolve_root(self) -> pathlib.Path:
        root = self.vlm_ckpt_dir or os.getenv("OPENPI_VLM_CKPT_DIR")
        if root:
            path = pathlib.Path(root).expanduser()
        else:
            path = _repo_root() / "assets" / "vlm_ckpt"
        if not path.exists():
            raise FileNotFoundError(
                f"vlm_ckpt directory not found: {path}. Set vlm_ckpt_dir or OPENPI_VLM_CKPT_DIR."
            )
        return path

    def _load_siglip_img(self, root: pathlib.Path) -> dict[str, np.ndarray]:
        npz_path = root / self.siglip_npz_relpath
        if not npz_path.exists():
            raise FileNotFoundError(f"SigLIP npz not found: {npz_path}")
        prefix = "params/img/"
        out: dict[str, np.ndarray] = {}
        # NpzFile is lazy: only the img-tower arrays are realised (the text tower is skipped).
        with np.load(npz_path, allow_pickle=False) as npz:
            for key in npz.files:
                if key.startswith(prefix):
                    out["img/" + key[len(prefix):]] = np.asarray(npz[key])
        if not out:
            raise ValueError(f"No 'params/img/*' keys found in SigLIP npz: {npz_path}")
        return out

    def _resolve_gemma_ckpt(self, root: pathlib.Path) -> pathlib.Path:
        base = root / self.gemma_ckpt_relpath
        if not base.exists():
            raise FileNotFoundError(f"Gemma checkpoint dir not found: {base}")
        # The top-level dir IS the Google base Gemma3-270M orbax checkpoint
        # (FLAT layout: ``transformer/{embedder,layer_*,final_norm}``, bfloat16).
        # Do NOT descend into any ``step_*`` sub-dir -- that is a separate (ValueModel
        # training) artifact and would load value-tuned, not base, Gemma weights.
        return base

    @staticmethod
    def _gemma_flat_to_llm(flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map flattened Gemma checkpoint leaves to the model's ``llm/*`` keys.

        Detects the Gemma subtree prefix and re-roots it under ``llm/``. Handles:
          * Google base Gemma orbax: leaves ``transformer/{embedder,layer_*,final_norm}/...``;
          * a ValueModel-format checkpoint: leaves ``params/llm/...`` (img/value_head ignored);
          * an already-``llm/``-rooted tree;
          * a raw Gemma checkpoint: leaves ``embedder/...`` / ``layer_*/...`` / ``final_norm/...``.
        """
        keys = list(flat)
        if any(k.startswith("transformer/") for k in keys):
            prefix = "transformer/"
        elif any(k.startswith("params/llm/") for k in keys):
            prefix = "params/llm/"
        elif any(k.startswith("llm/") for k in keys):
            prefix = "llm/"
        elif any(k.startswith(("embedder/", "final_norm/")) or k.startswith("layer_") for k in keys):
            prefix = ""
        else:
            raise ValueError(f"Could not locate Gemma weights in checkpoint. Sample keys: {keys[:6]}")

        out: dict[str, np.ndarray] = {}
        for k, v in flat.items():
            if prefix:
                if k.startswith(prefix):
                    out["llm/" + k[len(prefix):]] = np.asarray(v)
            elif k.startswith(("embedder/", "final_norm/")) or k.startswith("layer_"):
                out["llm/" + k] = np.asarray(v)
        return out

    def _load_gemma_llm(self, root: pathlib.Path) -> dict[str, np.ndarray]:
        import jax
        import orbax.checkpoint as ocp

        ckpt_path = self._resolve_gemma_ckpt(root)
        # restore_type=np.ndarray -> host numpy, so the checkpoint's original
        # (multi-device) sharding is irrelevant. This orbax version requires the
        # restore item to match the full on-disk tree, so we restore everything
        # (incl. any ema_params / step) and then pick out the Gemma 'llm' subtree.
        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(str(ckpt_path))
            item = dict(metadata)
            restored = ckptr.restore(
                str(ckpt_path),
                ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item
                    ),
                ),
            )
        # Prefer the upstream gemma converter (FLAT transformer/* -> nested
        # {embedder, layer_*, final_norm}, einsum '/w' unwrapped by param_remapper);
        # fall back to a direct prefix mapping if the gemma lib path is unavailable.
        nested = _maybe_convert_gemma_ckpt_tree(restored)
        if nested is not None:
            flat = flax.traverse_util.flatten_dict(nested, sep="/")
            out = {"llm/" + k: np.asarray(v) for k, v in flat.items()}
            if out:
                logger.info("ValueModelWeightLoader: gemma tree converted via gemma.gm.ckpts (%d leaves)", len(out))
                return out
        flat = flax.traverse_util.flatten_dict(restored, sep="/")
        return self._gemma_flat_to_llm(flat)

    def load(self, params: at.Params) -> at.Params:
        root = self._resolve_root()

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        result = dict(flat_ref)

        siglip = self._load_siglip_img(root)
        gemma = self._load_gemma_llm(root)

        filled: set[str] = set()
        shape_mismatch: list[str] = []
        unused: list[str] = []

        for source_name, source in (("siglip(img)", siglip), ("gemma(llm)", gemma)):
            n_used = 0
            for k, v in source.items():
                tk = k
                # Gemma stores some einsum weights as ".../w" (e.g. mlp/gating_einsum/w),
                # while this repo's Gemma module exposes the param directly (mlp/gating_einsum).
                # Drop a trailing "/w" when only the stripped key exists in the model.
                if tk not in flat_ref and tk.endswith("/w") and tk[: -len("/w")] in flat_ref:
                    tk = tk[: -len("/w")]
                if tk not in flat_ref:
                    unused.append(f"{source_name}:{k}")
                    continue
                ref = flat_ref[tk]
                if tuple(v.shape) != tuple(ref.shape):
                    shape_mismatch.append(f"{tk}: source{tuple(v.shape)} != model{tuple(ref.shape)}")
                    continue
                result[tk] = v.astype(ref.dtype) if v.dtype != ref.dtype else v
                filled.add(tk)
                n_used += 1
            logger.info("ValueModelWeightLoader: loaded %d leaves from %s", n_used, source_name)

        if shape_mismatch:
            raise ValueError(
                "ValueModelWeightLoader: shape mismatch between source and model params:\n  "
                + "\n  ".join(shape_mismatch)
            )

        # Every backbone leaf (img/* and llm/*) MUST be filled from the pretrained source.
        backbone_ref = [k for k in flat_ref if k.startswith("img/") or k.startswith("llm/")]
        missing = sorted(k for k in backbone_ref if k not in filled)
        if missing:
            raise ValueError(
                f"ValueModelWeightLoader: {len(missing)} backbone leaves were NOT filled from "
                "pretrained weights (frozen-random backbone would result):\n  "
                + "\n  ".join(missing[:20])
                + ("\n  ..." if len(missing) > 20 else "")
            )

        kept_random = sorted(k for k in flat_ref if k not in filled)
        logger.info(
            "ValueModelWeightLoader: %d backbone leaves loaded, %d value-head/fusion leaves kept random (%s)",
            len(filled),
            len(kept_random),
            ", ".join(sorted({k.split("/")[0] for k in kept_random})),
        )
        if unused:
            logger.warning(
                "ValueModelWeightLoader: %d source keys had no matching model param and were ignored "
                "(expected: SigLIP MAPHead/text-tower, extra ValueModel layers in the Gemma checkpoint). "
                "First few: %s",
                len(unused),
                ", ".join(unused[:8]),
            )

        return flax.traverse_util.unflatten_dict(result, sep="/")

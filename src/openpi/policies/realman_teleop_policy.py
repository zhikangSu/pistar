import dataclasses
import io
from typing import Any

import einops
import numpy as np
from PIL import Image

from openpi import transforms


def make_realman_teleop_example() -> dict:
    """Creates a random input example for the RealMan teleop policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class RealmanTeleopInputs(transforms.DataTransformFn):
    """Inputs for RealMan dual-arm teleop data.

    Expected inputs after repacking:
    - images: dict with cam_high, cam_left_wrist, cam_right_wrist
    - state: [14]
    - actions: [action_horizon, 14]
    """

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        base_image = _to_hwc_uint8(in_images["cam_high"])

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": _to_hwc_uint8(in_images["cam_left_wrist"]),
            "right_wrist_0_rgb": _to_hwc_uint8(in_images["cam_right_wrist"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "adv_ind" in data:
            inputs["adv_ind"] = data["adv_ind"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealmanTeleopOutputs(transforms.DataTransformFn):
    """Outputs for RealMan dual-arm teleop policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}


def _to_hwc_uint8(image: Any) -> np.ndarray:
    image = _materialize_image(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={image.shape}")
    return np.ascontiguousarray(image)


def _materialize_image(image: Any) -> np.ndarray:
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            with Image.open(io.BytesIO(image["bytes"])) as image_obj:
                return np.asarray(image_obj.convert("RGB"))
        if image.get("path"):
            with Image.open(image["path"]) as image_obj:
                return np.asarray(image_obj.convert("RGB"))
        raise ValueError(f"Unsupported image dict keys: {list(image.keys())}")

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    arr = np.asarray(image)
    if arr.shape == ():
        return _materialize_image(arr.item())
    return arr

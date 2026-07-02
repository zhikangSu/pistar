from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup.
            # 几何引导的 reward_fn(函数)/guide_scale/start_ratio(在 Python if/标量分支里用) 必须标 static，
            # 否则 jax.jit 会把它们当 traced 数组而报 "Error interpreting argument ... reward_fn"。
            # 缺省(reward_fn=None,guide_scale=0.0)时也安全：static None/0.0，if 短路、零回归。
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=("reward_fn", "guide_scale", "start_ratio", "absolute_ee", "joint_ee"),
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # ---- 几何 reward 去噪引导：从 obs 里取出每-episode 的目标坐标（不进强类型 Observation struct）----
        # client 每次 infer 时可在 obs 里附 "cube_xyz"/"plate_xyz" (3,) 数组。这里在 _input_transform
        # （RepackTransform 会丢弃 map 外的 key）之前 pop 出来，稍后路由进 sample_kwargs。
        # 仅当 sample_kwargs 已配 guide 引导（reward_fn/guide_scale）时才生效；否则无害忽略。
        steer_xyz = {}
        for _k in ("cube_xyz", "plate_xyz"):
            if _k in inputs:
                steer_xyz[_k] = inputs.pop(_k)
        # 可选:每-请求覆盖 guide_scale(静态标量)。用于实时调参、以及 BC(0) vs VLS(>0) 对照可视化——
        # 同一 serve 既能出引导轨迹又能出不引导轨迹,不必重启或开两个 serve。缺省=用 serve 启动时的固定值。
        override_guide_scale = inputs.pop("guide_scale", None)
        # 可选:固定采样噪声种子。flow-matching 每次 infer 抽新噪声→同输入也会有~cm 级抖动。传同一
        # noise_seed 给 BC(guide=0) 与 VLS(guide>0) 两次推理,差异才纯是引导效果(不被采样噪声淹没)。
        noise_seed = inputs.pop("noise_seed", None)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            if noise_seed is not None:
                sample_rng_or_pytorch_device = jax.random.key(int(np.asarray(noise_seed).reshape(-1)[0]))
            else:
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        # 路由 per-episode 目标坐标进 sample_kwargs（加 batch 维对齐 (b,3)）。
        # 仅 JAX 模型支持几何引导；PyTorch 路径不注入（其 sample_actions 不识别该 kwarg）。
        if steer_xyz and not self._is_pytorch_model:
            for _k, _v in steer_xyz.items():
                _v = jnp.asarray(_v)
                if _v.ndim == 1:  # (3,) -> (1, 3)
                    _v = _v[None, ...]
                sample_kwargs[_k] = _v
        # per-request guide_scale 覆盖（静态标量；不同值各编译一次并缓存）。
        if override_guide_scale is not None and not self._is_pytorch_model:
            sample_kwargs["guide_scale"] = float(np.asarray(override_guide_scale).reshape(-1)[0])
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

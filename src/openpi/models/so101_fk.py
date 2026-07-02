#!/usr/bin/env python
"""SO101 可微正运动学(JAX) —— VLS(关节版)梯度引导的唯一可微环节。

q(5 转动关节弧度) → gripper_frame_link 在 base 系的 4x4 位姿。对 q 可 jax.grad/jax.jacobian。
参数从 /home/meow/HIL-RL--SO101/assets/so101/so101_new_calib.urdf 提取(5 revolute, axis 均为局部 z;
+ 固定 gripper_frame_joint)。与 placo(lerobot.model.kinematics) 同一 URDF → Gate 1a 对齐验证。

⚠️ 单位:本 FK 取【弧度】。placo 客户端 fk(kin, q_range, calib) 内部 RANGE_M100_100→度→forward_kinematics。
   对齐时 q_rad = joints_range_to_degrees(q_range, calib) * pi/180。
⚠️ 夹爪不进 FK 链(开合不改 TCP 基座位姿)。
⚠️ 各 joint 的【常量】origin 变换(不随 q 变)用 **numpy** 预算成 concrete 数组,**不能用 jnp**:
   若本模块在某次 jit trace 内才首次 import(serve 只经 pi0.py 间接 import、非 eager),jnp 常量会被
   建成 tracer 并泄漏出 trace(UnexpectedTracerError)。numpy 常量永远 concrete、与 trace 无关,
   且与 traced 的 q 旋转做 matmul 仍完全可微。只有依赖 q 的那一块(_Rz)用 jnp。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# URDF 链 base_link→...→gripper_frame_link 各 joint 的 origin(xyz, rpy)。5 revolute axis 均 (0,0,1)。
_JOINTS = [  # (name, xyz, rpy)
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624),   (3.14159, 4.18253e-17, -3.14159)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542),   (-1.5708, -1.5708, 0.0)),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),     (-3.63608e-16, 8.74301e-16, 1.5708)),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),      (4.02456e-15, 8.67362e-16, -1.5708)),
    ("wrist_roll",    (5.55112e-17, -0.0611, 0.0181),      (1.5708, 0.0486795, 3.14159)),
]
_GRIPPER_FRAME = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))  # 固定 gripper_frame_joint


def _rpy_to_R_np(rpy):
    """URDF rpy = 固定轴(外旋) XYZ = Rz(yaw)·Ry(pitch)·Rx(roll)。纯 numpy(常量预算用)。"""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    return Rz @ Ry @ Rx


def _T_np(xyz, R):
    """4x4 齐次变换,纯 numpy(常量预算用)。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _Rz4(q):
    """绕局部 z 转 q 的 4x4 齐次旋转(jnp;q 可为 tracer,FK 唯一依赖 q 的环节)。"""
    c, s = jnp.cos(q), jnp.sin(q)
    return jnp.array([[c, -s, 0.0, 0.0],
                      [s,  c, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])


# 各 joint 的常量 origin 变换(不随 q 变)—— 纯 numpy concrete,永不变 tracer。
_ORIGIN_T = np.stack([_T_np(xyz, _rpy_to_R_np(rpy)) for _, xyz, rpy in _JOINTS])  # (5,4,4)
_GRIPPER_T = _T_np(_GRIPPER_FRAME[0], _rpy_to_R_np(_GRIPPER_FRAME[1]))            # (4,4)


def fk(q_rad):
    """q_rad:(5,) 5 个转动关节弧度 → gripper_frame_link 在 base 系 4x4 位姿。可 jax.grad。"""
    T = jnp.eye(4)
    for i in range(5):
        # child = parent · origin(常量 numpy) · Rz(q)(jnp,可微)  —— revolute axis = 局部 z
        T = T @ _ORIGIN_T[i] @ _Rz4(q_rad[i])
    T = T @ _GRIPPER_T
    return T


def fk_pos(q_rad):
    """只取 EE 位置 (3,)（reward 用）。"""
    return fk(q_rad)[:3, 3]


if __name__ == "__main__":
    q = jnp.zeros(5)
    print("fk(0) EE pos =", np.round(np.asarray(fk_pos(q)), 4))

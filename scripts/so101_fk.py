#!/usr/bin/env python
"""SO101 可微正运动学(JAX) —— VLS(关节版)梯度引导的唯一可微环节。

q(5 转动关节弧度) → gripper_frame_link 在 base 系的 4x4 位姿。纯 jnp、可 jax.grad/jax.jacobian。
参数从 /home/meow/HIL-RL--SO101/assets/so101/so101_new_calib.urdf 提取(5 revolute, axis 均为局部 z;
+ 固定 gripper_frame_joint)。与 placo(lerobot.model.kinematics) 同一 URDF → Gate 1a 对齐验证。

⚠️ 单位:本 FK 取【弧度】。placo 客户端 fk(kin, q_range, calib) 内部 RANGE_M100_100→度→forward_kinematics。
   对齐时 q_rad = joints_range_to_degrees(q_range, calib) * pi/180。
⚠️ 夹爪不进 FK 链(开合不改 TCP 基座位姿)。
"""
from __future__ import annotations

import jax.numpy as jnp

# URDF 链 base_link→...→gripper_frame_link 各 joint 的 origin(xyz, rpy)。5 revolute axis 均 (0,0,1)。
_JOINTS = [  # (name, xyz, rpy)
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624),   (3.14159, 4.18253e-17, -3.14159)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542),   (-1.5708, -1.5708, 0.0)),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),     (-3.63608e-16, 8.74301e-16, 1.5708)),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),      (4.02456e-15, 8.67362e-16, -1.5708)),
    ("wrist_roll",    (5.55112e-17, -0.0611, 0.0181),      (1.5708, 0.0486795, 3.14159)),
]
_GRIPPER_FRAME = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))  # 固定 gripper_frame_joint


def _rpy_to_R(rpy):
    """URDF rpy = 固定轴(外旋) XYZ = Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    r, p, y = rpy
    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)
    cy, sy = jnp.cos(y), jnp.sin(y)
    Rx = jnp.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = jnp.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rz = jnp.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    return Rz @ Ry @ Rx


def _T(xyz, R):
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(jnp.asarray(xyz, dtype=R.dtype))
    return T


def _Rz(q):
    c, s = jnp.cos(q), jnp.sin(q)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# 预算各 joint 的常量 origin 变换(不随 q 变)。
_ORIGIN_T = [_T(xyz, _rpy_to_R(rpy)) for _, xyz, rpy in _JOINTS]
_GRIPPER_T = _T(_GRIPPER_FRAME[0], _rpy_to_R(_GRIPPER_FRAME[1]))


def fk(q_rad):
    """q_rad:(5,) 5 个转动关节弧度 → gripper_frame_link 在 base 系 4x4 位姿。可 jax.grad。"""
    T = jnp.eye(4)
    for i in range(5):
        # child = parent · origin · Rz(q)  (revolute axis = 局部 z)
        T = T @ _ORIGIN_T[i] @ _T((0.0, 0.0, 0.0), _Rz(q_rad[i]))
    T = T @ _GRIPPER_T
    return T


def fk_pos(q_rad):
    """只取 EE 位置 (3,)（reward 用）。"""
    return fk(q_rad)[:3, 3]


if __name__ == "__main__":
    import numpy as np
    q = jnp.zeros(5)
    print("fk(0) EE pos =", np.round(np.asarray(fk_pos(q)), 4))

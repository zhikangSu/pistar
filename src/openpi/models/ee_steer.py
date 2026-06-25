"""几何 reward —— 去噪引导用的可微抓放 reward（纯 jnp，对 EE 轨迹/夹爪可微）。

搬自 /home/meow/SO101/scripts/geom_reward_poc.py 的 grasp_place_reward。核心命题：
当 pistar 的 action 本身就是 EE-delta 时，EE 轨迹 = start_ee + cumsum(action[:, :3])
是纯张量、天然对 action 可微（无 FK），所以建立在 EE 轨迹上的几何 reward 对 action
有非 0 梯度。这正是 value 路死掉（∂V/∂a≡0，不吃 action）、EE 路活下来的根本区别。

用法（pi0.py sample_actions 去噪引导）：
  from openpi.models import ee_steer
  R = ee_steer.grasp_place_reward(cube_xyz, plate_xyz, traj, grip, z_table)
  其中 traj:(B,T,3) = start_ee + cumsum(action[..., :3]); grip:(B,T) = action[..., 3]。
  R 是标量，对 traj/grip（进而对 action）可微。
"""
import jax
import jax.numpy as jnp


def grasp_place_reward(cube_xyz, plate_xyz, traj, grip, z_table=0.0):
    """分阶段抓放 reward（可微，纯 jnp + sigmoid 软门控，无 Python if）。返回标量。

    Args:
        cube_xyz:  (3,) jnp, cube 在 base 系坐标（米）。
        plate_xyz: (3,) jnp, plate 在 base 系坐标（米）—— MVP 抓取阶段未直接使用，预留放置阶段。
        traj:      (B, T, 3) jnp, EE 轨迹 = start_ee + cumsum(action[..., :3])。
        grip:      (B, T) jnp, 每步夹爪指令（action[..., 3]）。
        z_table:   float, 桌面高度（预留，MVP 未使用）。
    """
    B, T, _ = traj.shape
    t = jnp.arange(T)
    w = jnp.where(t >= int(0.6 * T), 3.0, 1.0)
    w = w / w.sum()                                              # 时间权重,后段更重

    pre_grasp = cube_xyz + jnp.array([0.0, 0.0, 0.05])          # cube 正上方 5cm
    d_approach = jnp.linalg.norm(traj - pre_grasp, axis=-1)     # (B,T)
    r_approach = -(d_approach ** 2 * w).sum(axis=1)             # R1 接近

    d_grasp = jnp.linalg.norm(traj[:, -1, :] - cube_xyz, axis=-1)
    r_grasp = -(d_grasp ** 2) * 2.0                            # R2 末端下压到 cube

    near_cube = jax.nn.sigmoid(-(d_approach - 0.03) / 0.01)     # 离 cube<3cm 软门控
    r_grip_close = (near_cube * grip).mean(axis=1)             # R3 近 cube 处鼓励夹爪闭合
    # 注:doc 伪代码这里写 .sum(50步)→量级~25 会淹没几何项；改 .mean 平衡,让"末端贴cube"主导

    reward = r_approach + r_grasp + 0.1 * r_grip_close          # MVP: 抓取(几何项主导)
    return reward.mean()

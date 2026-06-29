#!/usr/bin/env python
"""Gate 1a/1b: 验证 so101_fk.py 的 JAX FK 与 placo(lerobot.model.kinematics) 一致 + 雅可比可用。

env 拆分:placo 在 lerobot env、jax 在 pistar .venv。两步:
  1) lerobot-python gate1a_validate_fk.py --mode placo  # 随机 1000 组关节角(度)→placo FK→存 npz
  2) pistar/.venv/python gate1a_validate_fk.py --mode jax    # 我的 JAX FK 对比 + 雅可比有限差分校验
Gate 1a: max EE 位置误差 < 1e-4 m。Gate 1b: jax.jacobian 有限非 NaN 且与数值雅可比吻合(rtol 1e-4)。
"""
import argparse
import os
import sys

import numpy as np

NPZ = "/tmp/gate1a_fk.npz"
N = 1000
# SO101 转动关节限位(度),取自部署 client SO101_JOINT_MIN/MAX。
JMIN = np.array([-36.0, -107.0, -37.0, 41.0, -46.0])
JMAX = np.array([66.0, 45.0, 99.0, 99.0, 65.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["placo", "jax"])
    a = ap.parse_args()

    if a.mode == "placo":
        # 复用部署 client 的 placo FK（calibration=None → q 当作度直接喂 forward_kinematics）。
        import importlib.util
        scr = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("O", os.path.join(scr, "so101_openpi_robot_client_orient.py"))
        O = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(O)
        kin = O.make_kinematics(O.DEFAULT_URDF)
        rng = np.random.default_rng(0)
        deg = rng.uniform(JMIN, JMAX, size=(N, 5))   # 随机 5 关节角(度)
        ee = np.zeros((N, 3))
        for i in range(N):
            q6 = np.concatenate([deg[i], [0.0]])      # +夹爪(不影响 gripper_frame_link)
            ee[i] = O.fk(kin, q6, None)[:3, 3]         # placo FK(度)
        np.savez(NPZ, deg=deg, ee_placo=ee)
        print(f"✅ placo: 存 {N} 组到 {NPZ}  ee[0]={np.round(ee[0],4)}")

    else:  # jax
        import jax, jax.numpy as jnp
        scr = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, scr)
        import so101_fk as FK
        d = np.load(NPZ)
        deg, ee_placo = d["deg"], d["ee_placo"]
        rad = np.deg2rad(deg)
        ee_jax = np.array([np.asarray(FK.fk_pos(jnp.asarray(rad[i]))) for i in range(len(rad))])
        err = np.linalg.norm(ee_jax - ee_placo, axis=1)
        print(f"=== Gate 1a: JAX FK vs placo ({len(rad)} 组) ===")
        print(f"  位置误差(m): mean {err.mean():.2e}  max {err.max():.2e}  (Gate: max<1e-4)")
        print(f"  {'✅ PASS' if err.max() < 1e-4 else '❌ FAIL — 检查 rpy 约定/轴/单位'}")
        print(f"  ee_jax[0]={np.round(ee_jax[0],4)}  ee_placo[0]={np.round(ee_placo[0],4)}")
        # Gate 1b: 雅可比可用 + 与有限差分吻合
        q0 = jnp.asarray(rad[0])
        J = np.asarray(jax.jacobian(FK.fk_pos)(q0))      # (3,5)
        eps = 1e-5
        Jnum = np.zeros((3, 5))
        for k in range(5):
            dq = np.zeros(5); dq[k] = eps
            Jnum[:, k] = (np.asarray(FK.fk_pos(q0 + dq)) - np.asarray(FK.fk_pos(q0 - dq))) / (2 * eps)
        jerr = np.abs(J - Jnum).max()
        print(f"=== Gate 1b: jax.jacobian vs 有限差分 ===")
        print(f"  雅可比 finite={np.all(np.isfinite(J))}  max|J_autodiff - J_numeric|={jerr:.2e} (Gate: <1e-4)")
        print(f"  {'✅ PASS' if (np.all(np.isfinite(J)) and jerr < 1e-4) else '❌ FAIL'}")


if __name__ == "__main__":
    main()

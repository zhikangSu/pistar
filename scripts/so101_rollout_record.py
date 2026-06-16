#!/usr/bin/env python3
"""SO101 real-robot DAgger ROLLOUT recorder for the openpi PiStar policy server.

Runs in the **conda `lerobot` env** on the laptop. Builds on the proven pure-inference
loop in `so101_openpi_robot_client.py` (imported, not duplicated) and adds:

  1. SO101 **leader-arm intervention** — faithful port of the HIL-RL/SiLRI
     `SO101LeaderIntervention` leader logic (torque-ordering, mirror-before-torque,
     manual takeover, smooth release). The leader is the SAME physical SO101 leader the
     user teleoperates/records with (port 5C4C125442).
  2. **pynput hotkeys** — Space=toggle autonomous↔intervention, Enter=start episode,
     y=end+SUCCESS, n=end+FAILURE, d=DISCARD (drop), a=align (mirror leader→follower so
     the operator can set the start pose), q=quit.
  3. A **raw v3.0 rollout recorder** — writes ONLY raw observations/actions/flags:
        observation.images.fixed (505x480 native) / observation.images.wrist (640x480)
        observation.state (joint6) / action (joint6: autonomous=policy, intervention=leader)
        intervention (per-frame 0/1) / success (per-frame constant, set at episode end)
     **No pistar label math here** (value_label/reward/reward_label/adv_ind/resize) — that
     lives in ONE place: code-reader's `convert_so101_v3_to_pistar.py --mode rollout`, which
     reads this v3.0 set and emits the pistar v2.1 schema. Zero-drift by construction.

Architecture (openpi, NOT lerobot async_inference):
    [this recorder] --websocket(openpi-client)--> [serve_policy.py server, has the policy]
    [SO101 follower + leader + 2 cameras on the laptop]

Units: leader.get_action(), follower.get_observation()[`{j}.pos`], the policy action chunk,
and the demo dataset `action`/`state` columns are ALL in the same LeRobot calibrated units
(RANGE_M100_100 arm + 0-100 gripper). So recording the raw command/state directly is
schema-consistent with the demos — no normalization needed.

Safety: requires explicit confirmation; `--max-relative-target` clamps per-step joint jump
during AUTONOMOUS control (the cap is intentionally bypassed during human takeover, exactly
like SiLRI, then restored); gripper intervention deadband avoids an accidental release on
takeover entry.

Usage
-----
    # logic + dataset self-check (NO robot, NO leader, NO server):
    python scripts/so101_rollout_record.py --self-check

    # real run (robot WILL move):
    python scripts/so101_rollout_record.py \
        --server-host 127.0.0.1 --port 8000 \
        --repo-id meow/so101_plate_rollout_round1 \
        --max-relative-target 15
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Reuse the proven pure-inference client helpers (camera/robot config, obs mapping, [:6]).
from so101_openpi_robot_client import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_ROBOT_ID,
    DEFAULT_ROBOT_PORT,
    JOINT_ORDER,
    actions_to_joint_command,
    build_openpi_obs,
    build_robot,
)

# The user's SO101 leader arm (the one used for teleop/record_pretty.py).
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125442-if00"
DEFAULT_LEADER_ID = "so101_leader"

# Leader hold gains (match SiLRI SO101LeaderIntervention defaults).
# Leader autonomous follow-hold gain (P = D coefficient). In autonomy the leader gently mirrors
# the follower so takeover is seamless; the operator HAND-SUPPORTS the leader's weight, so the
# servo only needs a light torque to track — keep this LOW (low current → won't overload the
# gravity-loaded joints id_=2). Overridable via --leader-hold-gain; 0 = no hold (leader limp).
DEFAULT_LEADER_HOLD_GAIN = 6
# Leader runtime torque ceiling (Feetech STS `Torque_Limit`, 0–1000 = 0–100%). Capping it LOW is
# the real overload fix: the servo physically can't reach the over-current that trips Feetech
# overload protection, so id_=2 never latches. The operator carries the weight, so a low cap still
# tracks fine. Overridable via --leader-torque-limit; the leader follows gently and just lags
# (operator's hand absorbs it) rather than ever tripping.
DEFAULT_LEADER_TORQUE_LIMIT = 400

# Takeover follow = SiLRI-style DIRECT send (cmd = raw leader joints, no script-side smoothing). The
# only rate limit is the follower's own `--max-relative-target` clamp inside send_action (the single
# safety net). Since the leader low-gain-mirrors the follower in autonomy, at takeover leader≈follower
# so the first direct step is tiny (no jump). (`_eased_target` below is still used by reset `a` align.)
ALIGN_STEP_DEG = 5.0       # per-tick follower ease toward the leader during reset alignment (gentler)
ALIGN_TOL_DEG = 2.0        # alignment considered done when every joint is within this of the leader
ALIGN_MAX_STEPS = 150      # safety cap on a single `a` alignment ramp (~5s @30fps)

STATE_ACTION_NAMES = [f"{j}.pos" for j in JOINT_ORDER]  # matches the demo dataset columns


# ─────────────────────────────────────────────────────────────────────────────
# Hotkey state (pynput global listener sets these flags; the control loop reads them)
# ─────────────────────────────────────────────────────────────────────────────
class HotkeyState:
    def __init__(self):
        self.intervention = False      # Space toggles autonomous(False) <-> intervention(True)
        self.start_episode = False     # Enter (consumed by loop)
        self.end_success = False       # y
        self.end_failure = False       # n
        self.discard = False           # d/D
        self.align = False             # a/A
        self.quit = False              # q
        self.keys_down = set()         # currently-held keys → suppress OS auto-repeat (debounce)

    def reset_episode_edges(self):
        self.start_episode = False
        self.end_success = False
        self.end_failure = False
        self.discard = False
        self.align = False


def make_on_release(hk: "HotkeyState"):
    """Clear the held-key flag so the NEXT physical press is a fresh edge (pairs with on_press)."""
    def on_release(key):
        hk.keys_down.discard(str(key))
    return on_release


def make_on_press(hk: "HotkeyState"):
    """Build the pynput on_press callback. Edge-triggered: each physical press fires ONCE.

    pynput re-fires on_press during OS key auto-repeat (held key); without this guard a slightly
    long Space press would toggle intervention several times and could flip back to autonomous —
    which re-enables the leader hold torque (the "leader won't go limp after Space" bug).
    """
    def on_press(key):
        try:
            k = str(key)
            if k in hk.keys_down:           # auto-repeat of a still-held key → ignore
                return
            hk.keys_down.add(k)
            if k == "Key.space":
                hk.intervention = not hk.intervention
                print(f"[按键] Space：{'人工介入' if hk.intervention else 'policy 自主'}", flush=True)
            elif k == "Key.enter":
                hk.start_episode = True
                print("[按键] Enter：开始本条 episode 录制。", flush=True)
            elif k in ("'y'", "'Y'"):
                hk.end_success = True
                print("[按键] y：本条标记【成功】并结束。", flush=True)
            elif k in ("'n'", "'N'"):
                hk.end_failure = True
                print("[按键] n：本条标记【失败】并结束。", flush=True)
            elif k in ("'d'", "'D'"):
                hk.discard = True
                print("[按键] d：本条【作废】丢弃（不写入数据集）。", flush=True)
            elif k in ("'a'", "'A'"):
                hk.align = True
                print("[按键] a：对齐——从臂跟随主臂当前位姿（设 episode 初始位）。", flush=True)
            elif k in ("'q'", "'Q'"):
                hk.quit = True
                print("[按键] q：请求退出。", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 键盘回调异常: {exc}", flush=True)

    return on_press


# ─────────────────────────────────────────────────────────────────────────────
# Leader arm — faithful port of SiLRI SO101LeaderIntervention's leader plumbing,
# adapted to drive a bare SO101Follower object (no gym env / so101_station).
# ─────────────────────────────────────────────────────────────────────────────
class LeaderArm:
    """SO101 leader arm, **low-gain mirroring** the follower during autonomy.

    The operator hand-supports the leader's weight, so the servo only needs a light torque to
    track the follower — `hold_gain` (P=D coefficient) is kept LOW so the current stays low and
    the gravity-loaded joint id_=2 (shoulder_lift/elbow) does not overload. In autonomy the leader
    mirrors the follower (seamless takeover: leader ≈ follower when the operator grabs it); on
    takeover entry the leader is released limp so the operator moves it freely, and the follower
    follows with a bounded eased step (`_eased_target`).

    Overload prevention/recovery (Feetech STS):
    - PREVENT: connect() caps the runtime `Torque_Limit` low (`torque_limit`) so the servo current
      can't reach the overload-protection threshold → id_=2 never trips.
    - RECOVER: on a trip the servo CLAMPS `Torque_Limit` down and a bare `Torque_Enable` toggle does
      NOT lift it (that's why a trip "sticks" until power cycle — the Feetech protocol has no reboot
      instruction). So _enable_torque RE-WRITES `Torque_Limit` before enabling, which restores torque
      without a power cycle. If it still won't recover, the operator power-cycles the leader and
      `--resume`s (the run survives).
    - `--leader-hold-gain 0` → leader stays fully limp (no hold; loses seamless takeover but zero
      overload risk) as an escape hatch.
    """

    def __init__(self, port: str, leader_id: str, hold_gain: int = DEFAULT_LEADER_HOLD_GAIN,
                 torque_limit: int = DEFAULT_LEADER_TORQUE_LIMIT):
        self._port = port
        self._id = leader_id
        self.hold_gain = int(hold_gain)
        self.torque_limit = int(torque_limit)
        self.hold_enabled = int(hold_gain) > 0   # gain 0 → never hold (fully limp escape hatch)
        self.leader = None
        self.motor_names = None
        self.torque_enabled = False
        self.torque_faulted = False
        self._torque_retry_after = 0.0
        self._torque_cooldown_s = 3.0
        self._last_warn_at = 0.0

    # -- construction (no hardware) / connection ------------------------------
    def build(self):
        """Construct the SO101Leader object WITHOUT connecting (safe for self-check)."""
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        # use_degrees left default(False) -> calibrated units, matching follower/dataset.
        cfg = SO101LeaderConfig(port=self._port, id=self._id)
        self.leader = SO101Leader(cfg)
        return self.leader

    def connect(self):
        if self.leader is None:
            self.build()
        self.leader.connect()
        self.motor_names = list(self.leader.bus.motors)
        for motor in self.motor_names:
            try:
                self.leader.bus.write("P_Coefficient", motor, self.hold_gain)
                self.leader.bus.write("I_Coefficient", motor, 0)
                self.leader.bus.write("D_Coefficient", motor, self.hold_gain)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[LeaderArm] gain config failed for %s: %s", motor, exc)
        # Cap runtime torque so the leader can never reach the Feetech overload current (the real
        # fix — lerobot leaves the leader at full default torque; only the follower gripper is capped).
        self._write_torque_limit()

    def _write_torque_limit(self):
        """Write the runtime Torque_Limit cap on all leader motors (prevent trip + restore after one)."""
        if self.motor_names is None:
            return
        for motor in self.motor_names:
            try:
                self.leader.bus.write("Torque_Limit", motor, self.torque_limit)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[LeaderArm] Torque_Limit write failed for %s: %s", motor, exc)

    def disconnect(self):
        if self.leader is not None:
            try:
                self._disable_torque(force=True)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.leader.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # -- reads ----------------------------------------------------------------
    def read_joints(self) -> np.ndarray:
        """Leader joint positions in JOINT_ORDER (calibrated units), shape (6,)."""
        action = self.leader.get_action()
        return np.array([float(action[f"{j}.pos"]) for j in JOINT_ORDER], dtype=np.float32)

    # -- torque ---------------------------------------------------------------
    def _warn(self, message, exc):
        logging.warning("[LeaderArm] %s: %s", message, exc)
        now = time.perf_counter()
        if now - self._last_warn_at >= 2.0:
            print(f"[警告] {message}：{exc}", flush=True)
            self._last_warn_at = now

    def _enable_torque(self) -> bool:
        if not self.hold_enabled:            # --leader-hold-gain 0 → never hold (fully limp)
            return False
        if self.torque_faulted:
            return False
        if self.torque_enabled:
            return True
        if time.perf_counter() < self._torque_retry_after:
            return False  # cooling down after an overload trip
        try:
            # Restore the runtime Torque_Limit FIRST — an overload trip clamps it down and a bare
            # Torque_Enable toggle won't lift it (the "sticks until power-cycle" symptom). Re-writing
            # it here recovers torque without a power cycle.
            self._write_torque_limit()
            self.leader.bus.enable_torque(num_retry=2)
        except Exception as exc:  # noqa: BLE001
            self.torque_enabled = False
            if "overload" in str(exc).lower():
                self._relieve_overload()
                self._torque_retry_after = time.perf_counter() + self._torque_cooldown_s
                self._warn("主臂过载保护触发，已卸力 + 复位扭矩上限冷却后自动重试", exc)
            else:
                self.torque_faulted = True
                self._warn("主臂上扭矩失败，保持无扭矩；若主臂卡住请断电重启后用 --resume 续跑", exc)
            return False
        self.torque_enabled = True
        self._torque_retry_after = 0.0
        return True

    def _relieve_overload(self):
        try:
            self.leader.bus.disable_torque(num_retry=2)
        except Exception:  # noqa: BLE001
            pass
        self.torque_enabled = False

    def _disable_torque(self, force=False) -> bool:
        ok = True
        # force=True (release_for_operator) must ALWAYS hit the hardware Torque_Enable=0 — never let
        # the faulted-flag early-return skip it, or the leader could stay stiff at takeover.
        if self.torque_faulted and not self.torque_enabled and not force:
            return False
        if self.torque_enabled or force:
            try:
                self.leader.bus.disable_torque(num_retry=2)
            except Exception as exc:  # noqa: BLE001
                ok = False
                if "overload" in str(exc).lower():
                    # Overload latch: the servo is ALREADY in protection (effectively limp).
                    # Treat as transient — cooldown + auto-retry later; do NOT permanent-fault
                    # (that was the bug that bricked the arm and forced a manual restart).
                    self._torque_retry_after = time.perf_counter() + self._torque_cooldown_s
                    self._warn("主臂过载(关扭矩时)：已按无扭矩处理，冷却后自动恢复，无需重启", exc)
                else:
                    self.torque_faulted = True
                    self._warn("主臂关扭矩失败，按无扭矩处理", exc)
            finally:
                self.torque_enabled = False
        return ok

    # -- servo ----------------------------------------------------------------
    def mirror_to_follower(self, follower_joints: dict):
        """Drive leader to match the follower's current joints (write goal BEFORE torque).

        follower_joints: {`{joint}.pos`: value} from robot.get_observation().
        Writing Goal_Position while torque is still off avoids a jerk to a stale firmware
        goal when torque enables (exactly the SiLRI ordering).

        If hold is disabled (--leader-hold-gain 0), this is a no-op (leader stays fully limp).
        """
        if self.motor_names is None or not self.hold_enabled:
            return False
        goal = {name: float(follower_joints[f"{name}.pos"]) for name in self.motor_names}
        try:
            self.leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:  # noqa: BLE001
            self._warn("主臂同步 follower 目标失败", exc)
            return False
        return self._enable_torque()

    def hold_at(self, target_joints: np.ndarray):
        """Hold leader at an operator-selected pose (target in JOINT_ORDER)."""
        if self.motor_names is None:
            return False
        goal = {name: float(target_joints[i]) for i, name in enumerate(self.motor_names)}
        try:
            self.leader.bus.sync_write("Goal_Position", goal)
        except Exception as exc:  # noqa: BLE001
            self._warn("主臂保持当前位置失败", exc)
            return False
        return self._enable_torque()

    def release_for_operator(self):
        """Leave the leader limp so the operator can move it freely (takeover / align)."""
        self._disable_torque(force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout recorder — buffers a whole episode, then commits with the success constant.
# Writes RAW v3.0 only (no pistar label math). Discard drops the buffer (SiLRI 'D').
# ─────────────────────────────────────────────────────────────────────────────
def rollout_features() -> dict:
    """v3.0 feature schema for the raw rollout set (read by --mode rollout converter)."""
    return {
        "observation.images.fixed": {
            "dtype": "video", "shape": (505, 480, 3), "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_ACTION_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": STATE_ACTION_NAMES},
        "intervention": {"dtype": "int64", "shape": (1,), "names": None},
        "success": {"dtype": "int64", "shape": (1,), "names": None},
    }


class RolloutRecorder:
    def __init__(self, repo_id: str, root: str | None, fps: float, task: str):
        self.repo_id = repo_id
        self.root = root
        self.fps = int(round(fps))
        self.task = task
        self.dataset = None
        self._buffer: list[dict] = []

    def _resolved_root(self) -> Path:
        """Where the dataset lives on disk (explicit --root, else HF cache by repo-id)."""
        if self.root is not None:
            return Path(self.root)
        from lerobot.utils.constants import HF_LEROBOT_HOME
        return Path(HF_LEROBOT_HOME) / self.repo_id

    def _exists(self) -> bool:
        return (self._resolved_root() / "meta" / "info.json").exists()

    def open(self, resume: bool = False):
        """Open the rollout dataset for writing.

        not resume + exists  -> hard error (never silently overwrite).
        not resume + missing -> LeRobotDataset.create (fresh).
        resume     + exists  -> LeRobotDataset.resume (append; episode_index continues).
        resume     + missing -> treated as fresh create.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        exists = self._exists()
        if exists and not resume:
            raise FileExistsError(
                f"dataset 已存在: {self._resolved_root()}\n"
                f"  - 续接已有集请加 --resume\n"
                f"  - 或换一个 --repo-id 新建独立集（不覆盖旧的）"
            )

        if resume and exists:
            # Official lerobot 0.5.2 resume: loads existing metadata + builds a DatasetWriter
            # that appends new episodes (episode_index auto-continues from the existing count).
            # lerobot 0.5.2 resume() REQUIRES an explicit root (unlike create(), which accepts
            # None → HF cache). Resolve it ourselves (--root, else HF_LEROBOT_HOME/repo_id) so
            # `--resume` works without --root and never writes into the Hub snapshot cache.
            self.dataset = LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=str(self._resolved_root()),
                video_backend="pyav",
            )
            prev = self.dataset.meta.total_episodes
            self._validate_features()
            print(f"[recorder] --resume 续接已有集（{prev} 条），新条从 episode_index={prev} 起追加。", flush=True)
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                robot_type="so101",
                features=rollout_features(),
                use_videos=True,
                video_backend="pyav",
            )
            if resume:
                print("[recorder] --resume 指定但集不存在，按新建处理。", flush=True)
        return self.dataset

    def _validate_features(self):
        """On resume, the existing set must have the rollout schema (keys + shapes)."""
        existing = self.dataset.meta.features
        want = rollout_features()
        problems = []
        for key, spec in want.items():
            if key not in existing:
                problems.append(f"缺字段 {key}")
                continue
            ex_shape = tuple(existing[key].get("shape", ()))
            wt_shape = tuple(spec["shape"])
            if ex_shape != wt_shape:
                problems.append(f"{key} shape {ex_shape} != 期望 {wt_shape}")
        if problems:
            raise ValueError(
                "续接的数据集 schema 与 rollout_features 不一致，拒绝追加（避免污染）:\n  "
                + "\n  ".join(problems)
            )

    def add(self, fixed_img, wrist_img, state6, action6, intervention: int):
        """Buffer one raw frame (success is unknown until the episode ends)."""
        self._buffer.append({
            "observation.images.fixed": np.asarray(fixed_img),
            "observation.images.wrist": np.asarray(wrist_img),
            "observation.state": np.asarray(state6, dtype=np.float32),
            "action": np.asarray(action6, dtype=np.float32),
            "intervention": np.array([int(intervention)], dtype=np.int64),
        })

    def episode_len(self) -> int:
        return len(self._buffer)

    def discard(self):
        n = len(self._buffer)
        self._buffer = []
        return n

    def finalize(self, success: int) -> int:
        """Commit the buffered episode with a constant `success` column, then save."""
        n = len(self._buffer)
        if n == 0:
            return 0
        succ = np.array([int(success)], dtype=np.int64)
        for frame in self._buffer:
            frame = dict(frame)
            frame["success"] = succ
            frame["task"] = self.task
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self._buffer = []
        return n

    def close(self):
        """Flush dataset metadata/video buffers to disk (must run before the set is readable)."""
        if self.dataset is not None:
            self.dataset.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Self-check (no robot / no leader / no server)
# ─────────────────────────────────────────────────────────────────────────────
def self_check() -> int:
    print("=== self-check: construction / hotkeys / dataset roundtrip (no hardware) ===")
    ok = True

    # 1) robot + cameras construct (reuse client builder; no connect)
    try:
        robot = build_robot(DEFAULT_ROBOT_PORT, DEFAULT_ROBOT_ID, max_relative_target=15)
        cams = robot.config.cameras
        assert cams["fixed"].width == 480 and cams["fixed"].height == 505 and cams["fixed"].crop_top == 135
        assert cams["wrist"].width == 640 and cams["wrist"].height == 480
        print("[ok] robot+cameras built (fixed 480x505 ROTATE_270 crop135 / wrist 640x480)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] robot/camera build: {exc}")
        ok = False

    # 2) leader constructs without connecting + overload-config gating
    try:
        leader = LeaderArm(DEFAULT_LEADER_PORT, DEFAULT_LEADER_ID)
        obj = leader.build()
        assert obj is not None
        assert leader.hold_enabled is True and leader.torque_limit == DEFAULT_LEADER_TORQUE_LIMIT
        # gain 0 → fully limp (mirror is a no-op, torque never enabled)
        limp = LeaderArm(DEFAULT_LEADER_PORT, DEFAULT_LEADER_ID, hold_gain=0)
        assert limp.hold_enabled is False, "hold_gain=0 must disable hold (limp escape hatch)"
        assert limp.mirror_to_follower({}) is False and limp._enable_torque() is False
        # custom torque cap threads through
        capped = LeaderArm(DEFAULT_LEADER_PORT, DEFAULT_LEADER_ID, torque_limit=250)
        assert capped.torque_limit == 250
        print(f"[ok] SO101Leader built; torque_limit cap={DEFAULT_LEADER_TORQUE_LIMIT}; "
              f"hold-gain 0 → limp (mirror/enable no-op) verified")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] leader build/gating: {exc}")
        ok = False

    # 2b) takeover max_relative_target bypass (SiLRI parity): None during takeover, restored after
    try:
        class _FakeCfg:
            max_relative_target = 15
        class _FakeRobot:
            config = _FakeCfg()
        fake = _FakeRobot()
        cap = _CapBypass(fake)
        assert fake.config.max_relative_target == 15 and not cap.active
        cap.bypass()
        assert fake.config.max_relative_target is None and cap.active, "takeover must set cap None (1:1)"
        cap.bypass()  # idempotent
        assert fake.config.max_relative_target is None
        cap.restore()
        assert fake.config.max_relative_target == 15 and not cap.active, "release must restore the autonomous cap"
        cap.restore()  # idempotent
        assert fake.config.max_relative_target == 15
        print("[ok] takeover cap bypass: max_relative_target → None during takeover, restored on release "
              "(autonomous keeps its clamp) = SiLRI _start_intervention parity")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] cap bypass: {exc}")
        ok = False

    # 3) hotkey callback wiring + auto-repeat debounce (edge-triggered)
    try:
        hk = HotkeyState()
        on_press = make_on_press(hk)
        on_release = make_on_release(hk)

        def tap(k):  # one full physical press: press + release
            on_press(k); on_release(k)

        tap("Key.space"); assert hk.intervention is True, "space should enable intervention"
        tap("Key.space"); assert hk.intervention is False, "space should toggle back"
        # auto-repeat: holding space (repeated on_press WITHOUT release) must toggle only ONCE
        on_press("Key.space"); on_press("Key.space"); on_press("Key.space")
        assert hk.intervention is True, "held space (auto-repeat) must toggle once, not flip back"
        on_release("Key.space")
        tap("Key.enter"); assert hk.start_episode is True
        tap("'y'"); assert hk.end_success is True
        tap("'n'"); assert hk.end_failure is True
        tap("'d'"); assert hk.discard is True
        tap("'a'"); assert hk.align is True
        tap("'q'"); assert hk.quit is True
        hk.reset_episode_edges()
        assert not (hk.start_episode or hk.end_success or hk.discard or hk.align)
        print("[ok] hotkeys: Space toggle / Enter / y / n / d / a / q wired; auto-repeat debounced "
              "(held key toggles once → no flip-back that would re-stiffen the leader)")
    except AssertionError as exc:
        print(f"[FAIL] hotkey wiring: {exc}")
        ok = False

    # 3b) follower-side easing math (bounded follow) + step<=0 direct-send semantics
    try:
        step = ALIGN_STEP_DEG  # `_eased_target` is used by reset `a` alignment (takeover is direct-send)
        cur = np.zeros(6, dtype=np.float32)
        far_dist = 5.0 * step                                    # exactly 5 bounded steps away
        far = np.array([far_dist, -far_dist, far_dist, 0, 0, 0], dtype=np.float32)
        stepped = _eased_target(cur, far, step)
        assert float(np.max(np.abs(stepped - cur))) <= step + 1e-6, "ease exceeds step cap"
        near = cur + (step * 0.1)                                # within one step → reaches goal
        assert np.allclose(_eased_target(cur, near, step), near), "small move should reach goal"
        assert int(np.ceil(far_dist / step)) == 5, "a far target should converge in bounded steps"
        print(f"[ok] align easing: bounded {step:g}°/tick (far {far_dist:g}° → 5 steps); "
              f"takeover default = full SiLRI direct 1:1 (raw leader incl gripper, max_relative_target bypassed)")
    except AssertionError as exc:
        print(f"[FAIL] easing math: {exc}")
        ok = False

    # 4) dataset roundtrip — create v3.0, buffer fake frames, finalize(success), reload, assert
    tmp = Path(tempfile.mkdtemp(prefix="so101_rollout_selfcheck_"))
    try:
        rec = RolloutRecorder(repo_id="selfcheck/rollout", root=str(tmp / "ds"), fps=15, task=DEFAULT_PROMPT)
        rec.open()
        n_frames = 16
        for t in range(n_frames):
            fixed = np.random.randint(0, 255, (505, 480, 3), dtype=np.uint8)
            wrist = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            state = np.arange(6, dtype=np.float32) + t
            action = np.arange(6, dtype=np.float32) + t + 0.5
            intervention = 1 if t >= n_frames // 2 else 0   # second half = intervention
            rec.add(fixed, wrist, state, action, intervention)
        assert rec.episode_len() == n_frames
        committed = rec.finalize(success=1)
        assert committed == n_frames
        rec.close()  # flush metadata/video buffers before the set is readable

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("selfcheck/rollout", root=str(tmp / "ds"))
        feats = set(ds.features)
        need = {"observation.images.fixed", "observation.images.wrist", "observation.state",
                "action", "intervention", "success"}
        missing = need - feats
        assert not missing, f"missing features {missing}"
        assert ds.num_frames == n_frames and ds.num_episodes == 1
        s0, s_last = ds[0], ds[n_frames - 1]
        assert tuple(s0["observation.images.fixed"].shape) == (3, 505, 480), s0["observation.images.fixed"].shape
        assert tuple(s0["observation.images.wrist"].shape) == (3, 480, 640)
        assert int(s0["intervention"]) == 0 and int(s_last["intervention"]) == 1, "intervention column wrong"
        assert int(s0["success"]) == 1 and int(s_last["success"]) == 1, "success constant wrong"
        assert tuple(s0["observation.state"].shape) == (6,) and tuple(s0["action"].shape) == (6,)
        print(f"[ok] dataset roundtrip: {ds.num_episodes} ep / {ds.num_frames} frames, "
              f"features={sorted(need)}, images decode CHW, intervention+success columns correct")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[FAIL] dataset roundtrip: {exc}")
        traceback.print_exc()
        ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 4b) --resume root resolution: without --root, resume() must still get an explicit root
    # (HF_LEROBOT_HOME/repo_id), not None — lerobot 0.5.2 resume() rejects None.
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        rr = RolloutRecorder("meow/_selfcheck_norootset", root=None, fps=30, task=DEFAULT_PROMPT)
        resolved = rr._resolved_root()
        assert resolved == Path(HF_LEROBOT_HOME) / "meow/_selfcheck_norootset", resolved
        assert str(resolved) and resolved is not None, "resume root must be explicit (not None)"
        print(f"[ok] --resume root: root=None resolves to HF cache ({resolved}) so resume() gets explicit root")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] resume root resolution: {exc}")
        ok = False

    # 5) resume roundtrip — create→close→--resume reopen→append→close→reload; assert totals
    tmp2 = Path(tempfile.mkdtemp(prefix="so101_rollout_resume_"))
    try:
        root = str(tmp2 / "ds")

        def _add_episode(rec, n, base, succ):
            for t in range(n):
                rec.add(
                    np.random.randint(0, 255, (505, 480, 3), dtype=np.uint8),
                    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    np.arange(6, dtype=np.float32) + base + t,
                    np.arange(6, dtype=np.float32) + base + t + 0.5,
                    1 if t >= n // 2 else 0,
                )
            return rec.finalize(success=succ)

        # round 1: fresh create, 1 episode (8 frames), success
        rec1 = RolloutRecorder("selfcheck/resume", root=root, fps=15, task=DEFAULT_PROMPT)
        rec1.open(resume=False)
        _add_episode(rec1, 8, 0, 1)
        rec1.close()
        assert rec1._exists(), "dataset should exist after round 1"

        # opening non-resume on an existing set must HARD ERROR (never overwrite)
        rec_bad = RolloutRecorder("selfcheck/resume", root=root, fps=15, task=DEFAULT_PROMPT)
        raised = False
        try:
            rec_bad.open(resume=False)
        except FileExistsError:
            raised = True
        assert raised, "open(resume=False) on existing set must raise FileExistsError"

        # round 2: --resume append, 1 more episode (6 frames), failure
        rec2 = RolloutRecorder("selfcheck/resume", root=root, fps=15, task=DEFAULT_PROMPT)
        rec2.open(resume=True)
        _add_episode(rec2, 6, 100, 0)
        rec2.close()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("selfcheck/resume", root=root)
        assert ds.num_episodes == 2, f"resume episodes {ds.num_episodes} != 2"
        assert ds.num_frames == 8 + 6, f"resume frames {ds.num_frames} != 14"
        eps = sorted(set(int(x) for x in ds.hf_dataset["episode_index"]))
        assert eps == [0, 1], f"episode_index not continuous after resume: {eps}"
        assert {"intervention", "success"} <= set(ds.features), "features lost on resume"
        print(f"[ok] resume roundtrip: round1(1ep/8f) + --resume round2(1ep/6f) "
              f"= {ds.num_episodes}ep/{ds.num_frames}f, episode_index={eps} continuous; "
              f"open(resume=False)-on-existing correctly errored")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[FAIL] resume roundtrip: {exc}")
        traceback.print_exc()
        ok = False
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    # 6) openpi client import (optional until installed)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401
        print("[ok] openpi_client import OK")
    except ImportError:
        print("[warn] openpi_client not installed yet — `pip install -e packages/openpi-client` before a real run")

    print("=== self-check", "PASSED" if ok else "FAILED", "===")
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# Real run
# ─────────────────────────────────────────────────────────────────────────────
def _follower_obs_to_state(robot_obs: dict) -> np.ndarray:
    return np.array([float(robot_obs[f"{j}.pos"]) for j in JOINT_ORDER], dtype=np.float32)


def _eased_target(cur: np.ndarray, goal: np.ndarray, step_deg: float) -> np.ndarray:
    """A follower target that moves at most `step_deg` per joint from `cur` toward `goal`.

    Pure function (jump-safety independent of --max-relative-target): even if `goal` (the limp
    leader pose) is far, the returned command is a bounded step, so the follower ramps smoothly.
    """
    cur = np.asarray(cur, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    return cur + np.clip(goal - cur, -float(step_deg), float(step_deg))


def _ramp_follower_to_leader(robot, leader: "LeaderArm", period: float, hk: "HotkeyState",
                             step_deg: float = ALIGN_STEP_DEG, max_steps: int = ALIGN_MAX_STEPS,
                             align_gripper: bool = True) -> bool:
    """Smoothly drive the follower to the (limp) leader's current pose, one bounded step/tick.

    Re-reads the leader each tick so it tracks the operator if they keep moving. Returns True
    when the aligned joints are within ALIGN_TOL_DEG of the leader, or False on step-cap / hk abort.
    Used by reset `a` alignment and by takeover-engage (no jump even from a drooped leader).

    align_gripper=False (takeover engage): hold the follower's current gripper — do NOT sweep it
    to the leader's gripper during engage, so we never accidentally open/close on a held object
    (per-tick gripper is governed by the deadband once following).
    """
    arm = slice(0, 5)  # 5 arm joints; index 5 is the gripper
    for _ in range(max_steps):
        if hk is not None and (hk.quit or hk.discard):
            return False
        goal = leader.read_joints()
        cur = _follower_obs_to_state(robot.get_observation())
        if not align_gripper:
            goal[5] = cur[5]  # keep gripper where it is during engage
        if float(np.max(np.abs(goal[arm] - cur[arm]))) <= ALIGN_TOL_DEG:
            return True
        robot.send_action(actions_to_joint_command(_eased_target(cur, goal, step_deg)))
        time.sleep(period)
    return False


class ClientHolder:
    """Holds the openpi websocket client and supports a one-shot reconnect on a blip."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=self.host, port=self.port)
        return self.client

    def reconnect(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=self.host, port=self.port)

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, obs):
        return self.client.infer(obs)


def _infer_with_retry(client_holder: "ClientHolder", obs) -> dict:
    """Run inference; on a transient failure try ONE websocket reconnect then retry once.

    A second failure propagates to the caller, which aborts the current episode.
    """
    try:
        return client_holder.infer(obs)
    except Exception as first:  # noqa: BLE001
        print(f"[警告] 推理失败({type(first).__name__}: {str(first)[:80]});尝试重连 server 一次…", flush=True)
        client_holder.reconnect()
        return client_holder.infer(obs)


def _wait_for_verdict(hk: "HotkeyState", what: str) -> str:
    """Freeze the robot (stop sending; follower holds via torque) and wait for y/n/d/q."""
    print(f"[episode] {what}：机械臂保持当前位（已停发指令）。按 y=成功  n=失败  d=作废  q=退出。", flush=True)
    while True:
        if hk.quit:
            return "quit"
        if hk.discard:
            return "discard"
        if hk.end_success:
            return "success"
        if hk.end_failure:
            return "failure"
        time.sleep(0.05)


class _CapBypass:
    """Bypass the follower's max_relative_target during takeover (SiLRI `_start_intervention`).

    bypass(): save the current cap and set it to None → takeover is unclamped, true 1:1.
    restore(): put the saved cap back → autonomy keeps its safety clamp.
    Idempotent both ways; restore() on any episode exit (finally) prevents leaking None into autonomy.
    """

    def __init__(self, robot):
        self.robot = robot
        self.saved = None

    @property
    def active(self) -> bool:
        return self.saved is not None

    def bypass(self):
        if self.saved is None:
            self.saved = self.robot.config.max_relative_target
            self.robot.config.max_relative_target = None

    def restore(self):
        if self.saved is not None:
            try:
                self.robot.config.max_relative_target = self.saved
            except Exception:  # noqa: BLE001
                pass
            self.saved = None


def run_episode(args, robot, leader: LeaderArm, client_holder: "ClientHolder",
                recorder: RolloutRecorder, hk: HotkeyState) -> str:
    """Run one rollout episode.

    Returns 'success' | 'failure' | 'discard' | 'quit' | 'error'.
    On a transient per-step I/O failure (camera read / server drop / robot comms blip):
    stop sending commands (safety), abort THIS episode -> 'error' (main discards its
    buffer and returns to reset; the whole run does NOT crash, saved episodes are kept).

    Images are recorded at NATIVE resolution; resize_with_pad->224 happens ONLY in
    code-reader's converter (and server-side ResizeImages at inference). No resize here.
    """
    period = 1.0 / args.fps
    exec_h = max(1, args.exec_horizon)
    step_deg = args.takeover_step_deg     # 0 (default) = DIRECT raw send (full SiLRI); >0 = bounded ease
    chunk = None
    ptr = 10**9            # force first inference
    was_intervening = False
    step = 0
    cap = _CapBypass(robot)   # follower max_relative_target, bypassed during takeover (SiLRI parity)

    # In autonomy the leader low-gain-mirrors the follower (operator hand-supports its weight) so
    # takeover is seamless. Start by releasing it; the first autonomous tick re-asserts the hold.
    leader.release_for_operator()
    print(f"[episode] 录制中。Space=介入/自主切换  y=成功结束  n=失败结束  d=作废  (max {args.max_steps} 步)", flush=True)

    try:
        while step < args.max_steps:
            tick = time.perf_counter()
            if hk.quit:
                return "quit"
            if hk.discard:
                return "discard"
            if hk.end_success:
                return "success"
            if hk.end_failure:
                return "failure"

            # ── per-step I/O wrapped: any transient blip aborts THIS episode cleanly ──
            try:
                robot_obs = robot.get_observation()
                state6 = _follower_obs_to_state(robot_obs)
                intervening = hk.intervention

                # --- intervention edge transitions (SiLRI _start/_finish_intervention parity) ---
                if intervening and not was_intervening:
                    # Release leader limp so the operator moves it freely, AND bypass the follower's
                    # per-tick max_relative_target → takeover is true 1:1 (exactly SiLRI
                    # _start_intervention). Jump-safe because the autonomous low-gain mirror keeps
                    # leader ≈ follower at entry. Restored on release / episode end (finally).
                    leader.release_for_operator()
                    cap.bypass()                                   # follower max_relative_target → None (1:1)
                    print("[接管] 人工接管开启：移动主臂遙操从臂（1:1 直发，无钳）。", flush=True)
                elif not intervening and was_intervening:
                    cap.restore()                                  # restore autonomous safety clamp
                    ptr = 10**9                                    # re-infer fresh after takeover
                    print("[接管] 人工接管结束：恢复 policy 自主（主臂低扭矩跟随从臂，限速恢复）。", flush=True)
                was_intervening = intervening

                # --- choose + send command ---
                if intervening:
                    # Belt-and-suspenders: the leader MUST stay limp during takeover.
                    if leader.torque_enabled:
                        leader.release_for_operator()
                    leader_joints = leader.read_joints()
                    # Full SiLRI: command the raw leader joints INCLUDING the gripper, directly. No
                    # deadband — the autonomous mirror keeps the leader gripper ≈ follower so there's
                    # no pop at entry. step_deg>0 is an optional gentler bounded mode.
                    cmd6 = leader_joints.astype(np.float32, copy=True) if step_deg <= 0 \
                        else _eased_target(state6, leader_joints, step_deg)
                else:
                    if ptr >= (0 if chunk is None else chunk.shape[0]) or ptr >= exec_h:
                        obs = build_openpi_obs(robot_obs, args.prompt, args.adv_ind)
                        result = _infer_with_retry(client_holder, obs)  # 1 reconnect attempt inside
                        chunk = np.asarray(result["actions"])          # (H, 7)
                        ptr = 0
                    cmd6 = np.asarray(chunk[ptr]).reshape(-1)[:6]
                    ptr += 1
                    # Autonomous: leader gently mirrors the follower (low gain) → stays ≈ follower.
                    leader.mirror_to_follower(robot_obs)

                robot.send_action(actions_to_joint_command(cmd6))
            except Exception as exc:  # noqa: BLE001
                print(f"\n[异常] per-step I/O 失败：{type(exc).__name__}: {str(exc)[:120]}", flush=True)
                print("[异常] 已停止给机器人发指令（follower 靠自身扭矩 hold 末位）；本条中止、回到 reset。", flush=True)
                logging.warning("run_episode per-step failure", exc_info=True)
                return "error"

            recorder.add(robot_obs["fixed"], robot_obs["wrist"], state6, cmd6, int(intervening))
            step += 1

            dt = time.perf_counter() - tick
            if period > dt:
                time.sleep(period - dt)

        # Reached --max-steps: freeze + let the operator decide (don't auto-misjudge a near-success).
        return _wait_for_verdict(hk, f"到达 --max-steps({args.max_steps}) 上限")
    finally:
        cap.restore()   # never leak a bypassed max_relative_target into autonomy / next episode


def main() -> int:
    ap = argparse.ArgumentParser(description="SO101 DAgger rollout recorder (openpi PiStar server)")
    ap.add_argument("--self-check", action="store_true", help="validate logic+dataset without hardware/server")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--adv-ind", default="positive", choices=["positive", "negative", "none"],
                    help="condition fed to the policy at inference (NOT written to the dataset)")
    ap.add_argument("--repo-id", default="meow/so101_rollout_round1")
    ap.add_argument("--root", default=None, help="dataset dir (default: HF cache by repo-id)")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing dataset (episode_index continues); "
                         "without it, an existing dataset is a hard error (never overwrites)")
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="control loop / recorded fps. MUST equal the demo fps (this project = 30) "
                         "so rollout and demo share the same per-frame time scale: lerobot writes "
                         "timestamp = frame_index/fps (uniform), and the VLM N-step advantage is "
                         "counted in FRAMES — a 15 vs 30 mismatch would desync demo/rollout time scales.")
    ap.add_argument("--exec-horizon", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=700,
                    help="per-episode step cap (@30fps ~= 23s; covers this project's longest demo "
                         "646 frames / 21.5s + headroom). On reaching it the arm FREEZES and waits "
                         "for y/n/d (not auto-fail). Raise for longer tasks.")
    ap.add_argument("--max-relative-target", type=float, default=None,
                    help="per-step joint jump clamp (deg) during AUTONOMOUS; bypassed during takeover")
    ap.add_argument("--takeover-step-deg", type=float, default=0.0,
                    help="takeover follow mode. 0 (default) = full SiLRI: DIRECT raw leader→follower, "
                         "1:1, no script clamp and max_relative_target bypassed during takeover. "
                         ">0 = optional gentler bounded mode (ease ≤N°/tick).")
    ap.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    ap.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    ap.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    ap.add_argument("--leader-hold-gain", type=int, default=DEFAULT_LEADER_HOLD_GAIN,
                    help="leader P/D coefficient for the LOW-torque autonomous follow-hold "
                         "(operator hand-supports the weight, so keep it low → low current → no "
                         "overload). Raise if the leader sags too much; 0 = no hold (leader limp).")
    ap.add_argument("--leader-torque-limit", type=int, default=DEFAULT_LEADER_TORQUE_LIMIT,
                    help="leader runtime Torque_Limit cap (Feetech STS, 0-1000=0-100%%). LOW caps "
                         "current below the overload-trip threshold → id_=2 won't latch. Lower it "
                         "more if it still overloads; raise if the leader can't follow at all.")
    ap.add_argument("--no-confirm", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    from pynput import keyboard

    hk = HotkeyState()
    listener = keyboard.Listener(on_press=make_on_press(hk), on_release=make_on_release(hk))
    listener.start()

    robot = build_robot(args.robot_port, args.robot_id, args.max_relative_target)
    leader = LeaderArm(args.leader_port, args.leader_id, hold_gain=args.leader_hold_gain,
                       torque_limit=args.leader_torque_limit)
    recorder = RolloutRecorder(args.repo_id, args.root, args.fps, args.prompt)

    # Fail fast on the dataset existence/resume policy BEFORE touching hardware/server.
    if recorder._exists() and not args.resume:
        print(f"❌ dataset 已存在: {recorder._resolved_root()}\n"
              f"   续接加 --resume，或换 --repo-id 新建。", flush=True)
        listener.stop(); return 1

    print("=" * 70)
    print("⚠️  REAL ROBOT — the SO101 follower WILL move (policy + human takeover).")
    print(f"    server   : ws://{args.server_host}:{args.port}   adv_ind(infer)={args.adv_ind}")
    print(f"    dataset  : {args.repo_id}  (root={args.root or 'HF cache'})  resume={args.resume}")
    print(f"    episodes : {args.num_episodes}  fps={args.fps}  max_steps={args.max_steps}")
    print(f"    max_relative_target(autonomous)={args.max_relative_target}"
          f"{'  (UNSET — recommend 15 on first runs)' if args.max_relative_target is None else ''}")
    print("    hotkeys  : Space=介入/自主  Enter=开始本条  y=成功  n=失败  d=作废  a=对齐  q=退出")
    print("    Clear workspace. Keep a hand on the e-stop / power.")
    print("=" * 70)
    if not args.no_confirm and input("Type 'GO' to start (anything else aborts): ").strip() != "GO":
        print("aborted."); listener.stop(); return 1

    print(f"[client] connecting ws://{args.server_host}:{args.port} ...")
    client_holder = ClientHolder(args.server_host, args.port)
    client_holder.connect()
    print(f"[client] server metadata: {client_holder.get_server_metadata()}")
    print("[robot] connecting follower + leader ...")
    robot.connect()
    leader.connect()
    leader.release_for_operator()   # start limp; autonomy re-asserts the low-gain follow-hold per tick
    recorder.open(resume=args.resume)

    saved = 0
    try:
        ep = 0
        while ep < args.num_episodes and not hk.quit:
            # ---- reset/align phase: operator positions the leader, 'a' aligns, Enter starts ----
            hk.reset_episode_edges()
            hk.intervention = False
            leader.release_for_operator()
            print(f"\n[reset {saved + 1}/{args.num_episodes}] 摆好场景与主臂初始位："
                  "a=从臂平滑跟随主臂  Enter=开始录制  q=退出", flush=True)
            while not hk.start_episode and not hk.quit:
                if hk.align:
                    hk.align = False
                    # Operator has hand-positioned the (limp) leader to the desired start pose.
                    # ONE press of `a` smoothly ramps the follower all the way there (bounded
                    # step/tick, re-reads the leader each tick), instead of one clamped jump.
                    period = 1.0 / args.fps
                    print("[对齐] 从臂平滑移动到主臂位姿中…", flush=True)
                    reached = _ramp_follower_to_leader(robot, leader, period, hk, step_deg=ALIGN_STEP_DEG)
                    print(f"[对齐] {'已到位' if reached else '已尽量逼近(到步数上限)'}。再按 a 微调，或 Enter 开始。",
                          flush=True)
                time.sleep(0.05)
            if hk.quit:
                break

            hk.reset_episode_edges()
            outcome = run_episode(args, robot, leader, client_holder, recorder, hk)
            hk.reset_episode_edges()
            hk.intervention = False

            n = recorder.episode_len()
            if outcome == "quit":
                print(f"[episode] 退出请求；丢弃未完成的 {n} 帧。", flush=True)
                recorder.discard(); break
            if outcome == "error":
                dropped = recorder.discard()
                print(f"[episode] 本条因异常作废，丢弃 {dropped} 帧（不计入、不增计数），回到 reset 可重录。"
                      f"  已保存 {saved} 条不受影响。", flush=True)
                continue
            if outcome == "discard":
                dropped = recorder.discard()
                print(f"[episode] 作废，丢弃 {dropped} 帧（不计入数据集，请重录）。", flush=True)
                continue
            success = 1 if outcome == "success" else 0
            committed = recorder.finalize(success)
            saved += 1
            ep += 1
            print(f"[episode] 已保存第 {saved} 条：{committed} 帧，success={success}。", flush=True)
    except KeyboardInterrupt:
        print("\n[run] interrupted (Ctrl-C).", flush=True)
        recorder.discard()
    finally:
        print("[cleanup] flushing dataset + disconnecting ...", flush=True)
        try:
            recorder.close()  # flush metadata/video so the saved episodes are readable
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] dataset finalize: {exc}")
        try:
            leader.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] leader disconnect: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] robot disconnect: {exc}")
        listener.stop()
    print(f"[run] done — saved {saved} episode(s) to {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

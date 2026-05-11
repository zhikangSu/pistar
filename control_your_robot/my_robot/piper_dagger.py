import sys
sys.path.append("./")

import numpy as np
import time

from my_robot.base_robot import Robot
from my_robot.camera_config import get_piper_camera_serials
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

# Start positions are in radians.
START_POSITION_ANGLE_FOLLOWER_ARM = [
    0,
    -0.4208,
    0.0324,
    0.0780,
    0.3558,
    0.0078,
]

# Master reset pose captured from the current can0 arm state.
START_POSITION_ANGLE_MASTER_ARM = [
    0.069639,
    -0.035343,
    0.038153,
    -0.064368,
    0.069691,
    -0.010228,
]

# Master-slave linkage config (0x470).
MASTER_ROLE = 0xFA  # teaching input arm
FOLLOWER_ROLE = 0xFC  # motion output arm
FEEDBACK_OFFSET = 0x00
CTRL_OFFSET = 0x00
LINKAGE_OFFSET = 0x00

condition = {
    "robot": "piper_dagger",
    "save_path": "./save/",
    "task_name": "dagger",
    "save_format": "hdf5",
    "save_freq": 10,
}


class PiperDAgger(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.camera_serials = get_piper_camera_serials("dagger")
        self.controllers = {
            "arm": {
                "left_arm": PiperController("left_arm"),   # follower
                "right_arm": PiperController("right_arm"),  # master
            },
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }
        self._mirror_last_joint_cmd = None
        self._mirror_last_gripper_cmd = None
        # Deadband thresholds to reduce jitter during intervention
        # Joint: 10000 ≈ 0.175 rad ≈ 10 degrees
        # Gripper: 2000 ≈ 0.029 (2.9% of range)
        self._mirror_joint_deadband_cmd = 10000
        self._mirror_gripper_deadband_cmd = 2000
        # Rate limiting & optional low-pass filter to reduce jitter
        self._mirror_min_send_interval = 0.02  # seconds
        self._mirror_filter_alpha = None  # 0~1, None disables smoothing
        self._mirror_last_send_time = 0.0
        self._intervention_anchor_joint_cmd = None
        self._intervention_anchor_gripper_cmd = None
        self._intervention_active = False
        self._intervention_start_deadband_cmd = 15000
        self._intervention_start_gripper_deadband_cmd = 3000
        self._intervention_move_count = 0
        self._intervention_move_required = 3
        self._policy_enabled = True
        self._mirror_master_speed_percent = 100
        self._sync_master_with_policy_commands = True
        self._mirror_master_joint_baseline = None
        self._mirror_master_gripper_baseline = None
        self._mirror_follower_joint_baseline = None
        self._mirror_follower_gripper_baseline = None
        # Use a stronger follower gripper effort during rollout so grasping force
        # stays consistent while the arm is moving.
        self._follower_gripper_effort = 5000
        self._master_gripper_effort = 1000

    @staticmethod
    def _joint_to_cmd(joint):
        joint = np.array(joint, dtype=float)
        return (joint * 57295.7795).astype(int)  # 1000*180/pi

    @staticmethod
    def _gripper_to_cmd(gripper):
        return int(float(gripper) * 70 * 1000)

    def _send_arm_target(
        self,
        controller,
        move_data,
        *,
        speed_percent: int | None = None,
        gripper_effort: int,
    ):
        if controller is None or move_data is None:
            return

        if speed_percent is not None:
            controller.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)

        joint = move_data.get("joint")
        if joint is not None:
            joint_cmd = self._joint_to_cmd(joint)
            controller.JointCtrl(
                int(joint_cmd[0]),
                int(joint_cmd[1]),
                int(joint_cmd[2]),
                int(joint_cmd[3]),
                int(joint_cmd[4]),
                int(joint_cmd[5]),
            )

        if "gripper" in move_data:
            gripper_cmd = self._gripper_to_cmd(move_data["gripper"])
            controller.GripperCtrl(gripper_cmd, int(gripper_effort), 0x01, 0)

    def reset(self):
        """Reset both arms to start positions with proper state management"""
        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller

        print("[reset] preparing arms for reset...")

        # Step 1: Exit any special modes for both arms
        for ctrl, name in ((master, "master"), (follower, "follower")):
            try:
                ctrl.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
                time.sleep(0.15)
                ctrl.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
                time.sleep(0.15)
            except Exception as exc:
                print(f"[reset] {name} exit drag mode failed: {exc}")

        # Step 2: Configure both as followers (software controllable)
        try:
            master.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
            time.sleep(0.2)
            follower.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
            time.sleep(0.3)
        except Exception as exc:
            print(f"[reset] MasterSlaveConfig failed: {exc}")

        # Step 3: Set control mode with slower speed for smoother reset
        reset_speed_percent = 30
        try:
            master.MotionCtrl_2(0x01, 0x01, reset_speed_percent, 0x00)
            time.sleep(0.1)
            follower.MotionCtrl_2(0x01, 0x01, reset_speed_percent, 0x00)
            time.sleep(0.1)
        except Exception as exc:
            print(f"[reset] MotionCtrl_2 failed: {exc}")

        # Step 4: Enable arms
        try:
            master.EnableArm(7)
        except Exception:
            pass
        time.sleep(0.1)
        try:
            follower.EnableArm(7)
        except Exception:
            pass
        time.sleep(0.3)  # Wait for enable to take effect

        print("[reset] moving to start positions...")

        # Step 5: Reset both arms to their configured start poses.
        try:
            master_reset_pose = np.array(START_POSITION_ANGLE_MASTER_ARM, dtype=float)
            follower_reset_pose = np.array(START_POSITION_ANGLE_FOLLOWER_ARM, dtype=float)
            self.controllers["arm"]["right_arm"].reset(
                master_reset_pose.copy(), speed_percent=reset_speed_percent
            )
            self.controllers["arm"]["left_arm"].reset(
                follower_reset_pose.copy(), speed_percent=reset_speed_percent
            )
            time.sleep(0.5)  # Wait for movement to complete
        except Exception as exc:
            print(f"[reset] position reset failed: {exc}")

        print("[reset] reset complete")

       

    def set_up(self):
        super().set_up()

        import time

        # Master arm on can0 (human operates), follower arm on can1 (executes task)
        self.controllers["arm"]["right_arm"].set_up("can0")   # master -> can0 (human drags)
        self.controllers["arm"]["left_arm"].set_up("can1")    # follower -> can1 (executes)

        # 等待 CAN 总线稳定（刚上电时需要更长时间）
        print("[setup] Waiting for CAN bus to stabilize...")
        time.sleep(3.0)  # 增加到 3 秒

        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller

        print("[setup] Configuring both arms as followers (0xFC)...")

        # Exit drag teaching mode first
        try:
            # Master arm: forcefully exit all special modes
            master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
            time.sleep(0.2)
            master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
            time.sleep(0.2)

            # Follower arm: clear all modes
            follower.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching (just in case)
            time.sleep(0.2)
            follower.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
            time.sleep(0.3)
        except Exception as e:
            print(f"[setup] Warning: Exit drag mode failed: {e}")
            print("[setup] Please restart the program if initialization fails")

        # CRITICAL: Force reset from MASTER role to FOLLOWER role
        # If master arm was in 0xFA (MASTER) mode, need explicit transition
        try:
            print("[setup] Force resetting master arm from any previous role...")
            # First set to FOLLOWER (this may fail if already in a weird state)
            master.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)
            # Set again to ensure it takes effect
            master.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)

            print("[setup] Configuring follower arm...")
            follower.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)
        except Exception as e:
            print(f"[setup] Warning: MasterSlaveConfig failed: {e}")
            print("[setup] Please restart the program if initialization fails")

        # Enable joint control mode with lower speed to prevent jitter
        try:
            master.MotionCtrl_1(0x00, 0x00, 0x00)
            time.sleep(0.1)
            follower.MotionCtrl_1(0x00, 0x00, 0x00)
            time.sleep(0.1)
            # Use slow speed (15%) during setup to prevent sudden movements
            master.MotionCtrl_2(0x01, 0x01, 15, 0x00)
            time.sleep(0.1)
            follower.MotionCtrl_2(0x01, 0x01, 15, 0x00)
            time.sleep(0.2)

            # Enable arms to stabilize
            try:
                master.EnableArm(7)
            except Exception:
                pass
            time.sleep(0.1)
            try:
                follower.EnableArm(7)
            except Exception:
                pass
            time.sleep(0.2)
        except Exception as e:
            print(f"[setup] Warning: MotionCtrl failed: {e}")
            print("[setup] Please restart the program if initialization fails")

        print("[setup] Both arms configured as followers")

        self.sensors["image"]["cam_head"].set_up(self.camera_serials["head"])
        self.sensors["image"]["cam_wrist"].set_up(self.camera_serials["wrist"])

        self.set_collect_type({
            "arm": ["joint", "qpos", "gripper"],
            "image": ["color"],
        })

        self.reassert_follower_hold()
        print("piper_dagger set up success - both arms in follower mode")

    def _configure_both_as_followers(self):
        """Configure both arms as followers (software controllable)"""
        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller
        if master is None or follower is None:
            raise RuntimeError("Controllers are not initialized")

        # Step 1: Exit drag teaching mode first (if in that mode)
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.1)
        master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        follower.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        time.sleep(0.1)

        # Step 2: Configure both as follower role
        master.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        follower.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        time.sleep(0.2)

        # Step 3: Enable joint control mode for both arms
        master.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # Enable joint control, max speed
        follower.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # Enable joint control, max speed
        time.sleep(0.1)
        try:
            master.EnableArm(7)
        except Exception:
            pass
        try:
            follower.EnableArm(7)
        except Exception:
            pass

        self.reassert_follower_hold()
        print("[reset] Both arms configured as followers")

    def move_follower(self, move_data, bypass_policy: bool = False):
        # Allow teleop/intervention to bypass policy gating.
        if not self._policy_enabled and not bypass_policy:
            return
        follower_controller = self.controllers["arm"]["left_arm"].controller
        self._send_arm_target(
            follower_controller,
            move_data,
            speed_percent=100,
            gripper_effort=self._follower_gripper_effort,
        )
        if self._sync_master_with_policy_commands and self._policy_enabled and not bypass_policy:
            # During autonomous rollout, send the same target to the master arm immediately
            # instead of waiting for feedback-based mirroring, which lags behind the follower.
            master_controller = self.controllers["arm"]["right_arm"].controller
            self._send_arm_target(
                master_controller,
                move_data,
                speed_percent=100,
                gripper_effort=self._master_gripper_effort,
            )

    def move_master(self, move_data):
        master_controller = self.controllers["arm"]["right_arm"].controller
        self._send_arm_target(
            master_controller,
            move_data,
            speed_percent=100,
            gripper_effort=self._master_gripper_effort,
        )

    def set_policy_enabled(self, enabled: bool):
        self._policy_enabled = enabled

    def get_master_state(self):
        return self.controllers["arm"]["right_arm"].get_state()

    def get_follower_state(self):
        return self.controllers["arm"]["left_arm"].get_state()

    def reset_intervention_tracking(self):
        """Clear intervention tracking state when switching modes."""
        self._intervention_anchor_joint_cmd = None
        self._intervention_anchor_gripper_cmd = None
        self._intervention_active = False
        self._intervention_move_count = 0
        # CRITICAL: Also reset mirror tracking to prevent stale data
        self._mirror_last_joint_cmd = None
        self._mirror_last_gripper_cmd = None
        self._mirror_last_send_time = 0.0
        self._mirror_master_joint_baseline = None
        self._mirror_master_gripper_baseline = None
        self._mirror_follower_joint_baseline = None
        self._mirror_follower_gripper_baseline = None

    def set_mirror_sync_baseline(self):
        """Capture the current master/follower poses as the relative-sync baseline."""
        master_state = self.get_master_state()
        follower_state = self.get_follower_state()
        self._mirror_master_joint_baseline = np.array(master_state["joint"], dtype=float)
        self._mirror_master_gripper_baseline = float(master_state["gripper"])
        self._mirror_follower_joint_baseline = np.array(follower_state["joint"], dtype=float)
        self._mirror_follower_gripper_baseline = float(follower_state["gripper"])

    def hold_follower_position(self):
        """Stop follower movement by commanding its current feedback state."""
        state = self.get_follower_state()
        follower_controller = self.controllers["arm"]["left_arm"].controller

        # Lock the current position and keep the same stronger gripper effort used
        # during follower rollout.
        self._send_arm_target(
            follower_controller,
            {
                "joint": state["joint"],
                "gripper": state["gripper"],
            },
            gripper_effort=self._follower_gripper_effort,
        )

    def reassert_follower_hold(self):
        """Re-apply follower hold after role/mode switches so gripper force stays on."""
        try:
            time.sleep(0.05)
            self.hold_follower_position()
        except Exception as exc:
            print(f"[warn] failed to reassert follower hold: {exc}")


    def mirror_master_to_follower(self):
        """Mirror master arm position to follower arm during intervention"""
        master_controller = self.controllers["arm"]["right_arm"].controller
        follower_controller = self.controllers["arm"]["left_arm"].controller

        # CRITICAL: Use control frames (GetArmJointCtrl) in drag mode, not feedback frames
        ctrl = master_controller.GetArmJointCtrl()
        if ctrl.Hz > 0:
            joint_ctrl = ctrl.joint_ctrl
            joint_cmd = np.array([
                joint_ctrl.joint_1,
                joint_ctrl.joint_2,
                joint_ctrl.joint_3,
                joint_ctrl.joint_4,
                joint_ctrl.joint_5,
                joint_ctrl.joint_6,
            ], dtype=int)
            # FIXED: Use correct gripper control field
            gripper_ctrl = master_controller.GetArmGripperCtrl()
            gripper_cmd = gripper_ctrl.gripper_ctrl.grippers_angle
        else:
            state = self.get_master_state()
            joint_cmd = (state["joint"] * 57295.7795).astype(int)  # 1000*180/3.1415926
            gripper_cmd = int(state["gripper"] * 70 * 1000)

        if not self._intervention_active:
            if self._intervention_anchor_joint_cmd is None:
                self._intervention_anchor_joint_cmd = joint_cmd.copy()
                self._intervention_anchor_gripper_cmd = gripper_cmd
                return

            anchor_delta = np.max(np.abs(joint_cmd - self._intervention_anchor_joint_cmd))
            anchor_gripper_delta = abs(gripper_cmd - self._intervention_anchor_gripper_cmd)
            if (anchor_delta < self._intervention_start_deadband_cmd and
                    anchor_gripper_delta < self._intervention_start_gripper_deadband_cmd):
                self._intervention_move_count = 0
                self.hold_follower_position()
                return

            self._intervention_move_count += 1
            if self._intervention_move_count < self._intervention_move_required:
                return

            self._intervention_active = True

        if self._mirror_last_joint_cmd is not None:
            joint_delta = np.max(np.abs(joint_cmd - self._mirror_last_joint_cmd))
            gripper_delta = abs(gripper_cmd - self._mirror_last_gripper_cmd)
            if (joint_delta < self._mirror_joint_deadband_cmd and
                    gripper_delta < self._mirror_gripper_deadband_cmd):
                return

        now = time.time()
        if now - self._mirror_last_send_time < self._mirror_min_send_interval:
            return

        if self._mirror_filter_alpha is not None and self._mirror_last_joint_cmd is not None:
            alpha = float(self._mirror_filter_alpha)
            joint_cmd = (alpha * joint_cmd + (1.0 - alpha) * self._mirror_last_joint_cmd).astype(int)
            gripper_cmd = int(alpha * gripper_cmd + (1.0 - alpha) * self._mirror_last_gripper_cmd)

        self._mirror_last_joint_cmd = joint_cmd.copy()
        self._mirror_last_gripper_cmd = gripper_cmd
        self._mirror_last_send_time = now

        self._send_arm_target(
            follower_controller,
            {
                "joint": joint_cmd / 57295.7795,
                "gripper": gripper_cmd / (70 * 1000),
            },
            gripper_effort=self._follower_gripper_effort,
        )

    def mirror_follower_to_master(self):
        """Mirror follower arm position to master arm for observation.

        The master arm keeps its current pose as the synchronization baseline, then
        tracks follower deltas relative to that baseline instead of jumping to the
        follower's absolute reset pose.
        """
        if self._sync_master_with_policy_commands and self._policy_enabled:
            return

        follower_state = self.get_follower_state()
        master_state = self.get_master_state()
        master_controller = self.controllers["arm"]["right_arm"].controller

        if self._mirror_master_joint_baseline is None:
            self.set_mirror_sync_baseline()
            return

        joint_delta = np.array(follower_state["joint"], dtype=float) - self._mirror_follower_joint_baseline
        target_joint = self._mirror_master_joint_baseline + joint_delta
        target_gripper = (
            self._mirror_master_gripper_baseline
            + float(follower_state["gripper"])
            - self._mirror_follower_gripper_baseline
        )

        # Convert joint angles to controller format
        j1, j2, j3, j4, j5, j6 = target_joint * 57295.7795  # 1000*180/3.1415926
        j1, j2, j3, j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        master_controller.MotionCtrl_2(0x01, 0x01, self._mirror_master_speed_percent, 0x00)
        master_controller.JointCtrl(j1, j2, j3, j4, j5, j6)

        # Mirror gripper
        gripper = int(target_gripper * 70 * 1000)
        master_controller.GripperCtrl(gripper, self._master_gripper_effort, 0x01, 0)

    def enable_master_drag_mode(self):
        """Enable intervention mode: master arm becomes draggable teaching arm"""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        # Step 1: Exit any existing modes first
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.1)
        master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        time.sleep(0.1)

        # Step 2: Configure as FOLLOWER first (clean state transition)
        master.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        time.sleep(0.2)

        # Step 3: Configure as MASTER role (0xFA) for true zero-force dragging
        master.MasterSlaveConfig(MASTER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        time.sleep(0.2)

        # Step 4: CRITICAL - Reduce drag teaching friction (lower = lighter)
        try:
            if hasattr(master, "GripperTeachingPendantParamConfig"):
                master.GripperTeachingPendantParamConfig(
                    teaching_range_per=100,
                    max_range_config=70,
                    teaching_friction=1,  # Very light friction
                )
                time.sleep(0.1)
        except Exception as e:
            print(f"[warn] failed to set teaching friction: {e}")

        # Step 5: Enter drag teaching mode
        # MotionCtrl_1(emergency_stop, track_ctrl, grag_teach_ctrl)
        # grag_teach_ctrl: 0x01 = Start teaching record (enter drag teaching mode)
        master.MotionCtrl_1(0x00, 0x00, 0x01)  # Enable drag teaching mode
        time.sleep(0.2)

        self.reset_intervention_tracking()
        print("[mode] master arm drag teaching mode enabled - FREE TO DRAG")

    def disable_master_drag_mode(self):
        """Disable intervention mode: master arm becomes software-controllable follower"""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        print("[mode] disabling master drag mode...")
        # The follower never leaves follower/joint-control mode during intervention.
        # Reconfiguring it here causes a brief release of the gripper hold, so keep
        # the follower untouched and simply re-assert its current target.
        self.reassert_follower_hold()

        # Step 1: Exit drag teaching mode
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.15)
        master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        time.sleep(0.15)

        self.reset_intervention_tracking()

        # Step 2: Switch back to follower role for software control
        master.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        time.sleep(0.25)

        # Step 3: Enable normal joint control with slow speed (15%)
        master.MotionCtrl_2(0x01, 0x01, 15, 0x00)
        time.sleep(0.1)
        try:
            master.EnableArm(7)
        except Exception:
            pass
        time.sleep(0.15)

        self.reassert_follower_hold()
        print("[mode] master arm back to follower mode (0xFC) - software control ready")

    def reset_to_follower_mode(self):
        """Reset both arms to follower mode (called at episode end)"""
        self._configure_both_as_followers()

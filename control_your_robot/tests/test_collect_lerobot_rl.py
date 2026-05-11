"""
测试 CollectLeRobotRL 的 schema 和 episode 标签逻辑。
不依赖真实硬件，重点验证逐帧监督字段和旧 schema 拦截。
"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot.data.collect_lerobot_rl import (
    ADV_IND_KEY,
    INTERVENTION_KEY,
    REWARD_KEY,
    REWARD_LABEL_KEY,
    VALUE_LABEL_KEY,
    CollectLeRobotRL,
)


print("✓ CollectLeRobotRL 导入成功")


class FakeDataset:
    """轻量级 LeRobotDataset 替身，用于捕获写入帧。"""

    def __init__(self, features=None):
        self.features = features or {}
        self.hf_features = self.features
        self.frames = []
        self.saved_episodes = 0
        self.num_episodes = 0

    def add_frame(self, frame, task=None):
        stored = dict(frame)
        if task is not None:
            stored["task"] = task
        self.frames.append(stored)

    def save_episode(self):
        self.saved_episodes += 1

    def start_image_writer(self, num_processes=0, num_threads=0):
        return None


def generate_mock_data(step):
    """生成模拟的机器人数据。"""
    controllers_data = {
        "left_arm": {
            "joint": np.linspace(0.1, 0.6, 6, dtype=np.float32) + step * 0.01,
            "gripper": np.array([0.5 + step * 0.01], dtype=np.float32),
        }
    }
    sensors_data = {}
    return controllers_data, sensors_data


def build_feature_schema(value_label_key=VALUE_LABEL_KEY, include_new_fields=True):
    """构造测试用 schema。"""
    features = {
        "state": {},
        "actions": {},
        INTERVENTION_KEY: {},
        value_label_key: {},
    }
    if include_new_fields:
        features.update(
            {
                REWARD_KEY: {},
                REWARD_LABEL_KEY: {},
                ADV_IND_KEY: {},
            }
        )
    return features


def create_collector(
    dataset=None,
    move_check=False,
    penalty_value=-1.0,
    output_dir=None,
):
    """创建测试收集器，可注入 fake dataset。"""
    collector = CollectLeRobotRL(
        repo_id="test_dataset",
        output_dir=str(output_dir or Path("./test_output")),
        task_name="test_task",
        fps=10,
        robot_type="piper",
        state_dim=7,
        action_dim=7,
        image_size=(480, 640),
        camera_keys={},
        move_check=move_check,
        tolerance=0.0001,
        penalty_value=penalty_value,
    )
    if dataset is not None:
        collector.dataset = dataset
        collector.dataset_created = True
        collector.value_label_key = VALUE_LABEL_KEY
    return collector


def collect_episode(collector, num_frames, intervention_pattern=None):
    """为 collector 填充一个 episode。"""
    for step in range(num_frames):
        controllers_data, sensors_data = generate_mock_data(step)
        is_intervention = False
        if intervention_pattern is not None:
            is_intervention = intervention_pattern[step]
        collector.collect(controllers_data, sensors_data, is_intervention=is_intervention)


def tensor_scalar(value):
    """从 tensor / ndarray / 标量中提取 float。"""
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def frame_scalars(frames, key):
    """提取逐帧标量字段。"""
    return [tensor_scalar(frame[key]) for frame in frames]


def frame_strings(frames, key):
    """提取逐帧字符串字段。"""
    return [frame[key] for frame in frames]


def test_dataset_schema_creation():
    """测试新建数据集时包含 reward/reward_label/adv_ind schema。"""
    print("\n" + "=" * 60)
    print("测试 1: 新数据集 schema")
    print("=" * 60)

    captured = {}
    original_create = LeRobotDataset.create

    def fake_create(*args, **kwargs):
        captured["features"] = kwargs["features"]
        return FakeDataset(features=kwargs["features"])

    LeRobotDataset.create = fake_create
    try:
        with TemporaryDirectory() as tmpdir:
            collector = create_collector(output_dir=tmpdir)
            controllers_data, sensors_data = generate_mock_data(0)
            collector.collect(controllers_data, sensors_data, is_intervention=False)

        features = captured["features"]
        assert REWARD_KEY in features, "缺少 reward schema"
        assert REWARD_LABEL_KEY in features, "缺少 reward_label schema"
        assert ADV_IND_KEY in features, "缺少 adv_ind schema"
        assert features[ADV_IND_KEY]["dtype"] == "string", "adv_ind dtype 应为 string"
        print("✓ 新 schema 包含 reward / reward_label / adv_ind")
        return True
    finally:
        LeRobotDataset.create = original_create


def test_existing_schema_validation():
    """测试旧 schema 会在写入前被拦截。"""
    print("\n" + "=" * 60)
    print("测试 2: 旧 schema 拦截")
    print("=" * 60)

    dataset = FakeDataset(features=build_feature_schema(include_new_fields=False))
    collector = create_collector(dataset=dataset)

    controllers_data, sensors_data = generate_mock_data(0)
    try:
        collector.collect(controllers_data, sensors_data, is_intervention=False)
    except RuntimeError as exc:
        message = str(exc)
        assert REWARD_KEY in message, "错误信息应包含 reward"
        assert REWARD_LABEL_KEY in message, "错误信息应包含 reward_label"
        assert ADV_IND_KEY in message, "错误信息应包含 adv_ind"
        print("✓ 旧 schema 在 collect 阶段被正确拦截")
        return True

    raise AssertionError("旧 schema 未被拦截")


def test_teleop_success_labels():
    """测试主从采集规则：终帧 reward=1，adv_ind=positive。"""
    print("\n" + "=" * 60)
    print("测试 3: teleop 成功标签")
    print("=" * 60)

    dataset = FakeDataset(features=build_feature_schema())
    collector = create_collector(dataset=dataset)

    collect_episode(collector, num_frames=5, intervention_pattern=[True] * 5)
    collector.save_episode(success=True, adv_ind_value="positive")

    assert dataset.saved_episodes == 1, "应保存 1 个 episode"
    rewards = frame_scalars(dataset.frames, REWARD_KEY)
    reward_labels = frame_scalars(dataset.frames, REWARD_LABEL_KEY)
    adv_ind = frame_strings(dataset.frames, ADV_IND_KEY)
    interventions = frame_scalars(dataset.frames, INTERVENTION_KEY)

    assert rewards == [0.0, 0.0, 0.0, 0.0, 1.0], f"teleop reward 错误: {rewards}"
    assert np.allclose(reward_labels, [-0.2, -0.2, -0.2, -0.2, 0.0]), f"teleop reward_label 错误: {reward_labels}"
    assert adv_ind == ["positive"] * 5, f"teleop adv_ind 错误: {adv_ind}"
    assert interventions == [1.0] * 5, f"teleop intervention 错误: {interventions}"
    print("✓ teleop 成功标签正确")
    return True


def test_rollout_success_labels():
    """测试 rollout 成功规则：终帧 reward=1，adv_ind=none。"""
    print("\n" + "=" * 60)
    print("测试 4: rollout 成功标签")
    print("=" * 60)

    dataset = FakeDataset(features=build_feature_schema())
    collector = create_collector(dataset=dataset)

    collect_episode(collector, num_frames=4, intervention_pattern=[False] * 4)
    collector.save_episode(success=True, adv_ind_value="none")

    rewards = frame_scalars(dataset.frames, REWARD_KEY)
    reward_labels = frame_scalars(dataset.frames, REWARD_LABEL_KEY)
    adv_ind = frame_strings(dataset.frames, ADV_IND_KEY)

    assert rewards == [0.0, 0.0, 0.0, 1.0], f"rollout success reward 错误: {rewards}"
    assert np.allclose(reward_labels, [-0.25, -0.25, -0.25, 0.0]), f"rollout success reward_label 错误: {reward_labels}"
    assert adv_ind == ["none"] * 4, f"rollout success adv_ind 错误: {adv_ind}"
    print("✓ rollout 成功标签正确")
    return True


def test_rollout_failure_labels_and_value_regression():
    """测试 rollout 失败规则，并确认 value_label 逻辑未回归。"""
    print("\n" + "=" * 60)
    print("测试 5: rollout 失败标签和值回归")
    print("=" * 60)

    dataset = FakeDataset(features=build_feature_schema())
    collector = create_collector(dataset=dataset, penalty_value=-2.0)

    collect_episode(collector, num_frames=4, intervention_pattern=[False] * 4)
    collector.save_episode(success=False, adv_ind_value="none")

    rewards = frame_scalars(dataset.frames, REWARD_KEY)
    reward_labels = frame_scalars(dataset.frames, REWARD_LABEL_KEY)
    adv_ind = frame_strings(dataset.frames, ADV_IND_KEY)
    value_labels = frame_scalars(dataset.frames, VALUE_LABEL_KEY)

    assert rewards == [0.0, 0.0, 0.0, 0.0], f"rollout failure reward 错误: {rewards}"
    assert np.allclose(reward_labels, [-0.25, -0.25, -0.25, -1.0]), f"rollout failure reward_label 错误: {reward_labels}"
    assert adv_ind == ["none"] * 4, f"rollout failure adv_ind 错误: {adv_ind}"
    assert np.allclose(value_labels, [-2.0, -2.0, -2.0, -2.0]), f"value_label 回归: {value_labels}"
    print("✓ rollout 失败标签和值回归正确")
    return True


def test_move_detection():
    """测试移动检测仍然有效。"""
    print("\n" + "=" * 60)
    print("测试 6: 移动检测")
    print("=" * 60)

    dataset = FakeDataset(features=build_feature_schema())
    collector = create_collector(dataset=dataset, move_check=True)

    controllers_data, sensors_data = generate_mock_data(0)
    collector.collect(controllers_data, sensors_data, False)
    print(f"  帧1: 收集 (缓存: {len(collector.episode_buffer)})")

    collector.collect(controllers_data, sensors_data, False)
    print(f"  帧2: 静止，应跳过 (缓存: {len(collector.episode_buffer)})")

    controllers_data, sensors_data = generate_mock_data(10)
    collector.collect(controllers_data, sensors_data, False)
    print(f"  帧3: 移动 (缓存: {len(collector.episode_buffer)})")

    assert len(collector.episode_buffer) == 2, "应该只收集了 2 帧"
    print("✓ 移动检测测试通过")
    return True


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("CollectLeRobotRL 功能测试")
    print("=" * 60)

    tests = [
        ("新数据集 schema", test_dataset_schema_creation),
        ("旧 schema 拦截", test_existing_schema_validation),
        ("teleop 成功标签", test_teleop_success_labels),
        ("rollout 成功标签", test_rollout_success_labels),
        ("rollout 失败标签和值回归", test_rollout_failure_labels_and_value_regression),
        ("移动检测", test_move_detection),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as exc:
            print(f"\n✗ 测试失败: {exc}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n所有测试通过")
        return True

    print("\n部分测试失败")
    return False


if __name__ == "__main__":
    import os

    os.environ["INFO_LEVEL"] = "INFO"
    success = run_all_tests()
    sys.exit(0 if success else 1)

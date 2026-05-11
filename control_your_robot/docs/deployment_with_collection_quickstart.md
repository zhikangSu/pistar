# 部署时数据收集功能 - 快速开始

## 概述

基于 `example/deploy/piper_single_on_PI0.py` 的增强版本，支持在部署时收集强化学习数据。

## 核心功能

✅ **人工干预模式**: 按空格键实时切换自主/人工控制
✅ **干预标记**: 自动记录每帧是自主(0)还是人工(1)
✅ **价值标签**: 自动写入 `value_label`
✅ **奖励字段**: 自动写入 `reward` 和 `reward_label`
✅ **adv_ind**: rollout 数据自动写 `"none"`，留待后续覆盖
✅ **移动检测**: 干预模式下只记录移动帧，跳过静止帧
✅ **LeRobot 格式**: 直接生成 LeRobot 数据集，无需转换

## 快速开始

### 1. 安装依赖

**无需额外依赖！**

键盘监听使用 Python 内置库（`termios`, `select`, `tty`），在 Linux 系统上默认可用。

### 2. 配置参数

编辑 `example/deploy/piper_single_on_PI0_with_collection.py`:

```python
# 模型参数
MODEL_PATH = "/app/checkpoint/29999"
TASK_NAME = "Plug the black plug into the three-hole socket"

# 数据收集参数
REPO_ID = "piper_plug_task_rl"
OUTPUT_DIR = "/home/chaihoa/Desktop/experiment/lerobot_datasets"
FPS = 10
PENALTY_VALUE = -1.0  # 失败惩罚值
```

### 3. 运行脚本

```bash
uv run python example/deploy/piper_single_on_PI0_with_collection.py
```

### 4. 操作流程

```
1. 按 Enter → 开始 episode
2. 机器人自主执行任务
3. 按 空格 → 切换到人工干预（机器人停止）
4. 人工控制机器人（通过示教器等）
5. 按 空格 → 切换回自主模式
6. 按 Enter → 结束 episode
7. 输入 1（成功）或 0（失败）
8. 数据自动保存
```

## 数据格式

生成的 LeRobot 数据集包含以下字段：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `state` | float32 | (7,) | 机器人状态 |
| `actions` | float32 | (7,) | 动作（下一帧状态） |
| `intervention` | int64 | (1,) | 干预标记 (0/1) |
| `value_label` | float32 | (1,) | 价值标签；成功轨迹按 `-(T - t) / T`，失败轨迹为 `penalty_value` |
| `reward` | float32 | (1,) | 成功轨迹仅最后一帧为 `1`，失败轨迹全为 `0` |
| `reward_label` | float32 | (1,) | 非终帧固定 `-1 / T`，终帧按成功/失败取 `0` 或 `-1` |
| `adv_ind` | string | (1,) | rollout 数据当前固定为 `"none"` |
| `image` | uint8 | (3,480,640) | 头部相机 |
| `wrist_image` | uint8 | (3,480,640) | 手腕相机 |

## 标签计算

```python
# 成功 episode
value_label[t] = -(100 - t) / 100
reward[:-1] = 0
reward[-1] = 1
reward_label[:-1] = -1 / 100
reward_label[-1] = 0

# 失败 episode
value_label[:] = penalty_value
reward[:] = 0
reward_label[:-1] = -1 / 100
reward_label[-1] = -1

# rollout 默认占位
adv_ind[:] = "none"
```

## 文件结构

```
control_your_robot/
├── src/robot/data/
│   └── collect_lerobot_rl.py          # 扩展的数据收集类
├── example/deploy/
│   ├── piper_single_on_PI0.py         # 原始部署脚本
│   └── piper_single_on_PI0_with_collection.py  # 新增：带数据收集
└── docs/
    └── deployment_with_collection.md   # 详细文档
```

## 核心类

### CollectLeRobotRL

位置: `src/robot/data/collect_lerobot_rl.py`

```python
from robot.data.collect_lerobot_rl import CollectLeRobotRL

collector = CollectLeRobotRL(
    repo_id="dataset_name",
    output_dir="./datasets",
    task_name="task",
    fps=10,
    penalty_value=-1.0,
)

# 收集数据（带干预标记）
collector.collect(controllers_data, sensors_data, is_intervention=False)

# 保存 episode（rollout 默认写 adv_ind="none"）
collector.save_episode(success=True, adv_ind_value="none")
```

### InterventionController

位置: `example/deploy/piper_single_on_PI0_with_collection.py`

```python
intervention_ctrl = InterventionController()
intervention_ctrl.start_listener()  # 启动空格键监听

# 获取当前模式
is_intervention = intervention_ctrl.get_state()
```

## 常见问题

### Q: 键盘监听不工作？

```bash
# 使用 sudo 运行
sudo uv run python example/deploy/piper_single_on_PI0_with_collection.py

# 或添加用户到 input 组
sudo usermod -a -G input $USER
```

### Q: 如何调整移动检测灵敏度？

```python
collector = CollectLeRobotRL(
    ...
    tolerance=0.001,  # 增大容差（默认 0.0001）
)
```

### Q: 如何修改失败惩罚值？

```python
PENALTY_VALUE = -2.0  # 修改为 -2.0
```

## 与原始脚本对比

| 特性 | 原始脚本 | 新脚本 |
|------|----------|--------|
| 部署模型 | ✅ | ✅ |
| 数据收集 | ❌ | ✅ |
| 人工干预 | ❌ | ✅ |
| 干预标记 | ❌ | ✅ |
| 价值标签 | ❌ | ✅ |
| 移动检测 | ❌ | ✅ |

## 下一步

1. **训练强化学习模型**: 使用收集的数据训练 RL 策略
2. **数据分析**: 分析干预频率、成功率等指标
3. **迭代改进**: 根据数据反馈改进策略

## 参考文档

- 详细文档: `docs/deployment_with_collection.md`
- 原始部署脚本: `example/deploy/piper_single_on_PI0.py`
- LeRobot 收集示例: `example/collect/collect_lerobot_direct.py`
- OpenPI 价值标签: `/home/chaihoa/project_wang/openpi/scripts/add_value_labels.py`

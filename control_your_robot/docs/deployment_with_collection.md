# 部署时数据收集 + 强化学习标签

本文档介绍如何在部署 PI0 模型时同时收集强化学习数据，支持人工干预以及 `value_label`、`reward`、`reward_label`、`adv_ind` 等监督字段。

## 功能特性

### 1. 自主操作 + 人工干预切换
- **自主模式**: 机器人执行策略网络输出的动作
- **干预模式**: 机器人停止执行网络输出，由人工接管控制
- **切换方式**: 按空格键实时切换模式
- **数据记录**: 两种模式下的数据都会被记录，并标记干预标志

### 2. 干预标记 (Intervention Flag)
- **字段名**: `intervention`
- **数据类型**: `int64`
- **取值**:
  - `0`: 自主操作（机器人执行策略输出）
  - `1`: 人工干预（人工控制机器人）
- **用途**: 用于强化学习训练时区分自主和人工数据

### 3. 价值标签 (Value Labels)
- **字段名**: `value_label`
- **数据类型**: `float32`
- **计算公式**:
  - **中间帧**: `value_label = -(T - t) / T`，范围 `[-1, 0]`
    - `T`: episode 总帧数
    - `t`: 当前帧索引
  - **最后一帧**:
    - 成功: `value_label = 0`
    - 失败: `value_label = penalty_value` (默认 `-1.0`)
- **参考**: 与 `/home/chaihoa/project_wang/openpi/scripts/add_value_labels.py` 一致

### 4. 奖励字段
- **字段名**: `reward`
- **数据类型**: `float32`
- **计算规则**:
  - 成功 episode: 仅最后一帧 `reward = 1`，其余帧 `reward = 0`
  - 失败 episode: 所有帧 `reward = 0`

### 5. 奖励标签 (Reward Labels)
- **字段名**: `reward_label`
- **数据类型**: `float32`
- **计算规则**:
  - 所有非终帧: `reward_label = -1 / T`
  - 成功 episode 终帧: `reward_label = 0`
  - 失败 episode 终帧: `reward_label = -1`

### 6. adv_ind
- **字段名**: `adv_ind`
- **数据类型**: `string`
- **当前 rollout 默认值**: 每一帧都写字符串 `"none"`，留待后续覆盖

### 7. 移动检测
- **功能**: 在干预模式下，只记录机器人移动时的帧
- **目的**: 避免记录静止帧，减少数据冗余
- **容差**: 默认 `0.0001` 弧度

## 使用方法

### 基本用法

```bash
uv run python example/deploy/piper_single_on_PI0_with_collection.py
```

### 配置参数

在脚本中修改以下参数：

```python
# 模型参数
MODEL_PATH = "/app/checkpoint/29999"
TASK_NAME = "Plug the black plug into the three-hole socket"
TRAIN_CONFIG_NAME = "pi05_libero_local"
MAX_STEP = 200
NUM_EPISODE = 10

# 数据收集参数
REPO_ID = "piper_plug_task_rl"
OUTPUT_DIR = "/home/chaihoa/Desktop/experiment/lerobot_datasets"
FPS = 10
PENALTY_VALUE = -C.0  # 失败时的惩罚值,-C
```

### 操作流程

1. **启动脚本**
   ```bash
   uv run python example/deploy/piper_single_on_PI0_with_collection.py
   ```

2. **开始 Episode**
   - 按 `Enter` 键开始新的 episode
   - 机器人默认进入自主操作模式

3. **模式切换**
   - 按 `空格键` 切换自主/干预模式
   - 自主模式: 机器人执行策略输出
   - 干预模式: 机器人停止，等待人工控制

4. **结束 Episode**
   - 按 `Enter` 键结束当前 episode
   - 系统提示输入任务完成情况:
     - 输入 `1`: 任务成功（最后一帧 `reward = 1`，最后一帧 `reward_label = 0`）
     - 输入 `0`: 任务失败（所有帧 `reward = 0`，最后一帧 `reward_label = -1`）

5. **数据保存**
   - 系统自动保存 episode 数据到 LeRobot 格式
   - 包含 `intervention`、`value_label`、`reward`、`reward_label`、`adv_ind`

## 数据格式

### LeRobot 数据集结构

```
task_rl/
├── data/
│   ├── chunk-000/
│   │   ├── episode_0.parquet
│   │   ├── episode_1.parquet
│   │   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── episode_0_image.mp4
│   │   ├── episode_0_wrist_image.mp4
│   │   └── ...
├── meta_data/
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── info.json
```

### Parquet 文件字段

每个 episode 的 parquet 文件包含以下字段：

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `state` | float32 | (7,) | 机器人状态 (6关节 + 1夹爪) |
| `actions` | float32 | (7,) | 动作 (下一帧的 state) |
| `intervention` | int64 | (1,) | 干预标记 (0=自主, 1=干预) |
| `value_label` | float32 | (1,) | 价值标签。成功轨迹按 `-(T - t) / T`，失败轨迹为 `penalty_value` |
| `reward` | float32 | (1,) | 成功轨迹仅最后一帧为 `1`，失败轨迹全为 `0` |
| `reward_label` | float32 | (1,) | 非终帧固定 `-1 / T`，终帧按成功/失败取 `0` 或 `-1` |
| `adv_ind` | string | (1,) | rollout 数据当前固定为 `"none"` |
| `image` | uint8 | (3, 480, 640) | 头部相机图像 |
| `wrist_image` | uint8 | (3, 480, 640) | 手腕相机图像 |

### 价值标签示例

假设一个 episode 有 100 帧，任务成功：

```python
# 中间帧 (t=0 到 t=98)
value_label[0] = -(100 - 0) / 100 = -1.0
value_label[1] = -(100 - 1) / 100 = -0.99
value_label[50] = -(100 - 50) / 100 = -0.5
value_label[98] = -(100 - 98) / 100 = -0.02

# 最后一帧 (t=99)
value_label[99] = 0.0  # 成功
```

如果任务失败：

```python
value_label[:] = -1.0  # 失败 (penalty_value)
reward[:] = 0.0
reward_label[:-1] = -1 / 100
reward_label[-1] = -1.0
```

## 核心实现

### 1. CollectLeRobotRL 类

位置: `src/robot/data/collect_lerobot_rl.py`

扩展了 `CollectLeRobot`，添加了：
- 干预标记支持
- 价值标签计算
- 成功/失败区分

关键方法：

```python
# 收集数据（带干预标记）
collector.collect(controllers_data, sensors_data, is_intervention=True/False)

# 保存 episode（部署 rollout 通常固定写 adv_ind="none"）
collector.save_episode(success=True/False, adv_ind_value="none")
```

### 2. InterventionController 类

位置: `example/deploy/piper_single_on_PI0_with_collection.py`

管理自主/干预模式切换：

```python
intervention_ctrl = InterventionController()
intervention_ctrl.start_listener()  # 启动键盘监听

# 获取当前状态
is_intervention = intervention_ctrl.get_state()

# 切换模式（自动通过空格键触发）
intervention_ctrl.toggle_intervention()
```

### 3. 数据收集流程

```python
# 自主模式
if not is_intervention:
    # 执行策略输出
    action = model.get_action()
    robot.move(action)
    # 收集数据（标记为自主）
    collector.collect(data[0], data[1], is_intervention=False)

# 干预模式
else:
    # 机器人停止，不执行网络输出
    # 只收集数据（标记为干预）
    collector.collect(data[0], data[1], is_intervention=True)
```


## 参考资料

- [LeRobot 文档](https://github.com/huggingface/lerobot)
- [OpenPI 价值标签脚本](file:///home/chaihoa/project_wang/openpi/scripts/add_value_labels.py)
- [原始部署脚本](file:///home/chaihoa/project_wang/control_your_robot/example/deploy/piper_single_on_PI0.py)
- [LeRobot 数据收集示例](file:///home/chaihoa/project_wang/control_your_robot/example/collect/collect_lerobot_direct.py)

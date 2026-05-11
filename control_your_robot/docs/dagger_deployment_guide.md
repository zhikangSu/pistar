# DAgger部署配置指南

## 概述

本文档说明如何配置和使用DAgger（Dataset Aggregation）模式进行机器人策略训练数据收集。

DAgger是一种迭代式强化学习算法，通过以下方式改进策略：
1. **自主模式**：策略控制从臂执行任务，主臂镜像从臂位置（便于观察和随时接管）
2. **干预模式**：人工操作主臂，从臂跟随主臂执行遥操作
3. **数据收集**：记录 `intervention`、`value_label`、`reward`、`reward_label` 和 `adv_ind` 用于后续训练

## 硬件要求

### 必需硬件

1. **2根USB-CAN适配器**
   - 主臂使用一根（连接到can0）
   - 从臂使用一根（连接到can1）
   - **重要**：必须使用独立的CAN接口，单CAN线无法实现DAgger功能

2. **机械臂配置**
   - 主臂（Master Arm）：用于人工干预，连接到can0
   - 从臂（Follower Arm）：执行任务，连接到can1

3. **相机**
   - 头部相机（cam_head）：俯视视角
   - 手腕相机（cam_wrist）：第一人称视角
   - 相机挂载在从臂上

### 硬件连接示意图

```
电脑
├── USB-CAN 1 (can0) ─→ 主臂
└── USB-CAN 2 (can1) ─→ 从臂
```



## 配置步骤

### 1. 硬件配置


```python
# 相机序列号（使用rs-enumerate-devices查看）
CAMERA_SERIALS = {
    'head': '338622070768',    # 替换为你的头部相机序列号
    'wrist': '338622072453',   # 替换为你的手腕相机序列号
}

# 主臂起始位置（弧度）
START_POSITION_ANGLE_MASTER_ARM = [
    0,        # Joint 1
    -0.4208,  # Joint 2
    0.0324,   # Joint 3
    0.0780,   # Joint 4
    0.3558,   # Joint 5
    0.0078,   # Joint 6
]

# 从臂起始位置（弧度）
START_POSITION_ANGLE_FOLLOWER_ARM = [
    0,        # Joint 1
    -0.4208,  # Joint 2
    0.0324,   # Joint 3
    0.0780,   # Joint 4
    0.3558,   # Joint 5
    0.0078,   # Joint 6
]
```

### 2. 部署配置

编辑 `example/deploy/piper_dagger_on_PI0.py`：

```python
# 模型配置
MODEL_PATH = "/path/to/your/checkpoint"
TASK_NAME = "Your task name"
TRAIN_CONFIG_NAME = "pi05_libero_local"

# 数据收集配置
REPO_ID = "your_dataset_name"
OUTPUT_DIR = "/path/to/output"
FPS = 10                    # 数据收集频率
PENALTY_VALUE = -1.0        # 失败时的惩罚值
MIRROR_FPS = 50             # 主臂镜像频率（建议5倍于FPS）

# Episode配置
MAX_STEP = 200              # 每个episode最大步数
NUM_EPISODE = 10            # 总episode数
```

### 3. CAN接口检查

在运行前，确认CAN接口已正确配置：

```bash
# 检查CAN接口
ip link show can0
ip link show can1

# 如果未启动，手动启动
sudo ip link set can0 type can bitrate 500000
sudo ip link set can1 type can bitrate 500000
sudo ip link set can0 up
sudo ip link set can1 up
```

## 使用方法

### 启动DAgger部署

```bash
uv run python example/deploy/piper_dagger_on_PI0.py
```

### 操作流程

1. **初始化阶段**
   - 机器人自动初始化
   - 加载策略模型
   - 初始化数据收集器

2. **Episode执行**
   - 按`Enter`键开始episode
   - 默认进入自主模式（策略控制）
   - 按`空格键`切换模式：
     - 🟢 **自主模式**：策略控制从臂，主臂镜像从臂
     - 🟡 **干预模式**：按空格进入干预模式，手动操作主臂，从臂跟随
   - 按`Enter`键结束episode

3. **Episode标注**
   - 输入`1`：任务成功（最后一帧 `reward=1`，最后一帧 `reward_label=0`）
   - 输入`0`：任务失败（全帧 `reward=0`，最后一帧 `reward_label=-1`）

4. **数据保存**
   - 数据自动保存为LeRobot格式
   - 包含 `intervention`、`value_label`、`reward`、`reward_label`、`adv_ind`

### 控制技巧

#### 自主模式下的干预时机

- **策略即将失败**：看到机器人动作不合理时立即按空格键
- **关键步骤**：在任务关键步骤手动演示正确操作
- **安全保护**：避免碰撞或危险动作

#### 干预模式下的操作

- **平滑操作**：缓慢移动主臂，避免从臂抖动
- **观察从臂**：确保从臂正确跟随主臂
- **完成演示后**：按空格键切回自主模式

## 数据格式

### LeRobot + RL格式

生成的数据集包含以下字段：

```python
{
    # 标准LeRobot字段
    "observation.state": [...],          # 从臂状态（7维：6关节+1夹爪）
    "observation.image": [...],          # 头部相机图像
    "observation.wrist_image": [...],    # 手腕相机图像
    "action": [...],                     # 动作（7维）

    # RL扩展字段
    "intervention": 0/1,                 # 干预标记（0=自主，1=人工）
    "value_label": float,                # 价值标签（成功轨迹按 -(T - t) / T，失败轨迹为 penalty_value）
    "reward": float,                     # 奖励：成功 episode 仅最后一帧为 1，失败 episode 全为 0
    "reward_label": float,               # 中间帧固定为 -1 / T，终帧按成功/失败取 0 或 -1
    "adv_ind": "none",                   # rollout 数据每一帧固定写 "none"，留待后续覆盖

    # 元数据
    "episode_index": int,
    "frame_index": int,
    "timestamp": float,
}
```

### 标签规则

- `value_label`
  - 成功 episode: 除最后一帧外，按 `-(T - t) / T` 递增到 0
  - 失败 episode: 所有帧都写 `penalty_value`，默认 `-1.0`
- `reward`
  - 输入 `1`: 最后一帧为 `1`，其余帧为 `0`
  - 输入 `0`: 所有帧为 `0`
- `reward_label`
  - 除最后一帧外，所有帧都为 `-1 / T`
  - 输入 `1`: 最后一帧为 `0`
  - 输入 `0`: 最后一帧为 `-1`
- `adv_ind`
  - 该 DAgger rollout 脚本保存数据时固定写字符串 `"none"`
  - 这与推理时是否传入 `--adv-ind` 无关，后者只影响模型请求

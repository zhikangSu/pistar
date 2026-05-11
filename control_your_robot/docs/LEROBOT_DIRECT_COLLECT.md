# LeRobot 直接收集模式

## 概述

本系统提供**直接生成 LeRobot 格式**的数据收集功能，无需中间 HDF5 格式转换步骤。

## 新增文件

```
src/robot/data/collect_lerobot.py           # LeRobot 数据收集类
my_robot/base_robot_lerobot.py              # 机器人基类（LeRobot 版本）
my_robot/piper_single_lerobot.py            # Piper 单臂机器人（LeRobot 版本）
example/collect/collect_lerobot_direct.py   # 直接收集脚本
```

## 工作流程对比

### 旧流程（需要转换）
```
1. collect_mp_robot.py → 生成 HDF5 文件
2. convert_piper_to_lerobot_libero.py → 转换为 LeRobot 格式
```

### 新流程（直接生成）
```
collect_lerobot_direct.py → 直接生成 LeRobot 格式 ✅
```

## 快速开始

### 1. 修改配置参数

编辑 `example/collect/collect_lerobot_direct.py`:

```python
REPO_ID = "piper_plug_task"                    # 数据集 ID
OUTPUT_DIR = "./datasets/lerobot_datasets"     # 输出目录
TASK_NAME = "Plug the black plug into the three-hole socket"  # 任务名称
FPS = 10                                       # 采集频率
NUM_EPISODES = 10                              # 收集 episode 数量
```

### 2. 运行收集脚本

```bash
python example/collect/collect_lerobot_direct.py
```

### 3. 操作流程

1. **启动脚本** → 进程初始化
2. **按 Enter** → 开始收集当前 episode
3. **按 Enter** → 结束并保存当前 episode
4. **重复步骤 2-3** → 收集多个 episodes

### 4. 查看数据

收集完成后，数据直接保存为 LeRobot 格式：

```bash
OUTPUT_DIR/
└── REPO_ID/
    ├── data/
    ├── meta_data/
    ├── videos/
    └── ...
```

## 核心组件说明

### 1. CollectLeRobot 类

**路径**: `src/robot/data/collect_lerobot.py`

**功能**:
- 直接使用 LeRobot API 保存数据
- 自动处理图像编码
- 支持单臂/双臂机器人
- 内置移动检测

**关键方法**:
- `collect()` - 收集一帧数据
- `save_episode()` - 保存当前 episode
- `_extract_state()` - 从控制器数据提取状态
- `_extract_images()` - 从传感器数据提取图像

### 2. RobotLeRobot 基类

**路径**: `my_robot/base_robot_lerobot.py`

**功能**:
- 替代 `Robot` 基类
- 使用 `CollectLeRobot` 而不是 `CollectAny`
- 接口与原基类保持一致

### 3. PiperSingleLeRobot 类

**路径**: `my_robot/piper_single_lerobot.py`

**功能**:
- Piper 单臂机器人的 LeRobot 版本
- 自动配置 Libero 格式（7维单臂）
- 相机映射：
  - `cam_head` → `image`
  - `cam_wrist` → `wrist_image`

### 4. collect_lerobot_direct.py 脚本

**路径**: `example/collect/collect_lerobot_direct.py`

**功能**:
- 多进程数据收集
- 时间同步调度
- 交互式控制（Enter 键）

## 自定义机器人

如果您需要为其他机器人创建 LeRobot 收集器：

### 示例：双臂机器人

```python
from my_robot.base_robot_lerobot import RobotLeRobot

class MyDualArmRobot(RobotLeRobot):
    def __init__(self, repo_id, output_dir, task_name, fps=10):
        camera_keys = {
            "cam_head": "image",
            "cam_wrist_left": "wrist_image_left",
            "cam_wrist_right": "wrist_image_right",
        }

        super().__init__(
            repo_id=repo_id,
            output_dir=output_dir,
            task_name=task_name,
            fps=fps,
            robot_type="dual_arm",
            state_dim=14,  # 双臂：(6关节 + 1夹爪) × 2
            action_dim=14,
            image_size=(480, 640),
            camera_keys=camera_keys,
            move_check=True,
        )

        # 初始化控制器和传感器
        self.controllers = {
            "arm": {
                "left_arm": ...,
                "right_arm": ...,
            },
        }
        self.sensors = {
            "image": {
                "cam_head": ...,
                "cam_wrist_left": ...,
                "cam_wrist_right": ...,
            },
        }
```

## 数据格式

### LeRobot Libero 格式（单臂）

```python
features = {
    "image": {
        "dtype": "image",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "state": {
        "dtype": "float32",
        "shape": (7,),  # 6关节 + 1夹爪
        "names": ["joint_1", ..., "joint_6", "gripper"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_1", ..., "joint_6", "gripper"],
    },
}
```

### 与 HDF5 格式的映射

| HDF5 格式 | LeRobot 格式 |
|-----------|--------------|
| `left_arm/joint` | `state[:6]` |
| `left_arm/gripper` | `state[6]` |
| `cam_head/color` | `image` |
| `cam_wrist/color` | `wrist_image` |

## 优势

✅ **无需转换** - 直接生成目标格式
✅ **节省存储** - 避免中间 HDF5 文件
✅ **简化流程** - 一步到位
✅ **易于扩展** - 支持自定义机器人
✅ **保持兼容** - 接口与原系统一致

## 常见问题

### 1. 如何修改图像尺寸？

在创建机器人实例时指定：

```python
robot = PiperSingleLeRobot(
    image_size=(720,1280)
)
```

### 2. 如何关闭移动检测？

```python
robot = PiperSingleLeRobot(
    move_check=False,
)
```

### 3. 如何添加更多相机？

修改 `camera_keys` 映射：

```python
camera_keys = {
    "cam_head": "image",
    "cam_wrist": "wrist_image",
    "cam_third": "third_person_image",  # 新增相机
}
```

同时在 `sensors` 中添加对应的传感器。

## 性能优化

- **图像编码**: 支持 JPEG 压缩（自动检测）
- **多进程写入**: 10 个线程 + 5 个进程
- **移动检测**: 跳过静止帧，减少数据量

## 技术细节

### 数据流

```
控制器 + 传感器
    ↓
robot.get()
    ↓
robot.collect()
    ↓
CollectLeRobot.collect()
    ↓
episode_buffer (内存缓存)
    ↓
robot.finish()
    ↓
LeRobot.add_frame() × N
    ↓
LeRobot.save_episode()
    ↓
保存到磁盘
```

### Action 生成策略

```python
# action = 下一帧的 state
actions[i] = states[i + 1]
actions[-1] = states[-1]  # 最后一帧重复
```

## 与原系统兼容性

本系统**不影响**原有的 HDF5 收集流程：

- `collect_mp_robot.py` ✅ 仍然可用
- `CollectAny` ✅ 仍然可用
- `convert_piper_to_lerobot_libero.py` ✅ 仍然可用

您可以根据需求选择：
- **新系统**: 直接生成 LeRobot 格式
- **旧系统**: 先生成 HDF5，再转换

## 总结

本系统提供了一个**开箱即用**的 LeRobot 直接收集方案，借鉴了 `collect_mp_robot.py` 的多进程架构，但省去了中间转换步骤，让数据收集更加高效和简洁。

# LIBERO 推理 Rollout 导出为 LeRobot（PiStar）

本文档说明如何使用 `examples/libero/main.py`，将 LIBERO 推理阶段的 rollout 自动导出为 LeRobot 数据集。

## 已增加能力

`examples/libero/main.py` 已支持“评测时可选保存 rollout 数据集”。

开启后，每个 episode 会保存为一个 LeRobot episode，包含字段：

- `image`
- `wrist_image`
- `state`（8 维：eef_pos(3) + axis_angle(3) + gripper_qpos(2)）
- `actions`（7 维，策略输出）
- `intervention`（固定为 `0`，无人工干预）
- `value`（PiStar 风格价值标签）

`task` 字段保存为 LIBERO 的语言指令。

## Value 标签规则

与 DAgger 采集脚本一致：

- 成功轨迹：
  - `value[t] = -(T - t) / T`
  - 最后一帧强制为 `0.0`
- 失败轨迹：
  - 所有帧都为 `rollout_penalty_value`（默认 `-1.0`）

成功/失败自动根据 LIBERO 的 `done` 判断，不需要人工输入。

## 运行示例

在项目目录执行：

### 0) 先启动策略服务（必须）

`examples/libero/main.py` 会通过 websocket 请求策略推理，因此需要先在另一个终端启动 `serve_policy.py`。

例如（按你的 checkpoint 路径修改）：

```bash
cd /media/chaihoa/software1/project_wang/control_your_robot/src/robot/policy/pistar

python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_star_libero_infer \
  --policy.dir=checkpoints/pi05_star_libero/my_experiment/10000 \
  --port=8000
```

> `--port` 需要和下面 `main.py` 里的 `--args.port` 一致。

### 1) 启动 LIBERO rollout + LeRobot 导出

```bash
cd /media/chaihoa/software1/project_wang/control_your_robot/src/robot/policy/pistar

uv run examples/libero/main.py \
  --args.host 0.0.0.0 \
  --args.port 8000 \
  --args.task_suite_name libero_spatial \
  --args.num_trials_per_task 5 \
  --args.save_lerobot_rollout true \
  --args.rollout_repo_id ybpy/libero_rollout \
  --args.rollout_output_dir /tmp \
  --args.rollout_overwrite true \
  --args.rollout_penalty_value -1.0
```

## 导出参数说明

- `--args.save_lerobot_rollout`
  - 是否开启 rollout 导出。默认：`false`
- `--args.rollout_repo_id`
  - LeRobot 的 repo id（本地目录名也使用该值）
- `--args.rollout_output_dir`
  - 输出根目录
  - 不传则使用 `HF_LEROBOT_HOME`
- `--args.rollout_overwrite`
  - 设为 `true` 时，写入前删除已有同名数据集目录
- `--args.rollout_robot_type`
  - LeRobot 元数据里的机器人类型。默认：`panda`
- `--args.rollout_fps`
  - LeRobot 元数据里的 fps。默认：`10`
- `--args.rollout_penalty_value`
  - 失败轨迹的 value 标签。默认：`-1.0`

## 备注

- 该导出流程面向 LIBERO 仿真评测 rollout。
- 视频保存（`video_out_path`）与 LeRobot 导出互不影响，可同时开启。
- 若开启导出但环境未安装 `lerobot`，脚本会抛出导入错误。

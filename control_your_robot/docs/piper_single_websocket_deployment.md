# Piper Single WebSocket 部署说明

## 概述

这套部署用于把 `example/deploy/piper_single_on_PI0.py` 改成“远端推理，本地控制”模式：

- 远端服务器加载 OpenPI checkpoint 并提供 WebSocket 推理服务
- 本地机器连接机械臂和相机
- 本地采集观测后通过 WebSocket 发给服务器
- 服务器返回动作块
- 本地再通过 CAN 把动作发送给机械臂

对应脚本：

- 服务端: `scripts/serve_piper_single_pi05star_websocket.py`
- 本地端: `example/deploy/piper_single_on_PI0_websocket.py`

虽然服务端脚本名里还带 `pi05star`，但当前实现已经是通用版：

- 默认按 `pi05` 部署
- 如果本地传了 `--adv-ind`，则按 `PiStar` 请求方式发送

## 部署规则

### 默认模式：`pi05`

默认行为：

- 服务端默认 `--train-config pi05_piper`
- 本地端默认不发送 `adv_ind`

适合普通 `pi05` checkpoint。

### `PiStar` 模式

只要你在本地端传了 `--adv-ind`，请求里就会附带该字段。

常见值：

- `positive`
- `negative`

如果服务端实际加载的是 `PiStar` config，本地却没有传 `--adv-ind`，脚本会直接报错并阻止启动 episode。

## 运行前准备

### 1. 服务器准备

服务器需要：

- 能正常加载 OpenPI checkpoint
- 已安装 `openpi` 相关依赖
- 能访问对应 checkpoint 路径

### 2. 本地机器人端准备

本地机器需要：

- 机械臂已连接到 CAN
- 相机已连接
- `PiperSingle` 可正常初始化
- 能访问 websocket 服务端地址

### 3. 当前单臂输入说明

本地端默认使用：

- `left_arm` 关节状态
- `cam_wrist` 图像

如果没有 `cam_head`，脚本会自动补零图像给模型的 `observation/image`。

## 启动方法

### 1. 启动服务端

最简单的 `pi05` 启动方式：

```bash
python scripts/serve_piper_single_pi05star_websocket.py \
  --checkpoint-dir /path/to/checkpoint
```

等价于：

```bash
python scripts/serve_piper_single_pi05star_websocket.py \
  --checkpoint-dir /path/to/checkpoint \
  --train-config pi05_piper \
  --host 0.0.0.0 \
  --port 8000
```

如果要显式启动 `PiStar` checkpoint：

```bash
python scripts/serve_piper_single_pi05star_websocket.py \
  --checkpoint-dir /path/to/checkpoint \
  --train-config pi05_star_toy_326_infer
```



启动成功后会打印类似信息：

```text
Piper 单臂 websocket 推理服务
checkpoint: /path/to/checkpoint
train_config: pi05_piper
deploy_mode: pi05
listen: ws://0.0.0.0:8000
```

### 2. 启动本地机器人端

默认 `pi05` 用法：

```bash
python example/deploy/piper_single_on_PI0_websocket.py \
  --server-host <server_ip> \
  --server-port 8000 \
  --task-name "Put these toys into the box"
```

python example/deploy/piper_single_on_PI0_websocket.py \
    --server-host 173.0.147.3 \
    --server-port 8000


如果使用 `PiStar`，增加 `--adv-ind`：

```bash
python example/deploy/piper_single_on_PI0_websocket.py \
  --server-host <server_ip> \
  --server-port 8000 \
  --task-name "Put these toys into the box" \
  --adv-ind positive
```

## 常用参数

### 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint-dir` | 必填 | checkpoint 根目录 |
| `--train-config` | `pi05_piper` | OpenPI 配置名 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--default-prompt` | `None` | 请求里没有 prompt 时使用 |

### 本地端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server-host` | `127.0.0.1` | 服务器 IP |
| `--server-port` | `8000` | 服务器端口 |
| `--task-name` | `Put these toys into the box` | 任务名 |
| `--instruction` | `None` | 显式指定 prompt；不传则从任务文件随机采样 |
| `--adv-ind` | `None` | 可选；传入时按 PiStar 方式发送 |
| `--chunk-size` | `10` | 每次只执行动作块前多少步 |
| `--control-freq` | `10.0` | 本地 CAN 控制频率 |
| `--max-step` | `150` | 单个 episode 最大步数 |
| `--num-episode` | `10` | episode 数量 |

## 运行流程

### 本地端流程

1. 初始化 `PiperSingle`
2. 连接 WebSocket 服务端
3. 读取服务端 metadata
4. 根据 metadata 判断是否要求 `adv_ind`
5. 按 Enter 开始 episode
6. 本地采集状态和图像
7. 发送 observation 到服务端
8. 接收 `actions`
9. 本地执行前 `chunk-size` 步动作
10. 循环直到中断或达到 `max-step`

### 控制逻辑

本地仍保留以下逻辑：

- 关节限幅
- 夹爪限幅
- 本地 `10 Hz` 节拍控制
- 按 Enter 中断执行

也就是说，这套部署只把“推理”放到远端，没有把机械臂控制搬到远端。

## 输入输出格式

### 本地发送给服务端

默认发送：

```python
{
    "observation/state": state_7d,
    "observation/image": img_head,
    "observation/wrist_image": img_wrist,
    "prompt": instruction,
}
```

如果传了 `--adv-ind`，则额外发送：

```python
{
    "adv_ind": "positive"
}
```

### 服务端返回

服务端返回：

```python
response["actions"]
```

本地只执行每个 action 的前 7 维：

- 前 6 维：关节
- 第 7 维：夹爪

## 常见问题

### 1. 服务端启动了，但本地连不上

检查：

- `--server-host` 是否写成了正确的服务器 IP
- 端口 `8000` 是否开放
- 服务端是否真的监听在 `0.0.0.0`

可先在服务器上确认日志里有：

```text
listen: ws://0.0.0.0:8000
```

### 2. 本地报错说服务端需要 `adv_ind`

说明你连到的是 `PiStar` 服务端。

解决方法：

```bash
python example/deploy/piper_single_on_PI0_websocket.py \
  --server-host <server_ip> \
  --server-port 8000 \
  --adv-ind positive
```

### 3. 只有 wrist 相机，没有 head 相机

这是当前单臂默认场景，脚本已经兼容。

没有 `cam_head` 时，会自动用零图像补 `observation/image`。

### 4. 动作延迟大

优先检查：

- 网络延迟
- 服务端推理耗时
- `chunk-size` 是否过小
- `control-freq` 是否设置过高

建议先保持默认值：

- `chunk-size=10`
- `control-freq=10`

### 5. checkpoint 能本地跑，但 websocket 跑不起来

检查：

- `--checkpoint-dir` 是否指向 checkpoint 根目录，而不是 `params/` 或 `assets/`
- `--train-config` 是否与 checkpoint 匹配
- 如果是 `PiStar`，是否在客户端传了 `--adv-ind`

## 参考文件

- `example/deploy/piper_single_on_PI0.py`
- `example/deploy/piper_single_on_PI0_websocket.py`
- `scripts/serve_piper_single_pi05star_websocket.py`
- `src/robot/policy/openpi/docs/remote_inference.md`
- `skills/deploy-piper-single-websocket/SKILL.md`

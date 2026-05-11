---
name: deploy-piper-single-websocket
description: 为 Agilex Piper 单臂部署实现或修改基于 WebSocket 的远程推理链路，覆盖普通 `pi05` 和 `pi05star` 两种部署方式。Use when adapting `example/deploy/piper_single_on_PI0.py` so inference runs on a remote GPU/server, the robot-side process only collects observations and sends websocket requests, and the local machine still controls the arm over CAN. Also use when wiring `openpi` websocket server/client, selecting between `pi05` and `pi05star`, defining observation/action schemas including `adv_ind`, or validating server/local startup flow for single-arm OpenPI deployment.
---

# Piper Single Websocket Deploy

复用仓库里现有的单臂本地部署逻辑，不要重新发明一套协议。

优先使用 `openpi` 自带的 websocket 远程推理栈，而不是 `scripts/server.py` / `scripts/client.py` 里的 `socket + pickle` 示例。后者可以参考数据往返思路，但它不是 websocket。

## Workflow

1. 把本地部署脚本当成真源头。

读取 `example/deploy/piper_single_on_PI0.py`，保留它的三件核心事情：
- 本地 `PiperSingle()` 初始化、`robot.get()` 采集、`robot.move()` 经 CAN 控制。
- `input_transform()` 的单臂状态拼接和相机兼容逻辑。
- `output_transform()` 的关节限幅和夹爪限幅。

2. 把“模型对象”替换成 websocket client。

本地端不要实例化 `PI0_SINGLE(...)` 做推理，而是改为创建 `openpi_client.websocket_client_policy.WebsocketClientPolicy`，然后在循环里发送 observation，接收 `{"actions": ...}`。

3. 把“服务器端职责”和“机器人端职责”切开。

服务器端：
- 运行 `src/robot/policy/openpi/scripts/serve_policy.py`
- 加载 checkpoint
- 暴露 websocket 服务
- 只负责推理，不接 CAN，不接机器人 SDK

机器人端：
- 初始化 `PiperSingle`
- 采集 `cam_wrist` 和关节状态
- 构造 observation
- 通过 websocket 请求动作块
- 在本地做限幅、节拍控制和 CAN 下发

4. 在单臂场景里显式处理缺省头相机。

`my_robot/agilex_piper_single_base.py` 默认只有 `cam_wrist`。如果模型或远程服务仍沿用 `PI0_SINGLE` 的双图输入接口，就在本地端用 `np.zeros_like(img_wrist)` 补一个 `observation/image`，并把真实腕部图像放在 `observation/wrist_image`。

5. 在客户端发送前做图像规整。

优先复用 `openpi_client.image_tools`：
- `resize_with_pad(..., 224, 224)`
- `convert_to_uint8(...)`

这样可以减少带宽和时延，也更接近 openpi 训练期输入。

6. 保持动作执行策略和原脚本一致，除非用户明确要求改变。

默认做法：
- 服务器返回动作块后，只执行前一小段 open-loop 动作。
- 延续原脚本的 10 Hz 节拍。
- 仍然只取前 7 个控制量给单臂执行。
- 继续使用原脚本里的限位逻辑。

## Deployment Modes

### `pi05`

用于普通 `pi05` checkpoint / config。

规则：
- 训练配置名不带 `pistar=True`。
- 客户端 observation 不需要 `adv_ind`。
- 服务器启动时只需要普通 config 和 checkpoint 目录。

常见例子：
- `pi05_piper`
- 其他普通 `pi05_*` config

### `pi05star`

用于 `PiStar` 变体，也就是 `pi05 + pistar=True`。

规则：
- 训练配置通常形如 `pi05_star_*`。
- 客户端 observation 必须附带 `adv_ind`。
- `adv_ind` 的典型值是 `positive` 或 `negative`。
- 如果缺少 `adv_ind`，本地或服务端都应该尽早报错，不要静默降级成普通 `pi05`。

仓库里现成例子：
- `pi05_star_toy_326`
- `pi05_star_toy_326_infer`

## Decision Rules

- 用户说“websocket / websoket 远程部署”时，优先看 `src/robot/policy/openpi/docs/remote_inference.md`。
- 用户说“库里面有案例”时，优先指向 `openpi` 的 websocket server/client，而不是 `scripts/server.py`。
- 如果用户要保留 `piper_single_on_PI0.py` 的行为一致性，先复制它的 `input_transform()` / `output_transform()` 再改通信层。
- 如果用户明确说 `pi05`，不要注入 `adv_ind`。
- 如果用户明确说 `pi05star` 或 `PiStar`，强制要求 `adv_ind`，并在服务器端和客户端都保留这个字段。
- 如果用户没有明确说明，但当前脚本或 config 名是 `pi05_star_*`，按 `pi05star` 处理。
- 如果用户没有明确说明，且当前 config 是普通 `pi05_*`，按 `pi05` 处理。
- 如果用户没有要求改变服务端模型，默认沿用当前单臂 checkpoint、任务指令文件和对应的 config 模式。
- 如果用户想最小修改，优先新建一个 websocket 版部署脚本，不直接重写已有本地脚本。

## Implementation Notes

写代码时优先参考这些文件：
- `example/deploy/piper_single_on_PI0.py`
- `my_robot/agilex_piper_single_base.py`
- `src/robot/policy/openpi/docs/remote_inference.md`
- `src/robot/policy/openpi/scripts/serve_policy.py`
- `src/robot/policy/openpi/src/openpi/serving/websocket_policy_server.py`
- `src/robot/policy/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `src/robot/policy/openpi/packages/openpi-client/src/openpi_client/image_tools.py`

如果需要接口细节，读取：
- `references/source-map.md`
- `references/data-contract.md`

## Validation

至少验证下面这些点：
- 本地脚本仍能初始化 `PiperSingle`，并只在本地占用 CAN。
- websocket client 能连上服务端，并收到 metadata。
- 发送一次 observation 后，能收到 `actions` 字段。
- 返回动作块的 shape 和本地执行维度匹配；单臂执行时只消费前 7 维。
- 无 `cam_head` 时，补零图像路径可以正常工作。
- `pi05` 模式下不发送 `adv_ind` 也能工作。
- `pi05star` 模式下发送 `adv_ind` 后能正常返回动作；缺少 `adv_ind` 时应明确失败。
- 机器人端控制频率没有被网络调用拖垮；必要时改成“低频请求动作块 + 高频本地执行 chunk”。

## Common Pitfalls

- 不要把 `scripts/server.py` 当成 websocket 实现，它是原始 TCP 自定义协议。
- 不要把服务端也接入机器人控制；CAN 控制必须留在本地。
- 不要把单臂状态错误扩成 14 维再直接发给 `openpi` websocket 服务；远程观测键要按服务端 policy 期望来。
- 不要忽略 `prompt`；远程服务默认仍依赖文本指令。
- 不要把 `pi05star` 当成普通 `pi05`；少了 `adv_ind` 会直接破坏 PiStar 推理前提。
- 不要直接执行完整动作块；先沿用原部署脚本的短 chunk 策略。

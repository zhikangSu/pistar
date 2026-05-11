# Source Map

Read only the files needed for the current task.

## Core local deployment path

- `example/deploy/piper_single_on_PI0.py`
  - Current single-arm local deployment baseline.
  - Reuse its `input_transform()` and `output_transform()` semantics first.
- `my_robot/agilex_piper_single_base.py`
  - Confirms hardware ownership on the local side.
  - Current setup is `left_arm` on `can1` and `cam_wrist` as the only active camera.

## Preferred websocket remote inference path

- `src/robot/policy/openpi/docs/remote_inference.md`
  - Minimal openpi-supported remote inference workflow.
- `src/robot/policy/openpi/scripts/serve_policy.py`
  - Server entrypoint for loading a checkpoint and exposing a websocket service.
- `src/robot/policy/openpi/src/openpi/serving/websocket_policy_server.py`
  - Exact websocket request/response behavior.
- `src/robot/policy/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py`
  - Synchronous client used by the robot-side process.
- `src/robot/policy/openpi/packages/openpi-client/src/openpi_client/image_tools.py`
  - Resize and `uint8` conversion helpers for client-side preprocessing.

## Existing non-websocket examples

- `scripts/server.py`
- `scripts/client.py`
- `src/robot/utils/base/bisocket.py`

Use these only as references for control-loop structure or request/reply sequencing. Do not copy their transport layer when the user explicitly asks for websocket.

## PI0 single-arm behavior to preserve

- `src/robot/policy/openpi/src/openpi/inference_model.py`
  - `PI0_SINGLE.update_observation_window()` expects:
    - `observation/state`
    - `observation/image`
    - `observation/wrist_image`
    - `prompt`
    - `adv_ind` only when the config has `pistar=True`
  - `PI0_SINGLE.get_action()` returns `["actions"][:, :8]`
  - Existing local deployment then executes only the first 7 values for one arm.

## Mode selection and config examples

- `example/deploy/piper_single_on_PI0.py`
  - Current CLI already exposes `--train-config` and `--adv-ind`.
  - Defaults currently point to a `pi05star` example: `pi05_star_toy_326_infer`.
- `src/robot/policy/openpi/src/openpi/inference_model.py`
  - `_uses_adv_ind()` checks `train_config.model.pistar`.
  - `_validate_adv_ind()` enforces `adv_ind` for PiStar inference.
- `src/robot/policy/openpi/src/openpi/training/config.py`
  - `pi05_piper` is a normal `pi05` single-arm style config.
  - `pi05_star_toy_326` and `pi05_star_toy_326_infer` are PiStar configs.

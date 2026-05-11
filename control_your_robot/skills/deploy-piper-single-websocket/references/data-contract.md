# Data Contract

Use this contract when converting `example/deploy/piper_single_on_PI0.py` into a websocket client.

## Local observation source

From `PiperSingle.get()`:

- robot state:
  - `data[0]["left_arm"]["joint"]` -> 6 joint values
  - `data[0]["left_arm"]["gripper"]` -> 1 gripper value
- sensors:
  - `data[1]["cam_wrist"]["color"]` exists by default
  - `data[1]["cam_head"]["color"]` may be absent in the single-arm setup

## Client-side observation payload

Prefer this payload for remote openpi inference:

```python
{
    "observation/state": state_7d,
    "observation/image": img_head_224_uint8,
    "observation/wrist_image": img_wrist_224_uint8,
    "prompt": instruction,
}
```

Notes:

- `state_7d` is `6 joints + 1 gripper`.
- If no head camera exists, set `observation/image = np.zeros_like(img_wrist_224_uint8)`.
- Resize and pad images to `224 x 224` on the client side.
- Convert images to `uint8` before sending.

Mode-specific rule:

- `pi05`: do not require `adv_ind`.
- `pi05star`: must include `adv_ind`, typically `positive` or `negative`.

Example for `pi05star`:

```python
{
    "observation/state": state_7d,
    "observation/image": img_head_224_uint8,
    "observation/wrist_image": img_wrist_224_uint8,
    "prompt": instruction,
    "adv_ind": "positive",
}
```

## Server response contract

`WebsocketClientPolicy.infer(observation)` returns a dict. For PI0 deployment, expect:

```python
response["actions"]
```

Typical handling:

- treat `response["actions"]` as an action chunk shaped like `[T, action_dim]`
- execute only a short prefix locally
- preserve local CAN timing control

## Action execution contract for Piper single-arm

Preserve the existing local transform:

- only consume the first 7 values for execution
- values `0:6` map to joint targets
- value `6` maps to gripper target
- clamp joints with the limits already defined in `example/deploy/piper_single_on_PI0.py`
- clamp gripper to `[0.0, 1.0]`

Recommended execution pattern:

1. request one chunk over websocket
2. slice a short prefix, such as 10 actions
3. execute locally at 10 Hz
4. allow local stop/interruption logic to remain local

## Responsibility split

Server side:

- load checkpoint
- run policy inference
- return action chunk
- for `pi05star`, reject requests that omit `adv_ind`

Local robot side:

- own camera access
- own CAN bus access
- build observations
- decide whether the payload is `pi05` or `pi05star`
- clamp and rate-limit actions
- send `robot.move(...)`

import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

# Use TestArmController instead of PiperController for testing without hardware
from robot.controller.TestArm_controller import TestArmController
# Use TestVisionSensor instead of RealsenseSensor for testing without cameras
from robot.sensor.TestVision_sensor import TestVisonSensor

from robot.data.collect_any import CollectAny

CAMERA_SERIALS = {
    'head': '338622070768',  # Replace with actual serial number
    'wrist': '338622072453',   # Replace with actual serial number
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,   # Joint 1
    0.85220935,    # Joint 2
    -0.68542569,  # Joint 3
    0.,   # Joint 4
    0.78588684,  # Joint 5
    -0.05256932,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "robot":"test_piper_single",
    "save_path": "./datasets/",
    "task_name": "Test data collection without robot hardware",
    "save_format": "hdf5",
    "save_freq": 10,
}


class TestPiperSingle(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.condition = condition
        # Use TestArmController for mock testing
        self.controllers = {
            "arm":{
                "left_arm": TestArmController("left_arm", DoFs=6, INFO="DEBUG"),
            },
        }
        # Note: Using TestVisonSensor instead of RealsenseSensor for testing
        self.sensors = {
            "image":{
                "cam_head": TestVisonSensor("cam_head", INFO="DEBUG"),
                "cam_wrist": TestVisonSensor("cam_wrist", INFO="DEBUG"),
            },
        }

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self):
        super().set_up()

        # TestArmController doesn't need CAN interface, pass None
        self.controllers["arm"]["left_arm"].set_up(None)

        # TestVisonSensor doesn't need camera serials
        self.sensors["image"]["cam_head"].set_up(None, is_depth=False, encode_rgb=False)
        self.sensors["image"]["cam_wrist"].set_up(None, is_depth=False, encode_rgb=False)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })

        print("set up success!")

if __name__=="__main__":
    import time
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"  # Enable debug output

    robot = TestPiperSingle()
    robot.set_up()
    # collection test
    robot.reset()
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()

    # moving test
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)
    time.sleep(1)
    move_data = {
        "arm":{
            "left_arm":{
            "joint":[0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)

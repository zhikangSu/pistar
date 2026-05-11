[![中文](https://img.shields.io/badge/中文-简体-blue)](./README.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README_EN.md)

[Chinese WIKI](https://tian-nian.github.io/control_your_robot-doc/)

# 控制你的机器人!
该项目旨在与帮助各位进入具身智能领域后能快速上手一整套从控制机械臂开始, 到数据采集, 到最终VLA模型的训练与部署的流程.

## 快速上手!
由于本项目实现了部分测试样例, 如机械臂测试样例, 视觉模拟样例, 完整机器人模拟样例, 因此可以在没有任何实体的情况下快速了解本项目的整体框架.
由于没涉及任何本体, 所以安装环境只需要执行:
```
 pip install -r requirements.txt
```  
本项目有特殊的调试参数, 分为:"DEBUG", "INFO", "ERROR", 如果想要完整看到数据的流程, 可以设置为"DEBUG".
```bash
export INFO_LEVEL="DEBUG"
```
或者可以在对应main函数中引入:
```python
import os
os.environ["INFO_LEVEL"] = "DEBUG" # DEBUG , INFO, ERROR
```
1. 数据采集测试
```bash
# 多进程(通过时间同步器实现更严格的等时间距采集)
python example/collect/collect_mp_robot.py
# 多进程(对每个元件单独进程采集数据)
python example/collect/collect_mp_component.py
# 单线程(会存在一些由于函数执行导致的延迟堆积)
python example/collect/collect.py
```

2. 模型部署测试
```bash
# 跑一个比较直观的部署测试代码
python example/deploy/robot_on_test.py
# 实现的通用部署脚本
bash deploy.sh
# 数据回灌一致性测试
bash eval_offline.sh
```

3. 远程部署数据传输
```bash
# 先启动服务器, 模仿推理端(允许多次连接, 监听端口)
python scripts/server.py
# 本地, 获取数据并执行指令(示例只执行了10次)
python scripts/client.py
```

4. 一些有意思的代码
```python
# 采集对应的关键点, 并且进行轨迹重演
python scripts/collect_moving_ckpt.py 
# sapien仿真, 请参考planner/README.md
```

5. 调试对应的一些代码
```bash
# 由于controller与sensor有__init__.py, 所以需要按照-m形式执行代码
python -m controller.TestArm_controller
python -m sensor.TestVision_sensor
python -m my_robot.test_robot
```

6. 一些数据转化脚本
```bash
# 在执行完python example/collect/collect.py, 路径下有轨迹后
python scripts/convert2rdt_hdf5.py save/test_robot/ save/rdt/
```

### 🤖 设备支持情况

#### 🎛️ 控制器
**✅ 已实现**
| 机械臂         | 底盘               | 灵巧手       | 其他       |
|----------------|--------------------|--------------|------------|
| Agilex Piper   | Agilex Tracer2.0   | 🚧 开发中    | 📦 待补充  |
| RealMan 65B    | 📦 待补充          | 📦 待补充    | 📦 待补充  |
| daran aloha    | 📦 待补充          | 📦 待补充    | 📦 待补充  |

**🚧 准备支持**
| 机械臂    | 底盘       | 灵巧手     | 其他       |
|-----------|------------|------------|------------|
| JAKA      | 📦 待补充  | 📦 待补充  | 📦 待补充  |
| Franka    | 📦 待补充  | 📦 待补充  | 📦 待补充  |
| UR5e      | 📦 待补充  | 📦 待补充  | 📦 待补充  |

#### 📡 传感器
**✅ 已实现**
| 视觉传感器       | 触觉传感器    | 其他传感器  |
|------------------|---------------|-------------|
| RealSense 系列   | Vitac3D  | 📦 待补充   |
| 

**🚧 准备支持**
有需要新的传感器支持请提issue，也欢迎PR你的传感器配置！

## 表格形式
| 目录 | 说明 | 主要内容 |
|------|------|----------|
| **📂 controller** | 机器人控制器封装 | 机械臂、底盘等设备的控制 `class` |
| **📂 sensor** | 传感器封装 | 目前仅 `RealSense` 相机封装 |
| **📂 utils** | 工具函数库 | 辅助功能封装（如数学计算、日志等） |
| **📂 data** | 数据采集模块 | 数据记录、处理的 `class` |
| **📂 my_robot** | 机器人集成封装 | 完整机器人系统的组合 `class` |
| **📂 policy** | VLA 模型策略 | Vision-Language-Action 模型相关代码 |
| **📂 scripts** | 实例化脚本 | 主要运行入口、测试代码 |
| **📂 third_party** | 第三方依赖 | 需要编译的外部库 |
| **📂 planner** | 路径规划模块 | `curobo` 规划器封装 + 仿真机械臂代码 |
| **📂 example** | 示例代码 | 数据采集、模型部署等示例 |
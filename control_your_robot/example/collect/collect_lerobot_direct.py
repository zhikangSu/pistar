"""
直接生成 LeRobot 格式的数据收集脚本
借鉴 collect_mp_robot.py，但不需要后续转换步骤

使用方法:
python example/collect/collect_lerobot_direct.py
"""
import sys
sys.path.append("./")
import time

from multiprocessing import Process, Event, Barrier

from my_robot.piper_single_lerobot import PiperSingleLeRobot

from robot.utils.worker.time_scheduler import TimeScheduler
from robot.utils.worker.robot_worker import RobotWorker
from robot.utils.base.data_handler import is_enter_pressed


def RobotWorkerLeRobot(
    robot_class,
    robot_kwargs: dict,
    time_lock: Barrier,
    start_event: Event,
    finish_event: Event,
    exit_event: Event,
    saved_event: Event,
    process_name: str,
):
    """
    LeRobot 版本的机器人 Worker (支持多 Episode 复用进程)

    Args:
        robot_class: 机器人类（如 PiperSingleLeRobot）
        robot_kwargs: 机器人初始化参数
        time_lock: 时间同步锁
        start_event: 开始事件
        finish_event: 结束事件
        exit_event: 退出进程事件
        saved_event: 保存完成事件
        process_name: 进程名称
    """
    from robot.utils.base.data_handler import debug_print

    # 创建机器人实例
    robot = robot_class(**robot_kwargs)
    robot.set_up()

    debug_print(process_name, "初始化完成，等待主进程指令...", "INFO")

    try:
        while not exit_event.is_set():
            # 1. 等待开始信号
            if not start_event.is_set():
                time.sleep(0.1)
                continue
            
            # 收到开始信号，进入采集循环
            debug_print(process_name, "收到开始信号，开始采集...", "INFO")
            
            while not finish_event.is_set():
                if exit_event.is_set():
                    break
                    
                try:
                    time_lock.wait()
                except Exception as e:
                    debug_print(process_name, f"警告: Barrier 被中止 ({e})", "WARNING")
                    break

                if finish_event.is_set():
                    break

                try:
                    # 获取数据
                    data = robot.get()
                    # 收集数据
                    robot.collect(data)
                except Exception as e:
                    debug_print(process_name, f"错误: {e}", "ERROR")

            # 2. 收到结束信号，保存数据
            debug_print(process_name, "收到结束信号，正在保存...", "INFO")
            robot.finish()
            debug_print(process_name, "✓ 保存成功！", "INFO")
            
            # 通知主进程保存完毕
            saved_event.set()
            
            # 等待主进程重置信号
            while start_event.is_set() and not exit_event.is_set():
                time.sleep(0.01)

    except KeyboardInterrupt:
        debug_print(process_name, "用户中断", "WARNING")
    except Exception as e:
        debug_print(process_name, f"发生未捕获异常: {e}", "ERROR")
        import traceback
        debug_print(process_name, traceback.format_exc(), "ERROR")
    finally:
        debug_print(process_name, "Worker 退出", "INFO")


if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"  # DEBUG, INFO, ERROR

    # ==================== 配置参数 ====================
    REPO_ID = "piper_plug_task"
    OUTPUT_DIR = "/home/chaihoa/project_wang/dataset/lerobot"
    TASK_NAME = "Put these toys into the box"
    FPS = 10
    NUM_EPISODES = 100
    # ================================================

    print("=" * 60)
    print("LeRobot 直接收集模式 (进程复用优化版)")
    print("=" * 60)
    print(f"数据集 ID: {REPO_ID}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"任务名称: {TASK_NAME}")
    print(f"采集频率: {FPS} Hz")
    print(f"计划收集: {NUM_EPISODES} 个 episodes")
    print("=" * 60)

    # 创建多进程同步工具 (只创建一次)
    time_lock = Barrier(1 + 1)  # 1个robot进程 + 1个time_scheduler
    start_event = Event()
    finish_event = Event()
    exit_event = Event()
    saved_event = Event()

    # 机器人参数
    robot_kwargs = {
        "repo_id": REPO_ID,
        "output_dir": OUTPUT_DIR,
        "task_name": TASK_NAME,
        "fps": FPS,
        "move_check": True,
    }

    # 启动机器人进程 (只启动一次)
    robot_process = Process(
        target=RobotWorkerLeRobot,
        args=(
            PiperSingleLeRobot,
            robot_kwargs,
            time_lock,
            start_event,
            finish_event,
            exit_event,
            saved_event,
            "robot_worker_lerobot",
        ),
    )
    robot_process.start()
    
    # 启动时间调度器 (只启动一次)
    time_scheduler = TimeScheduler(work_barrier=time_lock, time_freq=FPS)

    try:
        for episode_id in range(NUM_EPISODES):
            print(f"\n{'='*60}")
            print(f"Episode {episode_id + 1}/{NUM_EPISODES}")
            print(f"{'='*60}")

            # 重置状态
            start_event.clear()
            finish_event.clear()
            saved_event.clear()
            
            print("按 Enter 键开始收集...", end="", flush=True)

            # 等待用户按 Enter 开始
            input("按 Enter 键开始收集...")
            print("开始收集! (按 Enter 结束)")
            start_event.set()

            # 启动计时器
            time_scheduler.start()

            # 等待用户按 Enter 结束
            input()
            finish_event.set()
            # 停止计时器 (暂停)
            time_scheduler.stop()
            # 关键修复：强制打破屏障，唤醒正卡在 wait() 的 worker 进程
            try:
                time_lock.abort()
            except Exception:
                pass

            print("等待数据保存...", end="", flush=True)
            # 等待保存完成
            while not saved_event.is_set():
                time.sleep(0.1)
            print(" 完成!")

            # 重置 Barrier，防止因 scheduler 强制退出导致的 BrokenBarrierError
            try:
                time_lock.reset()
            except Exception:
                pass

            print(f"✓ Episode {episode_id + 1} 完成！")
            print(f"  平均时间间隔: {time_scheduler.real_time_average_time_interval:.4f}s")
            
            # 手动重置一下 start_event，让子进程进入下一个循环等待
            start_event.clear()

        print("\n" + "=" * 60)
        print("✅ 所有 episodes 收集完成！")
        print(f"数据集保存在: {OUTPUT_DIR}/{REPO_ID}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n程序中断，正在退出...")
    finally:
        # 清理
        exit_event.set()
        # 释放可能卡住的锁
        try:
            time_lock.reset()
        except:
            pass
            
        if robot_process.is_alive():
            robot_process.join(timeout=2)
            if robot_process.is_alive():
                robot_process.terminate()
            robot_process.close()
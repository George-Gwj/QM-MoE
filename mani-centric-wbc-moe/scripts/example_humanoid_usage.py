#!/usr/bin/env python3
"""
人型机器人末端执行器任务使用示例

这个脚本展示了如何使用 play_humanoid_ee_task.py 来运行人型机器人的末端执行器任务。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_humanoid_task(ckpt_path, trajectory_path, robot_type="humanoid", 
                     visualize=True, record_video=False, num_steps=1000):
    """
    运行人型机器人末端执行器任务
    
    Args:
        ckpt_path (str): 模型检查点路径
        trajectory_path (str): 轨迹文件路径
        robot_type (str): 机器人类型 ("humanoid" 或 "biped")
        visualize (bool): 是否启用可视化
        record_video (bool): 是否录制视频
        num_steps (int): 运行步数
    """
    
    # 构建命令
    cmd = [
        "python", "scripts/play_humanoid_ee_task.py",
        "--ckpt_path", ckpt_path,
        "--trajectory_file_path", trajectory_path,
        "--robot_type", robot_type,
        "--num_steps", str(num_steps)
    ]
    
    if visualize:
        cmd.append("--visualize")
    
    if record_video:
        cmd.append("--record_video")
    
    print(f"运行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("任务执行成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"任务执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_dependencies():
    """检查依赖项"""
    required_files = [
        "scripts/play_humanoid_ee_task.py",
        "config/env/env_humanoid.yaml",
        "config/env/combo_humanoid_reaching.yaml",
        "config/env/tasks/humanoid_reaching.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("缺少以下文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def create_sample_trajectory(output_path):
    """创建示例轨迹文件"""
    import pickle
    import numpy as np
    
    # 创建简单的轨迹数据
    trajectory_data = {
        "positions": np.random.randn(100, 3) * 0.1,  # 随机位置
        "orientations": np.random.randn(100, 4),      # 随机姿态
        "times": np.linspace(0, 10, 100)             # 时间序列
    }
    
    # 保存轨迹文件
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    print(f"示例轨迹文件已创建: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="人型机器人末端执行器任务示例")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--trajectory_path", type=str, 
                       default="data/sample_humanoid_trajectory.pkl",
                       help="轨迹文件路径")
    parser.add_argument("--robot_type", type=str, default="humanoid",
                       choices=["humanoid", "biped"],
                       help="机器人类型")
    parser.add_argument("--visualize", action="store_true",
                       help="启用可视化")
    parser.add_argument("--record_video", action="store_true",
                       help="录制视频")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="运行步数")
    parser.add_argument("--create_sample", action="store_true",
                       help="创建示例轨迹文件")
    
    args = parser.parse_args()
    
    # 检查依赖项
    if not check_dependencies():
        print("请确保所有必需的文件都存在")
        sys.exit(1)
    
    # 创建示例轨迹文件
    if args.create_sample:
        os.makedirs(os.path.dirname(args.trajectory_path), exist_ok=True)
        create_sample_trajectory(args.trajectory_path)
    
    # 检查文件是否存在
    if not os.path.exists(args.ckpt_path):
        print(f"错误: 检查点文件不存在: {args.ckpt_path}")
        sys.exit(1)
    
    if not os.path.exists(args.trajectory_path):
        print(f"错误: 轨迹文件不存在: {args.trajectory_path}")
        print("使用 --create_sample 参数创建示例轨迹文件")
        sys.exit(1)
    
    # 运行任务
    success = run_humanoid_task(
        ckpt_path=args.ckpt_path,
        trajectory_path=args.trajectory_path,
        robot_type=args.robot_type,
        visualize=args.visualize,
        record_video=args.record_video,
        num_steps=args.num_steps
    )
    
    if success:
        print("\n任务完成! 检查输出文件:")
        print("- video.mp4 (如果启用了录制)")
        print("- logs.zarr (状态和动作数据)")
        print("- logs.pkl (配置和日志)")
        print("- exported/ (导出的模型)")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 
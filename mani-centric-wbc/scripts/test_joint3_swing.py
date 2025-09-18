#!/usr/bin/env python3
"""
测试 joint3 摆动功能的脚本
"""

import torch
import numpy as np

def test_joint3_identification():
    """测试 joint3 识别功能"""
    print("=== 测试 joint3 识别功能 ===")
    
    # 模拟 dof_names
    dof_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "link1", "link2", "joint3", "link4", "link5", "link6"
    ]
    
    # 测试 joint3 识别
    joint3_idx = -1
    for idx, name in enumerate(dof_names):
        if name == "joint3":
            joint3_idx = idx
            break
    
    print(f"找到 joint3 在索引: {joint3_idx}")
    print(f"关节名称: {dof_names[joint3_idx] if joint3_idx >= 0 else '未找到'}")
    
    return joint3_idx

def test_curriculum_stages():
    """测试课程学习阶段"""
    print("\n=== 测试课程学习阶段 ===")
    
    curriculum_configs = [
        {"stage": 1, "amplitude": 0.0, "frequency": 0.0, "description": "基础训练"},
        {"stage": 2, "amplitude": 0.2, "frequency": 0.5, "description": "小幅度摆动"},
        {"stage": 3, "amplitude": 0.5, "frequency": 0.5, "description": "中等幅度摆动"},
        {"stage": 4, "amplitude": 1.0, "frequency": 0.5, "description": "大幅度摆动"},
        {"stage": 5, "amplitude": 1.0, "frequency": 1.0, "description": "中等频率摆动"},
        {"stage": 6, "amplitude": 2.0, "frequency": 1.5, "description": "最大挑战"}
    ]
    
    for config in curriculum_configs:
        print(f"阶段 {config['stage']}: {config['description']}")
        print(f"  幅度: ±{config['amplitude']}rad, 频率: {config['frequency']}Hz")
    
    return curriculum_configs

def test_swing_trajectory():
    """测试摆动轨迹生成"""
    print("\n=== 测试摆动轨迹生成 ===")
    
    # 模拟参数
    init_pos = 0.0
    amplitude = 1.0
    frequency = 1.0
    dt = 0.01
    duration = 2.0  # 2秒
    
    # 生成时间序列
    time_steps = int(duration / dt)
    time = torch.linspace(0, duration, time_steps)
    
    # 生成摆动轨迹
    trajectory = init_pos + amplitude * torch.sin(2 * torch.pi * frequency * time)
    
    print(f"初始位置: {init_pos}")
    print(f"摆动幅度: ±{amplitude}rad")
    print(f"摆动频率: {frequency}Hz")
    print(f"时间步长: {dt}s")
    print(f"总时长: {duration}s")
    print(f"轨迹点数: {len(trajectory)}")
    
    # 显示前几个点
    print("前10个轨迹点:")
    for i in range(min(10, len(trajectory))):
        print(f"  t={time[i]:.3f}s: {trajectory[i]:.3f}rad")
    
    # 计算统计信息
    max_pos = torch.max(trajectory).item()
    min_pos = torch.min(trajectory).item()
    mean_pos = torch.mean(trajectory).item()
    
    print(f"\n轨迹统计:")
    print(f"  最大值: {max_pos:.3f}rad")
    print(f"  最小值: {min_pos:.3f}rad")
    print(f"  平均值: {mean_pos:.3f}rad")
    print(f"  实际幅度: ±{max_pos:.3f}rad")
    
    return trajectory

def test_pd_control():
    """测试PD控制器"""
    print("\n=== 测试PD控制器 ===")
    
    # 模拟参数
    kp = 100.0
    kd = 1.0
    
    # 模拟目标位置和当前状态
    target_pos = 0.5
    current_pos = 0.0
    current_vel = 0.0
    
    # 计算控制输出
    pos_error = target_pos - current_pos
    vel_error = -current_vel  # 目标速度为0
    
    control_output = kp * pos_error + kd * vel_error
    
    print(f"PD控制参数:")
    print(f"  kp (位置增益): {kp}")
    print(f"  kd (速度增益): {kd}")
    print(f"\n控制计算:")
    print(f"  目标位置: {target_pos}rad")
    print(f"  当前位置: {current_pos}rad")
    print(f"  当前速度: {current_vel}rad/s")
    print(f"  位置误差: {pos_error}rad")
    print(f"  速度误差: {vel_error}rad/s")
    print(f"  控制输出: {control_output}")
    
    return control_output

def main():
    """主函数"""
    print("Joint3 摆动功能测试")
    print("=" * 50)
    
    # 运行各项测试
    joint3_idx = test_joint3_identification()
    curriculum_configs = test_curriculum_stages()
    trajectory = test_swing_trajectory()
    control_output = test_pd_control()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    if joint3_idx >= 0:
        print(f"✓ joint3 识别成功，索引: {joint3_idx}")
    else:
        print("✗ joint3 识别失败")
    
    print(f"✓ 课程学习配置: {len(curriculum_configs)} 个阶段")
    print(f"✓ 摆动轨迹生成: {len(trajectory)} 个点")
    print(f"✓ PD控制器测试: 输出 {control_output:.3f}")

if __name__ == "__main__":
    main() 
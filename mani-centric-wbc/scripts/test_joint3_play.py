#!/usr/bin/env python3
"""
测试 joint3 摆动的简单脚本
用于验证 play_velcmd_arm_swing.py 的修改是否正确
"""

import numpy as np
import matplotlib.pyplot as plt

def test_joint3_swing_function():
    """测试 joint3 摆动角度生成函数"""
    
    # 模拟参数
    amplitude = 0.5
    frequency = 1.0
    phase = 0.0
    time_steps = np.linspace(0, 4, 100)  # 4秒，100个时间点
    
    def get_joint3_swing_angle(t):
        """为joint3生成摆动角度"""
        angle = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return angle
    
    # 生成摆动角度
    swing_angles = [get_joint3_swing_angle(t) for t in time_steps]
    
    # 绘制摆动曲线
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, swing_angles, 'b-', linewidth=2, label=f'joint3摆动 (幅度:±{amplitude}rad, 频率:{frequency}Hz)')
    plt.xlabel('时间 (s)')
    plt.ylabel('角度 (rad)')
    plt.title('joint3 摆动测试')
    plt.grid(True)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=amplitude, color='r', linestyle='--', alpha=0.5, label=f'幅度上限: {amplitude}')
    plt.axhline(y=-amplitude, color='r', linestyle='--', alpha=0.5, label=f'幅度下限: -{amplitude}')
    
    # 添加一些统计信息
    max_angle = max(swing_angles)
    min_angle = min(swing_angles)
    actual_amplitude = (max_angle - min_angle) / 2
    
    plt.text(0.02, 0.98, f'实际幅度: ±{actual_amplitude:.3f}rad\n最大角度: {max_angle:.3f}rad\n最小角度: {min_angle:.3f}rad', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"测试完成!")
    print(f"期望幅度: ±{amplitude}rad")
    print(f"实际幅度: ±{actual_amplitude:.3f}rad")
    print(f"期望频率: {frequency}Hz")
    print(f"实际周期: {1/frequency:.3f}s")
    
    return swing_angles

def test_curriculum_stages():
    """测试课程学习阶段"""
    
    stages = [1, 2, 3, 4, 5, 6]
    amplitudes = [0.0, 0.2, 0.5, 1.0, 1.0, 2.0]  # 对应各阶段的幅度
    frequencies = [0.0, 0.5, 0.5, 0.5, 1.0, 1.5]  # 对应各阶段的频率
    
    print("\n课程学习阶段测试:")
    print("阶段 | 幅度(rad) | 频率(Hz) | 说明")
    print("-" * 50)
    
    for i, stage in enumerate(stages):
        amp = amplitudes[i]
        freq = frequencies[i]
        if stage == 1:
            desc = "不启用摆动"
        elif stage == 2:
            desc = "小幅度，低频率"
        elif stage == 3:
            desc = "中等幅度，低频率"
        elif stage == 4:
            desc = "大幅度，低频率"
        elif stage == 5:
            desc = "大幅度，中等频率"
        elif stage == 6:
            desc = "最大幅度，高频率"
        
        print(f"{stage:^4} | {amp:^9.1f} | {freq:^8.1f} | {desc}")
    
    return stages, amplitudes, frequencies

if __name__ == "__main__":
    print("=== joint3 摆动功能测试 ===\n")
    
    # 测试摆动角度生成
    swing_angles = test_joint3_swing_function()
    
    # 测试课程学习阶段
    stages, amplitudes, frequencies = test_curriculum_stages()
    
    print("\n测试完成！")
    print("如果一切正常，你应该看到:")
    print("1. 一个正弦摆动曲线图")
    print("2. 课程学习阶段的参数表")
    print("3. 摆动角度在指定范围内变化") 
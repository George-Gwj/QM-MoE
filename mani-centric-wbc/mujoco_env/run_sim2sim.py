#!/usr/bin/env python3
"""
Sim2Sim转移脚本：将IsaacGym训练的策略转移到MuJoCo环境
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mujoco_env.env_velcmd_arm_swing_mujoco import create_mujoco_env

def main():
    parser = argparse.ArgumentParser(description='Sim2Sim转移：IsaacGym到MuJoCo')
    parser.add_argument('--model_path', type=str, required=True,
                       help='MuJoCo模型文件路径(.xml)')
    parser.add_argument('--policy_path', type=str, required=True,
                       help='训练好的策略文件路径(.pt)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='原始训练配置文件路径(.pkl或.yaml)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备 (cpu/cuda)')
    parser.add_argument('--render', action='store_true',
                       help='是否渲染环境')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='仿真步数')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='仿真时间步长')
    parser.add_argument('--vel_x', type=float, default=0.5,
                       help='X方向速度指令')
    parser.add_argument('--vel_y', type=float, default=0.0,
                       help='Y方向速度指令')
    parser.add_argument('--vel_z', type=float, default=0.0,
                       help='Z方向速度指令')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误：MuJoCo模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.policy_path):
        print(f"错误：策略文件不存在: {args.policy_path}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"错误：配置文件不存在: {args.config_path}")
        return
    
    print("=" * 60)
    print("Sim2Sim转移：IsaacGym到MuJoCo")
    print("=" * 60)
    print(f"MuJoCo模型: {args.model_path}")
    print(f"策略文件: {args.policy_path}")
    print(f"配置文件: {args.config_path}")
    print(f"设备: {args.device}")
    print(f"渲染: {args.render}")
    print(f"仿真步数: {args.num_steps}")
    print(f"时间步长: {args.dt}")
    print(f"速度指令: vx={args.vel_x}, vy={args.vel_y}, vz={args.vel_z}")
    print("=" * 60)
    
    try:
        # 创建MuJoCo环境
        print("正在创建MuJoCo环境...")
        env = create_mujoco_env(
            model_path=args.model_path,
            policy_path=args.policy_path,
            config_path=args.config_path,
            device=args.device,
            render=args.render,
            dt=args.dt
        )
        
        print("环境创建成功！")
        print(f"观测维度: {env.obs_dim}")
        print(f"动作维度: {env.action_dim}")
        print(f"机械臂关节数: {env.num_arm_dofs}")
        
        # 设置速度指令
        env.base_lin_vel_cmd = np.array([args.vel_x, args.vel_y, args.vel_z])
        print(f"设置速度指令: {env.base_lin_vel_cmd}")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print(f"初始观测维度: {obs.shape}")
        
        # 运行仿真
        print(f"开始运行仿真，共{args.num_steps}步...")
        
        total_reward = 0.0
        start_time = time.time()
        
        for step in range(args.num_steps):
            # 使用策略生成动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
                action_tensor = env.policy(obs_tensor)
                actions = action_tensor.cpu().numpy()[0]
            
            # 执行一步
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            
            # 打印进度
            if step % 100 == 0:
                print(f"步数: {step}/{args.num_steps}, 奖励: {reward:.4f}, 课程阶段: {env.curriculum_stage}")
                
                # 打印一些关键观测值
                if step == 0:
                    print(f"观测向量前10个值: {obs[:10]}")
                    print(f"动作向量前5个值: {actions[:5]}")
            
            # 渲染
            if args.render:
                env.render()
                time.sleep(args.dt)  # 控制渲染速度
            
            # 检查是否结束
            if done:
                print(f"环境在第{step}步结束")
                break
        
        # 打印结果
        end_time = time.time()
        simulation_time = end_time - start_time
        real_time_factor = (args.num_steps * args.dt) / simulation_time
        
        print("\n" + "=" * 60)
        print("仿真完成！")
        print("=" * 60)
        print(f"总步数: {args.num_steps}")
        print(f"总奖励: {total_reward:.4f}")
        print(f"平均奖励: {total_reward/args.num_steps:.4f}")
        print(f"仿真时间: {simulation_time:.2f}秒")
        print(f"实时因子: {real_time_factor:.2f}x")
        print(f"最终课程阶段: {env.curriculum_stage}")
        print("=" * 60)
        
        # 保存结果
        results = {
            'total_steps': args.num_steps,
            'total_reward': total_reward,
            'avg_reward': total_reward/args.num_steps,
            'simulation_time': simulation_time,
            'real_time_factor': real_time_factor,
            'final_curriculum_stage': env.curriculum_stage,
            'velocity_command': env.base_lin_vel_cmd.tolist()
        }
        
        output_dir = Path(args.policy_path).parent / 'sim2sim_results'
        output_dir.mkdir(exist_ok=True)
        
        import json
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"结果已保存到: {output_dir / 'results.json'}")
        
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
简化的MoE模型播放脚本
支持加载和运行混合专家(MoE)模型进行推理
"""

import os
import sys
import pickle
import argparse
from isaacgym import gymapi, gymutil

# 确保使用正确的legged_gym模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入路径设置
from setup_path import setup_correct_path
setup_correct_path()

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from legged_gym.env.isaacgym.env_add_baseinfo import IsaacGymEnv
from train import setup
import wandb


def recursively_replace_device(obj, device: str):
    """递归替换配置中的设备设置"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "device":
                obj[k] = device
            else:
                obj[k] = recursively_replace_device(v, device)
        return obj
    elif isinstance(obj, list):
        return [recursively_replace_device(v, device) for v in obj]
    else:
        return obj


def load_moe_model(ckpt_path, env, device):
    """加载MoE模型"""
    print(f"Loading MoE model from: {ckpt_path}")
    
    # 加载配置
    config = OmegaConf.create(
        pickle.load(
            open(os.path.join(os.path.dirname(ckpt_path), "config.pkl"), "rb")
        )
    )
    
    # 替换设备设置
    config = recursively_replace_device(
        OmegaConf.to_container(config, resolve=True),
        device=device,
    )
    
    # 创建runner并加载模型
    runner: OnPolicyRunner = hydra.utils.instantiate(
        config["runner"], env=env, eval_fn=None
    )
    runner.load(ckpt_path)
    
    return runner.alg.actor_critic, config


def play_moe():
    parser = argparse.ArgumentParser(description="Play with MoE model")
    parser.add_argument("--moe_ckpt_path", type=str, required=True, 
                       help="MoE模型权重路径")
    parser.add_argument("--trajectory_file_path", type=str, required=True,
                       help="轨迹文件路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="计算设备")
    parser.add_argument("--num_envs", type=int, default=1,
                       help="环境数量")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="运行步数")
    parser.add_argument("--visualize", action="store_true",
                       help="是否可视化")
    parser.add_argument("--show_weights", action="store_true",
                       help="显示MoE权重分布")
    parser.add_argument("--record_video", action="store_true",
                       help="录制视频")
    
    args = parser.parse_args()
    
    if args.visualize:
        args.num_envs = 1

    # 加载MoE模型配置
    print("Loading configuration...")
    config = OmegaConf.create(
        pickle.load(
            open(os.path.join(os.path.dirname(args.moe_ckpt_path), "config.pkl"), "rb")
        )
    )
    
    # 设置环境参数
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
    
    config = recursively_replace_device(
        OmegaConf.to_container(config, resolve=True),
        device=args.device,
    )
    config["_convert_"] = "all"
    config["wandb"]["mode"] = "offline"
    config["env"]["headless"] = not args.visualize
    config["env"]["graphics_device_id"] = int(args.device.split("cuda:")[-1]) if "cuda" in args.device else 0
    config["env"]["attach_camera"] = args.visualize
    config["env"]["sim_device"] = args.device
    config["env"]["dof_pos_reset_range_scale"] = 0
    config["env"]["controller"]["num_envs"] = args.num_envs
    config["env"]["cfg"]["env"]["num_envs"] = args.num_envs
    config["env"]["cfg"]["domain_rand"]["push_robots"] = False
    config["env"]["cfg"]["domain_rand"]["transport_robots"] = False

    # 设置环境为平面模式
    config["env"]["cfg"]["terrain"]["mode"] = "plane"
    config["env"]["cfg"]["init_state"]["pos_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["euler_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["lin_vel_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["ang_vel_noise"] = [0.0, 0.0, 0.0]
    config["env"]["tasks"]["reaching"]["sequence_sampler"]["file_path"] = args.trajectory_file_path
    config["env"]["constraints"] = {}

    setup(config, seed=config["seed"])

    # 创建环境
    print("Creating environment...")
    env: IsaacGymEnv = hydra.utils.instantiate(
        config["env"],
        sim_params=sim_params,
    )
    config["runner"]["ckpt_dir"] = wandb.run.dir
    
    # 加载MoE模型
    moe_policy, config = load_moe_model(args.moe_ckpt_path, env, args.device)
    
    print("Model loaded successfully!")
    print(f"Model type: {type(moe_policy).__name__}")
    
    # 重置环境
    obs, privileged_obs = env.reset()
    
    if args.visualize:
        env.render()

    # 权重统计
    weight_stats = {
        "base_weights": [],
        "arm_weights": [],
        "weight_entropy": []
    }
    
    # 动作和状态日志
    action_logs = {"actions": [], "moe_weights": []}
    state_logs = {"root_pos": [], "root_quat": [], "dof_pos": []}

    print(f"Starting simulation for {args.num_steps} steps...")
    
    with torch.inference_mode():
        for step_idx in range(args.num_steps):
            # 使用MoE策略进行推理
            actions = moe_policy.act_inference(obs)
            
            # 记录权重信息
            if args.show_weights:
                moe_weights = moe_policy.get_moe_weights(obs)
                base_weight = moe_weights[0, 0].item()
                arm_weight = moe_weights[0, 1].item()
                weight_entropy = -torch.sum(moe_weights * torch.log(moe_weights + 1e-8), dim=-1).mean().item()
                
                if step_idx % 50 == 0:  # 每50步打印一次
                    print(f"Step {step_idx:4d}: Base={base_weight:.3f}, Arm={arm_weight:.3f}, Entropy={weight_entropy:.3f}")
                
                weight_stats["base_weights"].append(base_weight)
                weight_stats["arm_weights"].append(arm_weight)
                weight_stats["weight_entropy"].append(weight_entropy)
                
                action_logs["moe_weights"].append(moe_weights.cpu().numpy())
            
            # 记录动作和状态
            action_logs["actions"].append(actions.cpu().numpy())
            state_logs["root_pos"].append(env.state.root_pos[0].cpu().numpy())
            state_logs["root_quat"].append(env.state.root_xyzw_quat[0].cpu().numpy())
            state_logs["dof_pos"].append(env.state.dof_pos[0].cpu().numpy())
            
            # 执行动作
            obs, privileged_obs, rews, dones, infos = env.step(actions)
            
            # 渲染
            if args.visualize:
                env.render()
            
            # 如果环境结束，重置
            if dones[0]:
                print(f"Episode ended at step {step_idx}")
                obs, privileged_obs = env.reset()
    
    # 保存结果
    print("Saving results...")
    results = {
        "config": config,
        "weight_stats": weight_stats,
        "action_logs": action_logs,
        "state_logs": state_logs,
    }
    
    # 保存为pickle文件
    results_path = os.path.join(wandb.run.dir, "moe_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 打印统计信息
    if args.show_weights:
        print("\n=== MoE Weight Statistics ===")
        print(f"Average base weight: {np.mean(weight_stats['base_weights']):.3f} ± {np.std(weight_stats['base_weights']):.3f}")
        print(f"Average arm weight: {np.mean(weight_stats['arm_weights']):.3f} ± {np.std(weight_stats['arm_weights']):.3f}")
        print(f"Average weight entropy: {np.mean(weight_stats['weight_entropy']):.3f} ± {np.std(weight_stats['weight_entropy']):.3f}")
        
        # 分析权重分布
        base_weights = np.array(weight_stats['base_weights'])
        arm_weights = np.array(weight_stats['arm_weights'])
        
        print(f"\nWeight Distribution Analysis:")
        print(f"Base weight > 0.8: {np.sum(base_weights > 0.8)} steps ({np.sum(base_weights > 0.8)/len(base_weights)*100:.1f}%)")
        print(f"Arm weight > 0.8: {np.sum(arm_weights > 0.8)} steps ({np.sum(arm_weights > 0.8)/len(arm_weights)*100:.1f}%)")
        print(f"Balanced (0.3 < weight < 0.7): {np.sum((base_weights > 0.3) & (base_weights < 0.7))} steps ({np.sum((base_weights > 0.3) & (base_weights < 0.7))/len(base_weights)*100:.1f}%)")
    
    print(f"Results saved to: {results_path}")
    print("Simulation completed!")


if __name__ == "__main__":
    play_moe()
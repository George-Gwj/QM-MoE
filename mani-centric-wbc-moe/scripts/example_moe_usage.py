#!/usr/bin/env python3
"""
MoE模型使用示例脚本
展示如何使用MoE模型进行推理和权重分析
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from legged_gym.rsl_rl.modules.actor_critic_moe import ActorCriticMoE
from legged_gym.rsl_rl.modules.obs_moe import ObsMoE
from legged_gym.rsl_rl.modules.actor_critic import ActorCritic


def create_dummy_models(obs_dim=109, action_dim=18, critic_obs_dim=241):
    """创建虚拟的专家模型用于测试"""
    
    # 创建基础专家模型
    base_actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, action_dim)
    )
    
    base_critic = torch.nn.Sequential(
        torch.nn.Linear(critic_obs_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    base_actor_critic = ActorCritic(
        actor=base_actor,
        critic=base_critic,
        num_actions=action_dim
    )
    
    # 创建手臂专家模型（结构相同）
    arm_actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, action_dim)
    )
    
    arm_critic = torch.nn.Sequential(
        torch.nn.Linear(critic_obs_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    arm_actor_critic = ActorCritic(
        actor=arm_actor,
        critic=arm_critic,
        num_actions=action_dim
    )
    
    # 创建MoE权重网络
    obs_moe = ObsMoE(
        obs_dim=obs_dim,
        hidden_dims=[128, 64],
        num_experts=2
    )
    
    return base_actor_critic, arm_actor_critic, obs_moe


def demonstrate_moe_inference():
    """演示MoE模型推理过程"""
    print("=== MoE模型推理演示 ===")
    
    # 创建模型
    base_ac, arm_ac, obs_moe = create_dummy_models()
    
    # 创建MoE模型
    moe_model = ActorCriticMoE(
        base_actor_critic=base_ac,
        arm_actor_critic=arm_ac,
        obs_moe=obs_moe,
        num_actions=18,
        freeze_experts=True
    )
    
    # 创建测试数据
    batch_size = 4
    obs_dim = 109
    obs = torch.randn(batch_size, obs_dim)
    
    print(f"输入观测形状: {obs.shape}")
    
    # 获取MoE权重
    weights = moe_model.get_moe_weights(obs)
    print(f"MoE权重形状: {weights.shape}")
    print(f"权重值:\n{weights}")
    
    # 进行推理
    actions = moe_model.act_inference(obs)
    print(f"输出动作形状: {actions.shape}")
    print(f"动作值:\n{actions}")
    
    # 分析权重分布
    base_weights = weights[:, 0]
    arm_weights = weights[:, 1]
    
    print(f"\n权重分析:")
    print(f"Base专家平均权重: {base_weights.mean():.3f}")
    print(f"Arm专家平均权重: {arm_weights.mean():.3f}")
    print(f"权重和: {(base_weights + arm_weights).mean():.3f}")
    
    return moe_model, obs, weights, actions


def analyze_weight_distribution(weights, num_samples=1000):
    """分析权重分布"""
    print("\n=== 权重分布分析 ===")
    
    # 生成更多样本进行统计分析
    obs_dim = 109
    obs_samples = torch.randn(num_samples, obs_dim)
    
    # 这里需要实际的MoE模型，我们使用随机权重模拟
    base_weights = torch.softmax(torch.randn(num_samples, 2), dim=-1)[:, 0]
    arm_weights = 1 - base_weights
    
    print(f"样本数量: {num_samples}")
    print(f"Base权重统计:")
    print(f"  均值: {base_weights.mean():.3f}")
    print(f"  标准差: {base_weights.std():.3f}")
    print(f"  最小值: {base_weights.min():.3f}")
    print(f"  最大值: {base_weights.max():.3f}")
    
    print(f"Arm权重统计:")
    print(f"  均值: {arm_weights.mean():.3f}")
    print(f"  标准差: {arm_weights.std():.3f}")
    print(f"  最小值: {arm_weights.min():.3f}")
    print(f"  最大值: {arm_weights.max():.3f}")
    
    # 分析权重偏好
    base_dominant = (base_weights > 0.7).sum().item()
    arm_dominant = (arm_weights > 0.7).sum().item()
    balanced = ((base_weights > 0.3) & (base_weights < 0.7)).sum().item()
    
    print(f"\n权重偏好分析:")
    print(f"Base专家主导 (>0.7): {base_dominant} ({base_dominant/num_samples*100:.1f}%)")
    print(f"Arm专家主导 (>0.7): {arm_dominant} ({arm_dominant/num_samples*100:.1f}%)")
    print(f"平衡使用 (0.3-0.7): {balanced} ({balanced/num_samples*100:.1f}%)")
    
    return base_weights, arm_weights


def plot_weight_analysis(base_weights, arm_weights):
    """绘制权重分析图表"""
    print("\n=== 生成权重分析图表 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 权重分布直方图
    axes[0, 0].hist(base_weights.numpy(), bins=50, alpha=0.7, label='Base', color='blue')
    axes[0, 0].hist(arm_weights.numpy(), bins=50, alpha=0.7, label='Arm', color='red')
    axes[0, 0].set_xlabel('权重值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('权重分布直方图')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 权重散点图
    axes[0, 1].scatter(base_weights.numpy(), arm_weights.numpy(), alpha=0.6)
    axes[0, 1].plot([0, 1], [1, 0], 'r--', alpha=0.5)  # 对角线
    axes[0, 1].set_xlabel('Base权重')
    axes[0, 1].set_ylabel('Arm权重')
    axes[0, 1].set_title('权重散点图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 权重时间序列（前100个样本）
    sample_size = min(100, len(base_weights))
    axes[1, 0].plot(range(sample_size), base_weights[:sample_size].numpy(), label='Base', alpha=0.7)
    axes[1, 0].plot(range(sample_size), arm_weights[:sample_size].numpy(), label='Arm', alpha=0.7)
    axes[1, 0].set_xlabel('样本索引')
    axes[1, 0].set_ylabel('权重值')
    axes[1, 0].set_title('权重时间序列')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 权重箱线图
    weight_data = [base_weights.numpy(), arm_weights.numpy()]
    axes[1, 1].boxplot(weight_data, labels=['Base', 'Arm'])
    axes[1, 1].set_ylabel('权重值')
    axes[1, 1].set_title('权重箱线图')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moe_weight_analysis.png', dpi=300, bbox_inches='tight')
    print("权重分析图表已保存为: moe_weight_analysis.png")
    
    return fig


def demonstrate_training_mode():
    """演示训练模式下的参数管理"""
    print("\n=== 训练模式参数管理 ===")
    
    base_ac, arm_ac, obs_moe = create_dummy_models()
    
    moe_model = ActorCriticMoE(
        base_actor_critic=base_ac,
        arm_actor_critic=arm_ac,
        obs_moe=obs_moe,
        num_actions=18,
        freeze_experts=True
    )
    
    # 检查可训练参数
    trainable_params = list(moe_model.parameters())
    print(f"可训练参数数量: {len(trainable_params)}")
    
    # 检查参数梯度状态
    for name, param in moe_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    # 模拟训练步骤
    obs = torch.randn(2, 109)
    actions = moe_model.act_inference(obs)
    
    # 计算损失（示例）
    target_actions = torch.randn_like(actions)
    loss = torch.nn.functional.mse_loss(actions, target_actions)
    print(f"示例损失: {loss.item():.4f}")
    
    # 反向传播（只有MoE权重网络会更新）
    loss.backward()
    
    print("训练模式演示完成")


def main():
    """主函数"""
    print("MoE模型使用示例")
    print("=" * 50)
    
    # 1. 演示推理过程
    moe_model, obs, weights, actions = demonstrate_moe_inference()
    
    # 2. 分析权重分布
    base_weights, arm_weights = analyze_weight_distribution(weights)
    
    # 3. 绘制分析图表
    try:
        plot_weight_analysis(base_weights, arm_weights)
    except ImportError:
        print("matplotlib未安装，跳过图表生成")
    
    # 4. 演示训练模式
    demonstrate_training_mode()
    
    print("\n示例完成！")
    print("使用说明:")
    print("1. 使用play_moe_simple.py进行简单推理")
    print("2. 使用play_moe_ee_task.py进行完整任务播放")
    print("3. 查看README_moe_play.md了解详细用法")


if __name__ == "__main__":
    main()
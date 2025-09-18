# 混合专家(MoE) Actor-Critic 使用指南

## 概述

本项目实现了基于混合专家(Mixture of Experts, MoE)的Actor-Critic模型，用于结合多个预训练的专家模型。该实现包含以下核心组件：

1. **ObsMoE**: 观测量编码器，将观测量通过MLP编码后输出专家权重
2. **ActorCriticMoE**: 混合专家Actor-Critic模型，结合多个预训练专家
3. **PPOMoE**: 专门用于训练MoE模型的PPO算法

## 核心特性

- **专家权重冻结**: 预训练的专家模型权重被冻结，只训练MoE权重网络
- **动态权重混合**: 根据当前观测量动态计算专家权重
- **梯度正确回传**: 确保只有MoE权重网络参与梯度更新
- **权重正则化**: 包含MoE权重分布的正则化项

## 文件结构

```
legged_gym/rsl_rl/
├── modules/
│   ├── obs_moe.py              # 观测量MoE编码器
│   ├── actor_critic_moe.py     # 混合专家Actor-Critic
│   └── __init__.py             # 模块导入
├── algorithms/
│   ├── ppo_moe.py              # MoE专用PPO算法
│   └── __init__.py             # 算法导入
└── config/
    ├── runner/alg/
    │   ├── ppo_moe.yaml        # PPO MoE配置
    │   └── actor_critic/
    │       ├── actor/moe.yaml  # MoE Actor配置
    │       └── critic/moe.yaml # MoE Critic配置
    └── train_moe_example.yaml  # 完整训练配置示例
```

## 使用方法

### 1. 准备预训练专家模型

首先需要训练两个专家模型（例如base专家和arm专家）：

```bash
# 训练基础专家模型
python scripts/train.py env.sim_device=cuda:0 env.graphics_device_id=0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl

python scripts/train.py --config-name=train env=combo_go2Piper_locomotion

# 训练手臂专家模型  
python scripts/train.py --config-name=train env=combo_go2Piper_reaching
```

### 2. 配置MoE训练

修改 `config/train_moe_example.yaml` 中的专家模型路径：

```yaml
runner:
  alg:
    actor_critic:
      base_ckpt_path: "path/to/base_expert.pth"  # 基础专家权重路径
      arm_ckpt_path: "path/to/arm_expert.pth"    # 手臂专家权重路径
```

### 3. 启动MoE训练

```bash
python scripts/train.py --config-name=train_moe_example
```

### 4. 推理和评估

```bash
python scripts/play.py --config-name=train_moe_example ckpt_path=path/to/moe_model.pth
```

## 配置参数说明

### ObsMoE 参数

- `obs_dim`: 输入观测量维度
- `hidden_dims`: MLP隐藏层维度列表，默认 [128, 64]
- `num_experts`: 专家数量，默认为2

### ActorCriticMoE 参数

- `base_actor_critic`: 基础专家ActorCritic配置
- `arm_actor_critic`: 手臂专家ActorCritic配置
- `obs_moe`: MoE权重网络配置
- `base_ckpt_path`: 基础专家模型权重路径
- `arm_ckpt_path`: 手臂专家模型权重路径
- `freeze_experts`: 是否冻结专家模型权重，默认True

### PPOMoE 参数

- `moe_weight_regularization`: MoE权重正则化系数，默认0.01
- 其他参数与标准PPO相同

## 工作原理

1. **观测量编码**: 输入观测量通过ObsMoE网络得到专家权重 `w = [w_base, w_arm]`
2. **专家推理**: 两个冻结的专家模型分别计算动作 `a_base`, `a_arm`
3. **权重混合**: 最终动作为 `a = w_base * a_base + w_arm * a_arm`
4. **梯度更新**: 只有ObsMoE网络参与梯度更新，专家模型权重保持冻结

## 调试和监控

### 查看专家权重分布

```python
# 在训练过程中监控权重分布
moe_weights = model.get_moe_weights(observations)
print(f"Base weight: {moe_weights[0]:.3f}, Arm weight: {moe_weights[1]:.3f}")
```

### 权重可视化

可以在训练过程中记录和可视化专家权重的变化，以了解模型如何在不同情况下选择专家。

## 注意事项

1. **权重路径**: 确保专家模型权重路径正确，且模型结构匹配
2. **观测量维度**: ObsMoE的输入维度必须与环境观测量维度一致
3. **专家兼容性**: 两个专家模型应该有相同的动作空间和网络结构
4. **训练稳定性**: 可以调整 `moe_weight_regularization` 参数来控制权重分布的多样性

## 扩展功能

### 添加更多专家

要支持超过2个专家，需要：

1. 修改 `ActorCriticMoE` 构造函数接受专家列表
2. 更新权重混合逻辑支持任意数量专家
3. 相应调整配置文件结构

### 自适应权重

可以考虑实现基于任务难度或性能的自适应权重调整机制。

## 故障排除

### 常见问题

1. **权重加载失败**: 检查专家模型路径和权重文件格式
2. **维度不匹配**: 确认观测量维度配置正确
3. **训练不收敛**: 尝试调整学习率和正则化参数
4. **专家权重不平衡**: 增加权重正则化系数或检查专家模型质量 
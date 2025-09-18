"""
混合专家(MoE) Actor-Critic模块。

该模块加载两个预训练的ActorCritic模型（base和arm），
通过ObsMoE计算权重来混合它们的输出。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional, Dict, Any
import copy

from .actor_critic import ActorCritic
from .obs_moe import ObsMoE


class ActorCriticMoE(nn.Module):
    """
    混合专家Actor-Critic模型。
    
    加载两个预训练的ActorCritic模型（ac_base和ac_arm），
    使用ObsMoE计算权重来混合它们的动作输出。
    只有ObsMoE部分参与训练，预训练模型权重被冻结。
    """
    
    is_recurrent = False
    
    def __init__(
        self,
        base_actor_critic: ActorCritic,
        arm_actor_critic: ActorCritic,
        obs_moe: ObsMoE,
        num_actions: int,
        base_ckpt_path: Optional[str] = None,
        arm_ckpt_path: Optional[str] = None,
        freeze_experts: bool = True,
        **kwargs,
    ):
        """
        初始化混合专家Actor-Critic模型。
        
        Args:
            base_actor_critic: 基础专家模型
            arm_actor_critic: 手臂专家模型
            obs_moe: 观测量MoE编码器
            num_actions: 动作维度
            base_ckpt_path: 基础专家模型权重路径
            arm_ckpt_path: 手臂专家模型权重路径
            freeze_experts: 是否冻结专家模型权重
        """
        super(ActorCriticMoE, self).__init__()
        
        self.num_actions = num_actions
        self.freeze_experts = freeze_experts
        
        # 深拷贝专家模型以避免共享权重
        self.ac_base = copy.deepcopy(base_actor_critic)
        self.ac_arm = copy.deepcopy(arm_actor_critic)
        
        # 加载预训练权重
        if base_ckpt_path is not None:
            self._load_expert_weights(self.ac_base, base_ckpt_path)
        if arm_ckpt_path is not None:
            self._load_expert_weights(self.ac_arm, arm_ckpt_path)
        
        # 冻结专家模型参数
        if freeze_experts:
            self._freeze_expert_parameters()
        
        # MoE权重计算器
        self.obs_moe = obs_moe
        
        # 用于存储混合后的分布
        self.distribution: Normal
        # 禁用参数验证以提高速度
        Normal.set_default_validate_args = False
    
    def _load_expert_weights(self, expert_model: ActorCritic, ckpt_path: str):
        """加载专家模型权重。"""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                expert_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                expert_model.load_state_dict(checkpoint)
            print(f"成功加载专家模型权重: {ckpt_path}")
        except Exception as e:
            print(f"加载专家模型权重失败 {ckpt_path}: {e}")
    
    def _freeze_expert_parameters(self):
        """冻结专家模型参数。"""
        for param in self.ac_base.parameters():
            param.requires_grad = False
        for param in self.ac_arm.parameters():
            param.requires_grad = False
        
        # 设置为评估模式
        self.ac_base.eval()
        self.ac_arm.eval()
    
    def reset(self, dones=None):
        """重置模型状态。"""
        self.ac_base.reset(dones)
        self.ac_arm.reset(dones)
    
    def forward(self, *args, **kwargs):
        """前向传播，默认调用推理模式。"""
        return self.act_inference(*args, **kwargs)
    
    @property
    def action_mean(self):
        """获取动作均值。"""
        return self.distribution.mean
    
    @property
    def action_std(self):
        """获取动作标准差。"""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """计算熵。"""
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations: torch.Tensor):
        """更新动作分布。"""
        # 计算MoE权重
        # import pdb;pdb.set_trace()
        moe_weights = self.obs_moe(observations)  # shape: (batch_size, 2) or (2,)
        
        # 获取两个专家的动作输出
        with torch.no_grad() if self.freeze_experts else torch.enable_grad():
            base_actions = self.ac_base.actor(observations)
            arm_actions = self.ac_arm.actor(observations)
        
        # 根据权重混合动作
        if len(moe_weights.shape) == 1:  # 单个样本
            mixed_actions = (moe_weights[0] * base_actions + 
                           moe_weights[1] * arm_actions)
        else:  # 批量样本
            mixed_actions = (moe_weights[:, 0:1] * base_actions + 
                           moe_weights[:, 1:2] * arm_actions)
        
        # 使用混合后的动作均值创建分布
        # 这里使用base模型的std作为标准差（也可以考虑混合std）
        std = self.ac_base.std
        self.distribution = Normal(mixed_actions, mixed_actions * 0.0 + std)
    
    def act(self, observations: torch.Tensor):
        """采样动作。"""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions: torch.Tensor):
        """计算动作的对数概率。"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations: torch.Tensor):
        """推理模式下的动作输出。"""
        # import pdb;pdb.set_trace()
        # 计算MoE权重
        moe_weights = self.obs_moe(observations)
        
        # 获取两个专家的动作输出
        with torch.no_grad():
            base_actions = self.ac_base.act_inference(observations)
            arm_actions = self.ac_arm.act_inference(observations)
        
        # 根据权重混合动作
        if len(moe_weights.shape) == 1:  # 单个样本
            mixed_actions = (moe_weights[0] * base_actions + 
                           moe_weights[1] * arm_actions)
        else:  # 批量样本
            mixed_actions = (moe_weights[:, 0:1] * base_actions + 
                           moe_weights[:, 1:2] * arm_actions)
        
        return mixed_actions
    
    def evaluate(self, observations: torch.Tensor):
        """评估状态价值。"""
        # 计算MoE权重
        # import pdb;pdb.set_trace()
        # print(observations.shape)#torch.Size([2048, 241]) Here
        values = self.ac_base.critic(observations)
        
        
        return values
    
    def get_moe_weights(self, observations: torch.Tensor):
        """获取MoE权重（用于分析和调试）。"""
        # import pdb;pdb.set_trace()
        return self.obs_moe(observations)
    
    def parameters(self):
        """返回可训练参数（仅MoE部分）。"""
        # import pdb;pdb.set_trace()
        if self.freeze_experts:
            return self.obs_moe.parameters()
        else:
            return super().parameters() 
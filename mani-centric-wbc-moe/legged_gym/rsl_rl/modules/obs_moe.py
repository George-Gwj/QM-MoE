"""
观测量混合专家(MoE)编码器模块。

该模块将观测量通过MLP编码，然后经过softmax层得到用于混合专家的权重。
"""

import torch
import torch.nn as nn
from typing import List


class ObsMoE(nn.Module):
    """
    观测量混合专家编码器。
    
    将输入观测量通过MLP编码后，经过softmax层输出权重，
    用于后续混合不同专家模型的输出。
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_experts: int = 2,
        **kwargs
    ):
        """
        初始化观测量MoE编码器。
        
        Args:
            obs_dim: 输入观测量维度
            hidden_dims: 隐藏层维度列表
            num_experts: 专家数量，默认为2（base和arm）
        """
        super(ObsMoE, self).__init__()
        
        self.obs_dim = obs_dim#109
        self.num_experts = num_experts
        # import pdb;pdb.set_trace()
        # 构建MLP网络
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层，输出专家权重
        layers.append(nn.Linear(input_dim, num_experts))
        
        self.mlp = nn.Sequential(*layers)
        
        # Softmax层用于归一化权重
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            observations: 输入观测量，形状为 (batch_size, obs_dim) 或 (obs_dim,)
            
        Returns:
            weights: 专家权重，形状为 (batch_size, num_experts) 或 (num_experts,)
        """
        # import pdb;pdb.set_trace()

        # 通过MLP编码
        logits = self.mlp(observations)#torch.Size([2048, 109])->torch.Size([2048, 241])#109 ok!
        
        # 通过softmax得到归一化权重
        weights = self.softmax(logits)
        
        return weights


class MLPObsMoE(ObsMoE):
    """
    基于MLP的观测量MoE编码器（与基类相同，保持接口一致性）。
    """
    pass


class ConvObsMoE(ObsMoE):
    """
    基于卷积的观测量MoE编码器（预留接口，当前使用MLP实现）。
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_channels: List[int] = [32, 64],
        num_experts: int = 2,
        **kwargs
    ):
        # 当前仍使用MLP实现，可根据需要扩展为卷积网络
        hidden_dims = [obs_dim // 2, obs_dim // 4] if obs_dim > 8 else [64, 32]
        super(ConvObsMoE, self).__init__(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            num_experts=num_experts,
            **kwargs
        ) 
class CriticObsMoE(nn.Module):
    """
    观测量混合专家编码器。
    
    将输入观测量通过MLP编码后，经过softmax层输出权重，
    用于后续混合不同专家模型的输出。
    """
    
    def __init__(
        self,
        critic_obs_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_experts: int = 2,
        **kwargs
    ):
        """
        初始化观测量MoE编码器。
        
        Args:
            obs_dim: 输入观测量维度
            hidden_dims: 隐藏层维度列表
            num_experts: 专家数量，默认为2（base和arm）
        """
        super(CriticObsMoE, self).__init__()
        
        self.critic_obs_dim = critic_obs_dim#241
        self.num_experts = num_experts
        # import pdb;pdb.set_trace()
        # 构建MLP网络
        layers = []
        input_dim = critic_obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层，输出专家权重
        layers.append(nn.Linear(input_dim, num_experts))
        
        self.mlp = nn.Sequential(*layers)
        
        # Softmax层用于归一化权重
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, critic_observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            observations: 输入观测量，形状为 (batch_size, obs_dim) 或 (obs_dim,)
            
        Returns:
            weights: 专家权重，形状为 (batch_size, num_experts) 或 (num_experts,)
        """
        # import pdb;pdb.set_trace()

        # 通过MLP编码
        logits = self.mlp(critic_observations)#torch.Size([2048, 109])->torch.Size([2048, 241])#109 ok!
        
        # 通过softmax得到归一化权重
        weights = self.softmax(logits)
        
        return weights


class MLPCriticObsMoE(CriticObsMoE):
    """
    基于MLP的观测量MoE编码器（与基类相同，保持接口一致性）。
    """
    pass


class ConvCriticObsMoE(CriticObsMoE):
    """
    基于卷积的观测量MoE编码器（预留接口，当前使用MLP实现）。
    """
    
    def __init__(
        self,
        critic_obs_dim: int,
        hidden_channels: List[int] = [32, 64],
        num_experts: int = 2,
        **kwargs
    ):
        # 当前仍使用MLP实现，可根据需要扩展为卷积网络
        hidden_dims = [critic_obs_dim // 2, critic_obs_dim // 4] if critic_obs_dim > 8 else [64, 32]
        super(ConvCriticObsMoE, self).__init__(
            critic_obs_dim=critic_obs_dim,
            hidden_dims=hidden_dims,
            num_experts=num_experts,
            **kwargs
        ) 
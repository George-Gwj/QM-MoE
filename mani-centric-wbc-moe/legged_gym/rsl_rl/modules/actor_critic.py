# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        num_actions: int,
        init_noise_std=1.0,
        **kwargs,
    ):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic
        self.num_actions = num_actions  # 动作维度，确保与专家模型输出一致



        # -------------------------- 新增：1. 定义mlp_composer_net --------------------------
        # 获取观测维度（从actor的input_dim获取，需确保actor有该属性，如MLP类定义）
        # self.obs_dim = actor.input_dim  # 若actor无input_dim，需手动传入obs_dim参数
        # 定义融合权重网络：输入(obs_dim) → 隐藏层(64) → 隐藏层(32) → 输出(2) → Softmax归一化
        self.mlp_composer_net = nn.Sequential(
            nn.Linear(109, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 输出2个权重（对应policy1和policy2）
            nn.Softmax(dim=-1)  # 归一化权重，确保sum(weights) = 1
        )

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        # self.mlp_composer_net = self.load_expert_policy_jit("/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/output/exported/")
        self.policy1 = self.load_expert_policy_jit("/home/george/code/umi-on-legs-compose/mani-centric-wbc/checkpoints/base/exported/")

        self.policy2 = self.load_expert_policy_jit("/home/george/code/umi-on-legs-compose/mani-centric-wbc/checkpoints/pushing/exported/")
        # 权重文件路径
        pretrained_path = "/home/george/code/umi-on-legs-compose/mani-centric-wbc/output/model_13000.pt"
            

        # 
        # -------------------------- 新增：2. 冻结专家模型参数 --------------------------
        self.freeze_expert_policy(self.policy1)
        self.freeze_expert_policy(self.policy2)
        # #policy2
        # # 导出策略（如果指定） 
        # import hydra
        # from omegaconf import OmegaConf
        # import pickle
        # import os
        # from isaacgym import gymapi, gymutil  # must be improved before torch
        # from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
        # from legged_gym.env.isaacgym.env_add_baseinfo import IsaacGymEnv

        # sim_params = gymapi.SimParams()
        # config = OmegaConf.create(
        # pickle.load(
        #     open(os.path.join(os.path.dirname("/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/checkpoints/base/model_20000.pt"), "config.pkl"), "rb")
        # )
        # )
        # env: IsaacGymEnv = hydra.utils.instantiate(
        # config["env"],
        # sim_params=sim_params,
        # )
        # runner2: OnPolicyRunner = hydra.utils.instantiate(
        #     config["runner"], env=env, eval_fn=None
        # )
        # # args.ckpt_path2 = "/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/checkpoints/base/model_20000.pt"
        # runner2.load("/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/checkpoints/base/model_20000.pt")
        # export_dir = os.path.join(os.path.dirname("/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/checkpoints/base/model_20000.pt"), 'exported')
        # # 确保目录存在
        # os.makedirs(export_dir, exist_ok=True)
        # # import pdb;pdb.set_trace()
        # # # 导出策略
        # self.export_policy_as_jit(runner2.alg.actor_critic, export_dir)#import policy
        # self.policy2 = runner2.alg.get_inference_policy(device=env.device)
    #policy2
    def freeze_expert_policy(self, policy):
        """冻结专家模型参数，禁止反向传播梯度"""
        for param in policy.parameters():
            param.requires_grad = False
        # 确保专家模型处于评估模式，避免BatchNorm等层的行为干扰
        policy.eval()
    def reset(self, dones=None):
        pass

    def forward(self, *args, **kwargs):
        return self.act_inference(*args, **kwargs)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def export_policy_as_jit(self,actor_critic, path):
        import os
        import copy
        """Export policy as TorchScript for C++ deployment"""
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, 'policy.pt')
        
        # 创建一个actor的克隆（避免修改原始模型）
        actor = copy.deepcopy(actor_critic.actor).to('cuda:0')
        
        # 创建输入示例（假设观测是向量）
        obs_dim = actor.input_dim if hasattr(actor, 'input_dim') else 48  # 默认观测维度
        dummy_input = torch.randn(1, obs_dim, device='cuda:0')
        
        # 导出为TorchScript - 优先使用trace
        try:
            traced_model = torch.jit.trace(actor, dummy_input)
            traced_model.save(output_path)
        except RuntimeError:
            # 回退到脚本模式
            scripted_model = torch.jit.script(actor)
            scripted_model.save(output_path)
        
        print(f'Exported policy as TorchScript to: {output_path}')
        
        # 同时导出模型配置
        config_path = os.path.join(path, 'model_config.json')
        with open(config_path, 'w') as f:
            import json
            config = {
                "obs_dim": obs_dim,
                "action_dim": actor.output_dim if hasattr(actor, 'output_dim') else 12,
                "device": "cuda:0"
            }
            json.dump(config, f)



    def load_expert_policy_jit(self, expert_policy_dir, device=None):
        import torch
        import json
        import os
        """
        从 TorchScript 文件加载专家策略
        :param expert_policy_dir: 包含 policy.pt 和 model_config.json 的目录
        :param device: 推理设备 ('cuda' 或 'cpu')
        :return: 已加载的 TorchScript 模型
        """
        # import pdb;pdb.set_trace()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 路径
        model_path = os.path.join(expert_policy_dir, 'policy.pt')
        config_path = os.path.join(expert_policy_dir, 'model_config.json')

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[INFO] Loaded model config: {config}")

        # 加载 TorchScript 模型
        try:
            policy = torch.jit.load(model_path, map_location=device)
            policy.eval()  # 设置为推理模式
            print(f"[INFO] Successfully loaded TorchScript model to {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}")

        return policy
    def update_distribution(self, observations: torch.Tensor):
        # 1. 计算当前批次每个样本的权重 [2048, 2]
        weights = self.mlp_composer_net(observations)  # 已通过Softmax归一化，确保每个样本的两个权重和为1
        
        # 2. 获取两个专家模型的输出（每个样本18维动作）
        mean1 = self.policy1(observations)  # [2048, 18]
        mean2 = self.policy2(observations)  # [2048, 18]
        
        # 3. 提取每个样本对应的权重，并扩展维度以匹配动作维度（关键步骤）
        # weights[:, 0]：取每个样本的第1个权重 → 形状[2048]
        # unsqueeze(1)：扩展为[2048, 1]，便于与[2048, 18]广播相乘
        # import pdb;pdb.set_trace()
        w1 = weights[:, 0].unsqueeze(1)  # [2048, 1]
        w2 = weights[:, 1].unsqueeze(1)  # [2048, 1]
        
        # 4. 融合动作：每个样本用自己的权重乘以对应的专家动作，再相加
        # 广播机制会自动将[2048, 1]与[2048, 18]匹配（每个动作维度都乘以该样本的权重）
        mean = mean1 * w1 + mean2 * w2  # 结果形状[2048, 18]，与批次和动作维度匹配
        
        # 5. 构建动作分布
        self.distribution = Normal(mean, self.std.expand_as(mean))

        # self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations):
        # import pdb;pdb.set_trace()
        self.update_distribution(observations)

        return self.distribution.sample()#actions 2048 18

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        # import pdb;pdb.set_trace()
        # /home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/output/model_13000.pt
        # 1. 计算当前批次每个样本的权重 [2048, 2]
        weights = self.mlp_composer_net(observations)  # 已通过Softmax归一化，确保每个样本的两个权重和为1
        
        # 2. 获取两个专家模型的输出（每个样本18维动作）
        mean1 = self.policy1(observations)  # [2048, 18]
        mean2 = self.policy2(observations)  # [2048, 18]
        
        # 3. 提取每个样本对应的权重，并扩展维度以匹配动作维度（关键步骤）
        # weights[:, 0]：取每个样本的第1个权重 → 形状[2048]
        # unsqueeze(1)：扩展为[2048, 1]，便于与[2048, 18]广播相乘
        # import pdb;pdb.set_trace()
        w1 = weights[:, 0].unsqueeze(1)  # [2048, 1]
        w2 = weights[:, 1].unsqueeze(1)  # [2048, 1]
        
        # 4. 融合动作：每个样本用自己的权重乘以对应的专家动作，再相加
        # 广播机制会自动将[2048, 1]与[2048, 18]匹配（每个动作维度都乘以该样本的权重）
        actions_mean = mean1 * w1 + mean2 * w2  # 结果形状[2048, 18]，与批次和动作维度匹配
        # actions_mean = self.actor(observations)
        # import pdb;pdb.set_trace()
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor):
        # import pdb;pdb.set_trace()
        value = self.critic(critic_observations)
        return value

    def get_moe_weights(self, observations: torch.Tensor):
        """获取MoE权重（用于分析和调试）。"""
        # 使用 mlp_composer_net 计算权重
        weights = self.mlp_composer_net(observations)
        return weights
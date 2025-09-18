"""
混合专家(MoE) PPO算法。

专门用于训练ActorCriticMoE模型的PPO算法，
只训练MoE权重网络，专家模型权重保持冻结。
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim

from legged_gym.rsl_rl.modules.actor_critic_moe import ActorCriticMoE
from legged_gym.rsl_rl.storage import RolloutStorage
from .ppo import PPO


class PPOMoE(PPO):
    """
    混合专家PPO算法。
    
    继承自标准PPO，但专门针对ActorCriticMoE模型进行优化，
    只训练MoE权重网络部分。
    """
    
    def __init__(
        self,
        actor_critic: ActorCriticMoE,
        num_learning_epochs: int,
        num_mini_batches: int,
        clip_param: float,
        gamma: float,
        lam: float,
        value_loss_coef: float,
        entropy_coef: float,
        learning_rate: float,
        max_grad_norm: float,
        use_clipped_value_loss: float,
        schedule: str,
        desired_kl: float,
        max_lr: float,
        min_lr: float,
        device: str,
        moe_weight_regularization: float = 0.01,
        **kwargs
    ):
        """
        初始化PPO MoE算法。
        
        Args:
            actor_critic: ActorCriticMoE模型
            moe_weight_regularization: MoE权重正则化系数
            其他参数与标准PPO相同
        """
        # 调用父类初始化，但传入MoE模型#HERE CORE TO ERROR
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            max_lr=max_lr,
            min_lr=min_lr,
            device=device,
        )
        
        self.moe_weight_regularization = moe_weight_regularization
        
        # 重新创建优化器，只优化MoE部分
        self.optimizer = optim.Adam(self.get_parameters(), lr=learning_rate)
    
    def get_parameters(self):
        """获取可训练参数（仅MoE权重网络）。"""
        return self.actor_critic.obs_moe.parameters()
    def update(self, learning_iter: int) -> Dict[str, float]:
        """更新模型参数，并返回详细的训练统计信息。"""
        update_stats = {}  # 用于累计每个 batch 的统计量

        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # 前向计算当前策略下的 log prob 和 value
            self.actor_critic.act(obs_batch, masks_batch, hid_states_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks_batch, hid_states_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 计算 MoE 权重正则化损失
            moe_weights = self.actor_critic.get_moe_weights(obs_batch)
            moe_entropy = -torch.sum(moe_weights * torch.log(moe_weights + 1e-8), dim=-1)
            moe_regularization_loss = -self.moe_weight_regularization * moe_entropy.mean()

            # Surrogate loss (PPO clipped)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_pred_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总损失
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + moe_regularization_loss
            )

            # 记录各个 loss 到 update_stats
            for name, loss_val in [
                ("value_loss", value_loss),
                ("surrogate_loss", surrogate_loss),
                ("moe_regularization_loss", moe_regularization_loss),
                ("entropy", entropy_batch.mean()),
            ]:
                if name not in update_stats:
                    update_stats[name] = 0.0
                update_stats[name] += loss_val.item()

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 归一化：除以总的更新次数
        num_updates = self.num_learning_epochs * self.num_mini_batches
        result = {k: v / num_updates for k, v in update_stats.items()}

        # 添加额外信息
        result.update({
            "learning_rate": self.learning_rate,
            "action_std": self.actor_critic.std.mean().item(),
        })

        # 清空 buffer（通常放在 epoch 结束后）
        self.storage.clear()

        return result
    # def update(self):
    #     """更新模型参数。"""
    #     mean_value_loss = 0
    #     mean_surrogate_loss = 0
    #     mean_moe_regularization_loss = 0
        
    #     if self.actor_critic.is_recurrent:
    #         generator = self.storage.recurrent_generator(
    #             self.num_mini_batches, self.num_learning_epochs
    #         )
    #     else:
    #         generator = self.storage.mini_batch_generator(
    #             self.num_mini_batches, self.num_learning_epochs
    #         )
        
    #     for (
    #         obs_batch,
    #         critic_obs_batch,
    #         actions_batch,
    #         target_values_batch,
    #         advantages_batch,
    #         returns_batch,
    #         old_actions_log_prob_batch,
    #         old_mu_batch,
    #         old_sigma_batch,
    #         hid_states_batch,
    #         masks_batch,
    #     ) in generator:
            
    #         self.actor_critic.act(obs_batch, masks_batch, hid_states_batch)
    #         actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
    #         import pdb;pdb.set_trace()
    #         value_batch = self.actor_critic.evaluate(
    #             critic_obs_batch, masks_batch, hid_states_batch
    #         )
    #         mu_batch = self.actor_critic.action_mean
    #         sigma_batch = self.actor_critic.action_std
    #         entropy_batch = self.actor_critic.entropy
            
    #         # 计算MoE权重正则化损失
    #         moe_weights = self.actor_critic.get_moe_weights(obs_batch)
    #         # 鼓励权重分布的多样性，避免总是选择一个专家
    #         moe_entropy = -torch.sum(moe_weights * torch.log(moe_weights + 1e-8), dim=-1)
    #         moe_regularization_loss = -self.moe_weight_regularization * moe_entropy.mean()
            
    #         # Surrogate loss
    #         ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
    #         surrogate = -torch.squeeze(advantages_batch) * ratio
    #         surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
    #             ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
    #         )
    #         surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
    #         # Value function loss
    #         if self.use_clipped_value_loss:
    #             value_pred_clipped = target_values_batch + (
    #                 value_batch - target_values_batch
    #             ).clamp(-self.clip_param, self.clip_param)
    #             value_losses = (value_batch - returns_batch).pow(2)
    #             value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
    #             value_loss = torch.max(value_losses, value_losses_clipped).mean()
    #         else:
    #             value_loss = (returns_batch - value_batch).pow(2).mean()
            
    #         # 总损失
    #         loss = (
    #             surrogate_loss
    #             + self.value_loss_coef * value_loss
    #             - self.entropy_coef * entropy_batch.mean()
    #             + moe_regularization_loss
    #         )
            
    #         # 梯度更新
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.get_parameters(), self.max_grad_norm)
    #         self.optimizer.step()
            
    #         mean_value_loss += value_loss.item()
    #         mean_surrogate_loss += surrogate_loss.item()
    #         mean_moe_regularization_loss += moe_regularization_loss.item()
        
    #     num_updates = self.num_learning_epochs * self.num_mini_batches
    #     mean_value_loss /= num_updates
    #     mean_surrogate_loss /= num_updates
    #     mean_moe_regularization_loss /= num_updates
    #     self.storage.clear()
        
    #     return mean_value_loss, mean_surrogate_loss, mean_moe_regularization_loss 
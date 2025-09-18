# 1. 首先导入Isaac Gym相关模块
import isaacgym
from isaacgym import gymapi, gymtorch

# 2. 导入legged_gym的具体环境类（使用IsaacVecEnv而非抽象的VecEnv）
import torch
import torch.nn as nn
from legged_gym.rsl_rl.algorithms import PPO
from legged_gym.env.isaacgym.env_add_baseinfo import IsaacGymEnv
from legged_gym.rsl_rl.storage.rollout_storage import RolloutStorage

# ---------------------------
# 1. 权重网络 (MLP)
# ---------------------------
class WeightNetwork(nn.Module):
    def __init__(self, obs_dim=109):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出2个权重 [w1, w2]
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.mlp(obs)

# ---------------------------
# 2. 融合环境（使用具体的IsaacVecEnv作为基类）
# ---------------------------
class SimpleFusionEnv(IsaacGymEnv):
    def __init__(self, num_envs=4, obs_dim=109):
        # 配置环境参数（符合IsaacVecEnv的要求）
        cfg = {
            "env": {
                "num_envs": num_envs,
                "num_observations": obs_dim,
                "num_actions": 2,  # 权重网络的输出维度
            },
            "sim": {
                "dt": 0.01,  # 仿真步长
                "substeps": 2
            }
        }
        
        # 初始化父类（IsaacVecEnv需要gymapi和配置）
        super().__init__(gymapi, cfg)
        
        self.obs_dim = obs_dim
        self.action_dim = 2
        self.device = self.device  # 继承自IsaacVecEnv的设备属性
        
        # 模拟预训练策略（替换为你的实际策略）
        self.policy1 = nn.Linear(obs_dim, 12).to(self.device)  # 四足策略
        self.policy2 = nn.Linear(obs_dim, 12).to(self.device)  # 机械臂策略
        
        # 初始化观测和done缓冲区
        self.obs_buf = torch.randn(num_envs, obs_dim, device=self.device)
        self.done_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    def reset(self):
        """重置环境"""
        self.obs_buf = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        self.done_buf[:] = False
        return self.obs_buf

    def step(self, weights):
        """执行融合动作"""
        # 从两个策略获取动作
        with torch.no_grad():
            action1 = self.policy1(self.obs_buf)  # 四足动作
            action2 = self.policy2(self.obs_buf)  # 机械臂动作
        
        # 融合动作
        fused_action = weights[:, 0:1] * action1 + weights[:, 1:2] * action2
        
        # 简单奖励计算
        stability = -torch.norm(self.obs_buf[:, :3])  # 稳定性指标
        tracking = -torch.norm(self.obs_buf[:, 10:13])  # 跟踪误差
        reward = 0.7 * stability + 0.3 * tracking
        
        # 更新观测（替换为真实数据）
        self.obs_buf = 0.9 * self.obs_buf + 0.1 * torch.randn_like(self.obs_buf)
        self.done_buf = torch.rand(self.num_envs, device=self.device) < 0.05  # 随机重置
        
        return self.obs_buf, reward, self.done_buf, {}

# ---------------------------
# 3. 训练循环
# ---------------------------
def train():
    num_envs = 4
    num_steps = 1024
    num_updates = 500
    obs_dim = 109

    # 初始化环境（现在使用正确的基类）
    env = SimpleFusionEnv(num_envs=num_envs, obs_dim=obs_dim)
    weight_net = WeightNetwork(obs_dim=obs_dim).to(env.device)
    
    # Actor-Critic包装器
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = weight_net
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def get_actions(self, obs):
            return self.actor(obs), self.critic(obs)
        
        def evaluate_actions(self, obs, actions):
            weights = self.actor(obs)
            log_prob = -torch.nn.functional.mse_loss(weights, actions, reduction="none").sum(-1)
            return log_prob, self.critic(obs), torch.tensor(0.0)

    actor_critic = ActorCritic().to(env.device)
    
    # PPO配置
    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.2,
        ppo_epoch=3,
        num_mini_batch=2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4
    )
    
    # 经验存储
    rollout_storage = RolloutStorage(
        num_steps, num_envs, obs_dim, 2, env.device, gamma=0.99
    )
    rollout_storage.obs[0].copy_(env.reset())

    # 训练循环
    for update in range(num_updates):
        for step in range(num_steps):
            with torch.no_grad():
                actions, values = actor_critic.get_actions(rollout_storage.obs[step])
            next_obs, rewards, dones, _ = env.step(actions)
            rollout_storage.insert(next_obs, actions, values, rewards, dones)
        
        with torch.no_grad():
            next_value = actor_critic.critic(rollout_storage.obs[-1])
        rollout_storage.compute_returns(next_value, False)
        ppo.update(rollout_storage)
        rollout_storage.after_update()

        if update % 50 == 0:
            print(f"Update {update}: 平均奖励 = {rollout_storage.rewards.mean():.2f}")

if __name__ == "__main__":
    train()
    
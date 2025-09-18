import mujoco_py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os
import pickle
from omegaconf import OmegaConf
import json

class IsaacGymToMuJoCoEnv:
    """将IsaacGym环境转换为MuJoCo环境的包装器"""
    
    def __init__(
        self,
        model_path: str,
        policy_path: str,
        config_path: str,
        device: str = "cpu",
        render: bool = True,
        dt: float = 0.01,
        num_envs: int = 1
    ):
        """
        初始化MuJoCo环境
        
        Args:
            model_path: MuJoCo模型文件路径(.xml)
            policy_path: 训练好的策略文件路径(.pt)
            config_path: 原始训练配置文件路径
            device: 计算设备
            render: 是否渲染
            dt: 仿真时间步长
            num_envs: 环境数量（MuJoCo通常为1）
        """
        self.device = device
        self.render = render
        self.dt = dt
        self.num_envs = num_envs
        
        # 加载MuJoCo模型
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        
        # 加载策略
        self.policy = self._load_policy(policy_path)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化环境状态
        self.curriculum_stage = getattr(self.config.env.cfg.curriculum, 'stage', 1)
        self.curriculum_transition_steps = getattr(self.config.env.cfg.curriculum, 'transition_steps', 2000000)
        self.global_step = 0
        
        # 机械臂参数
        self.arm_dof_indices = self._get_arm_dof_indices()
        self.num_arm_dofs = len(self.arm_dof_indices)
        self.arm_init_pos = np.zeros(self.num_arm_dofs)
        
        # 速度指令
        self.base_lin_vel_cmd = np.zeros(3)
        
        # 观测和动作维度
        self.obs_dim = self.config.env.num_observations
        self.action_dim = self.config.env.num_actions
        
        # 重置环境
        self.reset()
        
        # 设置渲染器
        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)
        
    def _load_policy(self, policy_path: str) -> nn.Module:
        """加载训练好的策略"""
        # 加载策略文件
        if policy_path.endswith('.pt'):
            # TorchScript格式
            policy = torch.jit.load(policy_path, map_location=self.device)
        else:
            # 普通PyTorch模型
            policy = torch.load(policy_path, map_location=self.device)
        
        policy.eval()
        return policy
    
    def _load_config(self, config_path: str) -> OmegaConf:
        """加载训练配置"""
        if config_path.endswith('.pkl'):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
        else:
            config = OmegaConf.load(config_path)
        return config
    
    def _get_arm_dof_indices(self) -> List[int]:
        """获取机械臂关节索引"""
        # 假设机械臂关节在最后6个
        total_dofs = self.model.nu
        if total_dofs >= 18:
            return list(range(total_dofs - 6, total_dofs))
        return []
    
    def _get_observations(self) -> np.ndarray:
        """构建观测向量，模拟IsaacGym环境的观测结构"""
        obs_list = []
        
        # 1. Setup观测 (如果有的话)
        # 这里简化处理，实际需要根据配置构建
        setup_obs = np.zeros(0)  # 根据实际配置调整
        obs_list.append(setup_obs)
        
        # 2. 状态观测
        # 根部角速度 (局部坐标系)
        root_ang_vel = self.sim.data.qvel[:3] * 0.25  # 应用scale
        obs_list.append(root_ang_vel)
        
        # 局部重力方向
        gravity = np.array([0, 0, -1])  # 简化处理
        obs_list.append(gravity)
        
        # 关节位置
        dof_pos = self.sim.data.qpos[7:7+self.action_dim]  # 跳过根部7个自由度
        obs_list.append(dof_pos)
        
        # 关节速度
        dof_vel = self.sim.data.qvel[6:6+self.action_dim] * 0.05  # 应用scale
        obs_list.append(dof_vel)
        
        # 3. 任务观测 (简化处理)
        task_obs = np.zeros(0)  # 根据实际任务配置调整
        obs_list.append(task_obs)
        
        # 4. 基座速度指令
        obs_list.append(self.base_lin_vel_cmd)
        
        # 5. 额外观测
        # 根部位置
        root_pos = self.sim.data.qpos[:3]
        obs_list.append(root_pos)
        
        # 根部速度
        root_vel = self.sim.data.qvel[:3]
        obs_list.append(root_vel)
        
        # Z轴旋转
        z_rotation = np.array([self.sim.data.qpos[6]])  # 四元数的z分量
        obs_list.append(z_rotation)
        
        # 6. 动作历史 (使用零向量)
        action_history = np.zeros(self.action_dim)
        obs_list.append(action_history)
        
        # 合并所有观测
        obs = np.concatenate(obs_list)
        
        # 确保观测维度正确
        if len(obs) != self.obs_dim:
            print(f"Warning: Expected obs dim {self.obs_dim}, got {len(obs)}")
            # 调整维度
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)))
            else:
                obs = obs[:self.obs_dim]
        
        return obs
    
    def _apply_actions(self, actions: np.ndarray):
        """应用动作到MuJoCo仿真器"""
        # 设置关节位置目标
        for i, action in enumerate(actions):
            if i < len(self.sim.data.ctrl):
                self.sim.data.ctrl[i] = action
    
    def _update_curriculum(self):
        """更新课程学习阶段"""
        if self.curriculum_stage == 1 and self.global_step >= self.curriculum_transition_steps:
            self.curriculum_stage = 2
            print(f"课程学习：进入阶段2 - 启用机械臂随机目标（不摆动）")
        elif self.curriculum_stage == 2 and self.global_step >= self.curriculum_transition_steps * 2:
            self.curriculum_stage = 3
            print(f"课程学习：进入阶段3 - 启用机械臂摆动（小幅度）")
        elif self.curriculum_stage == 3 and self.global_step >= self.curriculum_transition_steps * 3:
            self.curriculum_stage = 4
            print(f"课程学习：进入阶段4 - 加大机械臂摆动幅度")
        elif self.curriculum_stage == 4 and self.global_step >= self.curriculum_transition_steps * 4:
            self.curriculum_stage = 5
            print(f"课程学习：进入阶段5 - 超大机械臂摆动幅度")
    
    def _arm_swing_control(self):
        """机械臂摆动控制"""
        if self.num_arm_dofs == 0 or self.curriculum_stage == 1:
            return
        
        # 随机选择机械臂关节
        if self.num_arm_dofs > 0:
            rand_idx = np.random.randint(0, self.num_arm_dofs)
            arm_dof_idx = self.arm_dof_indices[rand_idx]
            
            # 根据课程阶段调整摆动幅度
            if self.curriculum_stage == 2:
                offset = 0.0
            elif self.curriculum_stage == 3:
                offset = (np.random.random() - 0.5) * 0.6
            elif self.curriculum_stage == 4:
                offset = (np.random.random() - 0.5) * 1.6
            elif self.curriculum_stage == 5:
                offset = (np.random.random() - 0.5) * 3.0
            else:
                offset = 0.0
            
            # 计算目标位置
            target = self.arm_init_pos[rand_idx] + offset
            
            # 应用PD控制
            current_pos = self.sim.data.qpos[7 + arm_dof_idx]
            current_vel = self.sim.data.qvel[6 + arm_dof_idx]
            
            kp = 100.0
            kd = 1.0
            
            torque = kp * (target - current_pos) + kd * (-current_vel)
            
            # 应用力矩
            if arm_dof_idx < len(self.sim.data.ctrl):
                self.sim.data.ctrl[arm_dof_idx] = torque
    
    def reset(self):
        """重置环境"""
        self.sim.reset()
        
        # 设置初始状态
        initial_qpos = np.zeros(self.model.nq)
        initial_qpos[2] = 0.3  # 设置初始高度
        
        # 设置机械臂初始角度
        if self.num_arm_dofs > 0:
            for i, idx in enumerate(self.arm_dof_indices):
                if 7 + idx < len(initial_qpos):
                    initial_qpos[7 + idx] = 0.0  # 机械臂初始角度
                    self.arm_init_pos[i] = 0.0
        
        self.sim.data.qpos[:] = initial_qpos
        self.sim.forward()
        
        # 重置课程学习
        self.curriculum_stage = 1
        self.global_step = 0
        
        return self._get_observations()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步仿真"""
        # 更新课程学习
        self._update_curriculum()
        
        # 应用动作
        self._apply_actions(actions)
        
        # 机械臂摆动控制
        self._arm_swing_control()
        
        # 仿真一步
        self.sim.step()
        
        # 获取观测
        obs = self._get_observations()
        
        # 计算奖励（简化处理）
        reward = 0.0
        
        # 检查是否结束
        done = False
        
        # 更新步数
        self.global_step += 1
        
        return obs, reward, done, {}
    
    def render(self):
        """渲染环境"""
        if self.render and hasattr(self, 'viewer'):
            self.viewer.render()
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'viewer'):
            self.viewer.close()

def create_mujoco_env(
    model_path: str,
    policy_path: str,
    config_path: str,
    **kwargs
) -> IsaacGymToMuJoCoEnv:
    """创建MuJoCo环境的工厂函数"""
    return IsaacGymToMuJoCoEnv(
        model_path=model_path,
        policy_path=policy_path,
        config_path=config_path,
        **kwargs
    ) 
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from isaacgym import gymapi, gymtorch

from .env_add_baseinfo import IsaacGymEnv
from .state import EnvState
from .control import Control

import logging

from .utils import torch_rand_float

class IsaacGymEnvVelCmdArmSwing(IsaacGymEnv):
    def __init__(
        self,
        cfg,
        sim_params,
        sim_device,
        headless,
        controller,
        state_obs,
        setup_obs,
        privileged_state_obs,
        privileged_setup_obs,
        tasks,
        constraints,
        seed: int,
        dof_pos_reset_range_scale: float,
        obs_history_len: int,
        vis_resolution: Tuple[int, int],
        env_spacing: float,
        ctrl_buf_len: int,
        max_action_value: float,
        ctrl_delay: Optional[torch.Tensor] = None,
        init_dof_pos: Optional[torch.Tensor] = None,
        graphics_device_id: Optional[int] = None,
        debug_viz: float = True,
        attach_camera: bool = True,
        dense_rewards: bool = True,
    ):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            sim_device=sim_device,
            headless=headless,
            controller=controller,
            state_obs=state_obs,
            setup_obs=setup_obs,
            privileged_state_obs=privileged_state_obs,
            privileged_setup_obs=privileged_setup_obs,
            tasks=tasks,
            constraints=constraints,
            seed=seed,
            dof_pos_reset_range_scale=dof_pos_reset_range_scale,
            obs_history_len=obs_history_len,
            vis_resolution=vis_resolution,
            env_spacing=env_spacing,
            ctrl_buf_len=ctrl_buf_len,
            max_action_value=max_action_value,
            ctrl_delay=ctrl_delay,
            init_dof_pos=init_dof_pos,
            graphics_device_id=graphics_device_id,
            debug_viz=debug_viz,
            attach_camera=attach_camera,
            dense_rewards=dense_rewards,
        )

        # 课程学习参数
        # 新增：课程学习多一个最前面的阶段，curriculum_stage=1时不做机械臂目标
        self.curriculum_stage = getattr(self.cfg.curriculum, 'stage', 1)  # 当前阶段
        self.curriculum_transition_steps = getattr(self.cfg.curriculum, 'transition_steps', 2000000)  # 阶段转换步数
        
        # 奖励权重（可从 cfg 读取，若不存在则使用默认）
        self.base_cmd_tracking_sigma: float = getattr(
            self.cfg.rewards, "base_cmd_tracking_sigma", 0.5
        )
        self.base_cmd_reward_scale: float = getattr(
            self.cfg.rewards, "base_cmd_reward_scale", 1.0
        )
        # 强制忽略 task（例如末端跟踪），reward系数设为0
        self.task_reward_attenuation: float = 0.0

        # 机械臂参数
        self.arm_dof_indices: List[int] = self._infer_arm_dof_indices() if hasattr(self, "_infer_arm_dof_indices") else []
        self.num_arm_dofs: int = len(self.arm_dof_indices)
        self.arm_init_pos: torch.Tensor = torch.zeros(
            (self.num_envs, self.num_arm_dofs), dtype=torch.float32, device=self.device
        ) if self.num_arm_dofs > 0 else torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        
        # 新增：为每个环境记录要控制的机械臂关节索引（在reset时确定）
        # 修改：现在每个环境控制三个关节
        self.arm_control_joint_idx: torch.Tensor = torch.zeros(
            (self.num_envs, 3), dtype=torch.long, device=self.device
        ) if self.num_arm_dofs > 0 else torch.zeros((self.num_envs, 0, 3), dtype=torch.long, device=self.device)
        
        # 机械臂摆动参数（根据课程阶段调整）
        self.arm_swing_amplitude = getattr(self.cfg.curriculum, 'arm_swing_amplitude', 0.5)
        self.arm_swing_enabled = getattr(self.cfg.curriculum, 'arm_swing_enabled', False)

    def _infer_arm_dof_indices(self) -> List[int]:
        """
        自动推断机械臂关节的dof下标。
        解决 AttributeError: 'IsaacGymEnvVelCmdArmSwing' object has no attribute '_infer_arm_dof_indices'
        """
        # 尝试从dof_names中找出机械臂关节
        # 假设机械臂关节名包含'link'或'joint'且编号大于腿部关节
        # 你可以根据实际的dof命名规则调整此处
        if not hasattr(self, "dof_names"):
            # 兼容性处理
            return []
        arm_indices = []
        for idx, name in enumerate(self.dof_names):
            # 这里假设机械臂关节名包含'link'或'joint'且编号大于12（前12个为腿部）
            # 你可以根据实际情况调整
            if ("link" in name or "joint" in name) and idx >= 12:
                arm_indices.append(idx)
        # 如果找不到，尝试直接取最后6个
        if not arm_indices and len(self.dof_names) >= 18:
            arm_indices = list(range(len(self.dof_names) - 6, len(self.dof_names)))
        return arm_indices

    def update_curriculum(self, global_step: int):
        """根据全局步数更新课程阶段"""
        # 新增：阶段1->2，2->3，3->4，4->5，5->6
        if self.curriculum_stage == 1 and global_step >= self.curriculum_transition_steps:
            # 阶段1 -> 阶段2：开始机械臂随机目标（原来的阶段1）
            self.curriculum_stage = 2
            self.arm_swing_enabled = False
            logging.info("课程学习：进入阶段2 - 启用机械臂随机目标（不摆动）")
        elif self.curriculum_stage == 2 and global_step >= self.curriculum_transition_steps * 2:
            # 阶段2 -> 阶段3：启用机械臂摆动（小幅度）
            self.curriculum_stage = 3
            self.arm_swing_enabled = True
            self.arm_swing_amplitude = 0.3
            logging.info("课程学习：进入阶段3 - 启用机械臂摆动（小幅度）")
        elif self.curriculum_stage == 3 and global_step >= self.curriculum_transition_steps * 3:
            # 阶段3 -> 阶段4：加大摆动幅度
            self.curriculum_stage = 4
            self.arm_swing_amplitude = 0.8
            logging.info("课程学习：进入阶段4 - 加大机械臂摆动幅度")
        elif self.curriculum_stage == 4 and global_step >= self.curriculum_transition_steps * 4:
            # 阶段4 -> 阶段5：更大幅度摆动 ±1.5rad
            self.curriculum_stage = 5
            self.arm_swing_amplitude = 3.0
            logging.info("课程学习：进入阶段5 - 超大机械臂摆动幅度（±1.5rad）")
        elif self.curriculum_stage == 5 and global_step >= self.curriculum_transition_steps * 5:
            # 阶段5 -> 阶段6：从单关节摆动升级为三关节摆动
            self.curriculum_stage = 6
            self.arm_swing_amplitude = 2.0  # 稍微降低幅度，因为控制三个关节更复杂
            logging.info("课程学习：进入阶段6 - 升级为三关节摆动控制")

    def _arm_swing_callback(self):
        """每个 decimation 步调用：根据课程阶段调整机械臂摆动
        新增：阶段1不做机械臂目标，阶段2及以后才做
        修改：使用reset时确定的关节索引，而不是每次都随机选择
        """
        if self.num_arm_dofs == 0:
            return

        # 新增：阶段1不做机械臂目标
        if self.curriculum_stage == 1:
            return

        num_envs = self.num_envs
        device = self.device

        # 使用reset时确定的机械臂关节索引，而不是每次都随机选择
        # 这样可以提高训练稳定性，让策略在一个episode内专注于控制同一个关节组合
        chosen_global = self.arm_control_joint_idx  # [num_envs]

        # PD 控制参数
        arm_control_cfg = getattr(self.cfg, 'arm_control', None)
        if arm_control_cfg is not None and isinstance(arm_control_cfg, dict):
            kp = arm_control_cfg.get('kp', 100.0)
            kd = arm_control_cfg.get('kd', 1.0)
        else:
            kp = 100.0
            kd = 1.0

        # 根据课程阶段调整摆动幅度
        if self.curriculum_stage == 2:
            # 阶段2：offset为0，形状为[num_envs]（只做随机目标，不摆动）
            offsets = torch.zeros((num_envs,), device=device)
        elif self.curriculum_stage == 3:
            # 阶段3：小幅度摆动 ±0.3rad
            offsets = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 0.6
        elif self.curriculum_stage == 4:
            # 阶段4：大幅度摆动 ±0.8rad
            offsets = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 1.6
        elif self.curriculum_stage == 5:
            # 阶段5：超大幅度摆动 ±1.5rad
            offsets = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 3.0
        elif self.curriculum_stage == 6:
            # 阶段6：三关节摆动，每个关节有不同的摆动幅度
            # 为三个关节分别生成摆动偏移
            offsets = torch.zeros((num_envs, 3), device=device)
            # 第一个关节：小幅度摆动 ±0.5rad
            offsets[:, 0] = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 1.0
            # 第二个关节：中等幅度摆动 ±1.0rad
            offsets[:, 1] = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 2.0
            # 第三个关节：大幅度摆动 ±1.5rad
            offsets[:, 2] = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * 3.0
        else:
            # 其他情况：使用self.arm_swing_amplitude
            offsets = (torch.rand((num_envs,), device=device, generator=self.generator) - 0.5) * self.arm_swing_amplitude
        
        # 初始角 + 摆动偏移
        # 需要将全局dof索引映射回局部arm_dof_indices的索引
        # 找到每个全局索引在arm_dof_indices中的位置
        arm_dof_idx_tensor = torch.tensor(self.arm_dof_indices, device=device, dtype=torch.long)
        
        if self.curriculum_stage == 6:
            # 阶段6：三关节摆动，需要处理三个关节
            local_indices = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
            for i in range(num_envs):
                for j in range(3):
                    # 找到每个全局索引在arm_dof_indices中的位置
                    local_idx = torch.where(arm_dof_idx_tensor == chosen_global[i, j])[0]
                    if len(local_idx) > 0:
                        local_indices[i, j] = local_idx[0]
            
            # 获取对应的初始角度 [num_envs, 3]
            init_sel = torch.zeros((num_envs, 3), device=device)
            for i in range(num_envs):
                for j in range(3):
                    init_sel[i, j] = self.arm_init_pos[i, local_indices[i, j]]
            
            targets = init_sel + offsets  # [num_envs, 3]
            
            # 按每个被选中的全局 dof 的软限位进行 clamp
            low = torch.zeros((num_envs, 3), device=device)
            high = torch.zeros((num_envs, 3), device=device)
            for i in range(num_envs):
                for j in range(3):
                    low[i, j] = self.curr_dof_pos_limits[chosen_global[i, j], 0]
                    high[i, j] = self.curr_dof_pos_limits[chosen_global[i, j], 1]
            targets = torch.clamp(targets, low, high)  # [num_envs, 3]
            
            # 当前角度和速度 [num_envs, 3]
            current_pos = torch.zeros((num_envs, 3), device=device)
            current_vel = torch.zeros((num_envs, 3), device=device)
            for i in range(num_envs):
                for j in range(3):
                    current_pos[i, j] = self.state.dof_pos[i, chosen_global[i, j]]
                    current_vel[i, j] = self.state.dof_vel[i, chosen_global[i, j]]
            
            # 计算 PD 力矩 [num_envs, 3]
            pos_error = targets - current_pos
            vel_error = -current_vel  # 目标速度为 0
            arm_torques = kp * pos_error + kd * vel_error
            
            # 创建所有关节的零力矩张量，一次性写入
            full_torques = torch.zeros((num_envs, self.num_dof), device=device, dtype=torch.float)
            env_idx = torch.arange(num_envs, device=device)
            for i in range(num_envs):
                for j in range(3):
                    full_torques[i, chosen_global[i, j]] = arm_torques[i, j]
            
        else:
            # 其他阶段：单关节摆动（保持原有逻辑）
            local_indices = torch.zeros_like(chosen_global)
            for i in range(num_envs):
                # 找到全局索引在arm_dof_indices中的位置
                local_idx = torch.where(arm_dof_idx_tensor == chosen_global[i])[0]
                if len(local_idx) > 0:
                    local_indices[i] = local_idx[0]
            
            # 获取对应的初始角度
            init_sel = self.arm_init_pos[
                torch.arange(num_envs, device=device),
                local_indices
            ]  # [num_envs]
            targets = init_sel + offsets  # [num_envs]

            # 按每个被选中的全局 dof 的软限位进行 clamp
            low = self.curr_dof_pos_limits[chosen_global, 0]  # [num_envs]
            high = self.curr_dof_pos_limits[chosen_global, 1]  # [num_envs]
            targets = torch.clamp(targets, low, high)  # [num_envs]

            # 当前角度和速度
            current_pos = self.state.dof_pos[
                torch.arange(num_envs, device=device),
                chosen_global
            ]  # [num_envs]
            current_vel = self.state.dof_vel[
                torch.arange(num_envs, device=device),
                chosen_global
            ]  # [num_envs]

            # 计算 PD 力矩
            pos_error = targets - current_pos  # [num_envs]
            vel_error = -current_vel           # [num_envs] 目标速度为 0
            arm_torques = kp * pos_error + kd * vel_error   # [num_envs]

            # 创建所有关节的零力矩张量，一次性写入
            full_torques = torch.zeros((num_envs, self.num_dof), device=device, dtype=torch.float)
            env_idx = torch.arange(num_envs, device=device)
            full_torques[env_idx, chosen_global] = arm_torques

        # 叠加到现有力矩并应用
        combined_torques = self.ctrl.torque + full_torques
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(combined_torques)
        )

    def compute_reward(self, state: EnvState, control: Control):
        """重写奖励计算，实现课程学习的目标"""
        return_dict = {
            "reward/total": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "reward/env": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "reward/constraint": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "reward/task": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
        }
        
        # 1. 计算 baselink 速度指令跟踪奖励
        vel_error = state.local_root_lin_vel[:, :2] - self.base_lin_vel_cmd[:, :2]
        

        base_cmd_reward = torch.exp(-torch.sum(vel_error**2, dim=1) / self.base_cmd_tracking_sigma**2)
        base_cmd_reward = base_cmd_reward * self.base_cmd_reward_scale
        return_dict["reward/base_cmd_tracking"] = base_cmd_reward
        return_dict["reward/total"] += base_cmd_reward
        return_dict["reward/env"] += base_cmd_reward
        
        # 2. 计算约束奖励（保持原有）
        for constraint_name, constraint in self.constraints.items():
            constraint_rewards = constraint.reward(state=state, control=control)
            for k, v in constraint_rewards.items():
                key = f"constraint/{constraint_name}/{k}"
                return_dict[key] = v
                return_dict["reward/total"] += v
                return_dict["reward/constraint"] += v
        
        # 3. 衰减的 task 奖励（末端跟踪）
        for task_name, task in self.tasks.items():
            raw_task_rewards = task.reward(state=state, control=control)
            attenuated = {k: v * self.task_reward_attenuation for k, v in raw_task_rewards.items()}
            task_rewards = {f"task/{task_name}/{k}": v for k, v in attenuated.items()}
            return_dict.update(task_rewards)
            summed = sum(task_rewards.values())
            return_dict["reward/total"] += summed
            return_dict["reward/task"] += summed
        
        return return_dict

    def reset_idx(self, env_ids):
        """重写 reset_idx 方法，记录机械臂初始角度"""
        # 调用父类的 reset_idx
        super().reset_idx(env_ids)
        
        # 记录机械臂关节的初始角度（在软限位范围内）
        if self.num_arm_dofs > 0:
            # 获取机械臂关节的软限位
            arm_limits_low = self.curr_dof_pos_limits[self.arm_dof_indices, 0]   # [num_arm_dofs]
            arm_limits_high = self.curr_dof_pos_limits[self.arm_dof_indices, 1]  # [num_arm_dofs]

            # 为每个环境采样机械臂初始角度（在软限位范围内）
            num = len(env_ids)
            arm_init_pos = torch_rand_float(
                lower=arm_limits_low,
                upper=arm_limits_high,
                shape=(num, self.num_arm_dofs),
                device=self.device,
                generator=self.generator,
            )
            
            # 将初始角度记录到对应环境
            self.arm_init_pos[env_ids] = arm_init_pos
        
        # 为每个环境随机选择要控制的机械臂关节
        if self.num_arm_dofs > 0:
            # 根据课程阶段选择关节数量
            if self.curriculum_stage == 6:
                # 阶段6：选择三个关节进行摆动控制
                for i, env_id in enumerate(env_ids):
                    # 为每个环境随机选择三个不同的关节索引
                    if self.num_arm_dofs >= 3:
                        # 如果有足够的关节，随机选择三个不同的
                        local_indices = torch.randperm(self.num_arm_dofs, device=self.device)[:3]
                    else:
                        # 如果关节数量不足，重复选择
                        local_indices = torch.randint(
                            low=0,
                            high=self.num_arm_dofs,
                            size=(3,),
                            device=self.device,
                            generator=self.generator,
                        )
                    
                    # 将局部索引映射到全局关节索引
                    global_joint_indices = [self.arm_dof_indices[idx.item()] for idx in local_indices]
                    self.arm_control_joint_idx[env_id] = torch.tensor(global_joint_indices, device=self.device)
            else:
                # 其他阶段：选择单个关节进行摆动控制
                for i, env_id in enumerate(env_ids):
                    # 为每个环境随机选择一个关节索引
                    local_idx = torch.randint(
                        low=0,
                        high=self.num_arm_dofs,
                        size=(1,),
                        device=self.device,
                        generator=self.generator,
                    )
                    
                    # 将局部索引映射到全局关节索引
                    global_joint_index = self.arm_dof_indices[local_idx.item()]
                    # 为了保持张量形状一致，将单个索引扩展为长度为3的张量
                    self.arm_control_joint_idx[env_id] = torch.tensor([global_joint_index, global_joint_index, global_joint_index], device=self.device)
        

    def step(
        self,
        action: torch.Tensor,
        return_vis: bool = False,
        callback: Optional[Callable[[IsaacGymEnv], None]] = None,
    ):
        # 更新课程阶段
        self.update_curriculum(self.global_step)
        
        # 新增：阶段1不做机械臂目标，阶段2及以后才做
        def combined_cb(_: IsaacGymEnv):
            self._arm_swing_callback()
            if callback is not None:
                callback(self)
        # import pdb;pdb.set_trace()
        return super().step(action=action, return_vis=return_vis, callback=combined_cb)
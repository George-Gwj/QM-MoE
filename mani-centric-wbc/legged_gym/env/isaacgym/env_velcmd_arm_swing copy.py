from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np
from isaacgym import gymapi, gymtorch

from .env_add_baseinfo import IsaacGymEnv
from .state import EnvState, EnvSetup
from .control import Control
from .obs import EnvObservationAttribute, EnvSetupAttribute

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
        self.curriculum_stage = getattr(self.cfg.curriculum, 'stage', 2)  # 当前阶段
        self.curriculum_transition_steps = getattr(self.cfg.curriculum, 'transition_steps', 2000000)  # 阶段转换步数
        
        # 奖励权重（可从 cfg 读取，若不存在则使用默认）
        self.base_cmd_tracking_sigma: float = getattr(
            self.cfg.rewards, "base_cmd_tracking_sigma", 0.5
        )
        self.base_cmd_reward_scale: float = getattr(
            self.cfg.rewards, "base_cmd_tracking_reward_scale", 3.0
        )
        
        # wz跟踪奖励参数
        self.wz_tracking_sigma: float = getattr(
            self.cfg.rewards, "wz_tracking_sigma", 0.3
        )
        self.wz_tracking_reward_scale: float = getattr(
            self.cfg.rewards, "wz_tracking_reward_scale", 1.5
        )
        
        # 高度跟踪奖励参数
        self.height_tracking_sigma: float = getattr(
            self.cfg.rewards, "height_tracking_sigma", 0.05
        )
        self.height_tracking_reward_scale: float = getattr(
            self.cfg.rewards, "height_tracking_reward_scale", 2.0
        )
        
        # 强制忽略 task（例如末端跟踪），reward系数设为0
        self.task_reward_attenuation: float = 0.0
        
        # 目标关节角度跟踪奖励参数
        self.arm_target_tracking_sigma: float = getattr(
            self.cfg.rewards, "arm_target_tracking_sigma", 0.1
        )
        self.arm_target_tracking_reward_scale: float = getattr(
            self.cfg.rewards, "arm_target_tracking_reward_scale", 2.0
        )
        
        # 课程学习中的机械臂奖励缩放
        self.arm_reward_scale_by_stage = getattr(self.cfg.curriculum, 'arm_reward_scale_by_stage', True)

        # 机械臂参数 - 目标关节角度
        self.arm_dof_indices = self._find_arm_dof_indices()
        self.arm_target_pos: torch.Tensor = torch.zeros(
            (self.num_envs, len(self.arm_dof_indices)), dtype=torch.float32, device=self.device
        )
        
        # 机械臂直接控制参数
        self.arm_direct_control = getattr(self.cfg.curriculum, 'arm_direct_control', True)  # 是否直接控制机械臂
        
        # 机械臂随机姿态参数
        self.arm_random_pose_enabled = getattr(self.cfg.curriculum, 'arm_random_pose_enabled', True)
        self.arm_pose_range_scale = getattr(self.cfg.curriculum, 'arm_pose_range_scale', 0.5)  # 机械臂目标关节角度50%范围
        
        # 基座速度指令缩放参数
        self.base_vel_cmd_scale = getattr(self.cfg.curriculum, 'base_vel_cmd_scale', 0.0)  # 初始基座速度为0
        
        # 基于性能的课程学习参数
        self.performance_based_curriculum = getattr(self.cfg.curriculum, 'performance_based_curriculum', True)
        self.curriculum_eval_freq = getattr(self.cfg.curriculum, 'curriculum_eval_freq', 1000)  # 每1000步评估一次
        self.curriculum_eval_episodes = getattr(self.cfg.curriculum, 'curriculum_eval_episodes', 10)  # 评估10个episode
        
        # 课程学习阈值
        self.curriculum_thresholds = {
            'base_cmd_tracking_error': getattr(self.cfg.curriculum, 'base_cmd_tracking_error_threshold', 0.1),
            'arm_target_tracking_error': getattr(self.cfg.curriculum, 'arm_target_tracking_error_threshold', 0.05),
            'orientation_error': getattr(self.cfg.curriculum, 'orientation_error_threshold', 0.1),
        }
        
        # 性能历史记录
        self.performance_history = {
            'base_cmd_tracking_error': [],
            'arm_target_tracking_error': [],
            'orientation_error': [],
        }
        
        # 课程学习状态
        self.curriculum_stable_steps = 0  # 当前阶段稳定步数
        self.curriculum_min_stable_steps = getattr(self.cfg.curriculum, 'min_stable_steps', 2000)  # 最少稳定步数
        
        # Episode计数
        self.episode_count = 0  # 总episode计数
        self.curriculum_stage_start_episode = 0  # 当前阶段开始的episode
        self.curriculum_stage_start_step = 0  # 当前阶段开始的步数
        
        # 高度指令参数
        self.height_range: Tuple[float, float] = getattr(
            self.cfg.rewards, "height_range", [0.15, 0.3]
        )
        
        # 高度指令张量
        self.base_height_cmd: torch.Tensor = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )

    def _find_arm_dof_indices(self) -> List[int]:
        """
        找到所有机械臂关节对应的dof索引
        假设机械臂关节名称包含 'link' 或 'joint' 且不是腿部关节
        """
        arm_indices = []
        if not hasattr(self, "dof_names"):
            return arm_indices
        
        # 腿部关节名称模式（通常不包含在机械臂中）
        leg_joint_patterns = ['hip', 'thigh', 'calf', 'foot', 'leg']
        
        for idx, name in enumerate(self.dof_names):
            # 检查是否是机械臂关节（包含link或joint，且不是腿部关节）
            is_arm_joint = ('link' in name.lower() or 'joint' in name.lower()) and \
                          not any(pattern in name.lower() for pattern in leg_joint_patterns)
            
            if is_arm_joint:
                arm_indices.append(idx)
        
        # 如果没找到，尝试使用默认的机械臂关节索引（假设前12个是腿部，后面是机械臂）
        if not arm_indices and len(self.dof_names) > 12:
            # 假设从第12个关节开始是机械臂
            arm_indices = list(range(12, min(len(self.dof_names), 18)))  # 最多6个机械臂关节
        
        return arm_indices

    def update_curriculum(self, global_step: int):
        """根据性能指标更新课程阶段 - 基座速度指令和机械臂目标关节角度"""
        if not self.performance_based_curriculum:
            # 使用基于步数的传统课程学习
            self._update_curriculum_by_steps(global_step)
            return
        
        # 基于性能的课程学习
        if global_step % self.curriculum_eval_freq == 0 and global_step > 0:
            self._evaluate_performance()
            self._check_curriculum_transition()

    def _update_curriculum_by_steps(self, global_step: int):
        """传统的基于步数的课程学习"""
        # 阶段1：只训练四足站立，需要基座速度稳定和姿态稳定
        if self.curriculum_stage == 1 and global_step >= self.curriculum_transition_steps:
            self.curriculum_stage = 2
            self.arm_pose_range_scale = 0.5  # 机械臂目标关节角度50%范围
            self.base_vel_cmd_scale = 0.0    # 基座速度为0
            logging.info("课程学习：进入阶段2 - 基座速度为0，机械臂目标关节角度50%范围（开始机械臂训练）")
        
        # 阶段2：基座速度小范围，机械臂目标关节角度50%范围
        elif self.curriculum_stage == 2 and global_step >= self.curriculum_transition_steps * 2:
            self.curriculum_stage = 3
            self.arm_pose_range_scale = 0.5  # 机械臂目标关节角度50%范围
            self.base_vel_cmd_scale = 0.3    # 基座速度30%范围
            logging.info("课程学习：进入阶段3 - 基座速度30%范围，机械臂目标关节角度50%范围")
        
        # 阶段3：基座速度中等范围，机械臂目标关节角度50%范围
        elif self.curriculum_stage == 3 and global_step >= self.curriculum_transition_steps * 3:
            self.curriculum_stage = 4
            self.arm_pose_range_scale = 0.5  # 机械臂目标关节角度50%范围
            self.base_vel_cmd_scale = 0.6    # 基座速度60%范围
            logging.info("课程学习：进入阶段4 - 基座速度60%范围，机械臂目标关节角度50%范围")
        
        # 阶段4：基座速度大范围，机械臂目标关节角度50%范围
        elif self.curriculum_stage == 4 and global_step >= self.curriculum_transition_steps * 4:
            self.curriculum_stage = 5
            self.arm_pose_range_scale = 0.5  # 机械臂目标关节角度50%范围
            self.base_vel_cmd_scale = 1.0    # 基座速度100%范围
            logging.info("课程学习：进入阶段5 - 基座速度100%范围，机械臂目标关节角度50%范围")

    def _evaluate_performance(self):
        """评估当前性能指标"""
        # 计算基座速度跟踪误差
        vel_error = torch.norm(self.state.local_root_lin_vel[:, :2] - self.base_lin_vel_cmd[:, :2], dim=1)
        base_cmd_error = torch.mean(vel_error).item()
        
        # 计算机械臂目标跟踪误差 - 已禁用
        arm_target_error = 0.0  # 固定为0，不计算机械臂误差
        
        # 计算姿态误差
        orientation_error = torch.mean(torch.norm(self.state.local_root_gravity[:, :2], dim=1)).item()
        
        # 记录性能历史
        self.performance_history['base_cmd_tracking_error'].append(base_cmd_error)
        self.performance_history['arm_target_tracking_error'].append(arm_target_error)
        self.performance_history['orientation_error'].append(orientation_error)
        
        # 保持历史记录长度
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
        
        logging.info(f"性能评估 - 基座速度误差: {base_cmd_error:.4f}, 姿态误差: {orientation_error:.4f}, 当前速度指令: {self.base_lin_vel_cmd},当前阶段: {self.curriculum_stage}")

    def _check_curriculum_transition(self):
        """检查是否满足进入下一阶段的条件"""
        if len(self.performance_history['base_cmd_tracking_error']) < 10:
            return  # 需要足够的历史数据
        
        # 计算最近10次的平均性能
        recent_base_error = np.mean(self.performance_history['base_cmd_tracking_error'][-10:])
        recent_arm_error = np.mean(self.performance_history['arm_target_tracking_error'][-10:])
        recent_orientation_error = np.mean(self.performance_history['orientation_error'][-10:])
        
        # 检查是否满足当前阶段的性能要求
        stage_requirements_met = self._check_stage_requirements(
            recent_base_error, recent_arm_error, recent_orientation_error
        )
        
        if stage_requirements_met:
            self.curriculum_stable_steps += self.curriculum_eval_freq
        else:
            self.curriculum_stable_steps = 0  # 重置稳定步数
        
        # 检查是否可以进入下一阶段
        if (stage_requirements_met and 
            self.curriculum_stable_steps >= self.curriculum_min_stable_steps and
            self.curriculum_stage < 5):
            self._advance_curriculum_stage()

    def _check_stage_requirements(self, base_error, arm_error, orientation_error):
        """检查当前阶段是否满足性能要求"""
        if self.curriculum_stage == 1:
            # 阶段1：只训练四足站立，需要基座速度稳定和姿态稳定
            return (base_error < self.curriculum_thresholds['base_cmd_tracking_error'] and
                    orientation_error < self.curriculum_thresholds['orientation_error'])
        
        elif self.curriculum_stage == 2:
            # 阶段2：基座速度为0，机械臂目标跟踪和姿态稳定 - 已禁用机械臂误差检查
            return (base_error < self.curriculum_thresholds['base_cmd_tracking_error'] and
                    orientation_error < self.curriculum_thresholds['orientation_error'])
        
        elif self.curriculum_stage == 3:
            # 阶段3：基座速度30%范围，需要基座速度跟踪 - 已禁用机械臂误差检查
            return (base_error < self.curriculum_thresholds['base_cmd_tracking_error'] and
                    orientation_error < self.curriculum_thresholds['orientation_error'])
        
        elif self.curriculum_stage == 4:
            # 阶段4：基座速度60%范围，需要基座速度跟踪 - 已禁用机械臂误差检查
            return (base_error < self.curriculum_thresholds['base_cmd_tracking_error'] and
                    orientation_error < self.curriculum_thresholds['orientation_error'])
        
        return False

    def _advance_curriculum_stage(self):
        """进入下一个课程阶段"""
        old_stage = self.curriculum_stage
        self.curriculum_stage += 1
        self.curriculum_stable_steps = 0  # 重置稳定步数
        
        # 计算当前阶段经过的episode和步数
        episodes_in_stage = self.episode_count - self.curriculum_stage_start_episode
        steps_in_stage = self.global_step - self.curriculum_stage_start_step
        
        # 更新课程参数
        if self.curriculum_stage == 1:
            self.arm_pose_range_scale = 0.5
            self.base_vel_cmd_scale = 0.0
            stage_description = "基座速度为0，只训练四足站立（基座速度稳定+姿态稳定）"
        
        elif self.curriculum_stage == 2:
            self.arm_pose_range_scale = 0.5
            self.base_vel_cmd_scale = 0.0
            stage_description = "基座速度为0，机械臂直接传递采样关节角度（目标关节角度50%范围）"
        
        elif self.curriculum_stage == 3:
            self.arm_pose_range_scale = 0.5
            self.base_vel_cmd_scale = 0.3
            stage_description = "基座速度30%范围，机械臂直接传递采样关节角度（目标关节角度50%范围）"
        
        elif self.curriculum_stage == 4:
            self.arm_pose_range_scale = 0.5
            self.base_vel_cmd_scale = 0.6
            stage_description = "基座速度60%范围，机械臂直接传递采样关节角度（目标关节角度50%范围）"
        
        elif self.curriculum_stage == 5:
            self.arm_pose_range_scale = 0.5
            self.base_vel_cmd_scale = 1.0
            stage_description = "基座速度100%范围，机械臂直接传递采样关节角度（目标关节角度50%范围）"
        
        # 更新阶段开始计数
        self.curriculum_stage_start_episode = self.episode_count
        self.curriculum_stage_start_step = self.global_step
        
        # 只在课程转换时打印详细信息
        print("=" * 80)
        print(f"🎓 课程学习转换 - 从阶段{old_stage}进入阶段{self.curriculum_stage}")
        print(f"📊 阶段{old_stage}统计信息:")
        print(f"   - 经过的episode数: {episodes_in_stage}")
        print(f"   - 经过的步数: {steps_in_stage}")
        print(f"   - 总episode数: {self.episode_count}")
        print(f"   - 总步数: {self.global_step}")
        print(f"🎯 阶段{self.curriculum_stage}目标: {stage_description}")
        print(f"⚙️  课程参数:")
        print(f"   - 机械臂目标关节角度范围: {self.arm_pose_range_scale * 100}%")
        print(f"   - 基座速度指令缩放: {self.base_vel_cmd_scale * 100}%")
        print("=" * 80)
        
        # 同时记录到日志
        logging.info(f"课程学习：从阶段{old_stage}进入阶段{self.curriculum_stage}")
        logging.info(f"阶段{old_stage}统计 - Episodes: {episodes_in_stage}, Steps: {steps_in_stage}")
        logging.info(f"阶段{self.curriculum_stage}目标: {stage_description}")

    def _generate_target_arm_poses(self, env_ids: torch.Tensor) -> torch.Tensor:
        """为指定环境生成目标机械臂关节角度，基于课程学习"""
        if not self.arm_dof_indices or not self.arm_random_pose_enabled:
            return torch.zeros((len(env_ids), len(self.arm_dof_indices)), device=self.device)
        
        num_envs = len(env_ids)
        random_poses = torch.zeros((num_envs, len(self.arm_dof_indices)), device=self.device)
        
        # 检查关节限位是否已经初始化
        if not hasattr(self, 'curr_dof_pos_limits') or self.curr_dof_pos_limits is None:
            print("警告: 关节限位未初始化，使用默认随机姿态参数")
            # 使用默认范围
            for i, dof_idx in enumerate(self.arm_dof_indices):
                random_poses[:, i] = torch_rand_float(
                    lower=-1.0, upper=1.0, shape=(num_envs,), 
                    device=self.device, generator=self.generator
                )
        else:
            # 基于初始位置附近随机化（类似父类逻辑）
            for i, dof_idx in enumerate(self.arm_dof_indices):
                joint_limit_low = self.curr_dof_pos_limits[dof_idx, 0]
                joint_limit_high = self.curr_dof_pos_limits[dof_idx, 1]
                joint_range = joint_limit_high - joint_limit_low
                
                # 获取初始位置
                init_pos = self.init_dof_pos[0, dof_idx]  # 假设所有环境的初始位置相同
                
                # 根据课程阶段调整随机范围
                effective_range = joint_range * self.arm_pose_range_scale
                
                # 在初始位置附近生成随机噪声（类似父类逻辑）
                noise = (
                    self.dof_pos_reset_range_scale
                    * torch.randn(
                        num_envs,
                        device=self.device,
                        generator=self.generator,
                    )
                    * effective_range
                )
                
                # 基于初始位置 + 噪声生成随机角度
                random_poses[:, i] = init_pos + noise
                
                # 确保不超出关节限位
                random_poses[:, i] = torch.clamp(
                    random_poses[:, i], joint_limit_low, joint_limit_high
                )
        
        return random_poses


    # 自定义奖励函数
    def _reward_base_cmd_tracking(self, state: EnvState, control: Control):
        """基座速度指令跟踪奖励"""
        vel_error = state.local_root_lin_vel[:, :2] - self.base_lin_vel_cmd[:, :2]
        return torch.exp(-torch.sum(vel_error**2, dim=1) / self.base_cmd_tracking_sigma**2)

    def _reward_wz_tracking(self, state: EnvState, control: Control):
        """wz角速度指令跟踪奖励"""
        wz_error = state.local_root_ang_vel[:, 2] - self.base_ang_vel_cmd[:, 2]
        return torch.exp(-torch.square(wz_error) / self.wz_tracking_sigma**2)

    def _reward_height_tracking(self, state: EnvState, control: Control):
        """高度指令跟踪奖励"""
        height_error = state.root_pos[:, 2] - self.base_height_cmd
        return torch.exp(-torch.square(height_error) / self.height_tracking_sigma**2)

    def _reward_ang_vel_xy_penalty(self, state: EnvState, control: Control):
        """roll和pitch角速度惩罚"""
        return torch.sum(torch.square(state.local_root_ang_vel[:, :2]), dim=1)

    def _reward_orientation_penalty(self, state: EnvState, control: Control):
        """姿态惩罚（非水平基座）"""
        return torch.sum(torch.square(state.local_root_gravity[:, :2]), dim=1)

    def _reward_arm_target_tracking(self, state: EnvState, control: Control):
        """机械臂目标关节角度跟踪奖励 - 已禁用，机械臂直接传递采样关节角度"""
        # 机械臂现在直接传递采样的关节角度，不需要policy学习机械臂控制
        # 因此机械臂奖励始终为0
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def compute_reward(self, state: EnvState, control: Control):
        """重写奖励计算，遵循父类框架"""
        return_dict = {
            "total": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "env": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "constraint": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
            "task": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
        }
        
        # 1. 计算环境奖励（基础奖励）- 使用父类框架
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            return_dict[name] = (
                self.reward_functions[i](state=state, control=control)
                * self.reward_scales[name]
            )
            return_dict["total"] += return_dict[name]
            return_dict["env"] += return_dict[name]
        
        # 2. 计算约束奖励（保持原有）
        for constraint_name, constraint in self.constraints.items():
            constraint_rewards = {
                f"constraint/{constraint_name}/{k}": v
                for k, v in constraint.reward(state=state, control=control).items()
            }
            return_dict.update(constraint_rewards)
            return_dict["total"] += sum(constraint_rewards.values())
            return_dict["constraint"] += sum(constraint_rewards.values())
        
        # # 3. 计算任务奖励（保持原有）
        # for task_name, task in self.tasks.items():
        #     raw_task_rewards = task.reward(state=state, control=control)
        #     attenuated = {k: v * self.task_reward_attenuation for k, v in raw_task_rewards.items()}
        #     task_rewards = {
        #         f"task/{task_name}/{k}": v
        #         for k, v in attenuated.items()
        #     }
        #     return_dict.update(task_rewards)
        #     return_dict["total"] += sum(task_rewards.values())
        #     return_dict["task"] += sum(task_rewards.values())
        
        # 4. 后处理（遵循父类逻辑）
        if self.cfg.rewards.only_positive_rewards:
            return_dict["total"][:] = torch.clip(return_dict["total"][:], min=0.0)
        return_dict["task_to_env_ratio"] = return_dict["task"].abs() / (
            return_dict["env"].abs() + 1e-10
        )
        return_dict["task_to_constraint_ratio"] = return_dict["task"].abs() / (
            return_dict["constraint"].abs() + 1e-10
        )
        
        # 5. 返回（遵循父类格式）
        return {f"reward/{k}": v * self.reward_dt_scale for k, v in return_dict.items()}

    def get_observations(
        self,
        state: EnvState,
        setup: EnvSetup,
        state_obs: Dict[str, EnvObservationAttribute],
        setup_obs: Dict[str, EnvSetupAttribute],
    ):
        """重写观测方法，添加高度指令到观测中"""
        obs_attrs = []
        for name, obs_attr in state_obs.items():
            value = obs_attr(struct=state, generator=self.generator)
            assert value.shape[-1] == obs_attr.dim
            obs_attrs.append(value)
        state_obs_tensor = torch.cat(
            obs_attrs,
            dim=1,
        )

        if len(self.tasks) > 0:
            all_task_obs = []
            for k, task in self.tasks.items():
                task_obs = task.observe(state=state)
                all_task_obs.append(task_obs)
            task_obs_tensor = torch.cat(
                all_task_obs,
                dim=1,
            )
        else:
            task_obs_tensor = torch.zeros(
                (self.num_envs, 0), dtype=torch.float, device=self.device
            )
        if len(setup_obs) > 0:
            obs_attrs = []
            for k, obs_attr in setup_obs.items():
                value = obs_attr(struct=setup, generator=self.generator).reshape(
                    self.num_envs, -1
                )
                assert value.shape[-1] == obs_attr.dim
                obs_attrs.append(value)
            setup_obs_tensor = torch.cat(obs_attrs, dim=1)
        else:
            setup_obs_tensor = torch.zeros(
                (self.num_envs, 0), dtype=torch.float, device=self.device
            )

        # 添加新的自定义观测
        additional_obs_tensor = self._get_additional_observations(state)

        return torch.cat(
            (
                setup_obs_tensor,
                state_obs_tensor,
                task_obs_tensor,
                # 追加 baselink 速度指令（vx, vy, vz），仅作为观测
                self.base_lin_vel_cmd,
                # 追加 baselink 角速度指令（wx, wy, wz），仅作为观测
                self.base_ang_vel_cmd,
                # 追加 baselink 高度指令（z），仅作为观测
                self.base_height_cmd.unsqueeze(-1),
                additional_obs_tensor,  # 新增的观测
                self.ctrl.action,
            ),
            dim=1,
        )

    def reset_idx(self, env_ids):
        """重写 reset_idx 方法，添加课程学习的基座速度指令和目标关节角度生成"""
        if len(env_ids) == 0:
            return

        # 1. 使用父类的标准重置逻辑
        super().reset_idx(env_ids)
        
        # 2. 根据课程学习重新设置基座速度指令
        if len(env_ids) > 0:
            num = len(env_ids)
            # 基座线速度指令（根据课程学习缩放）
            vx_vy = torch_rand_float(
                lower=-1.0,
                upper=1.0,
                shape=(num, 2),
                device=self.device,
                generator=self.generator,
            ) * self.base_vel_cmd_scale
            self.base_lin_vel_cmd[env_ids, 0:2] = vx_vy
            self.base_lin_vel_cmd[env_ids, 2] = 0.0
            
            # 基座角速度指令（根据课程学习缩放）
            wz = torch_rand_float(
                lower=-0.5,
                upper=0.5,
                shape=(num, 1),
                device=self.device,
                generator=self.generator,
            ) * self.base_vel_cmd_scale
            wx_wy_wz = torch.zeros((num, 3), device=self.device)
            wx_wy_wz[:, 2] = wz[:, 0]
            self.base_ang_vel_cmd[env_ids] = wx_wy_wz
        
        # 3. 生成目标机械臂关节角度（课程学习）
        if len(env_ids) > 0 and self.arm_dof_indices and self.arm_random_pose_enabled:
            target_arm_poses = self._generate_target_arm_poses(env_ids)
            self.arm_target_pos[env_ids] = target_arm_poses
        
        # 4. 设置高度指令（课程学习）
        if len(env_ids) > 0:
            height_cmd = torch_rand_float(
                lower=self.height_range[0],
                upper=self.height_range[1],
                shape=(len(env_ids),),
                device=self.device,
                generator=self.generator,
            )
            self.base_height_cmd[env_ids] = height_cmd

            
    def step(
        self,
        action: torch.Tensor,
        return_vis: bool = False,
        callback: Optional[Callable[[IsaacGymEnv], None]] = None,
    ):
        # 更新课程阶段
        self.update_curriculum(self.global_step)
        
        # 如果启用机械臂直接控制，需要修改action
        if self.arm_direct_control and self.arm_dof_indices:
            # 将机械臂采样的目标关节角度直接传递给action
            # 假设action的前面部分是腿部控制，后面部分是机械臂控制
            if action.shape[1] >= len(self.arm_dof_indices):
                # 如果action包含机械臂部分，替换为采样的目标关节角度
                action[:, -len(self.arm_dof_indices):] = self.arm_target_pos
            else:
                # 如果action不包含机械臂部分，需要扩展action
                leg_action = action
                full_action = torch.cat([leg_action, self.arm_target_pos], dim=1)
                action = full_action
        
        # 调用父类的step方法
        return super().step(action=action, return_vis=return_vis, callback=callback)
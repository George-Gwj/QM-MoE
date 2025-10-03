from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch3d.transforms as p3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_mul, to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.env.isaacgym.constraints import Constraint
from legged_gym.env.isaacgym.control import Control, PDController
from legged_gym.env.isaacgym.obs import EnvObservationAttribute, EnvSetupAttribute
from legged_gym.env.isaacgym.state_multi_actor import EnvSetup, EnvState
# from legged_gym.env.isaacgym.task import Task
from legged_gym.env.isaacgym.task_multi_actor import Task
from legged_gym.env.isaacgym.terrain import TerrainPerlin
from legged_gym.env.isaacgym.utils import quat_apply_yaw, torch_rand_float
from legged_gym.env.obs import ObservationAttribute
from legged_gym.rsl_rl.env import VecEnv

# ==== CBF 控制器导入（假设你有实现）====
# 暂时注释掉CBF控制器导入，先专注于障碍物功能
# try:
#     from legged_gym.env.isaacgym.cbf_controller import IsaacGymCBF
# except ImportError:
#     IsaacGymCBF = None  # 如果没有实现，后续会报错
IsaacGymCBF = None

# ==== 障碍物管理类 ====
class ObstacleManager:
    """IsaacGym障碍物管理器，对应MuJoCo中的障碍物"""
    
    def __init__(self, gym, sim, device, num_envs):
        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        
        # 障碍物句柄 - 现在每个环境都有障碍物
        self.obstacle_handles = {}  # {obstacle_name: [env0_handle, env1_handle, ...]}
        self.obstacle_bodies = {}
        
        # 障碍物参数（简化版本：只保留rectangle和第一个capsule）
        self.obstacle_params = {
            'capsule': [
                {'name': 'capsule', 'pos': [-0.05, 2.5, 0.68], 'size': [0.05, 0.3], 'euler': [0, -1.570796327, 0], 'color': [0, 0.9, 0, 0.5]},
            ],
            'rectangle': [
                {'name': 'rectangle', 'pos': [-0.05, 1.45, 1.8], 'size': [0.45, 0.3, 0.05], 'color': [0, 0, 0.9, 0.5]},
            ]
        }
        
        # 障碍物速度控制参数
        self.obstacle_velocities = torch.zeros(num_envs, 2, 3, device=device)  # 2个障碍物，每个3D速度
        self.obstacle_positions = torch.zeros(num_envs, 2, 3, device=device)   # 2个障碍物，每个3D位置
        
        # 创建障碍物 - 延迟创建，等待环境准备好
        # 注意：这里不能立即创建，因为gym和sim还没有初始化
        # self._create_obstacles()
        
    def _create_obstacles(self):
        """在IsaacGym中创建障碍物"""
        obstacle_count = 0
        
        # 创建球体障碍物
        for sphere in self.obstacle_params['sphere']:
            self._create_sphere_obstacle(sphere, obstacle_count)
            obstacle_count += 1
            
        # 创建胶囊体障碍物
        for capsule in self.obstacle_params['capsule']:
            self._create_capsule_obstacle(capsule, obstacle_count)
            obstacle_count += 1
            
        # 创建矩形障碍物
        for rectangle in self.obstacle_params['rectangle']:
            self._create_rectangle_obstacle(rectangle, obstacle_count)
            obstacle_count += 1
            
        print(f"✓ 成功创建 {obstacle_count} 个障碍物类型，每个环境 {self.num_envs} 个实例")
            
    def _create_sphere_obstacle(self, params, obstacle_id):
        """创建球体障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建球体资产 - 使用正确的IsaacGym API
                sphere_asset = self.gym.create_sphere(self.sim, params['size'])
                
                # 设置球体属性，确保它是动态的
                # self.gym.set_asset_density(sphere_asset, 1.0)  # 这个方法不存在
                
                # 创建球体实例 - 添加到特定环境
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                
                # 创建障碍物actor并添加到环境
                # 注意：这里需要使用环境句柄，而不是整数索引
                # 但是ObstacleManager没有访问环境句柄的权限
                # 所以我们需要修改设计，让障碍物创建在环境创建完成后进行
                print(f"⚠ 障碍物创建需要环境句柄，当前无法访问")
                self.obstacle_handles[params['name']].append(None)
                continue
                
                # 确保障碍物是动态的（不是静态的）
                # 设置刚体属性
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, sphere_handle)
                rigid_body_props[0].mass = 1.0  # 设置质量
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION  # 允许自碰撞
                self.gym.set_actor_rigid_body_properties(self.sim, sphere_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(sphere_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建球体障碍物 {params['name']} 失败: {e}")
                # 添加None占位符以保持索引一致
                self.obstacle_handles[params['name']].append(None)
            
    def _create_capsule_obstacle(self, params, obstacle_id):
        """创建胶囊体障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建胶囊体资产 - 使用正确的IsaacGym API
                capsule_asset = self.gym.create_capsule(
                    self.sim, 
                    params['size'][0], 
                    params['size'][1]
                )
                
                # 设置胶囊体属性
                # self.gym.set_asset_density(capsule_asset, 1.0)  # 这个方法不存在
                
                # 创建胶囊体实例 - 添加到特定环境
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                if 'euler' in params:
                    start_pose.r = gymapi.Quat.from_euler_zyx(*params['euler'])
                
                capsule_handle = self.gym.create_actor(
                    self.sim, 
                    capsule_asset, 
                    start_pose, 
                    f"{params['name']}_{env_id}", 
                    env_id, 
                    0
                )
                
                # 确保障碍物是动态的（不是静态的）
                # 设置刚体属性
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, capsule_handle)
                rigid_body_props[0].mass = 1.0  # 设置质量
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION  # 允许自碰撞
                self.gym.set_actor_rigid_body_properties(self.sim, capsule_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(capsule_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建胶囊体障碍物 {params['name']} 失败: {e}")
                # 添加None占位符以保持索引一致
                self.obstacle_handles[params['name']].append(None)
            
    def _create_rectangle_obstacle(self, params, obstacle_id):
        """创建矩形障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建矩形资产 - 使用正确的IsaacGym API
                box_asset = self.gym.create_box(
                    self.sim, 
                    params['size'][0], 
                    params['size'][1], 
                    params['size'][2]
                )
                
                # 设置矩形属性
                # self.gym.set_asset_density(box_asset, 1.0)  # 这个方法不存在
                
                # 创建矩形实例 - 添加到特定环境
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                
                box_handle = self.gym.create_actor(
                    self.sim, 
                    box_asset, 
                    start_pose, 
                    f"{params['name']}_{env_id}", 
                    env_id, 
                    0
                )
                
                # 确保障碍物是动态的（不是静态的）
                # 设置刚体属性
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, box_handle)
                rigid_body_props[0].mass = 1.0  # 设置质量
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION  # 允许自碰撞
                self.gym.set_actor_rigid_body_properties(self.sim, box_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(box_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建矩形障碍物 {params['name']} 失败: {e}")
                # 添加None占位符以保持索引一致
                self.obstacle_handles[params['name']].append(None)
            
    def update_obstacle_motion(self, dt: float):
        """更新障碍物运动（简化版本，障碍物保持静态）"""
        # 采用"只创建，不管理状态"的方式，障碍物保持静态
        # 如果需要动态障碍物，可以在这里添加简单的运动逻辑
        pass
        
    def _apply_obstacle_positions(self):
        """将计算的位置应用到IsaacGym仿真中（简化版本）"""
        # 采用"只创建，不管理状态"的方式，障碍物保持静态
        # 如果需要动态障碍物，可以在这里添加位置更新逻辑
        pass
        
    def get_obstacle_info(self) -> Dict[str, torch.Tensor]:
        """获取障碍物信息，供CBF控制器使用（简化版本，只返回静态参数）"""
        # 由于我们采用"只创建，不管理状态"的方式，这里返回静态参数
        return {
            'capsule': [torch.tensor(capsule['pos'], device=self.device) for capsule in self.obstacle_params['capsule']],
            'rectangle': [torch.tensor(rect['pos'], device=self.device) for rect in self.obstacle_params['rectangle']],
            'velocities': None  # 不管理速度
        }

PartialTask = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Task]
PartialConstraint = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Constraint]

import pytorch3d.transforms as p3d

class IsaacGymEnv(VecEnv):
    def __init__(
        self,
        cfg,
        sim_params,
        sim_device,
        headless,
        controller: PDController,
        state_obs: Dict[str, EnvObservationAttribute],
        setup_obs: Dict[str, EnvSetupAttribute],
        privileged_state_obs: Dict[str, EnvObservationAttribute],
        privileged_setup_obs: Dict[str, EnvSetupAttribute],
        tasks: Dict[str, PartialTask],
        constraints: Dict[str, PartialConstraint],
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
        # 初始化障碍物相关属性（必须在create_sim之前）
        self.obstacle_manager = None
        self.obstacles = None
        self.obs_v = None
        
        # 障碍物句柄和参数 - 延迟初始化，等待num_envs设置
        self.obstacle_handles = {}  # {obstacle_name: [env0_handle, env1_handle, ...]}
        self.obstacle_positions = None  # 将在num_envs设置后初始化
        self.obstacle_velocities = None  # 将在num_envs设置后初始化
        
        # 障碍物参数（对应MuJoCo XML中的设置）
        self.obstacle_params = {
            'capsule': [
                {'name': 'capsule', 'pos': [-0.05, 2.5, 0.68], 'size': [0.05, 0.3], 'euler': [0, -1.570796327, 0], 'color': [0, 0.9, 0, 0.5]},
            ],
            'rectangle': [
                {'name': 'rectangle', 'pos': [-0.05, 1.45, 1.8], 'size': [0.45, 0.3, 0.05], 'color': [0, 0, 0.9, 0.5]},
            ]
        }
        
        self.dof_pos_reset_range_scale = dof_pos_reset_range_scale

        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = debug_viz
        self.controller = controller
        self.controller.kp = self.controller.kp.repeat(cfg.env.num_envs, 1)
        self.controller.kd = self.controller.kd.repeat(cfg.env.num_envs, 1)
        self.init_kp = self.controller.kp.clone()
        self.init_kd = self.controller.kd.clone()
        self.gym_dt = (
            np.mean(self.controller.decimation_count_range) * self.sim_params.dt
        )
        self.reward_scales = self.cfg.rewards.scales
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.gym_dt)
        self.env_spacing = env_spacing
        self.dense_rewards = dense_rewards
        self.reward_dt_scale = self.sim_params.dt
        if not self.dense_rewards:
            self.reward_dt_scale *= np.mean(self.controller.decimation_count_range)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.gym_dt
        )
        self.cfg.domain_rand.transport_interval = np.ceil(
            self.cfg.domain_rand.transport_interval_s / self.gym_dt
        )

        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.sim_device = sim_device
        sim_device_type = "cuda" if "cuda" in self.sim_device else "cpu"
        if sim_device_type == "cuda":
            self.sim_device_id = int(self.sim_device.split(":")[1])
        else:
            self.sim_device_id = -1
        self.headless = headless

        if sim_device_type == "cuda":
            self.device: str = self.sim_device
            self.sim_params.use_gpu_pipeline = True
            self.sim_params.physx.use_gpu = True
        else:
            self.device: str = "cpu"
            self.sim_params.use_gpu_pipeline = False
            self.sim_params.physx.use_gpu = False
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # graphics device for rendering, -1 for no rendering
        if not attach_camera and headless:
            self.graphics_device_id = -1
        else:
            if graphics_device_id is None:
                self.graphics_device_id = self.sim_device_id
            else:
                self.graphics_device_id = graphics_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        # 现在可以初始化障碍物张量了
        self.obstacle_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)   # 2个障碍物，每个3D位置
        self.obstacle_velocities = torch.zeros(self.num_envs, 2, 3, device=self.device)  # 2个障碍物，每个3D速度

        self.init_dof_pos = (
            init_dof_pos if init_dof_pos is not None else self.controller.offset
        )
        self.init_dof_pos = self.init_dof_pos[None, :].to(self.device)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.state = EnvState.initialize(
            gym=self.gym,
            sim=self.sim,
            device=self.device,
            terrain_heights=(
                None
                if self.cfg.terrain.mode in {"none", "plane"}
                else torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.float
                )
            ),
        )
        # print body indices
        rb_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0]
        )
        for i, rb_name in enumerate(rb_names):
            logging.info(f"[{i:02d}] {rb_name}")

        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.actor_handles[0])
        for i, dof_name in enumerate(dof_names):
            logging.info(f"[{i:02d}] {dof_name}")

        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        self.ctrl = Control(
            buffer=torch.zeros(
                (self.num_envs, ctrl_buf_len, self.num_actions),
                dtype=torch.float,
                device=self.device,
            ),
            torque=torch.zeros(
                (self.num_envs, self.num_actions),
                dtype=torch.float,
                device=self.device,
            ),
        )
        self.max_action_value = max_action_value
        if ctrl_delay is not None:
            assert torch.allclose(
                torch.round(ctrl_delay / self.sim_params.dt),
                ctrl_delay / self.sim_params.dt,
            ), "ctrl_delay must be a multiple of the simulation dt"
            assert (ctrl_delay >= 0).all(), "ctrl_delay can't be negative"
            self.ctrl_delay_steps = torch.round(ctrl_delay / self.sim_params.dt)
        else:
            self.ctrl_delay_steps = torch.zeros(self.num_actions, device=self.device)

        # initialize some data used later on
        self.global_step = 0
        self.extras = {}
        self.state_obs = {
            k: v
            for k, v in sorted(state_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.setup_obs = {
            k: v
            for k, v in sorted(setup_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.privileged_state_obs = {
            k: v
            for k, v in sorted(privileged_state_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.privileged_setup_obs = {
            k: v
            for k, v in sorted(privileged_setup_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.tasks = {
            k: v(self.gym, self.sim, self.device, self.generator)
            for k, v in tasks.items()
            if type(v) is functools.partial
        }
        self.constraints = {
            k: v(self.gym, self.sim, self.device, self.generator)
            for k, v in constraints.items()
            if type(v) is functools.partial
        }
        self._prepare_reward_function()

        # attach camera to last environment

        self.vis_env = self.envs[0]
        self.vis_cam_handle = None
        if attach_camera:
            cam_props = gymapi.CameraProperties()
            cam_props.horizontal_fov = 70.0
            cam_props.far_plane = 10.0
            cam_props.near_plane = 1e-2
            cam_props.height = vis_resolution[0]
            cam_props.width = vis_resolution[1]
            cam_props.enable_tensors = self.device != "cpu"
            cam_props.use_collision_geometry = False

            self.vis_cam_handle = self.gym.create_camera_sensor(self.vis_env, cam_props)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(1.6, 1.4, 0.8)
            local_transform.r = gymapi.Quat.from_euler_zyx(3.141592653589793, 2.8, 0.8)
            self.gym.attach_camera_to_body(
                self.vis_cam_handle,
                self.vis_env,
                self.actor_handles[0],
                local_transform,
                gymapi.FOLLOW_POSITION,
            )
        assert not self.state.isnan()
        self.obs_history = torch.zeros(
            (self.num_envs, obs_history_len, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_history_len = obs_history_len

        # baselink 速度指令缓存：[vx, vy, vz]，仅用于观测，不在本文件内产生控制作用
        self.base_lin_vel_cmd = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        
        # baselink 角速度指令缓存：[wx, wy, wz]，仅用于观测，不在本文件内产生控制作用
        self.base_ang_vel_cmd = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

        # ====== CBF 控制器初始化 ======
        # 暂时注释掉CBF控制器，先专注于障碍物功能
        # if IsaacGymCBF is not None:
        #     self.cbf_controller = IsaacGymCBF(
        #         num_envs=self.num_envs,
        #         device=self.device,
        #         obstacle_type_num=[2, 6, 1],  # 2球6胶囊1矩形，可根据实际调整
        #         T_step=self.gym_dt,
        #         CBF_mode='11',  # 鲁棒+动态CBF
        #         use_statistic_obstacle=True
        #     )
        #     # 设置正确的控制维度
        #     self.cbf_controller.set_control_dimension(self.num_actions)
        # else:
        #     self.cbf_controller = None
        self.cbf_controller = None
        
        # ====== 障碍物运动模式初始化 ======
        # 默认设置为静态障碍物
        self.set_obstacle_motion_pattern("static")

        # 每个环境中障碍物数量的变量
        self.num_obstacles_per_env = 2



    def update_obstacles(self):
        """更新障碍物位置和速度
        这里实现障碍物的动态更新逻辑
        可以是随机运动、预定义轨迹等
        """
        # TODO: 实现障碍物动态更新逻辑
        pass

    @property
    def episode_step(self) -> torch.Tensor:
        return (self.state.episode_time / self.gym_dt).long()

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def render(self, sync_frame_time=False):
        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        # step graphics
        self.gym.step_graphics(self.sim)
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # fetch results
            # step graphics
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)
        if self.vis_cam_handle is None:
            raise RuntimeError("No camera attached")
        env = self.vis_env
        rgb = self.gym.get_camera_image(
            self.sim, env, self.vis_cam_handle, gymapi.IMAGE_COLOR
        )
        rgb = rgb.reshape(rgb.shape[0], -1, 4)
        return rgb[..., :3]

    def step(
        self,
        action: torch.Tensor,
        return_vis: bool = False,
        callback: Optional[Callable[[IsaacGymEnv]]] = None,
    ):
        """
        Apply actions, simulate, call

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        info = {}

        # ===== CBF 安全动作修正 =====
        # 暂时注释掉CBF控制器，先专注于障碍物功能
        # safe_action = action
        # if self.cbf_controller is not None and self.obstacles is not None and self.obs_v is not None:
        #     try:
        #         # 获取当前状态
        #         current_state = self._get_current_state()
        #         # 计算CBF约束
        #         cbf_constraints = self.cbf_controller.compute_cbf_constraints(
        #         #     state=current_state,
        #         #     obstacles=self.obstacles,
        #         #     obs_v=self.obs_v
        #         # )
        #         # 求解QP优化
        #         safe_action = self.cbf_controller.solve_qp_optimization(
        #         #     target_action=action,
        #         #     cbf_constraints=cbf_constraints
        #         # )
        #         # 使用安全动作
        #         action = safe_action
        #     except Exception as e:
        #         print(f"⚠ CBF控制器运行失败: {e}")
        #         safe_action = action

        self.ctrl.push(
            torch.clip(action, -self.max_action_value, self.max_action_value).to(
                self.device
            )
        )
        reward = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        # step physics and render each frame
        rendering = self.viewer is not None or return_vis

        # ===== 障碍物管理 =====
        # 更新障碍物的位置和速度状态
        self.update_obstacle_states()
        
        # 应用障碍物运动模式
        self._apply_obstacle_motion(self.gym_dt)

        if rendering and self.debug_viz:
            self.visualize(vis_env_ids=[0])  # the rendering env

        if return_vis and self.vis_cam_handle is not None:
            vis = self.render(sync_frame_time=False)
            if vis is not None:
                info["vis"] = vis
        decimation_count = self.controller.decimation_count
        for decimation_step in range(decimation_count):
            # handle delay by indexing into the buffer of past targets
            # since new actions are pushed to the front of the buffer,
            # the current target is further back in the buffer for larger
            # delays.
            curr_target_idx = torch.ceil(
                ((self.ctrl_delay_steps - decimation_step)) / decimation_count
            ).long()
            assert (curr_target_idx >= 0).all()
            
            # 获取当前要使用的action
            current_action = self.ctrl.buffer.permute(2, 1, 0)[
                torch.arange(self.num_actions, device=self.device),
                curr_target_idx,
                :,
            ].permute(1, 0)
            
            # 调用callback并获取修改后的action
            modified_action = None
            if callback is not None:
                modified_action = callback(self, current_action)
            
            # 使用修改后的action（如果有的话）
            if modified_action is not None:
                action_to_use = modified_action
            else:
                action_to_use = current_action
            
            self.ctrl.torque = self.controller(
                action=action_to_use,
                state=self.state,
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.ctrl.torque)
            )
            self.state.step(gym=self.gym, sim=self.sim)
            if self.cfg.terrain.mode in {"perlin"}:
                self.state.measured_terrain_heights = self._get_heights()
            if self.dense_rewards or decimation_step == decimation_count - 1:
                for task_name, task in self.tasks.items():
                    for k, v in task.step(state=self.state, control=self.ctrl).items():
                        stat_key = f"task/{task_name}/{k}"
                        if stat_key not in info:
                            info[stat_key] = v
                        else:
                            # compute mean (of decimation steps) in place
                            info[stat_key] = (info[stat_key] * decimation_step + v) / (
                                decimation_step + 1
                            )
                for constraint_name, constraint in self.constraints.items():
                    for k, v in constraint.step(
                        state=self.state, control=self.ctrl
                    ).items():
                        stat_key = f"constraint/{constraint_name}/{k}"
                        if stat_key not in info:
                            info[stat_key] = v
                        else:
                            # compute mean (of decimation steps) in place
                            info[stat_key] = (info[stat_key] * decimation_step + v) / (
                                decimation_step + 1
                            )
                reward_terms = self.compute_reward(state=self.state, control=self.ctrl)
                reward += reward_terms["reward/total"]
                for k, v in reward_terms.items():
                    if k in info:
                        info[k] += v
                    else:
                        info[k] = v
        self.global_step += 1
        if self.cfg.domain_rand.push_robots and (
            self.global_step % self.cfg.domain_rand.push_interval == 0
        ):
            """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
            self.state.root_state[:, 7:13] = torch_rand_float(
                -self.cfg.domain_rand.max_push_vel,
                self.cfg.domain_rand.max_push_vel,
                (self.num_envs, 6),
                device=self.device,
                generator=self.generator,
            )  # lin vel x/y/z, ang vel x/y/z
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.state.root_state)
            )
        if self.cfg.domain_rand.transport_robots and (
            self.global_step % self.cfg.domain_rand.transport_interval == 0
        ):
            """Randomly transports the robots to a new location"""
            self.state.root_state[:, 0:3] += (
                torch.randn(
                    self.num_envs,
                    3,
                    device=self.device,
                    generator=self.generator,
                )
                * self.cfg.domain_rand.transport_pos_noise_std
            )
            euler_noise = (
                torch.randn(
                    self.num_envs,
                    3,
                    device=self.device,
                    generator=self.generator,
                )
                * self.cfg.domain_rand.transport_euler_noise_std
            )
            quat_wxyz_transport = p3d.matrix_to_quaternion(
                p3d.euler_angles_to_matrix(euler_noise, "XYZ")
            )
            self.state.root_state[:, 3:7] = quat_mul(
                self.state.root_state[:, 3:7],
                quat_wxyz_transport[..., [1, 2, 3, 0]],  # reorder to xyzw
            )

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.state.root_state)
            )
        self.check_termination(state=self.state, control=self.ctrl)
        for constraint_name, constraint in self.constraints.items():
            info[f"constraint/{constraint_name}/termination"] = (
                constraint.check_termination(state=self.state, control=self.ctrl)
            )
            self.reset_buf |= info[f"constraint/{constraint_name}/termination"]
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        obs = self.get_observations(
            state=self.state,
            setup=self.setup,
            state_obs=self.state_obs,
            setup_obs=self.setup_obs,
        )
        self.obs_history = torch.cat(
            (self.obs_history[:, 1:, :], obs.unsqueeze(1)), dim=1
        )
        privileged_obs = self.get_observations(
            state=self.state,
            setup=self.setup,
            state_obs=self.privileged_state_obs,
            setup_obs=self.privileged_setup_obs,
        )

        info.update(self.extras)
        return (
            self.obs_history.view(self.num_envs, -1),
            privileged_obs,
            reward,
            self.reset_buf,
            info,
        )

    # def _get_current_state(self) -> torch.Tensor:
    #     """获取当前系统状态
    #     组合状态向量：[base_pos, base_quat, joint_angles]
    #     """
    #     # 使用所有可用的关节角度，而不是硬编码的5个
    #     num_joints = self.state.dof_pos.shape[1]  # 动态获取关节数量
    #     state = torch.cat([
    #         self.state.root_pos,         # [num_envs, 3]
    #         self.state.root_xyzw_quat,   # [num_envs, 4]
    #         self.state.dof_pos,          # [num_envs, num_joints]
    #     ], dim=1)
    #     return state

    def check_termination(self, state: EnvState, control: Control):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(
                state.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_step > self.max_episode_length
        )  # no terminal reward for time-outs
        # also reset if robot walks off the safe bounds
        walked_off_safe_bounds = torch.logical_or(
            (self.state.root_pos[:, :2] < self.safe_bounds[None, :, 0]).any(dim=1),
            (self.state.root_pos[:, :2] > self.safe_bounds[None, :, 1]).any(dim=1),
        )
        self.time_out_buf |= walked_off_safe_bounds
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        for task in self.tasks.values():
            task.reset_idx(env_ids)
        for constraint in self.constraints.values():
            constraint.reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0

        # 采样 baselink 速度指令（vx∈[-1,1], vy∈[-1,1], vz=0），仅加入观测使用
        num = len(env_ids)
        vx_vy = torch_rand_float(
            lower=-1.0,
            upper=1.0,
            shape=(num, 2),
            device=self.device,
            generator=self.generator,
        )
        self.base_lin_vel_cmd[env_ids, 0:2] = vx_vy
        self.base_lin_vel_cmd[env_ids, 2] = 0.0
        
        # 采样 baselink 角速度指令（wx=0, wy=0, wz∈[-0.5,0.5]），仅加入观测使用
        wz = torch_rand_float(
            lower=-0.5,
            upper=0.5,
            shape=(num, 1),
            device=self.device,
            generator=self.generator,
        )
        wx_wy_wz = torch.zeros((num, 3), device=self.device)
        wx_wy_wz[:, 2] = wz[:, 0]
        self.base_ang_vel_cmd[env_ids] = wx_wy_wz

        # reset controllers
        if self.cfg.domain_rand.randomize_pd_params:
            self.controller.kp[env_ids] = (
                torch_rand_float(
                    lower=self.cfg.domain_rand.kp_ratio_range[0],
                    upper=self.cfg.domain_rand.kp_ratio_range[1],
                    shape=(len(env_ids), self.controller.control_dim),
                    device=self.device,
                    generator=self.generator,
                )
                * self.init_kp[env_ids]
            )
            self.controller.kd[env_ids] = (
                torch_rand_float(
                    lower=self.cfg.domain_rand.kd_ratio_range[0],
                    upper=self.cfg.domain_rand.kd_ratio_range[1],
                    shape=(len(env_ids), self.controller.control_dim),
                    device=self.device,
                    generator=self.generator,
                )
                * self.init_kd[env_ids]
            )
            self.setup.kp[env_ids] = self.controller.kp[env_ids]
            self.setup.kd[env_ids] = self.controller.kd[env_ids]

        # reset buffers
        self.ctrl.buffer[env_ids] = 0.0
        self.state.episode_time[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # 重置障碍物状态
        if hasattr(self, 'obstacle_positions') and self.obstacle_positions is not None:
            self.obstacle_positions[env_ids] = 0.0
        if hasattr(self, 'obstacle_velocities') and self.obstacle_velocities is not None:
            self.obstacle_velocities[env_ids] = 0.0
            
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self, state: EnvState, control: Control):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        return_dict = {
            "total": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "env": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "constraint": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "task": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
        }
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            return_dict[name] = (
                self.reward_functions[i](state=state, control=control)
                * self.reward_scales[name]
            )
            return_dict["total"] += return_dict[name]
            return_dict["env"] += return_dict[name]
        for constraint_name, constraint in self.constraints.items():
            constraint_rewards = {
                f"constraint/{constraint_name}/{k}": v
                for k, v in constraint.reward(state=state, control=control).items()
            }
            return_dict.update(constraint_rewards)
            return_dict["total"] += sum(constraint_rewards.values())
            return_dict["constraint"] += sum(constraint_rewards.values())
        for task_name, task in self.tasks.items():
            task_rewards = {
                f"task/{task_name}/{k}": v
                for k, v in task.reward(state=state, control=control).items()
            }
            return_dict.update(task_rewards)
            return_dict["total"] += sum(task_rewards.values())
            return_dict["task"] += sum(task_rewards.values())
        if self.cfg.rewards.only_positive_rewards:
            return_dict["total"][:] = torch.clip(return_dict["total"][:], min=0.0)
        return_dict["task_to_env_ratio"] = return_dict["task"].abs() / (
            return_dict["env"].abs() + 1e-10
        )
        return_dict["task_to_constraint_ratio"] = return_dict["task"].abs() / (
            return_dict["constraint"].abs() + 1e-10
        )
        return {f"reward/{k}": v * self.reward_dt_scale for k, v in return_dict.items()}

    def get_observations(
        self,
        state: EnvState,
        setup: EnvSetup,
        state_obs: Dict[str, EnvObservationAttribute],
        setup_obs: Dict[str, EnvSetupAttribute],
    ):
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
                additional_obs_tensor,  # 新增的观测
                self.ctrl.action,
            ),
            dim=1,
        )

    @staticmethod
    def get_additional_obs_dim() -> int:
        """计算额外观测的维度
        
        Returns:
            额外观测的总维度
        """
        # 计算所有额外观测的维度
        additional_obs_dims = [
            3,   # root_lin_pos (世界坐标系)
            3,   # root_lin_vel (世界坐标系)
            1,   # z_rotation (yaw角度)
        ]
        return sum(additional_obs_dims)

    def _get_additional_observations(self, state: EnvState) -> torch.Tensor:
        """获取额外的观测
        
        Args:
            state: 环境状态
            
        Returns:
            额外观测张量
        """
        additional_obs_list = []
        
        # 1. 添加base线性位置观测 (世界坐标系)
        root_lin_pos = state.root_pos  # [num_envs, 3] - x, y, z
        additional_obs_list.append(root_lin_pos)
        
        # 2. 添加base线性速度观测 (局部坐标系) - 更适合轨迹跟踪任务
        root_lin_vel_local = state.local_root_lin_vel  # [num_envs, 3] - vx, vy, vz (局部坐标系)
        additional_obs_list.append(root_lin_vel_local)
        
        # 3. 添加z轴旋转观测 (从四元数提取z轴旋转)
        # 将四元数转换为欧拉角，然后提取z轴旋转
        # 四元数格式是 xyzw，需要转换为 wxyz
        quat_wxyz = state.root_xyzw_quat[:, [3, 0, 1, 2]]  # 重新排列为 wxyz
        euler_angles = p3d.matrix_to_euler_angles(
            p3d.quaternion_to_matrix(quat_wxyz), "XYZ"
        )
        z_rotation = euler_angles[:, 2:3]  # 提取z轴旋转 (yaw)
        additional_obs_list.append(z_rotation)
        
        # 将所有额外观测连接起来
        if additional_obs_list:
            additional_obs_tensor = torch.cat(additional_obs_list, dim=1)
        else:
            additional_obs_tensor = torch.zeros(
                (self.num_envs, 0), dtype=torch.float, device=self.device
            )
        
        return additional_obs_tensor

    def create_sim(self):
        """Creates simulation and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )
        self._create_envs()
        self.safe_bounds = torch.tensor([[-10e8, 10e8]] * 2).to(self.device)
        if self.cfg.terrain.mode == "none":
            return
        elif self.cfg.terrain.mode == "plane":
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.cfg.terrain.static_friction
            plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            plane_params.restitution = self.cfg.terrain.restitution
            self.gym.add_ground(self.sim, plane_params)
        elif self.cfg.terrain.mode == "perlin":
            self.terrain = TerrainPerlin(
                tot_cols=self.cfg.terrain.tot_cols,
                tot_rows=self.cfg.terrain.tot_rows,
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                zScale=self.cfg.terrain.zScale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                slope_threshold=self.cfg.terrain.slope_threshold,
            )
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = self.terrain.vertices.shape[0]
            tm_params.nb_triangles = self.terrain.triangles.shape[0]

            tm_params.transform.p.x = self.cfg.terrain.transform_x
            tm_params.transform.p.y = self.cfg.terrain.transform_y
            tm_params.transform.p.z = self.cfg.terrain.transform_z
            tm_params.static_friction = self.cfg.terrain.static_friction
            tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            tm_params.restitution = self.cfg.terrain.restitution
            self.gym.add_triangle_mesh(
                self.sim,
                self.terrain.vertices.flatten(order="C"),
                self.terrain.triangles.flatten(order="C"),
                tm_params,
            )
            self.height_points = self._init_height_points()
            self.height_samples = (
                torch.tensor(self.terrain.heightsamples)
                .view(self.terrain.tot_rows, self.terrain.tot_cols)
                .to(self.device)
            )
            bounds = np.array(
                (
                    self.terrain.vertices.min(axis=0),
                    self.terrain.vertices.max(axis=0),
                )
            )
            bounds[:, 0] += self.cfg.terrain.transform_x
            bounds[:, 1] += self.cfg.terrain.transform_y
            bounds[:, 2] += self.cfg.terrain.transform_z
            terrain_dims = bounds[1, :2] - bounds[0, :2]
            logging.info(
                f"Terrain dimensions: {terrain_dims[0]:.1f}m x {terrain_dims[1]:.1f}m"
            )
            assert (
                terrain_dims > self.cfg.terrain.safety_margin * 2
            ).all(), "Terrain too small for safety margin"
            self.env_origins = -self.env_origins
            self.env_origins[:, 0] += torch_rand_float(
                bounds[0, 0] + self.cfg.terrain.safety_margin,
                bounds[1, 0] - self.cfg.terrain.safety_margin,
                (self.num_envs, 1),
                device=self.device,
                generator=self.generator,
            )[:, 0]
            self.env_origins[:, 1] += torch_rand_float(
                bounds[0, 1] + self.cfg.terrain.safety_margin,
                bounds[1, 1] - self.cfg.terrain.safety_margin,
                (self.num_envs, 1),
                device=self.device,
                generator=self.generator,
            )[:, 0]
            self.env_origins[:, 2] += float(bounds[1, 2])
        else:
            raise ValueError(f"Unknown terrain mode {self.cfg.terrain.mode!r}")
            
        # ===== 障碍物创建已移动到_create_envs函数内部 =====

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            for i in range(len(props)):
                props[i].friction = self.setup.rigid_shape_friction[env_id, i]
        if len(self.cfg.domain_rand.randomize_restitution_rigid_bodies) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_restitution_rigid_body_ids
            ):
                props[body_id].restitution = torch_rand_float(
                    lower=self.cfg.domain_rand.restitution_coef_range[0],
                    upper=self.cfg.domain_rand.restitution_coef_range[1],
                    shape=(1,),
                    device=self.device,
                    generator=self.generator,
                ).item()
                self.setup.rigidbody_restitution_coef[env_id, idx] = props[
                    body_id
                ].restitution
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.asset_dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.curr_dof_pos_limits = self.asset_dof_pos_limits.clone()
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.asset_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.asset_dof_pos_limits[i, 1] = props["upper"][i].item()
            self._update_dof_limits(ratio=1.0)

        for i in range(len(props)):
            if self.cfg.domain_rand.randomize_dof_damping:
                props["damping"][i] = self.setup.dof_damping[env_id, i]
            if self.cfg.domain_rand.randomize_dof_friction:
                props["friction"][i] = self.setup.dof_friction[env_id, i]
            if self.cfg.domain_rand.randomize_dof_velocity:
                props["velocity"][i] = self.setup.dof_velocity[env_id, i]
        return props

    def _update_dof_limits(self, ratio: Union[float, torch.Tensor]):
        m = (self.asset_dof_pos_limits[:, 0] + self.asset_dof_pos_limits[:, 1]) / 2
        r = self.asset_dof_pos_limits[:, 1] - self.asset_dof_pos_limits[:, 0]
        # soft limits
        self.curr_dof_pos_limits[:, 0] = m - 0.5 * r * ratio
        self.curr_dof_pos_limits[:, 1] = m + 0.5 * r * ratio

    def _process_rigid_body_props(self, props, env_id):
        # from https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/domain_randomization.md
        # > Physx only allows 64000 unique physics materials in the
        # > scene at once. If more than 64000 materials are needed,
        # > increase num_buckets to allow materials to be shared
        # > between prims.
        if len(self.cfg.domain_rand.randomize_rigid_body_masses) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_rigid_body_masses_ids
            ):
                props[body_id].mass += torch_rand_float(
                    lower=self.cfg.domain_rand.added_mass_range[0],
                    upper=self.cfg.domain_rand.added_mass_range[1],
                    shape=(1,),
                    device=self.device,
                    generator=self.generator,
                ).item()
                props[body_id].mass = max(props[body_id].mass, 0.01)
                self.setup.rigidbody_mass[env_id, idx] = props[body_id].mass
        if len(self.cfg.domain_rand.randomize_rigid_body_com) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_rigid_body_com_ids
            ):
                props[body_id].com += gymapi.Vec3(
                    *torch_rand_float(
                        lower=self.cfg.domain_rand.rigid_body_com_range[0],
                        upper=self.cfg.domain_rand.rigid_body_com_range[1],
                        shape=(3,),
                        device=self.device,
                        generator=self.generator,
                    )
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.setup.rigidbody_com_offset[env_id, idx, 0] = props[body_id].com.x
                self.setup.rigidbody_com_offset[env_id, idx, 1] = props[body_id].com.y
                self.setup.rigidbody_com_offset[env_id, idx, 2] = props[body_id].com.z
        return props

    def _reset_root_states(self, env_ids):
        """Reset root states of the robot only"""
        # base position
        self.state.root_state[env_ids] = self.base_init_state
        if (self.init_pos_noise > 0).any():
            self.state.root_state[env_ids, 0:3] += torch_rand_float(
                -self.init_pos_noise,
                self.init_pos_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        if (self.init_euler_noise > 0).any():

            euler_displacement = torch_rand_float(
                -self.init_euler_noise,
                self.init_euler_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
            matrix = p3d.euler_angles_to_matrix(euler_displacement, "XYZ")
            quat_xyzw = p3d.matrix_to_quaternion(matrix)[..., [1, 2, 3, 0]]
            self.state.root_state[env_ids, 3:7] = quat_mul(
                self.state.root_state[env_ids, 3:7], quat_xyzw
            )
        if (self.init_lin_vel_noise > 0).any():
            self.state.root_state[env_ids, 7:10] += torch_rand_float(
                -self.init_lin_vel_noise,
                self.init_lin_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        if (self.init_ang_vel_noise > 0).any():
            self.state.root_state[env_ids, 10:13] += torch_rand_float(
                -self.init_ang_vel_noise,
                self.init_ang_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        self.state.root_state[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.state.root_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ----------------------------------------

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
        logging.info("Reward functions: " + ", ".join(self.reward_names))

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        if not hasattr(self.cfg.domain_rand, "randomize_restitution_rigid_bodies"):
            self.cfg.domain_rand.randomize_restitution_rigid_bodies = []
        self.setup = EnvSetup(
            kp=self.controller.kp.clone(),
            kd=self.controller.kd.clone(),
            rigidbody_mass=torch.ones(
                (self.num_envs, len(self.cfg.domain_rand.randomize_rigid_body_masses)),
                device=self.device,
            ),
            rigidbody_com_offset=torch.zeros(
                (self.num_envs, len(self.cfg.domain_rand.randomize_rigid_body_com), 3),
                device=self.device,
            ),
            rigidbody_restitution_coef=torch.ones(
                (
                    self.num_envs,
                    len(self.cfg.domain_rand.randomize_restitution_rigid_bodies),
                ),
                device=self.device,
            ),
            rigid_shape_friction=torch.zeros(
                (self.num_envs, len(rigid_shape_props_asset), 3), device=self.device
            ),
            dof_damping=torch.zeros((self.num_envs, self.num_dof), device=self.device),
            dof_friction=torch.zeros((self.num_envs, self.num_dof), device=self.device),
            dof_velocity=torch.zeros((self.num_envs, self.num_dof), device=self.device),
        )

        if self.cfg.domain_rand.randomize_friction:
            # prepare friction randomization
            friction_range = self.cfg.domain_rand.friction_range
            # from https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/domain_randomization.md
            # > Physx only allows 64000 unique physics materials in the
            # > scene at once. If more than 64000 materials are needed,
            # > increase num_buckets to allow materials to be shared
            # > between prims.
            # As far as I (huy) can tell, it only applies to friction
            # and restitution and not other properties (mass, com, etc.)
            # > material_properties (dim=3): Static friction, Dynamic
            # > friction, and Restitution.
            num_buckets = self.cfg.domain_rand.num_friction_buckets
            bucket_ids = torch.randint(
                low=0,
                high=num_buckets,
                size=(self.num_envs, len(rigid_shape_props_asset)),
                device=self.device,
                generator=self.generator,
            )
            friction_buckets = torch_rand_float(
                lower=friction_range[0],
                upper=friction_range[1],
                shape=(num_buckets, 1),
                device=self.device,
                generator=self.generator,
            )
            self.setup.rigid_shape_friction = friction_buckets[bucket_ids]
        if self.cfg.domain_rand.randomize_dof_damping:
            self.setup.dof_damping[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_damping_range[0],
                upper=self.cfg.domain_rand.dof_damping_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        if self.cfg.domain_rand.randomize_dof_friction:
            self.setup.dof_friction[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_friction_range[0],
                upper=self.cfg.domain_rand.dof_friction_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        if self.cfg.domain_rand.randomize_dof_velocity:
            self.setup.dof_velocity[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_velocity_range[0],
                upper=self.cfg.domain_rand.dof_velocity_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        self.cfg.domain_rand.randomize_rigid_body_masses_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_rigid_body_masses
        ]

        self.cfg.domain_rand.randomize_rigid_body_com_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_rigid_body_com
        ]
        self.cfg.domain_rand.randomize_restitution_rigid_body_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_restitution_rigid_bodies
        ]

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        self.init_pos_noise = to_torch(
            self.cfg.init_state.pos_noise, device=self.device, requires_grad=False
        )
        self.init_euler_noise = to_torch(
            self.cfg.init_state.euler_noise, device=self.device, requires_grad=False
        )
        self.init_lin_vel_noise = to_torch(
            self.cfg.init_state.lin_vel_noise, device=self.device, requires_grad=False
        )
        self.init_ang_vel_noise = to_torch(
            self.cfg.init_state.ang_vel_noise, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

        sensor_pose = gymapi.Transform()
        if not hasattr(self.cfg.asset, "force_sensor_links"):
            self.cfg.asset.force_sensor_links = self.cfg.asset.feet_names
        for name in self.cfg.asset.force_sensor_links:
            """
            From Legged Gym:
            > The contact forces reported by `net_contact_force_tensor` are
            > unreliable when simulating on GPU with a triangle mesh terrain.
            > A workaround is to use force sensors, but the force are
            > propagated through the sensors of consecutive bodies resulting
            > in an undesireable behaviour. However, for a legged robot it is
            > possible to add sensors to the feet/end effector only and get the
            > expected results. When using the force sensors make sure to
            > exclude gravity from trhe reported forces with
            > `sensor_options.enable_forward_dynamics_forces`
            """
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False
            sensor_options.enable_constraint_solver_forces = True
            sensor_options.use_world_frame = True
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(
                robot_asset, index, sensor_pose, sensor_options
            )

        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # create a grid of robots
        env_lower = gymapi.Vec3(
            -self.env_spacing,
            -self.env_spacing,
            0,
        )
        env_upper = gymapi.Vec3(
            self.env_spacing,
            self.env_spacing,
            self.env_spacing,
        )
        # 现在支持每个环境多个actor：机器人 + 障碍物
        self.actor_handles = []  # 每个环境的机器人actor句柄
        self.obstacle_actor_handles = []  # 每个环境的障碍物actor句柄列表
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            origin = self.gym.get_env_origin(env_handle)
            self.env_origins[i, 0] = origin.x
            self.env_origins[i, 1] = origin.y
            self.env_origins[i, 2] = origin.z
            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            # 创建机器人actor（第一个actor）
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            # 为当前环境创建障碍物
            try:
                obstacle_handles = self._create_obstacles_for_env(env_handle, i)
                self.obstacle_actor_handles.append(obstacle_handles)
                print(f"✓ 环境 {i} 障碍物创建成功，共 {len(obstacle_handles)} 个障碍物actor")
            except Exception as e:
                print(f"⚠ 环境 {i} 障碍物创建失败: {e}")
                import traceback
                traceback.print_exc()
                self.obstacle_actor_handles.append([])
            
            print(f"✓ 环境 {i} 创建完成，机器人actor: {actor_handle}")

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        if (self.termination_contact_indices == -1).any():
            raise ValueError(
                f"Could not find all termination links in actor {self.gym.get_actor_name(self.envs[0], 0)!r}"
            )
            
        # ===== 障碍物已在每个环境创建循环中创建完成 =====
        # 初始化障碍物信息
        try:
            self.obstacles = self._get_obstacle_info()
            self.obs_v = self.obstacles['velocities'].mean(dim=1)
            print("✓ 障碍物信息初始化完成")
        except Exception as e:
            print(f"⚠ 障碍物信息初始化失败: {e}")
            self.obstacles = None
            self.obs_v = None

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, state: EnvState, control: Control):
        # Penalize z axis base linear velocity
        return torch.square(state.local_root_lin_vel[:, 2])

    def _reward_ang_vel_xy(self, state: EnvState, control: Control):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(state.local_root_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self, state: EnvState, control: Control):
        # Penalize non flat base orientation
        return torch.sum(
            torch.square(state.local_root_gravity[:, :2]),
            dim=1,
        )

    def visualize(self, vis_env_ids: List[int]):
        """
        Draws all the trajectory position target lines.
        """
        self.gym.clear_lines(self.viewer)
        for task in self.tasks.values():
            task.visualize(
                state=self.state, viewer=self.viewer, vis_env_ids=vis_env_ids
            )
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        if self.cfg.terrain.mode in {"perlin"}:
            for i in vis_env_ids:
                base_pos = (self.state.root_pos[i, :3]).cpu().numpy()
                heights = self.state.measured_terrain_heights[i].cpu().numpy()
                height_points = (
                    quat_apply_yaw(
                        self.state.root_xyzw_quat[i].repeat(heights.shape[0]),
                        self.height_points[i],
                    )
                    .cpu()
                    .numpy()
                )
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(
                        sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                    )

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_pos_range = self.curr_dof_pos_limits[:, 1] - self.curr_dof_pos_limits[:, 0]
        dof_pos_range[torch.isnan(dof_pos_range) | torch.isinf(dof_pos_range)] = 1.0
        self.state.dof_pos[env_ids] = torch.clip(
            self.init_dof_pos
            + (
                self.dof_pos_reset_range_scale
                * torch.randn(
                    len(env_ids),
                    self.state.dof_pos.shape[1],
                    device=self.device,
                    generator=self.generator,
                )
                * dof_pos_range
            ),
            min=self.curr_dof_pos_limits[:, 0],
            max=self.curr_dof_pos_limits[:, 1],
        )
        self.state.prev_dof_pos[env_ids] = self.state.dof_pos[env_ids].clone()
        self.state.dof_vel[env_ids] = 0.0
        self.state.prev_dof_vel[env_ids] = 0.0

        env_ids_int32 = (self.num_obstacles_per_env + 1) * env_ids.to(dtype=torch.int32) # 每个环境有num_obstacles_per_env个障碍物，加上机器人本身
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(
                torch.stack(
                    (
                        self.state.dof_pos,
                        self.state.dof_vel,
                    ),
                    dim=-1,
                )
            ),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mode == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mode == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.state.root_xyzw_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + self.state.root_pos[env_ids].unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.state.root_xyzw_quat.repeat(1, self.num_height_points),
                self.height_points,
            ) + self.state.root_pos.unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def _init_height_points(self):
        """Returns points at which the height measurements are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def __del__(self):
        # NOTE: this destructor still results in segfaults upon exit.
        # Need to investigate further.
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)
            self.gym.destroy_sim(self.sim)
            
    def _create_obstacles_directly(self):
        """直接在IsaacGym中创建障碍物"""
        obstacle_count = 0
        
        # 创建球体障碍物
        for sphere in self.obstacle_params['sphere']:
            self._create_sphere_obstacle_directly(sphere, obstacle_count)
            obstacle_count += 1
            
        # 创建胶囊体障碍物
        for capsule in self.obstacle_params['capsule']:
            self._create_capsule_obstacle_directly(capsule, obstacle_count)
            obstacle_count += 1
            
        # 创建矩形障碍物
        for rectangle in self.obstacle_params['rectangle']:
            self._create_rectangle_obstacle_directly(rectangle, obstacle_count)
            obstacle_count += 1
            
        print(f"✓ 成功创建 {obstacle_count} 个障碍物类型，每个环境 {self.num_envs} 个实例")
        
    def _create_sphere_obstacle_directly(self, params, obstacle_id):
        """直接创建球体障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建球体资产
                sphere_asset = self.gym.create_sphere(self.sim, params['size'])
                
                # 设置球体属性
                # self.gym.set_asset_density(sphere_asset, 1.0)  # 这个方法不存在
                
                # 创建球体实例 - 添加到特定环境
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                
                # 创建障碍物actor并添加到环境
                # 注意：这里需要使用环境句柄，而不是整数索引
                env_handle = self.envs[env_id]  # 获取环境句柄
                sphere_handle = self.gym.create_actor(
                    env_handle,  # 使用环境句柄
                    sphere_asset, 
                    start_pose, 
                    f"{params['name']}_{env_id}", 
                    env_id, 
                    0
                )
                
                # 确保障碍物是动态的
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, sphere_handle)
                rigid_body_props[0].mass = 1.0
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
                self.gym.set_actor_rigid_body_properties(self.sim, sphere_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(sphere_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
                print(f"✓ 在环境 {env_id} 中创建球体障碍物 {params['name']} 成功")
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建球体障碍物 {params['name']} 失败: {e}")
                self.obstacle_handles[params['name']].append(None)
                
    def _create_capsule_obstacle_directly(self, params, obstacle_id):
        """直接创建胶囊体障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建胶囊体资产
                capsule_asset = self.gym.create_capsule(
                    self.sim, 
                    params['size'][0], 
                    params['size'][1]
                )
                
                # 设置胶囊体属性
                # self.gym.set_asset_density(capsule_asset, 1.0)  # 这个方法不存在
                
                # 创建胶囊体实例
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                if 'euler' in params:
                    start_pose.r = gymapi.Quat.from_euler_zyx(*params['euler'])
                
                # 创建障碍物actor并添加到环境
                env_handle = self.envs[env_id]
                capsule_handle = self.gym.create_actor(
                    env_handle,
                    capsule_asset, 
                    start_pose, 
                    f"{params['name']}_{env_id}", 
                    env_id, 
                    0
                )
                
                # 确保障碍物是动态的
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, capsule_handle)
                rigid_body_props[0].mass = 1.0
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
                self.gym.set_actor_rigid_body_properties(self.sim, capsule_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(capsule_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
                print(f"✓ 在环境 {env_id} 中创建胶囊体障碍物 {params['name']} 成功")
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建胶囊体障碍物 {params['name']} 失败: {e}")
                self.obstacle_handles[params['name']].append(None)
                
    def _create_rectangle_obstacle_directly(self, params, obstacle_id):
        """直接创建矩形障碍物"""
        # 初始化障碍物句柄列表
        if params['name'] not in self.obstacle_handles:
            self.obstacle_handles[params['name']] = []
        
        for env_id in range(self.num_envs):
            try:
                # 创建矩形资产
                box_asset = self.gym.create_box(
                    self.sim, 
                    params['size'][0], 
                    params['size'][1], 
                    params['size'][2]
                )
                
                # 设置矩形属性
                # self.gym.set_asset_density(box_asset, 1.0)  # 这个方法不存在
                
                # 创建矩形实例
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*params['pos'])
                
                # 创建障碍物actor并添加到环境
                env_handle = self.envs[env_id]
                box_handle = self.gym.create_actor(
                    env_handle,
                    box_asset, 
                    start_pose, 
                    f"{params['name']}_{env_id}", 
                    env_id, 
                    0
                )
                
                # 确保障碍物是动态的
                rigid_body_props = self.gym.get_actor_rigid_body_properties(self.sim, box_handle)
                rigid_body_props[0].mass = 1.0
                # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
                self.gym.set_actor_rigid_body_properties(self.sim, box_handle, rigid_body_props)
                
                # 存储句柄
                self.obstacle_handles[params['name']].append(box_handle)
                
                # 设置初始位置
                self.obstacle_positions[env_id, obstacle_id] = torch.tensor(params['pos'], device=self.device)
                
                print(f"✓ 在环境 {env_id} 中创建矩形障碍物 {params['name']} 成功")
                
            except Exception as e:
                print(f"⚠ 在环境 {env_id} 中创建矩形障碍物 {params['name']} 失败: {e}")
                self.obstacle_handles[params['name']].append(None)
                
    def _get_obstacle_info(self) -> Dict[str, torch.Tensor]:
        """获取障碍物信息，供CBF控制器使用（简化版本，只返回静态参数）"""
        # 由于我们采用"只创建，不管理状态"的方式，这里返回静态参数
        return {
            'capsule': [torch.tensor(capsule['pos'], device=self.device) for capsule in self.obstacle_params['capsule']],
            'rectangle': [torch.tensor(rect['pos'], device=self.device) for rect in self.obstacle_params['rectangle']],
            'velocities': None  # 不管理速度
        }

    def _create_obstacles_for_env(self, env_handle, env_id):
        """为单个环境创建障碍物（只创建，不管理状态）"""
        obstacle_count = 0
        obstacle_handles = []  # 存储当前环境的所有障碍物句柄
        
        # 创建胶囊体障碍物
        for capsule in self.obstacle_params['capsule']:
            handle = self._create_capsule_obstacle_for_env(env_handle, env_id, capsule, obstacle_count)
            if handle is not None:
                obstacle_handles.append(handle)
            obstacle_count += 1
            
        # 创建矩形障碍物
        for rectangle in self.obstacle_params['rectangle']:
            handle = self._create_rectangle_obstacle_for_env(env_handle, env_id, rectangle, obstacle_count)
            if handle is not None:
                obstacle_handles.append(handle)
            obstacle_count += 1
            
        print(f"✓ 环境 {env_id} 成功创建 {len(obstacle_handles)} 个障碍物actor")
        return obstacle_handles
        
    def _create_sphere_obstacle_for_env(self, env_handle, env_id, params, obstacle_id):
        """为单个环境创建球体障碍物"""
        try:
            # 初始化障碍物句柄列表
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            
            # 创建球体资产
            sphere_asset = self.gym.create_sphere(self.sim, params['size'])
            
            # 设置球体属性 - 使用正确的IsaacGym API
            # self.gym.set_asset_density(sphere_asset, 1.0)  # 这个方法不存在
            
            # 创建球体实例 - 添加到特定环境
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*params['pos'])
            
            # 创建障碍物actor并添加到环境
            sphere_handle = self.gym.create_actor(
                env_handle,  # 使用传入的环境句柄
                sphere_asset, 
                start_pose, 
                f"{params['name']}_{env_id}", 
                env_id, 
                0
            )
            
            # 确保障碍物是动态的
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, sphere_handle)
            rigid_body_props[0].mass = 1.0
            # 移除不存在的标志
            # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
            self.gym.set_actor_rigid_body_properties(env_handle, sphere_handle, rigid_body_props)
            
            # 存储句柄
            self.obstacle_handles[params['name']].append(sphere_handle)
            
            print(f"✓ 在环境 {env_id} 中创建球体障碍物 {params['name']} 成功")
            return sphere_handle
            
        except Exception as e:
            print(f"⚠ 在环境 {env_id} 中创建球体障碍物 {params['name']} 失败: {e}")
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            self.obstacle_handles[params['name']].append(None)
            return None
            
    def _create_capsule_obstacle_for_env(self, env_handle, env_id, params, obstacle_id):
        """为单个环境创建胶囊体障碍物"""
        try:
            # 初始化障碍物句柄列表
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            
            # 创建胶囊体资产
            capsule_asset = self.gym.create_capsule(
                self.sim, 
                params['size'][0], 
                params['size'][1]
            )
            
            # 设置胶囊体属性 - 使用正确的IsaacGym API
            # self.gym.set_asset_density(capsule_asset, 1.0)  # 这个方法不存在
            
            # 创建胶囊体实例
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*params['pos'])
            if 'euler' in params:
                start_pose.r = gymapi.Quat.from_euler_zyx(*params['euler'])
            
            # 创建障碍物actor并添加到环境
            capsule_handle = self.gym.create_actor(
                env_handle,
                capsule_asset, 
                start_pose, 
                f"{params['name']}_{env_id}", 
                env_id, 
                0
            )
            
            # 确保障碍物是动态的
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, capsule_handle)
            rigid_body_props[0].mass = 1.0
            # 移除不存在的标志
            # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
            self.gym.set_actor_rigid_body_properties(env_handle, capsule_handle, rigid_body_props)
            
            # 存储句柄
            self.obstacle_handles[params['name']].append(capsule_handle)
            
            print(f"✓ 在环境 {env_id} 中创建胶囊体障碍物 {params['name']} 成功")
            return capsule_handle
            
        except Exception as e:
            print(f"⚠ 在环境 {env_id} 中创建胶囊体障碍物 {params['name']} 失败: {e}")
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            self.obstacle_handles[params['name']].append(None)
            return None
            
    def _create_rectangle_obstacle_for_env(self, env_handle, env_id, params, obstacle_id):
        """为单个环境创建矩形障碍物"""
        try:
            # 初始化障碍物句柄列表
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            
            # 创建矩形资产
            box_asset = self.gym.create_box(
                self.sim, 
                params['size'][0], 
                params['size'][1], 
                params['size'][2]
            )
            
            # 设置矩形属性 - 使用正确的IsaacGym API
            # self.gym.set_asset_density(box_asset, 1.0)  # 这个方法不存在
            
            # 创建矩形实例
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*params['pos'])
            if 'euler' in params:
                start_pose.r = gymapi.Quat.from_euler_zyx(*params['euler'])
            
            # 创建障碍物actor并添加到环境
            box_handle = self.gym.create_actor(
                env_handle,
                box_asset, 
                start_pose, 
                f"{params['name']}_{env_id}", 
                env_id, 
                0
            )
            
            # 确保障碍物是动态的
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            rigid_body_props[0].mass = 1.0
            # 移除不存在的标志
            # rigid_body_props[0].flags = gymapi.RIGID_BODY_FLAG_USE_SELF_COLLISION
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, rigid_body_props)
            
            # 存储句柄
            self.obstacle_handles[params['name']].append(box_handle)
            
            print(f"✓ 在环境 {env_id} 中创建矩形障碍物 {params['name']} 成功")
            return box_handle
            
        except Exception as e:
            print(f"⚠ 在环境 {env_id} 中创建矩形障碍物 {params['name']} 失败: {e}")
            if params['name'] not in self.obstacle_handles:
                self.obstacle_handles[params['name']] = []
            self.obstacle_handles[params['name']].append(None)
            return None

    def update_obstacle_states(self):
        """更新障碍物的位置和速度状态
        从IsaacGym仿真中获取障碍物的实时状态
        """
        if not hasattr(self, 'obstacle_actor_handles') or not self.obstacle_actor_handles:
            return
            
        # 初始化障碍物状态张量
        if not hasattr(self, 'obstacle_positions') or self.obstacle_positions is None:
            self.obstacle_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        if not hasattr(self, 'obstacle_velocities') or self.obstacle_velocities is None:
            self.obstacle_velocities = torch.zeros(self.num_envs, 2, 3, device=self.device)
            
        # 获取所有障碍物的根状态
        all_obstacle_states = []
        obstacle_count = 0
        
        for env_id in range(self.num_envs):
            env_obstacles = self.obstacle_actor_handles[env_id]
            for obstacle_handle in env_obstacles:
                if obstacle_handle is not None:
                    try:
                        # 获取障碍物的根状态 [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
                        obstacle_state = self.gym.get_actor_root_state_tensor_indexed(
                            self.sim,
                            gymtorch.unwrap_tensor(torch.tensor([obstacle_handle], device=self.device, dtype=torch.int32)),
                            1
                        )
                        
                        # 提取位置和速度信息
                        pos = obstacle_state[0, :3]  # [x, y, z]
                        lin_vel = obstacle_state[0, 7:10]  # [vx, vy, vz]
                        
                        # 存储到对应的张量中
                        if obstacle_count < 2:  # 假设最多2个障碍物
                            self.obstacle_positions[env_id, obstacle_count] = pos
                            self.obstacle_velocities[env_id, obstacle_count] = lin_vel
                        
                        obstacle_count += 1
                        
                    except Exception as e:
                        print(f"⚠ 获取环境 {env_id} 障碍物 {obstacle_handle} 状态失败: {e}")
                        continue
                        
        # 更新障碍物信息
        self.obstacles = self._get_dynamic_obstacle_info()
        if self.obstacles['velocities'] is not None:
            self.obs_v = self.obstacles['velocities'].mean(dim=1)  # 平均速度

    def _get_dynamic_obstacle_info(self) -> Dict[str, torch.Tensor]:
        """获取动态障碍物信息，包括实时位置和速度"""
        if not hasattr(self, 'obstacle_positions') or self.obstacle_positions is None:
            return self._get_obstacle_info()  # 回退到静态信息
            
        return {
            'positions': self.obstacle_positions,  # [num_envs, num_obstacles, 3]
            'velocities': self.obstacle_velocities,  # [num_envs, num_obstacles, 3]
            'capsule': [self.obstacle_positions[:, 0, :]],  # 第一个障碍物（胶囊体）
            'rectangle': [self.obstacle_positions[:, 1, :]],  # 第二个障碍物（矩形）
        }

    def get_obstacle_positions(self) -> torch.Tensor:
        """获取所有障碍物的当前位置
        
        Returns:
            torch.Tensor: 形状为 [num_envs, num_obstacles, 3] 的张量
        """
        if hasattr(self, 'obstacle_positions') and self.obstacle_positions is not None:
            return self.obstacle_positions
        else:
            # 如果没有动态位置信息，返回静态参数
            static_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
            for i, capsule in enumerate(self.obstacle_params['capsule']):
                static_positions[:, i, :] = torch.tensor(capsule['pos'], device=self.device)
            for i, rect in enumerate(self.obstacle_params['rectangle']):
                static_positions[:, i+1, :] = torch.tensor(rect['pos'], device=self.device)
            return static_positions

    def get_obstacle_velocities(self) -> torch.Tensor:
        """获取所有障碍物的当前速度
        
        Returns:
            torch.Tensor: 形状为 [num_envs, num_obstacles, 3] 的张量
        """
        if hasattr(self, 'obstacle_velocities') and self.obstacle_velocities is not None:
            return self.obstacle_velocities
        else:
            # 如果没有动态速度信息，返回零速度
            return torch.zeros(self.num_envs, 2, 3, device=self.device)

    def get_obstacle_distances(self, robot_positions: torch.Tensor) -> torch.Tensor:
        """计算机器人与障碍物之间的距离
        
        Args:
            robot_positions: 机器人位置，形状为 [num_envs, 3]
            
        Returns:
            torch.Tensor: 距离矩阵，形状为 [num_envs, num_obstacles]
        """
        obstacle_positions = self.get_obstacle_positions()  # [num_envs, num_obstacles, 3]
        
        # 计算欧几里得距离
        # robot_positions: [num_envs, 3] -> [num_envs, 1, 3]
        # obstacle_positions: [num_envs, num_obstacles, 3]
        robot_pos_expanded = robot_positions.unsqueeze(1)  # [num_envs, 1, 3]
        
        distances = torch.norm(obstacle_positions - robot_pos_expanded, dim=2)  # [num_envs, num_obstacles]
        return distances

    def get_obstacle_relative_positions(self, robot_positions: torch.Tensor, robot_orientations: torch.Tensor) -> torch.Tensor:
        """获取障碍物相对于机器人的位置（在机器人坐标系中）
        
        Args:
            robot_positions: 机器人位置，形状为 [num_envs, 3]
            robot_orientations: 机器人方向（四元数），形状为 [num_envs, 4]
            
        Returns:
            torch.Tensor: 相对位置，形状为 [num_envs, num_obstacles, 3]
        """
        obstacle_positions = self.get_obstacle_positions()  # [num_envs, num_obstacles, 3]
        
        # 计算相对位置（世界坐标系）
        relative_positions_world = obstacle_positions - robot_positions.unsqueeze(1)  # [num_envs, num_obstacles, 3]
        
        # 转换到机器人坐标系
        # 四元数格式转换：xyzw -> wxyz
        quat_wxyz = robot_orientations[:, [3, 0, 1, 2]]  # 重新排列为 wxyz
        
        # 创建旋转矩阵
        rotation_matrices = p3d.quaternion_to_matrix(quat_wxyz)  # [num_envs, 3, 3]
        
        # 应用旋转：R^T * relative_pos
        # 注意：这里需要广播到障碍物维度
        rotation_matrices_expanded = rotation_matrices.unsqueeze(1)  # [num_envs, 1, 3, 3]
        relative_positions_robot = torch.matmul(
            rotation_matrices_expanded.transpose(-2, -1),  # R^T
            relative_positions_world.unsqueeze(-1)  # [num_envs, num_obstacles, 3, 1]
        ).squeeze(-1)  # [num_envs, num_obstacles, 3]
        
        return relative_positions_robot

    def set_obstacle_motion_pattern(self, motion_type: str = "static", **kwargs):
        """设置障碍物的运动模式
        
        Args:
            motion_type: 运动类型，可选值：
                - "static": 静态障碍物
                - "linear": 线性运动
                - "circular": 圆周运动
                - "sinusoidal": 正弦运动
            **kwargs: 运动参数
        """
        if not hasattr(self, 'obstacle_motion_config'):
            self.obstacle_motion_config = {}
            
        self.obstacle_motion_config['type'] = motion_type
        self.obstacle_motion_config.update(kwargs)
        
        print(f"✓ 设置障碍物运动模式: {motion_type}")
        
    def _apply_obstacle_motion(self, dt: float):
        """应用障碍物运动模式"""
        if not hasattr(self, 'obstacle_motion_config') or not self.obstacle_motion_config:
            return
            
        motion_type = self.obstacle_motion_config.get('type', 'static')
        
        if motion_type == 'static':
            return  # 静态障碍物，不需要更新
            
        elif motion_type == 'linear':
            self._apply_linear_motion(dt)
            
        elif motion_type == 'circular':
            self._apply_circular_motion(dt)
            
        elif motion_type == 'sinusoidal':
            self._apply_sinusoidal_motion(dt)
            
    def _apply_linear_motion(self, dt: float):
        """应用线性运动"""
        if not hasattr(self, 'obstacle_actor_handles') or not self.obstacle_actor_handles:
            return
            
        # 获取运动参数
        velocity = self.obstacle_motion_config.get('velocity', [0.1, 0.0, 0.0])
        velocity_tensor = torch.tensor(velocity, device=self.device)
        
        # 更新位置
        if hasattr(self, 'obstacle_positions') and self.obstacle_positions is not None:
            self.obstacle_positions += velocity_tensor.unsqueeze(0).unsqueeze(0) * dt
            
        # 更新速度
        if hasattr(self, 'obstacle_velocities') and self.obstacle_velocities is not None:
            self.obstacle_velocities[:] = velocity_tensor.unsqueeze(0).unsqueeze(0)
            
        # 应用到IsaacGym仿真
        self._apply_obstacle_positions_to_sim()
        
    def _apply_circular_motion(self, dt: float):
        """应用圆周运动"""
        if not hasattr(self, 'obstacle_actor_handles') or not self.obstacle_actor_handles:
            return
            
        # 获取运动参数
        center = self.obstacle_motion_config.get('center', [0.0, 0.0, 0.0])
        radius = self.obstacle_motion_config.get('radius', 1.0)
        angular_velocity = self.obstacle_motion_config.get('angular_velocity', 1.0)
        
        center_tensor = torch.tensor(center, device=self.device)
        radius_tensor = torch.tensor(radius, device=self.device)
        angular_vel_tensor = torch.tensor(angular_velocity, device=self.device)
        
        # 计算当前时间
        current_time = self.state.episode_time * dt
        
        # 更新位置
        if hasattr(self, 'obstacle_positions') and self.obstacle_positions is not None:
            for env_id in range(self.num_envs):
                for obstacle_id in range(2):  # 假设最多2个障碍物
                    angle = angular_vel_tensor * current_time[env_id]
                    x = center_tensor[0] + radius_tensor * torch.cos(angle)
                    y = center_tensor[1] + radius_tensor * torch.sin(angle)
                    z = center_tensor[2]
                    
                    self.obstacle_positions[env_id, obstacle_id] = torch.tensor([x, y, z], device=self.device)
                    
        # 计算速度
        if hasattr(self, 'obstacle_velocities') and self.obstacle_velocities is not None:
            for env_id in range(self.num_envs):
                for obstacle_id in range(2):
                    angle = angular_vel_tensor * current_time[env_id]
                    vx = -radius_tensor * angular_vel_tensor * torch.sin(angle)
                    vy = radius_tensor * angular_vel_tensor * torch.cos(angle)
                    vz = 0.0
                    
                    self.obstacle_velocities[env_id, obstacle_id] = torch.tensor([vx, vy, vz], device=self.device)
                    
        # 应用到IsaacGym仿真
        self._apply_obstacle_positions_to_sim()
        
    def _apply_sinusoidal_motion(self, dt: float):
        """应用正弦运动"""
        if not hasattr(self, 'obstacle_actor_handles') or not self.obstacle_actor_handles:
            return
            
        # 获取运动参数
        amplitude = self.obstacle_motion_config.get('amplitude', [0.5, 0.0, 0.0])
        frequency = self.obstacle_motion_config.get('frequency', 1.0)
        phase = self.obstacle_motion_config.get('phase', 0.0)
        
        amplitude_tensor = torch.tensor(amplitude, device=self.device)
        frequency_tensor = torch.tensor(frequency, device=self.device)
        phase_tensor = torch.tensor(phase, device=self.device)
        
        # 计算当前时间
        current_time = self.state.episode_time * dt
        
        # 更新位置
        if hasattr(self, 'obstacle_positions') and self.obstacle_positions is not None:
            for env_id in range(self.num_envs):
                for obstacle_id in range(2):
                    time_val = current_time[env_id]
                    x = amplitude_tensor[0] * torch.sin(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    y = amplitude_tensor[1] * torch.sin(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    z = amplitude_tensor[2] * torch.sin(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    
                    self.obstacle_positions[env_id, obstacle_id] = torch.tensor([x, y, z], device=self.device)
                    
        # 计算速度
        if hasattr(self, 'obstacle_velocities') and self.obstacle_velocities is not None:
            for env_id in range(self.num_envs):
                for obstacle_id in range(2):
                    time_val = current_time[env_id]
                    vx = 2 * torch.pi * frequency_tensor * amplitude_tensor[0] * torch.cos(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    vy = 2 * torch.pi * frequency_tensor * amplitude_tensor[1] * torch.cos(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    vz = 2 * torch.pi * frequency_tensor * amplitude_tensor[2] * torch.cos(2 * torch.pi * frequency_tensor * time_val + phase_tensor)
                    
                    self.obstacle_velocities[env_id, obstacle_id] = torch.tensor([vx, vy, vz], device=self.device)
                    
        # 应用到IsaacGym仿真
        self._apply_obstacle_positions_to_sim()
        
    def _apply_obstacle_positions_to_sim(self):
        """将计算的位置应用到IsaacGym仿真中"""
        if not hasattr(self, 'obstacle_actor_handles') or not self.obstacle_actor_handles:
            return
            
        if not hasattr(self, 'obstacle_positions') or self.obstacle_positions is None:
            return
            
        try:
            for env_id in range(self.num_envs):
                env_obstacles = self.obstacle_actor_handles[env_id]
                for obstacle_id, obstacle_handle in enumerate(env_obstacles):
                    if obstacle_handle is not None and obstacle_id < 2:
                        # 获取当前位置
                        current_pos = self.obstacle_positions[env_id, obstacle_id]
                        
                        # 创建新的根状态
                        new_root_state = torch.zeros(13, device=self.device)
                        new_root_state[:3] = current_pos  # 位置
                        new_root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # 四元数（默认）
                        new_root_state[7:10] = self.obstacle_velocities[env_id, obstacle_id] if self.obstacle_velocities is not None else torch.zeros(3, device=self.device)  # 线速度
                        new_root_state[10:13] = torch.zeros(3, device=self.device)  # 角速度
                        
                        # 设置障碍物的根状态
                        self.gym.set_actor_root_state_tensor_indexed(
                            self.sim,
                            gymtorch.unwrap_tensor(new_root_state.unsqueeze(0)),
                            gymtorch.unwrap_tensor(torch.tensor([obstacle_handle], device=self.device, dtype=torch.int32)),
                            1
                        )
                        
        except Exception as e:
            print(f"⚠ 应用障碍物位置到仿真失败: {e}")

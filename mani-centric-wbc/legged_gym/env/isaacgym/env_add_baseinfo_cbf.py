from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pytorch3d.transforms as p3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_mul, to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.env.isaacgym.constraints import Constraint
from legged_gym.env.isaacgym.control import Control, PDController
from legged_gym.env.isaacgym.obs import EnvObservationAttribute, EnvSetupAttribute
from legged_gym.env.isaacgym.state import EnvSetup, EnvState
from legged_gym.env.isaacgym.task import Task
from legged_gym.env.isaacgym.terrain import TerrainPerlin
from legged_gym.env.isaacgym.utils import quat_apply_yaw, torch_rand_float
from legged_gym.env.obs import ObservationAttribute
from legged_gym.rsl_rl.env import VecEnv

PartialTask = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Task]
PartialConstraint = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Constraint]

from cbf.cbf_controller import CBF_controller, DISTURBANCE_OBSERVER
from multiprocessing import Process, Queue
import casadi as ca
import heapq


class ParallelAStarPlanner:
    def __init__(self, device, grid_size=0.1, world_bounds=(-2, 9, -2, 2)):
        self.device = device
        self.grid_size = grid_size
        self.world_bounds = world_bounds
        self.robot_radius = 0.25  # 四足机器人半径
        self.safety_margin = 0.05 # 安全裕度（米）
        # 障碍物位置和尺寸 (x, y, width, length, height)
        self.obstacles = [
            (4.0, -0.7, 0.5, 0.05, 0.6),   # box2: 左侧障碍物
            (4.0, 0.7, 0.5, 0.05, 0.6),    # box3: 右侧障碍物  
            (2.0, 0.0, 0.3, 0.3, 0.3),     # box4: 前方障碍物
        ]
    

    def smooth_path_3d(self, path, weight_data=0.1, weight_smooth=0.3, tolerance=0.00001):
        """3D路径平滑处理（保持yaw角度）"""
        if path is None or len(path) < 3:
            return path
            
        # 分离坐标和角度
        path_array = np.array([[point[0], point[1], point[2]] for point in path])
        smoothed = np.copy(path_array)
        
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path_array)-1):
                for j in range(3):  # x, y, yaw
                    aux = smoothed[i][j]
                    smoothed[i][j] += weight_data * (path_array[i][j] - smoothed[i][j])
                    smoothed[i][j] += weight_smooth * (smoothed[i-1][j] + smoothed[i+1][j] - 2 * smoothed[i][j])
                    change += abs(aux - smoothed[i][j])
                    
        # 转换回列表格式
        return [(x, y, yaw) for x, y, yaw in smoothed]

    def create_grid(self, num_envs):
        """创建栅格地图"""
        x_min, x_max, y_min, y_max = self.world_bounds
        grid_width = int((x_max - x_min) / self.grid_size) + 1
        grid_height = int((y_max - y_min) / self.grid_size) + 1
        
        grid = torch.ones((num_envs, grid_height, grid_width), 
                         device=self.device, dtype=torch.bool)
        
        return grid, (x_min, y_min, grid_width, grid_height)
    
    def add_obstacles_to_grid(self, grid, grid_info):
        """将障碍物添加到栅格地图中（考虑机器人半径和安全裕度）"""
        x_min, y_min, grid_width, grid_height = grid_info
        
        for obstacle in self.obstacles:
            x_center, y_center, width, length, _ = obstacle
            
            # 计算膨胀后的障碍物尺寸（机器人半径 + 安全裕度）
            inflated_width = width + 2 * (self.robot_radius + self.safety_margin)
            inflated_length = length + 2 * (self.robot_radius + self.safety_margin)
            
            # 计算膨胀后的障碍物在栅格中的边界
            x_start = int((x_center - inflated_width/2 - x_min) / self.grid_size)
            x_end = int((x_center + inflated_width/2 - x_min) / self.grid_size) + 1
            y_start = int((y_center - inflated_length/2 - y_min) / self.grid_size)  
            y_end = int((y_center + inflated_length/2 - y_min) / self.grid_size) + 1
            
            # 确保边界在栅格范围内
            x_start = max(0, min(x_start, grid_width))
            x_end = max(0, min(x_end, grid_width))
            y_start = max(0, min(y_start, grid_height))
            y_end = max(0, min(y_end, grid_height))
            
            # 将障碍物区域标记为不可通行
            grid[:, y_start:y_end, x_start:x_end] = False
            
        return grid
    
    def world_to_grid(self, point, grid_info):
        """世界坐标转栅格坐标"""
        x, y = point[0], point[1]
        x_min, y_min, grid_width, grid_height = grid_info
        
        grid_x = int((x - x_min) / self.grid_size)
        grid_y = int((y - y_min) / self.grid_size)
        
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_point, grid_info):
        """栅格坐标转世界坐标"""
        grid_x, grid_y = grid_point
        x_min, y_min, grid_width, grid_height = grid_info
        
        x = grid_x * self.grid_size + x_min
        y = grid_y * self.grid_size + y_min
        
        return (x, y)
    
    def heuristic(self, a, b):
        """启发式函数"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, point, grid, grid_info, env_idx=0):
        """获取相邻栅格"""
        x, y = point
        grid_width, grid_height = grid_info[2], grid_info[3]
        
        neighbors = []
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (1,-1), (-1,1), (-1,-1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid_width and 0 <= ny < grid_height and 
                grid[env_idx, ny, nx]):
                cost = 1.0 if abs(dx) + abs(dy) == 1 else 1.414
                neighbors.append(((nx, ny), cost))
                
        return neighbors
    
    def a_star_single(self, start, goal, grid, grid_info, env_idx=0):
        """改进的A*算法 - 考虑机器人半径和路径方向"""
        # 提取2D坐标用于路径规划
        start_2d = (start[0], start[1]) if len(start) > 2 else start
        goal_2d = (goal[0], goal[1]) if len(goal) > 2 else goal
        
        start_grid = self.world_to_grid(start_2d, grid_info)
        goal_grid = self.world_to_grid(goal_2d, grid_info)
        
        # 检查起点和终点是否可通行（考虑机器人半径）
        if not self._is_position_valid(start_2d, grid, grid_info, env_idx) or \
        not self._is_position_valid(goal_2d, grid, grid_info, env_idx):
            print(f"Start or goal position is invalid for env {env_idx}")
            return None
        
        # A*算法数据结构初始化
        open_set = []  # 优先队列，存储待探索节点 (f_score, position)
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}  # 记录每个节点的前驱节点，用于重建路径
        g_score = {start_grid: 0}  # 从起点到当前节点的实际代价
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}  # 估计总代价 = g_score + 启发式
        
        # 已探索节点集合（可选，用于优化）
        closed_set = set()
        
        while open_set:
            # 从开放列表取出f_score最小的节点
            current_f, current = heapq.heappop(open_set)
            
            # 如果到达目标，重建路径
            if current == goal_grid:
                path_2d = self._reconstruct_path(came_from, current, start_grid, grid_info)
                # 为2D路径添加yaw角度
                path_with_yaw = self._add_yaw_to_path(path_2d, start, goal)
                return path_with_yaw
            
            # 将当前节点加入关闭列表
            closed_set.add(current)
            
            # 探索当前节点的所有邻居
            for neighbor, move_cost in self.get_neighbors(current, grid, grid_info, env_idx):
                # 如果邻居已在关闭列表中，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算从起点经过当前节点到邻居的代价
                tentative_g = g_score[current] + move_cost
                
                # 如果找到更优路径，更新邻居信息
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    # 如果邻居不在开放列表中，添加它
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 开放列表为空但未找到路径
        print(f"No path found from {start_2d} to {goal_2d} for env {env_idx}")
        return None

    def _is_position_valid(self, position, grid, grid_info, env_idx):
        """检查位置是否有效（考虑机器人半径）"""
        grid_pos = self.world_to_grid(position, grid_info)
        
        # 检查中心点是否可通行
        if not grid[env_idx, grid_pos[1], grid_pos[0]]:
            return False
        
        # 检查机器人半径范围内的所有点
        robot_radius_cells = int(self.robot_radius / self.grid_size) + 1
        
        for dx in range(-robot_radius_cells, robot_radius_cells + 1):
            for dy in range(-robot_radius_cells, robot_radius_cells + 1):
                # 计算实际距离（欧几里得距离）
                distance = np.sqrt(dx**2 + dy**2) * self.grid_size
                if distance <= self.robot_radius:
                    check_x = grid_pos[0] + dx
                    check_y = grid_pos[1] + dy
                    
                    # 检查边界
                    if (0 <= check_x < grid_info[2] and 0 <= check_y < grid_info[3]):
                        if not grid[env_idx, check_y, check_x]:
                            return False
        
        return True

    def _reconstruct_path(self, came_from, current, start_grid, grid_info):
        """从终点回溯重建路径"""
        path = []
        while current in came_from:
            world_pos = self.grid_to_world(current, grid_info)
            path.append(world_pos)
            current = came_from[current]
        
        # 添加起点
        world_pos = self.grid_to_world(start_grid, grid_info)
        path.append(world_pos)
        path.reverse()
        
        return path

    def _add_yaw_to_path(self, path_2d, start, goal):
        """为2D路径添加yaw角度"""
        if len(path_2d) < 2:
            return [(path_2d[0][0], path_2d[0][1], 0.0)] if path_2d else []
        
        path_with_yaw = []
        
        for i, (x, y) in enumerate(path_2d):
            if i == 0:
                # 起点使用起始yaw
                yaw = start[2] if len(start) > 2 else 0.0
            elif i == len(path_2d) - 1:
                # 终点使用目标yaw
                yaw = goal[2] if len(goal) > 2 else 0.0
            else:
                # 中间点计算前进方向
                next_x, next_y = path_2d[i + 1]
                yaw = np.arctan2(next_y - y, next_x - x)
            
            path_with_yaw.append((x, y, yaw))
        
        return path_with_yaw
    
    def plan_paths_parallel(self, starts, goals, num_envs):
        """为所有环境并行规划路径"""
        grid, grid_info = self.create_grid(num_envs)
        grid = self.add_obstacles_to_grid(grid, grid_info)
        
        all_paths = []
        for env_idx in range(num_envs):
            start = starts[env_idx] if torch.is_tensor(starts) else starts
            goal = goals[env_idx] if torch.is_tensor(goals) else goals
            
            path = self.a_star_single((start[0][0], start[0][1]), (goal[0][0], goal[0][1]), grid, grid_info, env_idx)
            all_paths.append(path)
            
        return all_paths
    
    def smooth_path(self, path, weight_data=0.1, weight_smooth=0.3, tolerance=0.00001):
        """路径平滑处理"""
        if path is None or len(path) < 3:
            return path
            
        path_array = np.array(path)
        smoothed = np.copy(path_array)
        
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path_array)-1):
                for j in range(2):  # x, y
                    aux = smoothed[i][j]
                    smoothed[i][j] += weight_data * (path_array[i][j] - smoothed[i][j])
                    smoothed[i][j] += weight_smooth * (smoothed[i-1][j] + smoothed[i+1][j] - 2 * smoothed[i][j])
                    change += abs(aux - smoothed[i][j])
                    
        return smoothed.tolist()

class SuppressOutput:
    """Context manager to suppress qpOASES output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class PDController:
    """通用PD控制器（比例-微分控制）"""
    
    def __init__(self, kp=1.0, kd=0.0, dt=0.01, output_limits=None):
        """
        Args:
            kp: 比例增益
            kd: 微分增益
            dt: 时间步长
            output_limits: 输出限制 (min, max)
        """
        self.kp = kp
        self.kd = kd
        self.output_limits = output_limits
        
        # 内部状态
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def update(self, setpoint, current_value):
        """
        更新PD控制器
        
        Args:
            setpoint: 目标值
            current_value: 当前值
            
        Returns:
            control_output: 控制输出
        """
        # 计算误差
        error = setpoint - current_value
        
        # 比例项
        proportional = self.kp * error
        
        # 微分项
        derivative = self.kd * (error - self.previous_error)
        
        # PD输出
        output = proportional + derivative
        
        # 应用输出限制
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # 更新内部状态
        self.previous_error = error
        self.previous_output = output
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.previous_error = 0.0
        self.previous_output = 0.0
    
    def set_params(self, kp=None, kd=None, dt=None):
        """动态设置PD参数"""
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd


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

        # 添加CBF相关的init
        self.T_step = 0.005
        self.O_T_step = 0.00085
        self.obstacle_type_num = [0, 0, 4]
        self.use_robust = True
        self.use_dynamic = False
        self.set_disturbance = False
        self.para_v_fault = 1.0
        
        # Determine CBF mode
        if self.use_robust and self.use_dynamic:
            self.mode = '11'
        elif self.use_robust and not self.use_dynamic:
            self.mode = '10'
        elif not self.use_robust and self.use_dynamic:
            self.mode = '01'
        else:
            self.mode = '00'
        
        # CBF process setup
        self.CBF_input = Queue(1)
        self.CBF_output = Queue(1)
        self.DOB_input = Queue(1)
        self.DOB_output = Queue(1)
        # Start CBF process (but don't start it yet, wait for CBF_start call)
        self.CBF_process = Process(target=self.CBF_process_func, args=(self.CBF_input, self.CBF_output, self.DOB_input), daemon=True)
        # Start disturbance observer process
        self.DOB_process = Process(target=self.observer_process_func, args=(self.DOB_input, self.DOB_output), daemon=True)

        # Initialize CBF controller
        self.cbf_controller = CBF_controller(
            obstacle_type_num=self.obstacle_type_num,
            T_step=self.T_step,
            CBF_mode=self.mode,
            O_T_step=self.O_T_step,
            use_statistic_obstacle = False
        )

        self.h_threshold = 0.05
        self.h_list_min = 1.0
        self.update_beta = False
        self.velocity_limite = np.array([1, 1, 0.5, 0.2, 0.2, 1.0, 3.14, 3.40, 3.14, 3.93, 3.93])
        
        # CBF control configuration (11 DOF: 6 base + 5 arm)
        self.u_len = 11  # go2-6dof velocity + piper-5dof velocity (same as MuJoCo)
        self.base_dofs = 6  # x, y, z, roll, pitch, yaw
        self.arm_controlled_dofs = 5  # 5 arm joints for velocity control
        self.leg_dofs = 12

        # 初始化PD控制器（只创建一次）
        self.pd_controllers = []
        velocity_output_limits = [
            [-0.4, 0.4], [-0.4, 0.4], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0,1.0],
            [-3.14, 3.14], [-3.40, 3.40], [-3.14, 3.14], [-3.93, 3.93], [-3.93, 3.93], [-3.93, 3.93]
        ]
        self.position_kp = np.array([2, 2, 2, 2, 2, 2, 
                                    21.3016, 27.2393, 50.8360, 15.9196, 22.5215,  5.1744])  # 大幅降低所有手臂关节Kp
        self.position_kd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                    1.6801, 2.6524, 3.5458, 2.7676, 0.6234, 0.5672])  # 大幅降低所有手臂关节Kd

        self.arm_vel_kp = np.array([8, 8, 8, 8, 8, 8])
        self.arm_vel_kd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.prev_velocity_error = np.zeros(6)
        for i in range(12):
            self.pd_controllers.append(
                PDController(kp=self.position_kp[i], kd=self.position_kd[i], output_limits=velocity_output_limits[i])
            )
        self.arm_velocity_pd_controllers = []
        arm_torques_limits = [ [-20.0, 20.0], [-20.0, 20.0], [-15.0, 15.0], [-7.0, 7.0], [-5.0, 5.0], [-5.0, 5.0]]
        for i in range(6):
            self.arm_velocity_pd_controllers.append(
                PDController(kp=self.arm_vel_kp[i], kd=self.arm_vel_kd[i], output_limits=arm_torques_limits[i])
            )
        self.current_joint_values = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, -0.6, 0.0, 0.0])
        self.current_joint_vel = np.zeros(11)
        self.CBF_filter_velocity = np.zeros(11)
        self.target_base_arm_vel = np.zeros(11)
        self.r_arm = np.array([.036,.029,.029,.029,.029,.029,0.25])
        self.x0,self.y0,self.rectangle_r  = self.caculate_rectangle_from_cuboid(0.5, 0.7, 0.05)
        self.x1,self.y1,self.rectangle_r1  = self.caculate_rectangle_from_cuboid(0.5, 0.05, 0.6)
        self.x2,self.y2,self.rectangle_r2  = self.caculate_rectangle_from_cuboid(0.5, 0.05, 0.6)
        self.x3,self.y3,self.rectangle_r3  = self.caculate_rectangle_from_cuboid(0.3, 0.3, 0.3)       
        self.r_safe_expand = 0.01
        self.safe_R_list = np.array([
            self.r_arm+self.rectangle_r+2.5*self.r_safe_expand,
            self.r_arm+self.rectangle_r1+2.5*self.r_safe_expand,
            self.r_arm+self.rectangle_r2+2.5*self.r_safe_expand,
            self.r_arm+self.rectangle_r3+2.5*self.r_safe_expand,
        ])
        # 障碍物速度
        self.obs_v = np.array([
            [0.0,0.0,0.0],
            [0.0,0.0,0.0],
            [0.0,0.0,0.0],
            [0.0,0.0,0.0],
        ])
        self.dt = None
        self.h_list = np.array([0.0,0.0,0.0])
        self.solve_time = []
        self.ut = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        # A* path planning
        self.planner = ParallelAStarPlanner(self.device)
            
        # 路径相关变量
        self.paths = [None] * self.num_envs
        self.smoothed_paths = [None] * self.num_envs
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.waypoint_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 起点和终点
        self.start_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_positions = torch.tensor([[7.0, 0.0, 0.0]] * self.num_envs, device=self.device)
        
        # 路径规划参数
        self.waypoint_threshold = 0.2
        self.replan_interval = 100
        self.replan_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 在初始化时就规划路径
        self._initial_path_planning()


    def _initial_path_planning(self):
        """初始路径规划 - 包含3D平滑"""
        starts = []
        goals = []
        
        for env_id in range(self.num_envs):
            # 使用初始位置作为起点（包含yaw角度）
            starts.append([0.0, 0.0, 0.0])  # 初始位置 (0,0,0) - 包含yaw
            goals.append([7.0, 0.0, 0.0])   # 目标位置 (7,0,0) - 包含yaw
        
        # 规划路径
        new_paths = self.planner.plan_paths_parallel(starts, goals, self.num_envs)
        
        # 更新路径
        for env_id in range(self.num_envs):
            if new_paths[env_id] is not None:
                # 使用3D平滑路径
                smoothed_path = self.planner.smooth_path_3d(new_paths[env_id])
                full_path = smoothed_path
                
                self.paths[env_id] = new_paths[env_id]  # 保留原始路径
                self.smoothed_paths[env_id] = full_path  # 使用平滑后的路径
                self.current_waypoint_idx[env_id] = 0
                print(f"Environment {env_id} path planned with {len(full_path)} waypoints (smoothed)")
                
                # 打印路径信息用于调试
                print(f"Env {env_id} original path points: {len(new_paths[env_id])}")
                print(f"Env {env_id} smoothed path points: {len(full_path)}")
                print(f"Env {env_id} path: {full_path[:3]}...{full_path[-3:] if len(full_path) > 6 else ''}")
            else:
                print(f"Warning: Path planning failed for env {env_id}")
                # 创建默认的直线路径作为fallback
                default_path = [
                    (0.0, 0.0, 0.0),  # 起点
                    (3.5, 0.0, 0.0),  # 中间点
                    (7.0, 0.0, 0.0)   # 终点
                ]
                self.smoothed_paths[env_id] = default_path
                self.current_waypoint_idx[env_id] = 0




    def caculate_rectangle_from_cuboid(self,a,b,h):
        arr = sorted([a, b, h], reverse=True)        
        return arr[0]-arr[2]/np.sqrt(2),arr[1]-arr[2]/np.sqrt(2),arr[2]/np.sqrt(2) # arr : a b h


    def CBF_start(self):
        """Start CBF process with initial data - based on MuJoCo version"""
        # Update obstacle data
        obstacles = self.update_obstacle_data()
        
        # Get current joint values and velocities
        self.target_base_arm_vel=self.update_base_arm_pos_pid(env_idx=0)
        current_base_arm_pos, current_base_arm_vel=self.update_joint_pos_vel()
        
        # Calculate obstacle velocities
        obs_v = self.obs_v 
        
        # Use 11D state directly (6 base + 5 arm)
        states_11 = current_base_arm_pos
        states_vel_11 = current_base_arm_vel

        # Prepare initial data for CBF process (same format as MuJoCo)
        input_data = {
            "obstacles": obstacles,  # Current obstacle positions
            "target": self.target_base_arm_vel,  # Target velocities (11)
            "current_group_joint_values": current_base_arm_pos,  # 11D state
            "current_group_joint_vel": current_base_arm_vel,  # 11D state velocity
            "safe_R_list": self.safe_R_list,  # Safe radii
            "obs_v": obs_v,  # Obstacle velocities with fault parameter
            "update_beta": True,  # Initialize beta
            "obstacle_type_num": self.obstacle_type_num,
            "T_step": self.T_step,
            "O_T_step": self.O_T_step,  # Use same time step for observer
            "h_threshold": self.h_threshold,
            "CBF_mode": self.mode,
            "out_limite": self.velocity_limite,  # Default output limit
            "dt": self.dt
        }
        
        # Send initial data and start CBF process
        self.CBF_input.put(input_data)
        self.CBF_process.start()

    def CBF_process_func(self, in_queue, out_queue, DOB_input):
        """CBF computation process - based on MuJoCo version"""
        initialize_CBF = True
        CBF_counter = 0
        
        # Wait for initial data
        while in_queue.empty():
            pass
        initial_data = in_queue.get()
        
        # Extract initialization parameters
        obstacle_type_num = initial_data["obstacle_type_num"]
        T_step = initial_data["T_step"]
        O_T_step = initial_data["O_T_step"]
        h_threshold = initial_data["h_threshold"]
        CBF_mode = initial_data["CBF_mode"]
        out_limite = initial_data["out_limite"]
        
        # Initialize CBF controller
        CBF_filter = CBF_controller(
            obstacle_type_num=obstacle_type_num,
            T_step=T_step,
            O_T_step=O_T_step,
            h_threshold=h_threshold,
            out_limite=out_limite,
            CBF_mode=CBF_mode
        )
        
        def initial_beta(obstacles, current_group_joint_values, safe_R_list, obs_v, states_velocity):
            """Initialize beta parameters for CBF"""
            try:
                h0_val = CBF_filter.caculate_barriers(
                    current_group_joint_values, 
                    obstacles[0],  # sphere obstacles
                    obstacles[1],  # capsule obstacles  
                    obstacles[2],  # rectangle obstacles
                    safe_R_list, 
                    obs_v, 
                    states_velocity
                )
                h0 = h0_val.toarray()     
                # Ensure h0 values are positive (avoid division by zero)
                for i in range(len(h0)):
                    if h0[i] <= 0:
                        h0[i] = 0.0001
                
                # Calculate beta values
                beta = (CBF_filter.w0 ** 2) / (2 * h0)
                CBF_filter.set_beta(beta)
                
            except Exception as e:
                print(f"Error in initial_beta: {e}")
                # Set default beta values if calculation fails
                default_beta = np.ones(len(obstacles[0]) + len(obstacles[1]) + len(obstacles[2])) * 0.1
                CBF_filter.set_beta(default_beta)
        
        # Main CBF processing loop
        while True:
            if not in_queue.empty() or initialize_CBF:
                start_time = time.time()
                
                # Reset counter periodically to prevent overflow
                if CBF_counter > 20000:
                    CBF_counter = 0
                
                if initialize_CBF:
                    input_data = initial_data
                    # Send initialization data to DOB

                    O_T_step=CBF_filter.O_T_step
                    alpha=CBF_filter.alpha
                    f=ca.DM(CBF_filter.Ax).toarray()
                    g1=ca.DM(CBF_filter.gx).toarray()
                    g2=ca.DM(CBF_filter.g2).toarray()
                    w0=CBF_filter.w0
                    
                    DOB_data = {
                        "O_T_step": O_T_step,
                        "alpha": alpha,
                        "f": f,
                        "g1": g1,
                        "g2": g2,
                        "w0": w0
                    }
                    DOB_input.put(DOB_data)
                    initialize_CBF = False
                else:
                    input_data = in_queue.get()
                
                # Extract data from input
                obstacles = input_data["obstacles"]
                target = input_data["target"]
                current_group_joint_values = input_data["current_group_joint_values"]
                current_group_joint_vel = input_data["current_group_joint_vel"]
                safe_R_list = input_data["safe_R_list"]
                obs_v = input_data["obs_v"]
                update_beta = input_data["update_beta"]
                dt = input_data["dt"]
                # Update beta if needed
                if update_beta or CBF_counter == 0:
                    initial_beta(obstacles, current_group_joint_values, safe_R_list, obs_v, current_group_joint_vel)
                # Solve CBF-QP using solve_QP5 method

                with SuppressOutput():
                    CBF_filter_velocity, h_list = CBF_filter.solve_QP5(
                        obstacles=obstacles,
                        states_input=current_group_joint_values,
                        states_velocity=current_group_joint_vel,
                        u_input=target,
                        safe_R_list=safe_R_list,
                        obs_v=obs_v,
                        dt=dt
                    )
                
                process_once_time = time.time() - start_time
                
                # Send output data
                output_data = {
                    "CBF_filter_velocity": CBF_filter_velocity,
                    "h_list": h_list,
                    "process_once_time": process_once_time
                }
                out_queue.put(output_data)
                CBF_counter += 1


    def observer_process_func(self,DOB_input,DOB_output):
        initialize_DOB = True
        while DOB_input.empty() is True:
            pass
        Dob_initial_data = self.DOB_input.get()
        O_T_step=Dob_initial_data["O_T_step"]
        alpha=Dob_initial_data["alpha"]
        f=Dob_initial_data["f"]
        g1=Dob_initial_data["g1"]
        g2=Dob_initial_data["g2"]
        w0=Dob_initial_data["w0"]
        Dob = DISTURBANCE_OBSERVER(O_T_step,alpha,f,g1,g2,w0)
        cal_list = []
        while True:
            # s = time.time()
            if DOB_input.empty() is False or initialize_DOB:
                s = time.time()
                if initialize_DOB:

                    initialize_DOB = False
                    out_data={"initialize_DOB":True}
                else:
                    in_data = DOB_input.get()
                    currrent_state=in_data["currrent_state"]
                    ut=in_data["ut"]
                    dt=Dob.update_d(np.array(ut),currrent_state)
                    out_data={"dt":dt}
                DOB_output.put(out_data)
                cal_list.append(time.time()-s)
            if len(cal_list) > 100:
                avg = sum(cal_list) / len(cal_list)
                print("Dob avg =", avg)
                cal_list = []

    def observer_start(self):
        while self.DOB_input.empty():
            pass
        self.DOB_process.start()

    def update_joint_pos_vel(self):
        """Update current joint values from simulation (11D: 6 base + 5 arm)"""
        # Refresh DOF state tensor to get latest joint positions and velocities
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Update base 6DOF position from robot base  
        base_pos = self.state.root_pos[0].cpu().numpy()  # 第0个环境的位置
        # 获取机器人基座四元数 (x, y, z, w)
        base_quat = self.state.root_xyzw_quat[0].cpu().numpy()  # 第0个环境的四元数
        
        # 将四元数转换为欧拉角 (roll, pitch, yaw)
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(base_quat)
        euler_angles = r.as_euler('xyz', degrees=False)
        
        # 组合位置和姿态为6DOF
        base_6dof = np.concatenate([base_pos, euler_angles])

        # 获取手臂关节位置和速度 (索引12-16，共5个DOF用于CBF控制)
        arm_5dof_pos = self.state.dof_pos[0, self.leg_dofs:self.leg_dofs+self.arm_controlled_dofs].cpu().numpy()

        # Update 11D joint values (6 base + 5 arm)
        self.current_joint_values = np.concatenate([base_6dof, arm_5dof_pos])
        
        # For velocities, base velocities come from rigid body state
        base_lin_vel = self.state.root_lin_vel[0].cpu().numpy()
        base_ang_vel = self.state.root_ang_vel[0].cpu().numpy()
        arm_5dof_vel = self.state.dof_vel[0, self.leg_dofs:self.leg_dofs+self.arm_controlled_dofs].cpu().numpy()
        base_6dof_vel = np.concatenate([base_lin_vel, base_ang_vel])

        self.current_joint_vel = np.concatenate([base_6dof_vel, arm_5dof_vel])

        return self.current_joint_values, self.current_joint_vel

    def update_obstacle_data(self):

        """Update obstacle positions and velocities"""
        # 将CUDA张量转换为CPU numpy数组
        # 注意：现在使用4个box作为障碍物（box1, box2, box3, box4）
        num_actor_per_env = 5
        obstacles_pos = self.state.root_pos[1:num_actor_per_env].cpu().numpy()
        obstacles_vel = self.state.root_lin_vel[1:num_actor_per_env].cpu().numpy()

        rectangle_obstacle_input = np.array([[obstacles_pos[0][0],obstacles_pos[0][1],obstacles_pos[0][2],
                                             obstacles_pos[0][0]-self.x0/2,obstacles_pos[0][1]+self.y0/2,obstacles_pos[0][2],
                                             obstacles_pos[0][0]+self.x0/2,obstacles_pos[0][1]+self.y0/2,obstacles_pos[0][2],
                                             obstacles_pos[0][0]-self.x0/2,obstacles_pos[0][1]-self.y0/2,obstacles_pos[0][2]],
                                             [obstacles_pos[1][0],obstacles_pos[1][1],obstacles_pos[1][2],
                                             obstacles_pos[1][0]-self.x1/2,obstacles_pos[1][1]+self.y1/2,obstacles_pos[1][2],
                                             obstacles_pos[1][0]+self.x1/2,obstacles_pos[1][1]+self.y1/2,obstacles_pos[1][2],
                                             obstacles_pos[1][0]-self.x1/2,obstacles_pos[1][1]-self.y1/2,obstacles_pos[1][2]],
                                             [obstacles_pos[2][0],obstacles_pos[2][1],obstacles_pos[2][2],
                                             obstacles_pos[2][0]-self.x2/2,obstacles_pos[2][1]+self.y2/2,obstacles_pos[2][2],
                                             obstacles_pos[2][0]+self.x2/2,obstacles_pos[2][1]+self.y2/2,obstacles_pos[2][2],
                                             obstacles_pos[2][0]-self.x2/2,obstacles_pos[2][1]-self.y2/2,obstacles_pos[2][2]],
                                             [obstacles_pos[3][0],obstacles_pos[3][1],obstacles_pos[3][2],
                                             obstacles_pos[3][0]-self.x3/2,obstacles_pos[3][1]+self.y3/2,obstacles_pos[3][2],
                                             obstacles_pos[3][0]+self.x3/2,obstacles_pos[3][1]+self.y3/2,obstacles_pos[3][2],
                                             obstacles_pos[3][0]-self.x3/2,obstacles_pos[3][1]-self.y3/2,obstacles_pos[3][2]],
                                             ]) 
    
        return [np.array([]),np.array([]),rectangle_obstacle_input]

    def get_constraint_info(self, constraint_idx):
        """解析约束索引对应的约束信息"""
        # 根据CBF控制器的结构解析约束
        # obstacle_type_num = [0, 0, 4] (sphere, capsule, rectangle)
        # n_conpoment = 7 (机械臂组件数)
        # n_base_cbf = 3 (基础CBF数量)
        # n_statistic_cbf = 8 (统计CBF数量)
        
        sphere_num = self.obstacle_type_num[0]  # 0
        capsule_num = self.obstacle_type_num[1]  # 0  
        rectangle_num = self.obstacle_type_num[2]  # 4
        n_conpoment = 7
        n_base_cbf = 3
        n_statistic_cbf = 8
        
        totle_num = sphere_num + capsule_num + rectangle_num  # 4
        obstacle_constraints = totle_num * n_conpoment  # 4 * 7 = 28
        
        if constraint_idx < obstacle_constraints:
            # 障碍物约束
            obstacle_idx = constraint_idx // n_conpoment
            component_idx = constraint_idx % n_conpoment
            
            if obstacle_idx < sphere_num:
                obstacle_type = "sphere"
                obstacle_id = obstacle_idx
            elif obstacle_idx < sphere_num + capsule_num:
                obstacle_type = "capsule" 
                obstacle_id = obstacle_idx - sphere_num
            else:
                obstacle_type = "rectangle"
                obstacle_id = obstacle_idx - sphere_num - capsule_num
            
            component_names = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
            component_name = component_names[component_idx] if component_idx < len(component_names) else f"component_{component_idx}"
            
            return f"{obstacle_type}_{obstacle_id}_{component_name}"
            
        elif constraint_idx < obstacle_constraints + n_base_cbf:
            # 基础CBF约束
            base_idx = constraint_idx - obstacle_constraints
            return f"base_cbf_{base_idx}"
            
        else:
            # 统计CBF约束
            stat_idx = constraint_idx - obstacle_constraints - n_base_cbf
            return f"statistic_cbf_{stat_idx}"

    def get_joint3_constraints(self):
        """获取joint3相关的所有约束值"""
        if not hasattr(self, 'h_list') or len(self.h_list) == 0:
            return {}
        
        joint3_constraints = {}
        
        # obstacle_type_num = [0, 0, 4]
        # n_conpoment = 7
        # joint3对应component_idx = 2 (link2)
        
        sphere_num = self.obstacle_type_num[0]  # 0
        capsule_num = self.obstacle_type_num[1]  # 0  
        rectangle_num = self.obstacle_type_num[2]  # 4
        n_conpoment = 7
        
        totle_num = sphere_num + capsule_num + rectangle_num  # 4
        
        # 查找所有joint3相关的约束
        for obstacle_idx in range(totle_num):
            constraint_idx = obstacle_idx * n_conpoment + 2  # joint3 = link2, component_idx = 2
            if constraint_idx < len(self.h_list):
                obstacle_type = "rectangle" if obstacle_idx >= sphere_num + capsule_num else "unknown"
                obstacle_id = obstacle_idx - sphere_num - capsule_num if obstacle_idx >= sphere_num + capsule_num else obstacle_idx
                constraint_name = f"{obstacle_type}_{obstacle_id}_link2"
                joint3_constraints[constraint_name] = self.h_list[constraint_idx]
        
        return joint3_constraints

    def update_base_arm_pos_pid(self,env_idx):  
        """使用路径点更新基座和手臂位置"""
        # 获取当前路径点作为目标
        current_waypoint = self.smoothed_paths[env_idx][self.current_waypoint_idx[env_idx]]

        # 使用路径点作为基座目标
        target_base_arm_joint = [
            current_waypoint[0].item(),  # x
            current_waypoint[1].item(),  # y 
            0.3,                         # z (固定高度)
            0.0, 0.0,                    # roll, pitch
            current_waypoint[2].item(),  # yaw
            0.0, 0.5, -0.6, 0.0, 0.0    # 手臂关节
        ]
        
        target_base_arm_vel = []
        for i in range(11):
            if i < 6:  # 基座
                vel = self.pd_controllers[i].update(
                    target_base_arm_joint[i], 
                    self.current_joint_values[i]
                )
                target_base_arm_vel.append(vel)
            else:  # 手臂
                vel = self.position_kp[i] * (target_base_arm_joint[i] - self.current_joint_values[i]) - self.position_kd[i] * self.current_joint_vel[i]
                target_base_arm_vel.append(vel)



        return target_base_arm_vel


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

    # 修正路径跟踪方法
    def _update_path_tracking(self):
        """更新路径跟踪状态"""
        num_actors_per_env = 5  # 每个环境有5个actor
        
        for env_id in range(self.num_envs):
            if (self.smoothed_paths[env_id] is None or 
                self.current_waypoint_idx[env_id] >= len(self.smoothed_paths[env_id])):
                continue
            
            # 获取当前路径点
            current_waypoint = self.smoothed_paths[env_id][self.current_waypoint_idx[env_id]]
            
            # 获取机器人当前位置（第一个actor是机器人）
            robot_actor_idx = env_id * num_actors_per_env
            robot_pos = self.state.root_state[robot_actor_idx, 0:3].cpu().numpy()  # [x, y, z]
            
            # 计算到当前路径点的距离（只考虑x,y）
            distance = np.sqrt(
                (robot_pos[0] - current_waypoint[0].item())**2 + 
                (robot_pos[1] - current_waypoint[1].item())**2
            )
            
            # 检查是否到达路径点
            if distance < self.waypoint_threshold:
                if self.current_waypoint_idx[env_id] < len(self.smoothed_paths[env_id]) - 1:
                    old_waypoint = self.smoothed_paths[env_id][self.current_waypoint_idx[env_id]]
                    self.current_waypoint_idx[env_id] += 1
                    new_waypoint = self.smoothed_paths[env_id][self.current_waypoint_idx[env_id]]
                    print(f"Env {env_id}: Reached waypoint {old_waypoint}, moving to {new_waypoint}")
                else:
                    self.waypoint_reached[env_id] = True
                    print(f"Env {env_id}: Reached final waypoint!")

    def _check_replan(self):
        """检查是否需要重新规划路径"""
        self.replan_counter += 1
        
        need_replan = []
        for env_id in range(self.num_envs):
            # 每replan_interval步重新规划一次，或者如果机器人偏离路径太远
            if self.replan_counter[env_id] >= self.replan_interval:
                need_replan.append(env_id)
                self.replan_counter[env_id] = 0
        
        if need_replan:
            self._replan_paths(need_replan)



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

        if rendering and self.debug_viz:
            self.visualize(vis_env_ids=[0])  # the rendering env

        if return_vis and self.vis_cam_handle is not None:
            vis = self.render(sync_frame_time=False)
            if vis is not None:
                info["vis"] = vis
        decimation_count = self.controller.decimation_count
        time_start = time.time()

        # 路径跟踪和更新
        self._update_path_tracking()  
        # 定期重新规划路径（如果需要）
        # self._check_replan()


        for decimation_step in range(decimation_count):


            # CBF 
            if self.CBF_output.empty() is False:
                out_data=self.CBF_output.get()
                CBF_filter_velocity=out_data["CBF_filter_velocity"]
                self.h_list=out_data["h_list"]
                process_once_time=out_data["process_once_time"]
                CBF_filter_velocity=np.array(CBF_filter_velocity.toarray())
                self.ut = CBF_filter_velocity.tolist()
                self.CBF_filter_velocity = np.array(self.ut).reshape(len(self.ut))
                self.h_list_min=self.h_list.min()
                self.solve_time.append(process_once_time)
                
                # # 找到h_list_min对应的约束
                # if hasattr(self, 'h_list') and len(self.h_list) > 0:
                #     min_idx = np.argmin(self.h_list)
                #     constraint_info = self.get_constraint_info(min_idx)
                #     print(f"\n=== 最小约束信息 ===")
                #     print(f"最小约束索引: {min_idx}")
                #     print(f"约束信息: {constraint_info}")
                #     print(f"约束值: {self.h_list[min_idx]:.6f}")
                #     print("==================")

            self.target_base_arm_vel=self.update_base_arm_pos_pid(env_idx=0)
            current_base_arm_pos, current_base_arm_vel=self.update_joint_pos_vel()

            self.obstacles=self.update_obstacle_data()
    
            input_data={"obstacles":self.obstacles,
            "target":self.target_base_arm_vel,
            "current_group_joint_values":current_base_arm_pos,
            "current_group_joint_vel":current_base_arm_vel,
            "safe_R_list":self.safe_R_list,
            "obs_v":self.obs_v*self.para_v_fault,
            "update_beta":self.update_beta,
            "dt":self.dt}
            self.CBF_input.put(input_data)

            if self.DOB_output.empty() is False:
                DOB_out=self.DOB_output.get()
                if "initialize_DOB" in DOB_out:
                    pass
                else:
                    self.dt=DOB_out["dt"]
                DOB_in={'currrent_state':current_base_arm_pos,'ut':self.ut}
                self.DOB_input.put(DOB_in)



            callback(self) if callback is not None else None
            # handle delay by indexing into the buffer of past targets
            # since new actions are pushed to the front of the buffer,
            # the current target is further back in the buffer for larger
            # delays.
            curr_target_idx = torch.ceil(
                ((self.ctrl_delay_steps - decimation_step)) / decimation_count
            ).long()
            assert (curr_target_idx >= 0).all()
            self.ctrl.torque = self.controller(
                action=self.ctrl.buffer.permute(2, 1, 0)[
                    torch.arange(self.num_actions, device=self.device),
                    curr_target_idx,
                    :,
                ].permute(1, 0),
                state=self.state,
            )
            arm_torques = []
            for i in range(5):
                # 直接使用CBF输出的速度作为目标速度，计算速度误差
                target_velocity = self.CBF_filter_velocity[i+6]
                current_velocity = self.current_joint_vel[i+6]
                velocity_error = target_velocity - current_velocity
                
                # 简化的速度控制：只使用比例项
                torque = self.arm_vel_kp[i] * velocity_error
                arm_torques.append(torque)
                
            # Convert list to tensor and replace elements 13-17 (indices 12-16) of self.ctrl.torque with arm_torques
            arm_torques_tensor = torch.tensor(arm_torques, device=self.device, dtype=self.ctrl.torque.dtype)
            self.ctrl.torque[..., 12:17] = arm_torques_tensor




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
            num_actors_per_env = 5
            robot_actor_ids = torch.arange(0, self.num_envs * num_actors_per_env, num_actors_per_env, device=self.device)
            self.state.root_state[robot_actor_ids, 7:13] = torch_rand_float(
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
            num_actors_per_env = 5
            robot_actor_ids = torch.arange(0, self.num_envs * num_actors_per_env, num_actors_per_env, device=self.device)
            self.state.root_state[robot_actor_ids, 0:3] += (
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
            self.state.root_state[robot_actor_ids, 3:7] = quat_mul(
                self.state.root_state[robot_actor_ids, 3:7],
                quat_wxyz_transport[..., [1, 2, 3, 0]],  # reorder to xyzw
            )

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.state.root_state)
            )
        self.check_termination(state=self.state, control=self.ctrl)
        
        # 检查约束终止
        constraint_terminations = {}
        for constraint_name, constraint in self.constraints.items():
            constraint_termination = constraint.check_termination(state=self.state, control=self.ctrl)
            info[f"constraint/{constraint_name}/termination"] = constraint_termination
            constraint_terminations[constraint_name] = constraint_termination
            self.reset_buf |= constraint_termination
        
        # 打印约束终止原因
        if self.reset_buf.any():
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            for env_id in env_ids:
                env_id_int = env_id.item()
                constraint_reasons = []
                
                for constraint_name, constraint_termination in constraint_terminations.items():
                    if constraint_termination[env_id_int]:
                        constraint_reasons.append(f"约束违反: {constraint_name}")
                
                if constraint_reasons:
                    print(f"\n⚠️  环境 {env_id_int} 约束终止: {', '.join(constraint_reasons)}")
                    print("=" * 50)
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

        # 把CBF解算的速度替代obs的vel_cmd
        obs_his = self.obs_history.view(self.num_envs, -1)
        # print("arm_filter_velocity",self.CBF_filter_velocity[6],self.CBF_filter_velocity[7],self.CBF_filter_velocity[8],
        #                             self.CBF_filter_velocity[9],self.CBF_filter_velocity[10])
        # print("self.CBF_filter_velocity",self.CBF_filter_velocity[0],self.CBF_filter_velocity[1],self.CBF_filter_velocity[5])
        obs_his[:,78] = self.CBF_filter_velocity[0] * 2
        # obs_his[:,78] = self.target_base_arm_vel[0] * 2
        # obs_his[:,78] = 0.5 * 2
        obs_his[:,79] = self.CBF_filter_velocity[1] * 2
        # obs_his[:,79] = self.target_base_arm_vel[1] * 2
        obs_his[:,80] = self.CBF_filter_velocity[5] * 2 # scale
        # obs_his[:,80] = self.target_base_arm_vel[5] * 2 # scale
        # obs_his[:,78] = 0.5 * 2 # vx
        # obs_his[:,79] = 0.0 # vy
        # obs_his[:,80] = 0.0 # wz
        # obs_his[:,81] = 0.3 # wx
        # print("########################################################")
        # print("self.CBF_filter_velocity",self.CBF_filter_velocity[0],self.CBF_filter_velocity[1],self.CBF_filter_velocity[5])
        # print("arm_torques",self.ctrl.torque[..., 12:])
        return (
            # self.obs_history.view(self.num_envs, -1),
            obs_his,
            privileged_obs,
            reward,
            self.reset_buf,
            info,
        )

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
        num_actors_per_env = 5
        walked_off_safe_bounds = torch.logical_or(
            (self.state.root_pos[::num_actors_per_env, :2] < self.safe_bounds[None, :, 0]).any(dim=1),
            (self.state.root_pos[::num_actors_per_env, :2] > self.safe_bounds[None, :, 1]).any(dim=1),
        )
        self.time_out_buf |= walked_off_safe_bounds
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)  # 这里已经包含了box重置
        for task in self.tasks.values():
            task.reset_idx(env_ids)
        for constraint in self.constraints.values():
            constraint.reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0

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
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


        # 重置路径规划相关状态
        self.current_waypoint_idx[env_ids] = 0
        self.waypoint_reached[env_ids] = False
        self.replan_counter[env_ids] = 0
        
        # 为重置的环境重新规划路径
        self._replan_paths(env_ids)

    def _replan_paths(self, env_ids):
        """为指定环境重新规划路径 - 修复3D路径处理"""
        starts = []
        goals = []
        
        for env_id in env_ids:
            # 获取机器人当前位置和朝向作为起点
            num_actor_per_env = 5
            root_pos = self.state.root_state[env_id*num_actor_per_env, :6]
            
            # 获取当前yaw角度
            robot_yaw = root_pos[5]
            
            starts.append((root_pos[0], root_pos[1], robot_yaw))
            goals.append((self.goal_positions[env_id, 0].item(), 
                        self.goal_positions[env_id, 1].item(), 
                        0.0))  # 目标朝向设为0
        
        # 规划路径
        new_paths = self.planner.plan_paths_parallel(starts, goals, len(env_ids))
        
        # 更新路径
        for i, env_id in enumerate(env_ids):
            if new_paths[i] is not None:
                # 使用3D平滑路径
                if hasattr(self.planner, 'smooth_path_3d'):
                    smoothed_path = self.planner.smooth_path_3d(new_paths[i])
                else:
                    # 如果没有3D平滑方法，直接使用原始路径
                    smoothed_path = new_paths[i]
                
                full_path = smoothed_path
                
                self.paths[env_id] = new_paths[i]  # 保留原始路径
                self.smoothed_paths[env_id] = full_path  # 使用平滑后的路径
                self.current_waypoint_idx[env_id] = 0
                
                print(f"Env {env_id} replanned path with {len(full_path)} waypoints")
                
            else:
                # 路径规划失败，使用直线路径
                print(f"Warning: Path planning failed for env {env_id}, using straight line")
                start = starts[i]
                goal = goals[i]
                self.smoothed_paths[env_id] = [
                    (start[0], start[1], start[2]),
                    (goal[0], goal[1], goal[2])
                ]
                self.current_waypoint_idx[env_id] = 0



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
        num_actors_per_env = 5
        for name, obs_attr in state_obs.items():
            value = obs_attr(struct=state, generator=self.generator)
            if name == "root_ang_vel":
                value = value[::num_actors_per_env]
            if name == "root_lin_vel":
                value = value[::num_actors_per_env]
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
                if k == "reaching":
                    task_obs = torch.zeros_like(task_obs)
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


        return torch.cat(
            (
                setup_obs_tensor,
                state_obs_tensor,
                task_obs_tensor,
                self.ctrl.action,
            ),
            dim=1,
        )

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
        # 计算机器人actor的索引（每4个actor一组：机器人, box1, box2, box3）
        num_actors_per_env = 5
        robot_actor_ids = env_ids * num_actors_per_env
        
        # 重置机器人状态
        self.state.root_state[robot_actor_ids] = self.base_init_state
        if (self.init_pos_noise > 0).any():
            self.state.root_state[robot_actor_ids, 0:3] += torch_rand_float(
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
            self.state.root_state[robot_actor_ids, 3:7] = quat_mul(
                self.state.root_state[robot_actor_ids, 3:7], quat_xyzw
            )
        if (self.init_lin_vel_noise > 0).any():
            self.state.root_state[robot_actor_ids, 7:10] += torch_rand_float(
                -self.init_lin_vel_noise,
                self.init_lin_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        if (self.init_ang_vel_noise > 0).any():
            self.state.root_state[robot_actor_ids, 10:13] += torch_rand_float(
                -self.init_ang_vel_noise,
                self.init_ang_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        self.state.root_state[robot_actor_ids, :3] += self.env_origins[env_ids]
        
        # 重置4个box状态
        box1_actor_ids = env_ids * num_actors_per_env + 1  # box1索引
        box2_actor_ids = env_ids * num_actors_per_env + 2  # box2索引
        box3_actor_ids = env_ids * num_actors_per_env + 3  # box3索引
        box4_actor_ids = env_ids * num_actors_per_env + 4  # box4索引
        self._reset_box_states_in_root_state(env_ids, box1_actor_ids, box2_actor_ids, box3_actor_ids, box4_actor_ids)
        
        # 设置所有actor的状态（包括机器人和4个box）
        # 创建包含机器人和所有box的actor索引
        all_actor_ids = torch.cat([
            robot_actor_ids,    # 机器人actor索引
            box1_actor_ids,     # box1 actor索引
            box2_actor_ids,     # box2 actor索引
            box3_actor_ids,     # box3 actor索引
            box4_actor_ids      # box4 actor索引
        ]).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.state.root_state),
            gymtorch.unwrap_tensor(all_actor_ids),
            len(all_actor_ids),
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

        # 创建box assets
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.density = 1000.0  # 水的密度
        box_asset_options.fix_base_link = True  # 设置为固定基座
        
        # 创建4个不同尺寸的box
        box_asset_1 = self.gym.create_box(
            self.sim, 0.5, 1.4, 0.01, box_asset_options
        )
        box_asset_2 = self.gym.create_box(
            self.sim, 0.5, 0.05, 0.6, box_asset_options
        )
        box_asset_3 = self.gym.create_box(
            self.sim, 0.5, 0.05, 0.6, box_asset_options
        )
        box_asset_4 = self.gym.create_box(
            self.sim, 0.3, 0.3, 0.3, box_asset_options
        )

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
        self.actor_handles = []
        self.box_handles = []  # 添加box actor句柄列表，每个环境4个box
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
            
            # 创建4个box actors
            box_handles_env = []
            
            # Box 1: 大box (0.5x0.7x0.05) - 在机器人前方
            box1_offset = gymapi.Vec3(4.0, 0.0, 0.6)
            # box1_offset = gymapi.Vec3(2.0, 0.0, 2.0)
            box1_pose = gymapi.Transform()
            box1_pose.p = start_pose.p + box1_offset
            box1_pose.r = start_pose.r
            
            box1_handle = self.gym.create_actor(
                env_handle,
                box_asset_1,
                box1_pose,
                f"box1_{i}",
                i,
                0,  # collision group
                0,  # collision filter
            )
            
            # Box 2: 小box (0.05x0.05x0.6) - 在机器人左侧
            box2_offset = gymapi.Vec3(4.0, -0.7, 0.3)
            # box2_offset = gymapi.Vec3(2.0, -0.35, 2.0)
            box2_pose = gymapi.Transform()
            box2_pose.p = start_pose.p + box2_offset
            box2_pose.r = start_pose.r
            
            box2_handle = self.gym.create_actor(
                env_handle,
                box_asset_2,
                box2_pose,
                f"box2_{i}",
                i,
                0,  # collision group
                0,  # collision filter
            )
            
            # Box 3: 小box (0.05x0.05x0.6) - 在机器人右侧
            box3_offset = gymapi.Vec3(4.0, 0.7, 0.3)
            # box3_offset = gymapi.Vec3(2.0, 0.35, 2.0)
            box3_pose = gymapi.Transform()
            box3_pose.p = start_pose.p + box3_offset
            box3_pose.r = start_pose.r
            
            box3_handle = self.gym.create_actor(
                env_handle,
                box_asset_3,
                box3_pose,
                f"box3_{i}",
                i,
                0,  # collision group
                0,  # collision filter
            )
            
            # Box 4: 小box (0.3x0.3x0.3) - 放在机器人前方的地上
            box4_offset = gymapi.Vec3(2.0, 0.0, 1.0)  # 高度为0.3的一半，放在地上
            box4_pose = gymapi.Transform()
            box4_pose.p = start_pose.p + box4_offset
            box4_pose.r = start_pose.r
            
            box4_handle = self.gym.create_actor(
                env_handle,
                box_asset_4,
                box4_pose,
                f"box4_{i}",
                i,
                0,  # collision group
                0,  # collision filter
            )

            # 设置box颜色
            colors = [
                gymapi.Vec3(1.0, 1.0, 1.0),  # 白色 - box1
                gymapi.Vec3(1.0, 1.0, 1.0),  # 白色 - box2
                gymapi.Vec3(1.0, 1.0, 1.0),  # 白色 - box3
                gymapi.Vec3(1.0, 1.0, 1.0),  # 白色 - box4
            ]
            
            for j, (box_handle, color) in enumerate(zip([box1_handle, box2_handle, box3_handle, box4_handle], colors)):
                self.gym.set_rigid_body_color(
                    env_handle, box_handle, 0, 
                    gymapi.MESH_VISUAL_AND_COLLISION, color
                )
                box_handles_env.append(box_handle)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.box_handles.append(box_handles_env)

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

        # env_ids_int32 就是 actor indice 
        num_actors_per_env = 5
        env_ids_int32 = (env_ids * num_actors_per_env).to(dtype=torch.int32)
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
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

    def _reset_box_states_in_root_state(self, env_ids, box1_actor_ids, box2_actor_ids, box3_actor_ids, box4_actor_ids):
        """在root_state中重置4个box物体的位置和状态"""
        num_actors_per_env = 5
        for i, env_id in enumerate(env_ids):
            if env_id < len(self.box_handles):
                # 获取机器人的位置（世界坐标）
                robot_actor_id = env_id * num_actors_per_env  # 修正：每个环境有5个actor
                robot_pos = self.state.root_state[robot_actor_id, 0:3]
                
                # 定义4个box的偏移位置
                box_offsets = [
                    torch.tensor([4.0, 0.0, 0.6], device=self.device),   # box1: 前方
                    torch.tensor((4.0, -0.7, 0.3), device=self.device),  # box2: 左侧
                    torch.tensor([4.0, 0.7, 0.3], device=self.device),   # box3: 右侧
                    torch.tensor([2.0, 0.0, 0.15], device=self.device),   # box4: 前方
                    # torch.tensor([2.0, 0.0, 2.0], device=self.device),   # box1: 前方
                    # torch.tensor([2.0, -0.35, 2.0], device=self.device),  # box2: 左侧
                    # torch.tensor([2.0, 0.35, 2.0], device=self.device),   # box3: 右侧
                ]
                
                box_actor_ids = [box1_actor_ids[i], box2_actor_ids[i], box3_actor_ids[i], box4_actor_ids[i]]
                
                # 重置每个box
                for box_actor_id, box_offset in zip(box_actor_ids, box_offsets):
                    box_pos = robot_pos + box_offset
                    box_pos[2] = box_offset[2]
                    # 无旋转
                    box_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # w, x, y, z
                    
                    # 设置box状态到root_state中
                    self.state.root_state[box_actor_id, 0:3] = box_pos
                    self.state.root_state[box_actor_id, 3:7] = box_quat
                    self.state.root_state[box_actor_id, 7:10] = 0.0  # 线速度
                    self.state.root_state[box_actor_id, 10:13] = 0.0  # 角速度

    def _reset_box_states(self, env_ids):
        """重置box物体的位置和状态（旧方法，保留兼容性）"""
        # 这个方法现在调用新的方法
        num_actors_per_env = 5
        box1_actor_ids = env_ids * num_actors_per_env + 1
        box2_actor_ids = env_ids * num_actors_per_env + 2
        box3_actor_ids = env_ids * num_actors_per_env + 3
        box4_actor_ids = env_ids * num_actors_per_env + 4
        self._reset_box_states_in_root_state(env_ids, box1_actor_ids, box2_actor_ids, box3_actor_ids, box4_actor_ids)

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
            ) + self.state.root_pos.unsqueeze(1)
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

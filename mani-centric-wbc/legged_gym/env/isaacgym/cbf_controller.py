import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class IsaacGymCBF:
    """IsaacGym版本的CBF控制器，替代CasADi实现"""
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        obstacle_type_num: List[int] = [2, 0, 0],
        T_step: float = 0.01,
        h_threshold: float = 0.08,
        CBF_mode: str = '00',
        use_statistic_obstacle: bool = True,
        **kwargs
    ):
        self.num_envs = num_envs
        self.device = device
        self.obstacle_type_num = obstacle_type_num
        self.T_step = T_step
        self.h_threshold = h_threshold
        self.CBF_mode = CBF_mode
        self.use_statistic_obstacle = use_statistic_obstacle
        
        # 系统参数 - 动态设置，而不是硬编码
        self.u_len = None  # 控制输入维度，将在运行时动态设置
        self.state_dim = None  # 状态维度，将在运行时动态设置
        self.n_component = 7  # 机械臂组件数
        self.n_base_cbf = 3  # 基础CBF数量
        
        # 障碍物数量
        self.sphere_obstacle_num = obstacle_type_num[0]
        self.capsule_obstacle_num = obstacle_type_num[1]
        self.rectangle_obstacle_num = obstacle_type_num[2]
        
        # 安全参数
        self.h_danger = 0.0018
        self.normal_gama = 20.0
        self.h_danger_p_gama = -100.0
        self.h_danger_n_gama = 500.0
        
        # 根据CBF模式调整参数
        self._set_cbf_mode_params()
        
        # 初始化张量缓冲区
        self._init_tensors()
        
        # 初始化静态障碍物
        self._init_static_obstacles()
        
    def set_control_dimension(self, u_len: int):
        """动态设置控制输入维度
        
        Args:
            u_len: 控制输入维度
        """
        self.u_len = u_len
        # 重新初始化相关张量
        self._init_tensors()
        
    def set_state_dimension(self, state_dim: int):
        """动态设置状态维度
        
        Args:
            state_dim: 状态维度
        """
        self.state_dim = state_dim
        
    def _set_cbf_mode_params(self):
        """根据CBF模式设置参数"""
        if self.CBF_mode == '00':
            self.h_danger_p_gama = -2.0
            self.h_danger_n_gama = 5.0
        elif self.CBF_mode == '10':
            self.h_danger_p_gama = -20.0
            self.h_danger_n_gama = 20.0
        elif self.CBF_mode == '01':
            self.h_danger_p_gama = -20.0
            self.h_danger_n_gama = 20.0
        elif self.CBF_mode == '11':
            self.h_danger_p_gama = -100.0
            self.h_danger_n_gama = 600.0
            
    def _init_tensors(self):
        """初始化PyTorch张量缓冲区"""
        # CBF约束数量
        self.n_CBF_constraints = (
            (self.sphere_obstacle_num + self.capsule_obstacle_num + self.rectangle_obstacle_num) 
            * self.n_component + self.n_base_cbf
        )
        
        if self.use_statistic_obstacle:
            self.n_CBF_constraints += 4 * self.n_component + 8  # 静态障碍物 + 环境限制
            
        # 如果u_len未设置，使用默认值
        u_len = self.u_len if self.u_len is not None else 18  # 默认使用18个关节
        # 如果state_dim未设置，使用默认值
        state_dim = self.state_dim if self.state_dim is not None else 25  # 默认使用25维状态
            
        # 初始化张量
        self.h = torch.zeros(self.num_envs, self.n_CBF_constraints, device=self.device)
        # dhdx的维度应该是 [num_envs, n_CBF_constraints, state_dim]，不是u_len
        self.dhdx = torch.zeros(self.num_envs, self.n_CBF_constraints, state_dim, device=self.device)
        self.H = torch.zeros(self.num_envs, self.n_CBF_constraints, device=self.device)
        self.G = torch.zeros(self.num_envs, self.n_CBF_constraints, u_len, device=self.device)
        self.K = torch.zeros(self.num_envs, self.n_CBF_constraints, u_len, device=self.device)
        
        # 安全半径列表
        self.safe_R_list = torch.ones(
            self.num_envs, 
            self.sphere_obstacle_num + self.capsule_obstacle_num + self.rectangle_obstacle_num,
            self.n_component,
            device=self.device
        ) * 0.1  # 默认安全半径
        
    def _init_static_obstacles(self):
        """初始化静态障碍物"""
        if self.use_statistic_obstacle:
            # 静态胶囊障碍物位置
            self.static_capsule_obstacles = torch.tensor([
                [0, 3.1, 0.90824, -1.6, 3.1, 0.091763],
                [0, 3.1, 0.90824, 1.6, 3.1, 0.091763],
                [-1.25, 5.5, 0.7, -1.25, 4.5, 0.7],
                [1.25, 5.5, 0.7, 1.25, 4.5, 0.7],
            ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
            
            # 安全半径 - 动态设置，而不是硬编码
            # 根据实际机器人配置调整
            self.r_arm = torch.tensor([0.036, 0.029, 0.029, 0.029, 0.029, 0.07, 0.4], device=self.device)
            self.r_safe_expand = 0.01
            self.capsule_r = torch.tensor([0.05, 0.05, 0.55, 0.55], device=self.device)
            
    def compute_forward_kinematics(self, base_pos: torch.Tensor, base_quat: torch.Tensor, 
                                 joint_angles: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算机械臂前向运动学
        
        Args:
            base_pos: [num_envs, 3] 基座位置
            base_quat: [num_envs, 4] 基座四元数
            joint_angles: [num_envs, num_joints] 关节角度
            
        Returns:
            机械臂各段位置字典
        """
        # 这里需要实现完整的运动学计算
        # 简化版本，实际需要完整的DH参数和变换矩阵
        arm_points = {}
        
        # 基座变换矩阵
        base_transform = self._quat_to_transform_matrix(base_quat, base_pos)
        
        # 计算机械臂各段位置
        # 这里需要根据具体的机械臂参数实现
        # 暂时返回占位符
        
        return arm_points
        
    def _quat_to_transform_matrix(self, quat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """四元数+位置转换为4x4变换矩阵"""
        # 四元数转旋转矩阵
        qw, qx, qy, qz = quat.unbind(-1)
        
        # 旋转矩阵
        R = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy),
            2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx),
            2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).view(-1, 3, 3)
        
        # 4x4变换矩阵
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        
        return T
        
    def compute_distance_to_obstacles(self, arm_points: Dict[str, torch.Tensor], 
                                    obstacles: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算机械臂到障碍物的距离
        
        Args:
            arm_points: 机械臂各段位置
            obstacles: 障碍物信息
            
        Returns:
            [num_envs, n_CBF_constraints] 距离矩阵
        """
        distances = torch.zeros(self.num_envs, self.n_CBF_constraints, device=self.device)
        
        # 实现距离计算逻辑
        # 这里需要根据障碍物类型分别处理
        
        return distances
        
    def compute_cbf_constraints(self, state: torch.Tensor, obstacles: Dict[str, torch.Tensor],
                              obs_v: torch.Tensor, dt: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算CBF约束
        
        Args:
            state: [num_envs, state_dim] 系统状态
            obstacles: 障碍物信息
            obs_v: 障碍物速度
            dt: 扰动估计
            
        Returns:
            CBF约束信息
        """
        # 提取状态信息
        base_pos = state[:, :3]
        base_quat = state[:, 3:7]
        # 动态获取关节数量，而不是硬编码12
        joint_angles = state[:, 7:]
        
        # 计算前向运动学
        arm_points = self.compute_forward_kinematics(base_pos, base_quat, joint_angles)
        
        # 计算距离
        distances = self.compute_distance_to_obstacles(arm_points, obstacles)
        
        # 计算CBF函数值
        self.h = distances**2 - self.safe_R_list.view(self.num_envs, -1)**2
        
        # 计算雅可比矩阵（简化版本）
        self._compute_jacobians(state)
        
        # 计算CBF约束
        self._compute_cbf_constraints(state, obs_v, dt)
        
        return {
            'h': self.h,
            'H': self.H,
            'G': self.G,
            'K': self.K
        }
        
    def _compute_jacobians(self, state: torch.Tensor):
        """计算雅可比矩阵（简化版本）"""
        # 这里需要实现完整的雅可比计算
        # 暂时使用数值微分或解析解
        
        # 检查u_len和state_dim是否已设置
        if self.u_len is None:
            # 动态设置u_len为控制输入维度（通常是关节数量）
            self.set_control_dimension(state.shape[1] - 7)  # 减去base_pos(3) + base_quat(4)
        if self.state_dim is None:
            # 动态设置state_dim为状态维度
            self.set_state_dimension(state.shape[1])
        
        # 数值微分方法
        eps = 1e-6
        for i in range(self.u_len):
            state_plus = state.clone()
            state_plus[:, i] += eps
            
            # 计算h(state_plus) - h(state) / eps
            # 这里需要重新计算前向运动学和距离
            # 暂时跳过，因为需要完整的运动学实现
            pass
            
    def _compute_cbf_constraints(self, state: torch.Tensor, obs_v: torch.Tensor, dt: Optional[torch.Tensor] = None):
        """计算CBF约束矩阵"""
        # 根据CBF模式计算H, G, K矩阵
        # 注意：这里需要确保dhdx的维度与state匹配
        # dhdx: [num_envs, n_CBF_constraints, state_dim]
        # state: [num_envs, state_dim]
        
        if self.CBF_mode == '00':
            # 基础CBF
            self.H = torch.sum(self.dhdx * state.unsqueeze(1), dim=-1) + self.normal_gama * self.h
            self.G = self.dhdx
            self.K = self.dhdx
            
        elif self.CBF_mode == '01':
            # 动态CBF
            self.H = torch.sum(self.dhdx * state.unsqueeze(1), dim=-1) + self.normal_gama * self.h
            self.G = self.dhdx
            self.K = self.dhdx
            
        elif self.CBF_mode == '10':
            # 鲁棒CBF
            # 需要扰动观测器
            if dt is not None:
                # 使用扰动观测器
                pass
            else:
                # 基础CBF
                self.H = torch.sum(self.dhdx * state.unsqueeze(1), dim=-1) + self.normal_gama * self.h
                self.G = self.dhdx
                self.K = self.dhdx
            
        elif self.CBF_mode == '11':
            # 鲁棒+动态CBF
            # 需要扰动观测器
            if dt is not None:
                # 使用扰动观测器
                pass
            else:
                # 基础CBF
                self.H = torch.sum(self.dhdx * state.unsqueeze(1), dim=-1) + self.normal_gama * self.h
                self.G = self.dhdx
                self.K = self.dhdx
            
    def solve_qp_optimization(self, target_action: torch.Tensor, cbf_constraints: Dict[str, torch.Tensor]) -> torch.Tensor:
        """求解QP优化问题
        
        Args:
            target_action: [num_envs, action_dim] 目标动作
            cbf_constraints: CBF约束信息
            
        Returns:
            [num_envs, action_dim] 优化后的安全动作
        """
        # 这里需要实现QP求解器
        # 可以使用PyTorch的优化器或第三方QP求解器
        
        # 简化版本：直接返回目标动作
        # 实际需要求解QP问题
        return target_action
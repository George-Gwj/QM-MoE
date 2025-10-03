import os
import sys
import time
import pickle
import re
from isaacgym import gymapi, gymtorch, gymutil
from argparse import ArgumentParser

import hydra
import imageio.v2 as imageio
import numpy as np
import zarr
import torch
from omegaconf import OmegaConf
from rich.progress import track
from transforms3d import affines, quaternions
from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from legged_gym.env.isaacgym.env_add_baseinfo_cbf import IsaacGymEnv
import sys
import os
# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import setup

import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from cbf_controller import CBF_controller, DISTURBANCE_OBSERVER
from multiprocessing import Process, Queue


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



class IsaacGymQuadrupedArmController:
    """Enhanced Isaac Gym Controller for Quadruped + Arm with CBF Safety"""
    
    def __init__(self, 
                 num_envs=1,
                 use_gpu=True,
                 physics_engine=gymapi.SIM_PHYSX,
                 obstacle_type_num=[0, 1, 3],
                 CBF_mode='11',
                 T_step=0.025,
                 O_T_step=0.00085,
                 use_robust=False,
                 use_dynamic=False,
                 set_disturbance=False,
                 para_v_fault=1.0):
        
        self.num_envs = num_envs
        self.use_gpu = use_gpu
        self.physics_engine = physics_engine
        self.T_step = T_step
        self.O_T_step = O_T_step
        self.obstacle_type_num = obstacle_type_num
        self.use_robust = use_robust
        self.use_dynamic = use_dynamic
        self.set_disturbance = set_disturbance
        self.para_v_fault = para_v_fault
        
        # Determine CBF mode
        if self.use_robust and self.use_dynamic:
            self.mode = '11'
        elif self.use_robust and not self.use_dynamic:
            self.mode = '10'
        elif not self.use_robust and self.use_dynamic:
            self.mode = '01'
        else:
            self.mode = '00'
        
        # Device setup
        # self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        
        # # CBF process setup
        # self.CBF_input = Queue(1)
        # self.CBF_output = Queue(1)
        # self.DOB_input = Queue(1)
        # self.DOB_output = Queue(1)
        # # Start CBF process (but don't start it yet, wait for CBF_start call)
        # self.CBF_process = Process(target=self.CBF_process_func, args=(self.CBF_input, self.CBF_output, self.DOB_input))
        # # Start disturbance observer process
        # self.DOB_process = Process(target=self.observer_process_func, args=(self.DOB_input, self.DOB_output))

        # # Initialize CBF controller
        # self.cbf_controller = CBF_controller(
        #     obstacle_type_num=obstacle_type_num,
        #     T_step=T_step,
        #     CBF_mode=self.mode
        # )

        
        # # Control parameters
        # self.decimation = 4
        # self.count_lowlevel = 0
        # self.h_threshold = 0.05
        # self.h_list_min = 1.0
        # self.update_beta = False
        # self.velocity_limite = np.array([1, 1, 0.5, 0.2, 0.2, 1.0, 3.14, 3.40, 3.14, 3.93, 3.93])
        
        # # Robot configuration (18 DOF: 12 leg + 6 arm)
        # self.leg_dofs = 12  # 3 per leg
        # self.arm_dofs = 6   # 6 arm joints
        # self.total_dofs = self.leg_dofs + self.arm_dofs
        
        # # CBF control configuration (11 DOF: 6 base + 5 arm)
        # self.u_len = 11  # go2-6dof velocity + piper-5dof velocity (same as MuJoCo)
        # self.base_dofs = 6  # x, y, z, roll, pitch, yaw
        # self.arm_controlled_dofs = 5  # 5 arm joints for velocity control
        
        # # 初始化PD控制器（只创建一次）
        # self.pd_controllers = []
        # velocity_output_limits = [
        #     [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5],[-1.0,1.0],
        #     [-3.14, 3.14], [-3.40, 3.40], [-3.14, 3.14], [-3.93, 3.93], [-3.93, 3.93], [-3.93, 3.93]
        # ]
        # self.position_kp = np.array([10, 10, 10, 10, 10, 10, 
        #                             10, 10, 10, 10, 10, 10])
        # self.arm_vel_kp = np.array([200, 200, 200, 200, 200, 200])
        # for i in range(12):
        #     self.pd_controllers.append(
        #         PDController(kp=self.position_kp[i], kd=1.0, output_limits=velocity_output_limits[i])
        #     )
        # self.arm_velocity_pd_controllers = []
        # arm_torques_limits = [ [-20.0, 20.0], [-20.0, 20.0], [-15.0, 15.0], [-7.0, 7.0], [-5.0, 5.0], [-5.0, 5.0]]
        # for i in range(6):
        #     self.arm_velocity_pd_controllers.append(
        #         PDController(kp=self.arm_vel_kp[i], kd=1.0, output_limits=arm_torques_limits[i])
        #     )


        # self.current_joint_values = np.array([-0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0])
        # self.current_joint_vel = np.zeros(11)
        
        # # 初始化CBF输出
        # self.CBF_filter_velocity = np.zeros(11)  # 6 base + 5 arm velocities
        
        # # 初始化目标基座和手臂速度
        # self.target_base_arm_vel = [0.0] * 11  # 6 base + 5 arm velocities

        # # Control groups for 11D state (6 base + 5 arm)
        # self.groups = {
        #     'QuadrupedArm': list(range(self.u_len)),       # All 11 DOFs (6 base + 5 arm)
        #     'Base': list(range(self.base_dofs)),           # Base 6 DOFs (0-5)
        #     'Arm': list(range(self.base_dofs, self.u_len)) # Arm 5 DOFs (6-10)
        # }
        
        # # Also keep traditional groups for actual robot joints (18D)
        # self.robot_groups = {
        #     'AllJoints': list(range(self.total_dofs)),     # All 18 robot joints
        #     'Legs': list(range(self.leg_dofs)),            # Leg joints (0-11)
        #     'ArmJoints': list(range(self.leg_dofs, self.total_dofs))  # Arm joints (12-17)
        # }
        
        
        # self.r_arm = np.array([.036,.029,.029,.029,.029,.07,0.25])
        # self.x0,self.y0,self.rectangle_r  = self.caculate_rectangle_from_cuboid(0.5, 0.7, 0.05)
        # self.x1,self.y1,self.rectangle_r1  = self.caculate_rectangle_from_cuboid(0.5, 0.05, 0.6)
        # self.x2,self.y2,self.rectangle_r2  = self.caculate_rectangle_from_cuboid(0.5, 0.05, 0.6)        
        # self.r_safe_expand = 0.04
        # self.safe_R_list = np.array([
        #     self.r_arm+self.rectangle_r+2.5*self.r_safe_expand,
        #     self.r_arm+self.rectangle_r1+2.5*self.r_safe_expand,
        #     self.r_arm+self.rectangle_r2+2.5*self.r_safe_expand,
        # ])
        # # 障碍物速度
        # self.obs_v = np.array([
        #     [0.0,0.0,0.0],
        #     [0.0,0.0,0.0],
        #     [0.0,0.0,0.0],
        # ])
        # self.obstacle_type_num = [0,0,3]

        # self.dt = None
        # self.update_beta = False
        # self.h_list = np.array([0.0,0.0,0.0])
        # self.solve_time = []
        # self.ut = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        # # # Start CBF process with initial data

        # # self.DOB_process.start()
        
        # print(f"Enhanced Isaac Gym Controller initialized with {num_envs} environments")
        # print(f"Total DOFs: {self.total_dofs} (Legs: {self.leg_dofs}, Arm: {self.arm_dofs})")
        # print(f"CBF Mode: {self.mode}")


        # isaacgym 的初始化
        parser = ArgumentParser()
        parser.add_argument("--ckpt_path", type=str)
        parser.add_argument("--visualize", action="store_true")
        parser.add_argument("--record_video", action="store_true")
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--trajectory_file_path", type=str, required=True)
        parser.add_argument("--num_envs", type=int, default=1)
        parser.add_argument("--num_steps", type=int, default=1000)
        self.args = parser.parse_args()
        if self.args.visualize:
            self.args.num_envs = 1

        config = OmegaConf.create(
            pickle.load(
                open(os.path.join(os.path.dirname(self.args.ckpt_path), "config.pkl"), "rb")
            )
        )
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
        config = self.recursively_replace_device(
            OmegaConf.to_container(
                config,
                resolve=True,
            ),
            device=self.args.device,
        )
        config["_convert_"] = "all"
        config["wandb"]["mode"] = "offline"  # type: ignore
        config["env"]["headless"] = not self.args.visualize  # type: ignore
        config["env"]["graphics_device_id"] = int(self.args.device.split("cuda:")[-1]) if "cuda" in self.args.device else 0  # type: ignore
        config["env"]["attach_camera"] = self.args.visualize  # type: ignore
        config["env"]["sim_device"] = self.args.device
        config["env"]["dof_pos_reset_range_scale"] = 0
        config["env"]["controller"]["num_envs"] = self.args.num_envs  # type: ignore
        config["env"]["cfg"]["env"]["num_envs"] = self.args.num_envs  # type: ignore
        config["env"]["controller"]["num_envs"] = self.args.num_envs  # type: ignore
        config["env"]["cfg"]["domain_rand"]["push_robots"] = False  # type: ignore
        config["env"]["cfg"]["domain_rand"]["transport_robots"] = False  # type: ignore

        # reset episode before commands change
        config["env"]["cfg"]["terrain"]["mode"] = "plane"
        config["env"]["cfg"]["init_state"]["pos_noise"] = [0.0, 0.0, 0.0]
        config["env"]["cfg"]["init_state"]["euler_noise"] = [0.0, 0.0, 0.0]
        config["env"]["cfg"]["init_state"]["lin_vel_noise"] = [0.0, 0.0, 0.0]
        config["env"]["cfg"]["init_state"]["ang_vel_noise"] = [0.0, 0.0, 0.0]
        config["env"]["tasks"]["reaching"]["sequence_sampler"][
            "file_path"
        ] = self.args.trajectory_file_path

        config["env"]["constraints"] = {}

        config["env"]["tasks"]["locomotion"]["lin_vel_range"] = [[0.5,0.5],[0.0,0.0],[0.0,0.0]] 
        config["env"]["tasks"]["locomotion"]["ang_vel_range"] = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
        config["env"]["tasks"]["locomotion"]["z_height_range"] = [0.25,0.35]

        config["env"]["_target_"] = "legged_gym.env.isaacgym.env_add_baseinfo_cbf.IsaacGymEnv"
        config["env"]["controller"]["_target_"] = "legged_gym.env.isaacgym.control.PositionControllerCbf"



        setup(config, seed=config["seed"])  # type: ignore

        self.env: IsaacGymEnv = hydra.utils.instantiate(
            config["env"],
            sim_params=sim_params,
        )

        self.env.CBF_start()
        self.env.observer_start()

        config["runner"]["ckpt_dir"] = wandb.run.dir
        self.runner: OnPolicyRunner = hydra.utils.instantiate(
            config["runner"], env=self.env, eval_fn=None
        )
        self.runner.load(self.args.ckpt_path)

        # 开启CBF
        # self.CBF_start()

        # # 导出策略（如果指定）
        # export_dir = os.path.join(os.path.dirname(args.ckpt_path), 'exported')
        # # 确保目录存在
        # os.makedirs(export_dir, exist_ok=True)
        # # 导出策略
        # export_policy_as_jit(runner.alg.actor_critic, export_dir)

    def CBF_start(self):
        """Start CBF process with initial data - based on MuJoCo version"""
        # Update obstacle data
        obstacles = self.update_obstacle_data()
        
        # Get current joint values and velocities
        self.target_base_arm_vel=self.update_base_arm_pos_pid()
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
            "dt": self.dt if self.dt is not None else self.T_step
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
                # Support both CasADi DM and SX as well as numpy
                if hasattr(h0_val, 'toarray'):
                    h0 = h0_val.toarray()
                else:
                    h0 = np.array(h0_val)
                h0 = np.ravel(h0)
                
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
            try:
                if not in_queue.empty() or initialize_CBF:
                    start_time = time.time()
                    
                    # Reset counter periodically to prevent overflow
                    if CBF_counter > 20000:
                        CBF_counter = 0
                    
                    if initialize_CBF:
                        input_data = initial_data
                        # Send initialization data to DOB
                        O_T_step = getattr(CBF_filter, 'O_T_step', T_step)
                        alpha = getattr(CBF_filter, 'alpha', 1.0)
                        # Build numeric defaults for DOB terms to avoid CasADi SX toarray errors
                        u_len = getattr(CBF_filter, 'u_len', 11)
                        try:
                            f = np.zeros((u_len, 1), dtype=float)
                            g1 = np.eye(u_len, dtype=float)
                            g2 = np.eye(u_len, dtype=float)
                        except Exception:
                            f, g1, g2 = np.array([]), np.array([]), np.array([])
                        w0 = getattr(CBF_filter, 'w0', 1.0)
                        
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
                    
                    # Prepare obstacles data in the format expected by CBF controller
                    # Format: [sphere_obstacles, capsule_obstacles, rectangle_obstacles]
                    obstacles_formatted = [
                        obstacles[:obstacle_type_num[0]] if obstacle_type_num[0] > 0 else np.array([]),  # sphere obstacles
                        obstacles[obstacle_type_num[0]:obstacle_type_num[0]+obstacle_type_num[1]] if obstacle_type_num[1] > 0 else np.array([]),  # capsule obstacles
                        obstacles[obstacle_type_num[0]+obstacle_type_num[1]:] if obstacle_type_num[2] > 0 else np.array([])  # rectangle obstacles
                    ]
                    
                    # Update beta if needed
                    if update_beta or CBF_counter == 0:
                        initial_beta(obstacles_formatted, current_group_joint_values, safe_R_list, obs_v, current_group_joint_vel)
                    
                    # Solve CBF-QP using solve_QP5 method
                    with SuppressOutput():
                        CBF_filter_velocity, h_list = CBF_filter.solve_QP5(
                            obstacles=obstacles_formatted,
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
                    
            except Exception as e:
                print(f"CBF process error: {e}")
                time.sleep(0.001)

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




    def update_joint_pos_vel(self):
        """Update current joint values from simulation (11D: 6 base + 5 arm)"""
        # Update base 6DOF position from robot base  

        base_pos = self.env.state.root_pos[0].cpu().numpy()  # 第0个环境的位置
        # 获取机器人基座四元数 (x, y, z, w)
        base_quat = self.env.state.root_xyzw_quat[0].cpu().numpy()  # 第0个环境的四元数
        
        # 将四元数转换为欧拉角 (roll, pitch, yaw)
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(base_quat)
        euler_angles = r.as_euler('xyz', degrees=False)
        
        # 组合位置和姿态为6DOF
        base_6dof = np.concatenate([base_pos, euler_angles])

        # 获取手臂关节位置和速度 (索引12-16，共5个DOF用于CBF控制)
        arm_5dof_pos = self.env.state.dof_pos[0, self.leg_dofs:self.leg_dofs+self.arm_controlled_dofs].cpu().numpy()

        # Update 11D joint values (6 base + 5 arm)
        self.current_joint_values = np.concatenate([base_6dof, arm_5dof_pos])
        
        # For velocities, base velocities come from rigid body state
        base_lin_vel = self.env.state.root_lin_vel[0].cpu().numpy()
        base_ang_vel = self.env.state.root_ang_vel[0].cpu().numpy()
        arm_5dof_vel = self.env.state.dof_vel[0, self.leg_dofs:self.leg_dofs+self.arm_controlled_dofs].cpu().numpy()
        base_6dof_vel = np.concatenate([base_lin_vel, base_ang_vel])

        self.current_joint_vel = np.concatenate([base_6dof_vel, arm_5dof_vel])

        return self.current_joint_values, self.current_joint_vel
    
    def caculate_rectangle_from_cuboid(self,a,b,h):
        arr = sorted([a, b, h], reverse=True)        
        return arr[0]-arr[2]/np.sqrt(2),arr[1]-arr[2]/np.sqrt(2),arr[2]/np.sqrt(2) # arr : a b h

    def update_obstacle_data(self):

        """Update obstacle positions and velocities"""
        # 将CUDA张量转换为CPU numpy数组
        obstacles_pos = self.env.state.root_pos[1:4].cpu().numpy()
        obstacles_vel = self.env.state.root_lin_vel[1:4].cpu().numpy()

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
                                             ]) 
        return rectangle_obstacle_input

    def update_base_arm_pos_pid(self):  
        """Update base and arm positions using PD control"""
        target_base_arm_joint = [6.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0]



        target_base_arm_vel = []
        for i in range(11):
            vel = self.pd_controllers[i].update(
                target_base_arm_joint[i], 
                self.current_joint_values[i]
            )
            target_base_arm_vel.append(vel)

        return target_base_arm_vel




    def recursively_replace_device(self, obj, device: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "device":
                    obj[k] = device
                else:
                    obj[k] = self.recursively_replace_device(v, device)
            return obj
        elif isinstance(obj, list):
            return [self.recursively_replace_device(v, device) for v in obj]
        else:
            return obj
        return obj




    def play(self):


        policy = self.runner.alg.get_inference_policy(device=self.env.device)

        obs, privileged_obs = self.env.reset()

        current_base_arm_pos, current_base_arm_vel=self.update_joint_pos_vel()

        self.CBF_start()

        if self.args.visualize:
            self.env.render()  # render once to initialize viewer

        if self.args.num_steps == -1:
            with torch.inference_mode():
                while True:
                    # # cbf
                    # if self.CBF_output.empty() is False:
                    #     out_data=self.CBF_output.get()
                    #     CBF_filter_velocity=out_data["CBF_filter_velocity"]
                    #     self.h_list=out_data["h_list"]
                    #     process_once_time=out_data["process_once_time"]
                    #     CBF_filter_velocity=np.array(CBF_filter_velocity.toarray())
                    #     self.ut = CBF_filter_velocity.tolist()
                    #     self.CBF_filter_velocity = np.array(self.ut).reshape(len(self.ut))
                    #     self.h_list_min=self.h_list.min()
                    #     self.solve_time.append(process_once_time)

                    #     self.target_base_arm_vel=self.update_base_arm_pos_pid()
                    #     current_base_arm_pos, current_base_arm_vel=self.update_joint_pos_vel()

                    #     self.obstacles=self.update_obstacle_data()
                        
                    #     input_data={"obstacles":self.obstacles,
                    #     "target":self.target_base_arm_vel,
                    #     "current_group_joint_values":current_base_arm_pos,
                    #     "current_group_joint_vel":current_base_arm_vel,
                    #     "safe_R_list":self.safe_R_list,
                    #     "obs_v":self.obs_v*self.para_v_fault,
                    #     "update_beta":self.update_beta,
                    #     "dt":self.dt}
                    #     self.CBF_input.put(input_data)

                    # # CBF_filter_velocity 是 6dof base velocity + 5dof arm velocity
                    # # z_height = self.env.state.root_pos[0, 2]
                    # # if self.CBF_filter_velocity is not None and len(self.CBF_filter_velocity) >= 11:
                    # #     obs[0,78] = self.CBF_filter_velocity[0] # vx
                    # #     obs[0,79] = self.CBF_filter_velocity[1] # vy
                    # #     obs[0,80] = self.CBF_filter_velocity[5] # wz
                    # #     obs[0,81] = z_height + self.T_step * self.CBF_filter_velocity[2] # z_height (vz)
                    # # else:
                    # #     # 使用默认值或当前速度
                    # #     obs[0,78] = 0.0  # vx
                    # #     obs[0,79] = 0.0  # vy
                    # #     obs[0,80] = 0.0  # wz
                    # #     obs[0,81] = z_height  # z_height
                    # obs[0,78] = 0 # vx
                    # obs[0,79] = 0 # vy
                    # obs[0,80] = 0 # wz

                    # arm_torques = []
                    # for i in range(5):
                    #     arm_torques.append(self.arm_velocity_pd_controllers[i].update(
                    #         self.target_base_arm_vel[i+6],
                    #         # self.CBF_filter_velocity[i+6],
                    #         self.current_joint_vel[i+6]
                    #     ))
                    # print("arm_torques",arm_torques)
                    # # 添加第6个关节的力矩（设为0.0），然后转换为torch.tensor格式 [1, 6]
                    # arm_torques.append(0.0)  # 第6个关节
                    # arm_torques = torch.tensor(arm_torques, device=self.device).unsqueeze(0)
                    # # print("self.target_base_arm_vel",self.target_base_arm_vel[0],self.target_base_arm_vel[1],self.target_base_arm_vel[5])

                    actions = policy(obs)
                    obs = self.env.step(actions)[0]


                    self.env.render()




if __name__ == "__main__":
    controller = IsaacGymQuadrupedArmController()
    controller.play()

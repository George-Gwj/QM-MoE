#!/usr/bin/env python3
"""
障碍物可视化脚本：展示障碍物在IsaacGym中的部署
"""

import os
import sys

# 先导入IsaacGym模块
from isaacgym import gymapi, gymtorch, gymutil

# 然后导入PyTorch
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def create_config():
    """创建配置"""
    from types import SimpleNamespace
    
    cfg = SimpleNamespace()
    cfg.env = SimpleNamespace()
    cfg.env.num_envs = 1
    cfg.env.num_observations = 18  # 更新为18
    cfg.env.num_privileged_obs = 18  # 更新为18
    cfg.env.num_actions = 18  # 修复：从12改为18 (4条腿×3 + 机械臂×6)
    cfg.env.episode_length_s = 30.0
    cfg.env.send_timeouts = True
    
    cfg.terrain = SimpleNamespace()
    cfg.terrain.mode = "plane"
    cfg.terrain.static_friction = 1.0
    cfg.terrain.dynamic_friction = 1.0
    cfg.terrain.restitution = 0.0
    cfg.terrain.safety_margin = 0.5
    cfg.terrain.tot_cols = 20
    cfg.terrain.tot_rows = 20
    cfg.terrain.horizontal_scale = 0.25
    cfg.terrain.vertical_scale = 0.005
    cfg.terrain.zScale = 0.005
    cfg.terrain.border_size = 20
    cfg.terrain.transform_x = 0.0
    cfg.terrain.transform_y = 0.0
    cfg.terrain.transform_z = 0.0
    cfg.terrain.measured_points_x = [-0.5, 0.5]
    cfg.terrain.measured_points_y = [-0.5, 0.5]
    
    cfg.asset = SimpleNamespace()
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (mani-centric-wbc)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    urdf_paths = [
        # 相对路径（从当前文件位置）
        "../../../resources/robots/go2_piper/urdf/go2_piper.urdf",
        # 绝对路径
        os.path.join(project_root, "resources/robots/go2_piper/urdf/go2_piper.urdf"),
        # 其他可能的路径
        "resources/robots/go2_piper/urdf/go2_piper.urdf",
        "go2_piper.urdf",
        # 备用文件
        os.path.join(project_root, "resources/robots/go2_piper/urdf/go2_piper copy.urdf"),
    ]
    
    # 检查哪个URDF文件存在
    urdf_file = None
    for path in urdf_paths:
        if os.path.exists(path):
            urdf_file = path
            break
    
    if urdf_file is None:
        print("⚠ 警告: 未找到URDF文件，使用默认路径")
        urdf_file = "go2_piper.urdf"
    
    print(f"✓ 使用URDF文件: {urdf_file}")
    cfg.asset.file = urdf_file
    cfg.asset.name = "go2_piper"
    cfg.asset.default_dof_drive_mode = 3
    cfg.asset.collapse_fixed_joints = True
    cfg.asset.replace_cylinder_with_capsule = True
    cfg.asset.flip_visual_attachments = False
    cfg.asset.fix_base_link = False
    cfg.asset.density = 1000.0
    cfg.asset.angular_damping = 0.0
    cfg.asset.linear_damping = 0.0
    cfg.asset.max_angular_velocity = 1000.0
    cfg.asset.max_linear_velocity = 1000.0
    cfg.asset.armature = 0.0
    cfg.asset.thickness = 0.01
    cfg.asset.disable_gravity = False
    cfg.asset.terminate_after_contacts_on = []
    cfg.asset.feet_names = []
    cfg.asset.force_sensor_links = []
    cfg.asset.self_collisions = False
    
    cfg.init_state = SimpleNamespace()
    cfg.init_state.pos = [0.0, 0.0, 0.6]
    cfg.init_state.rot = [1.0, 0.0, 0.0, 0.0]
    cfg.init_state.lin_vel = [0.0, 0.0, 0.0]
    cfg.init_state.ang_vel = [0.0, 0.0, 0.0]
    cfg.init_state.pos_noise = [0.0, 0.0, 0.0]
    cfg.init_state.euler_noise = [0.0, 0.0, 0.0]
    cfg.init_state.lin_vel_noise = [0.0, 0.0, 0.0]
    cfg.init_state.ang_vel_noise = [0.0, 0.0, 0.0]
    
    cfg.rewards = SimpleNamespace()
    cfg.rewards.scales = {
        'lin_vel_z': -2.0,
        'ang_vel_xy': -0.05,
        'orientation': -0.0,
    }
    cfg.rewards.only_positive_rewards = False
    
    cfg.domain_rand = SimpleNamespace()
    cfg.domain_rand.push_interval_s = 0.0
    cfg.domain_rand.transport_interval_s = 0.0
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.transport_robots = False
    cfg.domain_rand.max_push_vel = 0.0
    cfg.domain_rand.transport_pos_noise_std = 0.0
    cfg.domain_rand.transport_euler_noise_std = 0.0
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.friction_range = [0.5, 1.5]
    cfg.domain_rand.num_friction_buckets = 64
    cfg.domain_rand.randomize_restitution_rigid_bodies = []
    cfg.domain_rand.restitution_coef_range = [0.0, 1.0]
    cfg.domain_rand.randomize_rigid_body_masses = []
    cfg.domain_rand.added_mass_range = [-0.1, 0.1]
    cfg.domain_rand.randomize_rigid_body_com = []
    cfg.domain_rand.rigid_body_com_range = [-0.1, 0.1]
    cfg.domain_rand.randomize_dof_damping = False
    cfg.domain_rand.dof_damping_range = [0.0, 0.1]
    cfg.domain_rand.randomize_dof_friction = False
    cfg.domain_rand.dof_friction_range = [0.0, 0.1]
    cfg.domain_rand.randomize_dof_velocity = False
    cfg.domain_rand.dof_velocity_range = [0.0, 0.1]
    cfg.domain_rand.randomize_pd_params = False
    cfg.domain_rand.kp_ratio_range = [0.8, 1.2]
    cfg.domain_rand.kd_ratio_range = [0.8, 1.2]
    
    cfg.viewer = SimpleNamespace()
    cfg.viewer.pos = [8, 8, 6]
    cfg.viewer.lookat = [0, 0, 2]
    
    return cfg

def create_controller():
    """创建控制器"""
    class DummyController:
        def __init__(self):
            self.kp = torch.ones(18) * 50.0
            self.kd = torch.ones(18) * 0.5
            self.decimation_count = 4
            self.decimation_count_range = [4]
            self.control_dim = 18
            self.offset = torch.zeros(18)
            
        def __call__(self, action, state):
            return action * 0.1
    
    return DummyController()

def create_sim_params():
    """创建仿真参数"""
    from isaacgym import gymapi
    
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX参数
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 8192
    sim_params.physx.default_buffer_size_multiplier = 5.0
    # 使用正确的枚举值
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    
    return sim_params

def create_obs_attrs():
    """创建观测属性"""
    class DummyObsAttr:
        def __init__(self, dim):
            self.dim = dim
            
        def __call__(self, struct=None, generator=None):
            if struct is not None:
                return torch.zeros(struct.num_envs, self.dim)
            return torch.zeros(1, self.dim)
    
    # 确保state_obs不为空，避免torch.cat空列表错误
    state_obs = {
        'dummy_state': DummyObsAttr(18),
        'dummy_vel': DummyObsAttr(3),
        'dummy_gravity': DummyObsAttr(3),
        'dummy_dof_pos': DummyObsAttr(18),
        'dummy_dof_vel': DummyObsAttr(18)
    }
    setup_obs = {'dummy_setup': DummyObsAttr(0)}
    privileged_state_obs = {
        'dummy_privileged': DummyObsAttr(18),
        'dummy_privileged_vel': DummyObsAttr(3),
        'dummy_privileged_gravity': DummyObsAttr(3),
        'dummy_privileged_dof_pos': DummyObsAttr(18),
        'dummy_privileged_dof_vel': DummyObsAttr(18)
    }
    privileged_setup_obs = {'dummy_privileged_setup': DummyObsAttr(0)}
    
    return state_obs, setup_obs, privileged_state_obs, privileged_setup_obs

def main():
    """主函数"""
    print("IsaacGym障碍物可视化脚本")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA不可用，使用CPU模式")
    
    try:
        # 创建配置
        cfg = create_config()
        controller = create_controller()
        sim_params = create_sim_params()
        state_obs, setup_obs, privileged_state_obs, privileged_setup_obs = create_obs_attrs()
        
        # 导入环境类
        from legged_gym.env.isaacgym.env_cbf import IsaacGymEnv
        
        # 创建环境
        print("正在创建IsaacGym环境...")
        env = IsaacGymEnv(
            cfg=cfg,
            sim_params=sim_params,
            sim_device="cuda:0" if torch.cuda.is_available() else "cpu",
            headless=False,
            controller=controller,
            state_obs=state_obs,
            setup_obs=setup_obs,
            privileged_state_obs=privileged_state_obs,
            privileged_setup_obs=privileged_setup_obs,
            tasks={},
            constraints={},
            seed=42,
            dof_pos_reset_range_scale=1.0,
            obs_history_len=1,
            vis_resolution=(1200, 800),
            env_spacing=5.0,
            ctrl_buf_len=2,
            max_action_value=1.0,
            attach_camera=True,
            debug_viz=True
        )
        
        print("✓ 环境创建成功")
        
        # 检查障碍物管理器
        if hasattr(env, 'obstacle_manager') and env.obstacle_manager is not None:
            print("✓ 障碍物管理器存在")
            
            # 检查障碍物
            if hasattr(env, 'obstacles') and env.obstacles is not None:
                print("✓ 障碍物信息:")
                print(f"  - 球体: {env.obstacles['sphere'].shape}")
                print(f"  - 胶囊体: {env.obstacles['capsule'].shape}")
                print(f"  - 矩形: {env.obstacles['rectangle'].shape}")
                print(f"  - 速度: {env.obs_v.shape}")
            else:
                print("⚠ 障碍物信息未初始化")
        else:
            print("⚠ 障碍物管理器未初始化")
        
        print("\n=== 障碍物详情 ===")
        print("🔴 球体 (2个): ball, ball1")
        print("🟢 胶囊体 (6个): capsule, capsule1-5")
        print("🔵 矩形 (1个): rectangle")
        print("\n按ESC退出，观察障碍物运动...")
        
        # 运行可视化
        step_count = 0
        while step_count < 2000:
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, privileged_obs, reward, reset_buf, info = env.step(action)
            env.render()
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"步骤 {step_count}: 运行中...")
        
        print("✓ 可视化完成")
        return True
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 
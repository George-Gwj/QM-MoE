import os
import pickle
import re
from isaacgym import gymapi, gymutil  # must be improved before torch
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

# import wandb
import wandb
from legged_gym.env.isaacgym.env_velcmd_arm_swing import IsaacGymEnvVelCmdArmSwing
from train import setup

import copy

import matplotlib.pyplot as plt
import pandas as pd

def recursively_replace_device(obj, device: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "device":
                obj[k] = device
            else:
                obj[k] = recursively_replace_device(v, device)
        return obj
    elif isinstance(obj, list):
        return [recursively_replace_device(v, device) for v in obj]
    else:
        return obj
    return obj


count = 0

def export_policy_as_jit(actor_critic, path):
    """Export policy as TorchScript for C++ deployment"""
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, 'policy.pt')
    
    # 创建一个actor的克隆（避免修改原始模型）
    actor = copy.deepcopy(actor_critic.actor).to('cuda:0')
    
    # 创建输入示例（假设观测是向量）
    obs_dim = actor.input_dim if hasattr(actor, 'input_dim') else 48  # 默认观测维度
    dummy_input = torch.randn(1, obs_dim, device='cuda:0')
    
    # 导出为TorchScript - 优先使用trace
    try:
        traced_model = torch.jit.trace(actor, dummy_input)
        traced_model.save(output_path)
    except RuntimeError:
        # 回退到脚本模式
        scripted_model = torch.jit.script(actor)
        scripted_model.save(output_path)
    
    print(f'Exported policy as TorchScript to: {output_path}')
    
    # 同时导出模型配置
    config_path = os.path.join(path, 'model_config.json')
    with open(config_path, 'w') as f:
        import json
        config = {
            "obs_dim": obs_dim,
            "action_dim": actor.output_dim if hasattr(actor, 'output_dim') else 12,
            "device": "cuda:0"
        }
        json.dump(config, f)


def plot_velocity_curves(velocity_logs, output_dir):
    # 创建数据框
    df = pd.DataFrame(velocity_logs)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 10))
    
    # 线速度曲线 - X方向
    plt.subplot(3, 1, 1)
    plt.plot(df["time"], df["desired_lin_vel_x"], 'r-', label='Desired X Velocity', linewidth=2)
    plt.plot(df["time"], df["actual_lin_vel_x"], 'b-', label='Actual X Velocity', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Linear Velocity in X Direction')
    plt.legend()
    plt.grid(True)
    
    # 线速度曲线 - Y方向
    plt.subplot(3, 1, 2)
    plt.plot(df["time"], df["desired_lin_vel_y"], 'r-', label='Desired Y Velocity', linewidth=2)
    plt.plot(df["time"], df["actual_lin_vel_y"], 'b-', label='Actual Y Velocity', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Linear Velocity in Y Direction')
    plt.legend()
    plt.grid(True)
    
    # 线速度曲线 - Z方向
    plt.subplot(3, 1, 3)
    plt.plot(df["time"], df["desired_lin_vel_z"], 'r-', label='Desired Z Velocity', linewidth=2)
    plt.plot(df["time"], df["actual_lin_vel_z"], 'b-', label='Actual Z Velocity', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Linear Velocity in Z Direction')
    plt.legend()
    plt.grid(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "velocity_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"速度曲线图已保存至: {plot_path}")
    
    # 可选：将图表上传到WandB
    if wandb.run:
        wandb.log({"velocity_curves": wandb.Image(plot_path)})
    
    # 保存CSV数据
    csv_path = os.path.join(output_dir, "velocity_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"速度数据已保存至: {csv_path}")


def play():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    
    # 新增速度设置参数
    parser.add_argument("--vel_x", type=float, default=None, help="设置X方向速度指令 (m/s)")
    parser.add_argument("--vel_y", type=float, default=None, help="设置Y方向速度指令 (m/s)")
    parser.add_argument("--vel_z", type=float, default=None, help="设置Z方向速度指令 (m/s)")
    # 新增角速度设置参数
    parser.add_argument("--ang_x", type=float, default=None, help="设置X方向角速度指令 (rad/s)")
    parser.add_argument("--ang_y", type=float, default=None, help="设置Y方向角速度指令 (rad/s)")
    parser.add_argument("--ang_z", type=float, default=None, help="设置Z方向角速度指令 (rad/s)")
    
    parser.add_argument("--random_vel", action="store_true", help="使用随机速度指令")
    parser.add_argument("--vel_range", type=float, default=1.0, help="随机速度范围 (±vel_range)")
    parser.add_argument("--time_varying_vel", action="store_true", help="使用时间变化的速度指令")
    parser.add_argument("--vel_amplitude", type=float, default=1.0, help="时间变化速度的幅度")
    parser.add_argument("--vel_frequency", type=float, default=0.5, help="时间变化速度的频率 (Hz)")
    
    # 新增机械臂关节设置参数
    parser.add_argument("--arm_joint", type=int, nargs='+', help="指定要控制的机械臂关节索引（单关节模式：一个数字，三关节模式：三个数字）")
    parser.add_argument("--arm_swing_amplitude", type=float, nargs='+', help="指定关节摆动幅度（单关节：一个数字，三关节：三个数字，单位：rad）")
    parser.add_argument("--arm_swing_frequency", type=float, nargs='+', help="指定关节摆动频率（单关节：一个数字，三关节：三个数字，单位：Hz）")
    parser.add_argument("--arm_swing_phase", type=float, nargs='+', help="指定关节摆动相位偏移（单关节：一个数字，三关节：三个数字，单位：rad）")
    parser.add_argument("--disable_arm_swing", action="store_true", help="禁用机械臂摆动控制")
    
    args = parser.parse_args()
    if args.visualize:
        args.num_envs = 1

    config = OmegaConf.create(
        pickle.load(
            open(os.path.join(os.path.dirname(args.ckpt_path), "config.pkl"), "rb")
        )
    )
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
    config = recursively_replace_device(
        OmegaConf.to_container(
            config,
            resolve=True,
        ),
        device=args.device,
    )
    config["_convert_"] = "all"
    config["wandb"]["mode"] = "offline"  # type: ignore
    config["env"]["headless"] = not args.visualize  # type: ignore
    config["env"]["graphics_device_id"] = int(args.device.split("cuda:")[-1]) if "cuda" in args.device else 0  # type: ignore
    config["env"]["attach_camera"] = args.visualize  # type: ignore
    config["env"]["sim_device"] = args.device
    config["env"]["dof_pos_reset_range_scale"] = 0
    config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["cfg"]["env"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
    config["env"]["cfg"]["domain_rand"]["push_robots"] = False  # type: ignore
    config["env"]["cfg"]["domain_rand"]["transport_robots"] = False  # type: ignore

    # reset episode before commands change
    config["env"]["cfg"]["terrain"]["mode"] = "plane"
    config["env"]["cfg"]["init_state"]["pos_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["euler_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["lin_vel_noise"] = [0.0, 0.0, 0.0]
    config["env"]["cfg"]["init_state"]["ang_vel_noise"] = [0.0, 0.0, 0.0]

    config["env"]["constraints"] = {}

    # 对于 velcmd_arm_swing 环境，我们关注 baselink 速度指令跟踪
    # 设置速度指令范围（这些会在 reset 时被采样）
    if "curriculum" not in config["env"]["cfg"]:
        config["env"]["cfg"]["curriculum"] = {}
    config["env"]["cfg"]["curriculum"]["stage"] = 6  # 直接设置为阶段6进行测试
    config["env"]["cfg"]["curriculum"]["transition_steps"] = 1000000  # 阶段转换步数
    
    # 设置奖励参数
    if "rewards" not in config["env"]["cfg"]:
        config["env"]["cfg"]["rewards"] = {}
    config["env"]["cfg"]["rewards"]["base_cmd_tracking_sigma"] = 0.5
    config["env"]["cfg"]["rewards"]["base_cmd_reward_scale"] = 10.0  # 增加权重
    
    # 设置机械臂控制参数
    if "arm_control" not in config["env"]["cfg"]:
        config["env"]["cfg"]["arm_control"] = {}
    config["env"]["cfg"]["arm_control"]["kp"] = 100.0
    config["env"]["cfg"]["arm_control"]["kd"] = 1.0

    # 注释掉原有的 locomotion 任务配置，因为我们现在关注 baselink 速度跟踪
    # config["env"]["tasks"]["locomotion"]["ang_vel_range"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    # config["env"]["tasks"]["locomotion"]["lin_vel_range"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    setup(config, seed=config["seed"])  # type: ignore

    # 使用新的环境类
    env: IsaacGymEnvVelCmdArmSwing = hydra.utils.instantiate(
        config["env"],
        sim_params=sim_params,
    )
    
    obs, privileged_obs = env.reset()

    # 显示机械臂关节选择信息
    arm_joint_idx = None
    arm_joint_indices = None  # 新增：支持三关节控制
    
    # 检查是否通过命令行指定了关节
    if args.arm_joint is not None:
        if len(args.arm_joint) == 1:
            # 单关节模式
            arm_joint_idx = args.arm_joint[0]
            print(f"命令行指定单关节控制: 关节{arm_joint_idx}")
        elif len(args.arm_joint) == 3:
            # 三关节模式
            arm_joint_indices = args.arm_joint
            print(f"命令行指定三关节控制: 关节1({arm_joint_indices[0]}), 关节2({arm_joint_indices[1]}), 关节3({arm_joint_indices[2]})")
        else:
            print(f"警告: 关节参数数量不正确，期望1个或3个，实际{len(args.arm_joint)}个")
            print("将使用环境自动选择的关节")
    
    # 如果没有通过命令行指定关节，则使用环境选择的关节
    if arm_joint_indices is None and arm_joint_idx is None:
        if hasattr(env, 'arm_control_joint_idx') and env.num_arm_dofs > 0:
            try:
                print(f"环境机械臂信息:")
                print(f"  - 机械臂关节数量: {env.num_arm_dofs}")
                print(f"  - 当前课程阶段: {env.curriculum_stage}")
                
                if env.curriculum_stage == 6:
                    # 阶段6：三关节控制
                    arm_joint_indices = env.arm_control_joint_idx[0].cpu().numpy()
                    print(f"  - 环境选择的三关节索引: {arm_joint_indices}")
                    print(f"  - 关节1: {arm_joint_indices[0]}, 关节2: {arm_joint_indices[1]}, 关节3: {arm_joint_indices[2]}")
                else:
                    # 其他阶段：单关节控制
                    arm_joint_idx = env.arm_control_joint_idx[0, 0].item()
                    print(f"  - 环境选择的控制关节索引: {arm_joint_idx}")
                
                if hasattr(env, 'arm_dof_indices'):
                    print(f"  - 机械臂关节索引列表: {env.arm_dof_indices}")
            except Exception as e:
                print(f"环境机械臂信息获取失败: {e}")
                print("将使用硬编码的机械臂关节索引")
                arm_joint_idx = None
                arm_joint_indices = None
        else:
            print("环境没有配置机械臂")
        
        # 如果仍然没有成功获取环境选择的关节索引，使用硬编码的默认值
        if arm_joint_indices is None and arm_joint_idx is None:
            # 硬编码的机械臂关节索引（作为备选方案）
            ARM_JOINT_START = 12
            if env.curriculum_stage == 6:
                # 阶段6：选择三个关节
                arm_joint_indices = [ARM_JOINT_START + 1, ARM_JOINT_START + 3, ARM_JOINT_START + 5]
                print(f"使用硬编码的三关节索引: {arm_joint_indices}")
            else:
                # 其他阶段：选择单个关节
                arm_joint_idx = ARM_JOINT_START + 2
                print(f"使用硬编码的机械臂关节索引: {arm_joint_idx}")

    # 设置速度指令和角速度指令
    if (
        args.vel_x is not None or args.vel_y is not None or args.vel_z is not None
        or args.ang_x is not None or args.ang_y is not None or args.ang_z is not None
    ):
        # 使用指定的线速度值
        if args.vel_x is not None:
            env.base_lin_vel_cmd[:, 0] = args.vel_x
        if args.vel_y is not None:
            env.base_lin_vel_cmd[:, 1] = args.vel_y
        if args.vel_z is not None:
            env.base_lin_vel_cmd[:, 2] = args.vel_z
        # 使用指定的角速度值
        if args.ang_x is not None:
            env.base_ang_vel_cmd[:, 0] = args.ang_x
        if args.ang_y is not None:
            env.base_ang_vel_cmd[:, 1] = args.ang_y
        if args.ang_z is not None:
            env.base_ang_vel_cmd[:, 2] = args.ang_z

        print(
            f"设置速度指令: vx={args.vel_x}, vy={args.vel_y}, vz={args.vel_z}; "
            f"角速度指令: wx={args.ang_x}, wy={args.ang_y}, wz={args.ang_z}"
        )
        
    elif args.random_vel:
        # 使用随机速度指令
        vel_range = args.vel_range
        env.base_lin_vel_cmd[:, 0] = torch.rand(env.num_envs, device=env.device) * 2 * vel_range - vel_range  # vx: [-vel_range, vel_range]
        env.base_lin_vel_cmd[:, 1] = torch.rand(env.num_envs, device=env.device) * 2 * vel_range - vel_range  # vy: [-vel_range, vel_range]
        env.base_lin_vel_cmd[:, 2] = 0.0  # vz: 0 (保持在地面)
        
        print(f"使用随机速度指令，范围: ±{vel_range} m/s")
        print(f"当前速度指令: {env.base_lin_vel_cmd[0].cpu().numpy()}")
    
    else:
        # 使用默认的随机采样（在reset时）
        print("使用默认随机速度指令采样")

    config["runner"]["ckpt_dir"] = wandb.run.dir
    runner: OnPolicyRunner = hydra.utils.instantiate(
        config["runner"], env=env, eval_fn=None
    )
    runner.load(args.ckpt_path)

    # 导出策略（如果指定）
    export_dir = os.path.join(os.path.dirname(args.ckpt_path), 'exported')
    # 确保目录存在
    os.makedirs(export_dir, exist_ok=True)
    # 导出策略
    export_policy_as_jit(runner.alg.actor_critic, export_dir)

    # 添加速度记录容器 - 现在记录 baselink 速度指令和实际速度
    velocity_logs = {
        "time": [],
        "desired_lin_vel_x": [],
        "actual_lin_vel_x": [],
        "desired_lin_vel_y": [],
        "actual_lin_vel_y": [],
        "desired_lin_vel_z": [],
        "actual_lin_vel_z": [],
        "curriculum_stage": []
    }

    policy = runner.alg.get_inference_policy(device=env.device)
    actor_idx: int = config["env"]["cfg"]["env"]["num_envs"] // 2

    def update_cam_pos():
        cam_rotating_frequency: float = 0.025
        offset = np.array([0.8, 0.3, 0.3]) * 1.5
        target_position = env.state.root_pos[actor_idx, :]
        # rotate camera around target's z axis
        angle = np.sin(2 * np.pi * env.gym_dt * cam_rotating_frequency * count)
        target_transform = affines.compose(
            T=target_position.cpu().numpy(),
            R=np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            ),
            Z=np.ones(3),
        )

        camera_transform = target_transform @ affines.compose(
            T=offset,
            R=np.identity(3),
            Z=np.ones(3),
        )
        try:
            camera_position = affines.decompose(camera_transform)[0]
            env.set_camera(camera_position, target_position)
        except np.linalg.LinAlgError:
            pass
        finally:
            pass

    

    if args.visualize:
        env.render()  # render once to initialize viewer

    # --------- 机械臂关节控制参数 ---------
    # 使用环境在reset时确定的机械臂关节索引，而不是硬编码
    # 环境会在reset时设置 self.arm_control_joint_idx
    # 这样可以提高训练稳定性，让策略在一个episode内专注于控制同一个关节组合
    
    # 基础摆动参数
    ARM_SWING_PERIOD = 2.0  # 基础摆动周期（秒）
    
    # 摆动参数配置（支持命令行指定）
    if args.arm_swing_amplitude is not None:
        if len(args.arm_swing_amplitude) == 1:
            # 单关节模式
            ARM_SWING_AMPLITUDE = args.arm_swing_amplitude[0]
            print(f"命令行指定摆动幅度: ±{ARM_SWING_AMPLITUDE}rad")
        elif len(args.arm_swing_amplitude) == 3:
            # 三关节模式
            ARM_SWING_AMPLITUDES = args.arm_swing_amplitude
            print(f"命令行指定三关节摆动幅度: 关节1(±{ARM_SWING_AMPLITUDES[0]}rad), 关节2(±{ARM_SWING_AMPLITUDES[1]}rad), 关节3(±{ARM_SWING_AMPLITUDES[2]}rad)")
        else:
            print(f"警告: 摆动幅度参数数量不正确，期望1个或3个，实际{len(args.arm_swing_amplitude)}个")
            print("将使用默认摆动幅度")
    
    if args.arm_swing_frequency is not None:
        if len(args.arm_swing_frequency) == 1:
            # 单关节模式
            ARM_SWING_FREQUENCY = args.arm_swing_frequency[0]
            print(f"命令行指定摆动频率: {ARM_SWING_FREQUENCY}Hz")
        elif len(args.arm_swing_frequency) == 3:
            # 三关节模式
            ARM_SWING_FREQUENCIES = args.arm_swing_frequency
            print(f"命令行指定三关节摆动频率: 关节1({ARM_SWING_FREQUENCIES[0]}Hz), 关节2({ARM_SWING_FREQUENCIES[1]}Hz), 关节3({ARM_SWING_FREQUENCIES[2]}Hz)")
        else:
            print(f"警告: 摆动频率参数数量不正确，期望1个或3个，实际{len(args.arm_swing_frequency)}个")
            print("将使用默认摆动频率")
    
    if args.arm_swing_phase is not None:
        if len(args.arm_swing_phase) == 1:
            # 单关节模式
            ARM_SWING_PHASE = args.arm_swing_phase[0]
            print(f"命令行指定摆动相位: {ARM_SWING_PHASE}rad")
        elif len(args.arm_swing_phase) == 3:
            # 三关节模式
            ARM_SWING_PHASES = args.arm_swing_phase
            print(f"命令行指定三关节摆动相位: 关节1({ARM_SWING_PHASES[0]}rad), 关节2({ARM_SWING_PHASES[1]}rad), 关节3({ARM_SWING_PHASES[2]}rad)")
        else:
            print(f"警告: 摆动相位参数数量不正确，期望1个或3个，实际{len(args.arm_swing_phase)}个")
            print("将使用默认摆动相位")
    
    # 默认摆动参数（当没有通过命令行指定时使用）
    ARM_JOINT_CONFIGS = {
        0: {"amplitude": 0.5, "frequency_multiplier": 1.0, "phase_offset": 0.0},      # 关节1：小幅度摆动 ±0.5rad
        1: {"amplitude": 1.0, "frequency_multiplier": 1.2, "phase_offset": np.pi/3},  # 关节2：中等幅度摆动 ±1.0rad
        2: {"amplitude": 1.5, "frequency_multiplier": 0.8, "phase_offset": np.pi/2}   # 关节3：大幅度摆动 ±1.5rad
    }

    def get_arm_swing_angles(t):
        """根据课程阶段6的设计，为三个关节生成不同的摆动角度"""
        # 检查是否禁用摆动
        if args.disable_arm_swing:
            return []
        
        if env.curriculum_stage == 6 and arm_joint_indices is not None:
            # 阶段6：三关节摆动，每个关节有不同的摆动幅度
            swing_angles = []
            for i, joint_idx in enumerate(arm_joint_indices):
                # 优先使用命令行指定的参数
                if args.arm_swing_amplitude is not None and len(args.arm_swing_amplitude) == 3:
                    amplitude = args.arm_swing_amplitude[i]
                else:
                    amplitude = ARM_JOINT_CONFIGS[i]["amplitude"]
                
                if args.arm_swing_frequency is not None and len(args.arm_swing_frequency) == 3:
                    frequency = args.arm_swing_frequency[i]
                else:
                    frequency = 1.0 / ARM_SWING_PERIOD * ARM_JOINT_CONFIGS[i]["frequency_multiplier"]
                
                if args.arm_swing_phase is not None and len(args.arm_swing_phase) == 3:
                    phase_offset = args.arm_swing_phase[i]
                else:
                    phase_offset = ARM_JOINT_CONFIGS[i]["phase_offset"]
                
                angle = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
                swing_angles.append(angle)
            return swing_angles
        else:
            # 其他阶段：单关节摆动（保持原有逻辑）
            if arm_joint_idx is not None:
                # 优先使用命令行指定的参数
                if args.arm_swing_amplitude is not None and len(args.arm_swing_amplitude) == 1:
                    amplitude = args.arm_swing_amplitude[0]
                else:
                    # 使用默认摆动参数
                    ARM_SWING_MIN = -1.5
                    ARM_SWING_MAX = 1.5
                    amplitude = (ARM_SWING_MAX - ARM_SWING_MIN) / 2
                
                if args.arm_swing_frequency is not None and len(args.arm_swing_frequency) == 1:
                    frequency = args.arm_swing_frequency[0]
                else:
                    frequency = 1.0 / ARM_SWING_PERIOD
                
                if args.arm_swing_phase is not None and len(args.arm_swing_phase) == 1:
                    phase_offset = args.arm_swing_phase[0]
                else:
                    phase_offset = 0.0
                
                angle = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
                return [angle]
            return []

    # import pdb;pdb.set_trace()
#policy2
    # 导出策略（如果指定） 
    runner2: OnPolicyRunner = hydra.utils.instantiate(
        config["runner"], env=env, eval_fn=None
    )
    args.ckpt_path2 = "/home/group16/xuws/umi-on-legs-compose/mani-centric-wbc/checkpoints/pushing/model_18000.pt"
    runner2.load(args.ckpt_path2)
    export_dir = os.path.join(os.path.dirname(args.ckpt_path2), 'exported')
    # 确保目录存在
    os.makedirs(export_dir, exist_ok=True)
    # import pdb;pdb.set_trace()
    # # 导出策略
    export_policy_as_jit(runner2.alg.actor_critic, export_dir)#import policy
    policy2 = runner2.alg.get_inference_policy(device=env.device)
#policy2
#weights
    weights = torch.tensor([0.95, 0.05],device=env.device)  # 融合权重（可动态调整）
#weights
    if args.num_steps == -1:
        with torch.inference_mode():
            while True:
                actions = policy(obs)#Here is the policy to action
                actions2 = policy2(obs) 
                actions = weights[0] * actions + weights[1] * actions2
                # 设置速度指令
                if args.vel_x is not None or args.vel_y is not None or args.vel_z is not None:
                    # 使用指定的速度值
                    if args.vel_x is not None:
                        env.base_lin_vel_cmd[:, 0] = args.vel_x
                    if args.vel_y is not None:
                        env.base_lin_vel_cmd[:, 1] = args.vel_y
                    if args.vel_z is not None:
                        env.base_lin_vel_cmd[:, 2] = args.vel_z

                # 使用确定的机械臂关节进行摆动控制
                actions = actions.clone()  # 避免原地修改
                t = env.state.episode_time[0].item() if hasattr(env.state, "episode_time") else 0
                swing_angles = get_arm_swing_angles(t)
                
                # 根据课程阶段应用摆动控制
                if env.curriculum_stage == 6 and arm_joint_indices is not None:
                    # 阶段6：三关节摆动控制
                    for i, (joint_idx, swing_angle) in enumerate(zip(arm_joint_indices, swing_angles)):
                        if actions.shape[1] > joint_idx:
                            actions[:, joint_idx] = swing_angle
                    print(f"时间: {t:.2f}s, 三关节摆动 - 关节1({arm_joint_indices[0]}): {swing_angles[0]:.3f}rad, "
                          f"关节2({arm_joint_indices[1]}): {swing_angles[1]:.3f}rad, "
                          f"关节3({arm_joint_indices[2]}): {swing_angles[2]:.3f}rad")
                elif arm_joint_idx is not None and actions.shape[1] > arm_joint_idx:
                    # 其他阶段：单关节摆动控制
                    actions[:, arm_joint_idx] = swing_angles[0] if swing_angles else 0.0
                    print(f"时间: {t:.2f}s, 单关节摆动 - 关节{arm_joint_idx}角度: {swing_angles[0] if swing_angles else 0.0:.3f}rad")

                obs = env.step(actions)[0]
                update_cam_pos()
                env.render()

    state_logs = {
        "root_state": [],
        "root_pos": [],
        "root_xyzw_quat": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "rigid_body_pos": [],
        "rigid_body_xyzw_quat": [],
        "dof_pos": [],
        "dof_vel": [],
        "contact_forces": [],
        "episode_time": [],
    }
    action_logs = {
        "torque": [],
        "action": [],
        "obs": [],
    }
    # 对于 velcmd_arm_swing 环境，我们记录 baselink 速度指令
    velcmd_logs = {
        "base_lin_vel_cmd": [],
        "curriculum_stage": [],
        "arm_swing_amplitude": [],
        "arm_control_joint_idx": [],  # 记录环境选择的机械臂关节索引
        "num_arm_dofs": [],          # 记录机械臂关节数量
    }
    episode_logs = {}
    
    with imageio.get_writer(
        uri=os.path.join(wandb.run.dir, "video.mp4"), mode="I", fps=24
    ) as writer, torch.inference_mode():

        def render_cb(env, writer=writer):
            global count
            if env.state.time * 24 < count:
                return
            if args.visualize:
                update_cam_pos()
                env.visualize(vis_env_ids=[0])
                env.render()
                count += 1
                if args.record_video:
                    env.gym.write_viewer_image_to_file(
                        env.viewer, f"/{wandb.run.dir}/out.png"
                    )
                    img = imageio.imread(f"/{wandb.run.dir}/out.png")
                    writer.append_data(img)
            
            # 时间变化的速度指令
            if args.time_varying_vel:
                time = env.state.episode_time[0].item()
                env.base_lin_vel_cmd[:, 0] = args.vel_amplitude * torch.sin(2 * np.pi * args.vel_frequency * time)
                env.base_lin_vel_cmd[:, 1] = args.vel_amplitude * torch.cos(2 * np.pi * args.vel_frequency * time)
                env.base_lin_vel_cmd[:, 2] = 0.0
            
            # 记录状态信息
            for k, v in state_logs.items():
                v.append(getattr(env.state, k)[:].cpu().numpy())
            
            action_logs["torque"].append(env.ctrl.torque[:].cpu().numpy())
            action_logs["action"].append(actions.view(args.num_envs, -1).cpu().numpy())
            action_logs["obs"].append(obs.view(args.num_envs, -1).cpu().numpy())
            
            # 记录 velcmd_arm_swing 特有的信息
            velcmd_logs["base_lin_vel_cmd"].append(env.base_lin_vel_cmd[:].cpu().numpy())
            velcmd_logs["curriculum_stage"].append([env.curriculum_stage] * args.num_envs)
            velcmd_logs["arm_swing_amplitude"].append([env.arm_swing_amplitude] * args.num_envs)
            
            # 记录机械臂关节选择信息
            try:
                if hasattr(env, 'arm_control_joint_idx') and env.num_arm_dofs > 0:
                    velcmd_logs["arm_control_joint_idx"].append(env.arm_control_joint_idx[:].cpu().numpy())
                    velcmd_logs["num_arm_dofs"].append([env.num_arm_dofs] * args.num_envs)
                    
                    # 新增：记录机械臂摆动角度和状态信息
                    if env.curriculum_stage == 6 and arm_joint_indices is not None:
                        # 阶段6：记录三关节信息
                        # 移除机械臂摆动数据记录
                        pass
                    else:
                        # 其他阶段：记录单关节信息
                        # 移除机械臂摆动数据记录
                        pass
            except Exception as e:
                # 如果环境出现问题，记录我们使用的硬编码关节索引
                velcmd_logs["arm_control_joint_idx"].append([arm_joint_idx if arm_joint_idx is not None else -1] * args.num_envs)
                velcmd_logs["num_arm_dofs"].append([0] * args.num_envs)  # 表示使用硬编码索引

        for step_idx in track(range(args.num_steps), description="Playing velcmd_arm_swing"):
            actions = policy(obs)
            # 使用确定的机械臂关节进行摆动控制
            actions = actions.clone()  # 避免原地修改
            t = env.state.episode_time[0].item() if hasattr(env.state, "episode_time") else step_idx * env.gym_dt
            swing_angles = get_arm_swing_angles(t)
            
            # 根据课程阶段应用摆动控制
            if env.curriculum_stage == 6 and arm_joint_indices is not None:
                # 阶段6：三关节摆动控制
                for i, (joint_idx, swing_angle) in enumerate(zip(arm_joint_indices, swing_angles)):
                    if actions.shape[1] > joint_idx:
                        actions[:, joint_idx] = swing_angle
                # 只在特定步数打印，避免日志过多
                if step_idx % 100 == 0:
                    print(f"步数: {step_idx}, 三关节摆动 - 关节1({arm_joint_indices[0]}): {swing_angles[0]:.3f}rad, "
                          f"关节2({arm_joint_indices[1]}): {swing_angles[1]:.3f}rad, "
                          f"关节3({arm_joint_indices[2]}): {swing_angles[2]:.3f}rad")
            elif arm_joint_idx is not None and actions.shape[1] > arm_joint_idx:
                # 其他阶段：单关节摆动控制
                actions[:, arm_joint_idx] = swing_angles[0] if swing_angles else 0.0
            obs, privileged_obs, rews, dones, infos = env.step(
                actions, callback=render_cb
            )
            
            # 记录速度数据用于绘图
            current_time = env.state.episode_time[actor_idx].item()
            actual_lin_vel = env.state.local_root_lin_vel[actor_idx, :].cpu().numpy()
            desired_lin_vel = env.base_lin_vel_cmd[actor_idx, :].cpu().numpy()
            
            velocity_logs["time"].append(current_time)
            velocity_logs["desired_lin_vel_x"].append(desired_lin_vel[0])
            velocity_logs["actual_lin_vel_x"].append(actual_lin_vel[0])
            velocity_logs["desired_lin_vel_y"].append(desired_lin_vel[1])
            velocity_logs["actual_lin_vel_y"].append(actual_lin_vel[1])
            velocity_logs["desired_lin_vel_z"].append(desired_lin_vel[2])
            velocity_logs["actual_lin_vel_z"].append(actual_lin_vel[2])
            velocity_logs["curriculum_stage"].append(env.curriculum_stage)
            
            for k, v in infos.items():
                episode_logs.setdefault(k, []).append(v[:].cpu().numpy())

            if args.visualize and dones[actor_idx]:
                break
    
    # 绘制速度跟踪曲线
    plot_velocity_curves(velocity_logs, wandb.run.dir)
    
    # 移除机械臂摆动曲线绘制
    # plot_arm_swing_curves(velcmd_logs, wandb.run.dir)

    # 保存日志数据
    root = zarr.group(
        store=zarr.DirectoryStore(wandb.run.dir + "/logs.zarr"), overwrite=True
    )
    for k, v in {
        **state_logs,
        **action_logs,
        **episode_logs,
        **velcmd_logs,  # 新增 velcmd 日志
    }.items():
        v = np.array(v)
        k = k.replace("/", "_")
        if len(v.shape) == 2:
            root.create_dataset(
                k, data=v, chunks=(1, *list(v.shape)[1:]), dtype=v.dtype
            )
        else:
            root.create_dataset(k, data=v, dtype=v.dtype)
    
    pickle.dump(
        {
            "config": config,
            "state_logs": state_logs,
            "action_logs": action_logs,
            "episode_logs": episode_logs,
            "velcmd_logs": velcmd_logs,  # 新增 velcmd 日志
        },
        open(wandb.run.dir + "/logs.zarr", "wb"),
    )


if __name__ == "__main__":
    play()
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

import wandb
from legged_gym.env.isaacgym.env_add_baseinfo import IsaacGymEnv
from train import setup

import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


def plot_trajectory(ee_logs, save_path="trajectory_plot.png"):
    """绘制末端执行器轨迹图"""
    if not ee_logs["actual_positions"]:
        print("No trajectory data to plot")
        return
    
    # 转换为numpy数组
    actual_pos = np.array(ee_logs["actual_positions"])
    target_pos = np.array(ee_logs["target_positions"])
    time_steps = np.array(ee_logs["time_steps"])
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('End-Effector Trajectory Analysis', fontsize=16)
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
             'b-', label='Actual', linewidth=2)
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
             'r--', label='Target', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. X位置随时间变化
    ax2 = axes[0, 1]
    ax2.plot(time_steps, actual_pos[:, 0], 'b-', label='Actual X', linewidth=2)
    ax2.plot(time_steps, target_pos[:, 0], 'r--', label='Target X', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Y位置随时间变化
    ax3 = axes[1, 0]
    ax3.plot(time_steps, actual_pos[:, 1], 'b-', label='Actual Y', linewidth=2)
    ax3.plot(time_steps, target_pos[:, 1], 'r--', label='Target Y', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Y Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Z位置随时间变化
    ax4 = axes[1, 1]
    ax4.plot(time_steps, actual_pos[:, 2], 'b-', label='Actual Z', linewidth=2)
    ax4.plot(time_steps, target_pos[:, 2], 'r--', label='Target Z', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Z Position vs Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {save_path}")
    
    # 计算并打印跟踪误差统计
    position_error = np.linalg.norm(actual_pos - target_pos, axis=1)
    print(f"\nTrajectory Tracking Statistics:")
    print(f"Mean Position Error: {np.mean(position_error):.4f} m")
    print(f"Max Position Error: {np.max(position_error):.4f} m")
    print(f"RMS Position Error: {np.sqrt(np.mean(position_error**2)):.4f} m")
    
    # 显示图形（如果在可视化模式下）
    plt.show()


def play():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trajectory_file_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--vx", type=float, default=0.5, help="X velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Y velocity command (m/s)")
    parser.add_argument("--vz", type=float, default=0.0, help="Z velocity command (m/s)")
    parser.add_argument("--plot_trajectory", action="store_true", help="Plot end-effector trajectory")
    parser.add_argument("--plot_save_path", type=str, default="trajectory_plot.png", help="Path to save trajectory plot")
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
    config["env"]["tasks"]["reaching"]["sequence_sampler"][
        "file_path"
    ] = args.trajectory_file_path

    config["env"]["constraints"] = {}

    setup(config, seed=config["seed"])  # type: ignore

    env: IsaacGymEnv = hydra.utils.instantiate(
        config["env"],
        sim_params=sim_params,
    )
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

    obs, privileged_obs = env.reset()
    


    if args.visualize:
        env.render()  # render once to initialize viewer

    if args.num_steps == -1:
        with torch.inference_mode():
            while True:
                # 在每个step中更新速度指令，确保观测中的base_lin_vel_cmd保持最新
                # env.base_lin_vel_cmd[:, 0] = args.vx  # vx = 0.5 m/s
                # env.base_lin_vel_cmd[:, 1] = args.vy  # vy = 0.0 m/s
                # env.base_lin_vel_cmd[:, 2] = args.vz  # vz = 0.0 m/s
                # print("base_lin",env.state.local_root_lin_vel[:, 0])
                actions = policy(obs)
                # actions = torch.zeros(args.num_envs, 18, device=env.device)
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
    task_logs = {
        "target_positions": [],
        "target_quats_wxyz": [],
    }
    
    # 添加末端执行器位置记录
    ee_logs = {
        "actual_positions": [],
        "target_positions": [],
        "time_steps": [],
    }
    episode_logs = {}
    task = list(env.tasks.values())[0]
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
            for k, v in state_logs.items():
                v.append(getattr(env.state, k)[:].cpu().numpy())
            action_logs["torque"].append(env.ctrl.torque[:].cpu().numpy())
            action_logs["action"].append(actions.view(args.num_envs, -1).cpu().numpy())
            action_logs["obs"].append(obs.view(args.num_envs, -1).cpu().numpy())
            global_target_pose = torch.stack(
                [
                    task.get_target_pose(
                        times=env.state.episode_time + t_offset,
                        sim_dt=env.state.sim_dt,
                    )
                    # NOTE: only for visualization purposes,
                    # you can change the target pose times to any
                    # time interval which visualizes the gripper
                    # movements best
                    for t_offset in np.linspace(0.05, 1.0, 8)
                ],
                dim=1,
            )
            task_logs["target_positions"].append(
                global_target_pose[..., :3, 3].cpu().squeeze().numpy()
            )
            task_logs["target_quats_wxyz"].append(
                np.vstack(
                    [
                        quaternions.mat2quat(rot_mat)
                        for rot_mat in global_target_pose[..., :3, :3]
                        .cpu()
                        .squeeze()
                        .numpy()
                    ]
                )
            )
            
            # 记录末端执行器实际位置和期望位置
            if args.plot_trajectory:
                # 获取当前末端执行器位置（假设是最后一个rigid body）
                ee_pos = env.state.rigid_body_pos[actor_idx, -1, :].cpu().numpy()  # 最后一个rigid body的位置
                target_pos = task.get_target_pose(
                    times=env.state.episode_time,
                    sim_dt=env.state.sim_dt,
                )[actor_idx, :3, 3].cpu().numpy()  # 当前时刻的期望位置
                
                ee_logs["actual_positions"].append(ee_pos)
                ee_logs["target_positions"].append(target_pos)
                ee_logs["time_steps"].append(env.state.time)  # time是标量，不需要索引

        for step_idx in track(range(args.num_steps), description="Playing"):
            # 在每个step中更新速度指令，确保观测中的base_lin_vel_cmd保持最新
            env.base_lin_vel_cmd[:, 0] = args.vx  # vx = 0.5 m/s
            env.base_lin_vel_cmd[:, 1] = args.vy  # vy = 0.0 m/s
            env.base_lin_vel_cmd[:, 2] = args.vz  # vz = 0.0 m/s
            
            actions = policy(obs)
            obs, privileged_obs, rews, dones, infos = env.step(
                actions, callback=render_cb
            )
            for k, v in infos.items():
                episode_logs.setdefault(k, []).append(v[:].cpu().numpy())

            if args.visualize and dones[actor_idx]:
                break
    root = zarr.group(
        store=zarr.DirectoryStore(wandb.run.dir + "/logs.zarr"), overwrite=True
    )
    for k, v in {
        **state_logs,
        **action_logs,
        **episode_logs,
        **task_logs,
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
            "task_logs": task_logs,
        },
        open(wandb.run.dir + "/logs.pkl", "wb"),
    )
    
    # 绘制轨迹图
    if args.plot_trajectory:
        plot_save_path = os.path.join(wandb.run.dir, args.plot_save_path)
        plot_trajectory(ee_logs, plot_save_path)


if __name__ == "__main__":
    play()

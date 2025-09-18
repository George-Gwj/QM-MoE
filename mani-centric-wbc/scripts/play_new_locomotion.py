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
from legged_gym.env.isaacgym.env_add_baseinfo_origin import IsaacGymEnv
from train import setup

import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

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


def plot_velocity_curves(velocity_logs, save_path):
    """绘制vx, vy, wz的真实速度和期望速度曲线"""
    time_steps = np.arange(len(velocity_logs['actual_vx']))
    
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Robot Velocity Tracking Performance', fontsize=16)
    
    # 绘制vx曲线
    axes[0].plot(time_steps, velocity_logs['actual_vx'], 'b-', label='Actual vx', linewidth=2)
    axes[0].plot(time_steps, velocity_logs['target_vx'], 'r--', label='Target vx', linewidth=2)
    axes[0].set_ylabel('vx (m/s)')
    axes[0].set_title('Linear Velocity X')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制vy曲线
    axes[1].plot(time_steps, velocity_logs['actual_vy'], 'b-', label='Actual vy', linewidth=2)
    axes[1].plot(time_steps, velocity_logs['target_vy'], 'r--', label='Target vy', linewidth=2)
    axes[1].set_ylabel('vy (m/s)')
    axes[1].set_title('Linear Velocity Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 绘制wz曲线
    axes[2].plot(time_steps, velocity_logs['actual_wz'], 'b-', label='Actual wz', linewidth=2)
    axes[2].plot(time_steps, velocity_logs['target_wz'], 'r--', label='Target wz', linewidth=2)
    axes[2].set_ylabel('wz (rad/s)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_title('Angular Velocity Z')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Velocity curves saved to: {save_path}')


def play():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trajectory_file_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
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

    config["env"]["tasks"]["locomotion"]["lin_vel_range"] = [[0.5,0.5],[0.0,0.0],[0.0,0.0]] 
    config["env"]["tasks"]["locomotion"]["ang_vel_range"] = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    config["env"]["tasks"]["locomotion"]["z_height_range"] = [0.25,0.35]

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
                actions = policy(obs)
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
    velocity_logs = {
        "actual_vx": [],
        "actual_vy": [],
        "actual_wz": [],
        "target_vx": [],
        "target_vy": [],
        "target_wz": [],
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
            
            # 记录速度数据
            local_lin_vel = env.state.local_root_lin_vel[actor_idx].cpu().numpy()
            local_ang_vel = env.state.local_root_ang_vel[actor_idx].cpu().numpy()
            velocity_logs["actual_vx"].append(local_lin_vel[0])
            velocity_logs["actual_vy"].append(local_lin_vel[1])
            velocity_logs["actual_wz"].append(local_ang_vel[2])
            

            velocity_logs["target_vx"].append(env.tasks["locomotion"].target_lin_vel[actor_idx, 0].cpu().numpy())
            velocity_logs["target_vy"].append(env.tasks["locomotion"].target_lin_vel[actor_idx, 1].cpu().numpy())
            velocity_logs["target_wz"].append(env.tasks["locomotion"].target_ang_vel[actor_idx, 2].cpu().numpy())


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

        for step_idx in track(range(args.num_steps), description="Playing"):
            actions = policy(obs)
            obs, privileged_obs, rews, dones, infos = env.step(
                actions, callback=render_cb
            )
            for k, v in infos.items():
                episode_logs.setdefault(k, []).append(v[:].cpu().numpy())

            if args.visualize and dones[actor_idx]:
                break
    # 绘制速度曲线
    velocity_plot_path = os.path.join(wandb.run.dir, "velocity_curves.png")
    plot_velocity_curves(velocity_logs, velocity_plot_path)
    
    root = zarr.group(
        store=zarr.DirectoryStore(wandb.run.dir + "/logs.zarr"), overwrite=True
    )
    for k, v in {
        **state_logs,
        **action_logs,
        **episode_logs,
        **task_logs,
        **velocity_logs,
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
            "velocity_logs": velocity_logs,
        },
        open(wandb.run.dir + "/logs.pkl", "wb"),
    )


if __name__ == "__main__":
    play()

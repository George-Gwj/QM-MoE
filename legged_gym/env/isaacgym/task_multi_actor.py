from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_rotate_inverse
from matplotlib import pyplot as plt

from legged_gym.env.isaacgym.control import Control
from legged_gym.env.isaacgym.pose_sequence import SequenceSampler
# from legged_gym.env.isaacgym.state_multi_actor import EnvState
from legged_gym.env.isaacgym.state_multi_actor import EnvState
from legged_gym.env.isaacgym.utils import torch_rand_float


def check_should_reset(
    time_s: torch.Tensor,
    dt: float,
    reset_time_s: float,
):
    num_episode_steps = (time_s / dt).long()
    reset_every_n_steps = int(reset_time_s / dt)
    should_reset = (num_episode_steps % reset_every_n_steps) == 0
    return should_reset


class Task:
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
    ):
        self.gym = gym
        self.sim = sim
        self.num_envs = gym.get_env_count(sim)
        self.device = device
        self.generator = generator

    @abstractmethod
    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # per simulation step callback, returns a dictionary task metrics (accuracy, etc.)
        pass

    def reset_idx(self, env_ids: torch.Tensor):
        # per episode
        pass

    def visualize(self, state: EnvState, viewer: gymapi.Viewer, vis_env_ids: List[int]):
        pass

    @abstractmethod
    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def observe(self, state: EnvState) -> torch.Tensor:
        pass


class Link3DVelocity(Task):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        target_link_name: str,
        lin_vel_obs_scale: float,
        ang_vel_obs_scale: float,
        resampling_time: float,
        min_target_lin_vel: float,
        lin_vel_range: List[Tuple[float, float]],
        ang_vel_range: List[Tuple[float, float]],
        tracking_sigma: float,
        lin_vel_reward_scale: float,
        ang_vel_reward_scale: float,
        feet_air_time_reward_scale: float,
        lin_vel_reward_power: float,
        ang_vel_reward_power: float,
        feet_sensor_indices: List[int],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
        )
        env = gym.get_env(sim, 0)
        self.target_link_idx = gym.find_actor_rigid_body_handle(
            env,
            gym.get_actor_handle(env, 0),
            target_link_name,
        )
        assert self.target_link_idx != -1
        self.target_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.lin_vel_obs_scale = lin_vel_obs_scale
        self.ang_vel_obs_scale = ang_vel_obs_scale
        self.resampling_time = resampling_time
        self.min_target_lin_vel = min_target_lin_vel

        self.lin_vel_range = torch.tensor(lin_vel_range).to(self.device)
        self.ang_vel_range = torch.tensor(ang_vel_range).to(self.device)
        assert len(self.lin_vel_range) == 3
        assert len(self.ang_vel_range) == 3
        self.tracking_sigma = tracking_sigma

        self.lin_vel_reward_scale = lin_vel_reward_scale
        self.ang_vel_reward_scale = ang_vel_reward_scale
        self.lin_vel_reward_power = lin_vel_reward_power
        self.ang_vel_reward_power = ang_vel_reward_power
        self.feet_air_time_reward_scale = feet_air_time_reward_scale

        self.feet_sensor_indices = torch.tensor(
            feet_sensor_indices,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_sensor_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_sensor_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.target_lin_vel[env_ids, :] = torch_rand_float(
            self.lin_vel_range[:, 0],
            self.lin_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)
        self.target_ang_vel[env_ids, :] = torch_rand_float(
            self.ang_vel_range[:, 0],
            self.ang_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)

        # set small commands to zero
        self.target_lin_vel[env_ids] *= (
            torch.norm(self.target_lin_vel[env_ids], dim=1) > self.min_target_lin_vel
        ).unsqueeze(1)

    def reset_idx(self, env_ids: torch.Tensor):
        self.resample_commands(env_ids)
        self.feet_air_time[env_ids] = 0.0

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        env_ids = (
            check_should_reset(
                time_s=state.episode_time,
                dt=state.sim_dt,
                reset_time_s=self.resampling_time,
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.resample_commands(env_ids)
        return {
            "angular_vel_err": self.get_ang_vel_err(state=state),
            "linear_vel_err": self.get_lin_vel_err(state=state),
        }

    def observe(self, state: EnvState) -> torch.Tensor:
        obs_terms = [
            self.target_lin_vel * self.lin_vel_obs_scale,
            self.target_ang_vel * self.ang_vel_obs_scale,
        ]
        obs = torch.cat(
            obs_terms,
            dim=-1,
        )
        return obs

    def get_link_local_lin_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.rigid_body_lin_vel[:, self.target_link_idx],
        )

    def get_link_local_ang_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.rigid_body_ang_vel[:, self.target_link_idx],
        )

    def get_lin_vel_err(self, state: EnvState):
        return (self.target_lin_vel - self.get_link_local_lin_vel(state=state)).norm(
            dim=-1
        )

    def get_ang_vel_err(self, state: EnvState):
        return (self.target_ang_vel - self.get_link_local_ang_vel(state=state)).norm(
            dim=-1
        )

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # Tracking of linear velocity commands (xy axes)
        lin_vel_reward = torch.exp(
            -self.get_lin_vel_err(state=state) ** self.lin_vel_reward_power
            / self.tracking_sigma
        )
        # Tracking of angular velocity commands (yaw)
        yaw_reward = torch.exp(
            -self.get_ang_vel_err(state=state) ** self.ang_vel_reward_power
            / self.tracking_sigma
        )
        assert state.force_sensor_tensor is not None
        contact = state.force_sensor_tensor[:, self.feet_sensor_indices, 2].abs() > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += state.sim_dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.target_lin_vel[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt

        return {
            "lin_vel": lin_vel_reward * self.lin_vel_reward_scale,
            "yaw": yaw_reward * self.ang_vel_reward_scale,
            "feet_air_time": rew_airTime * self.feet_air_time_reward_scale,
        }


class Link2DVelocity(Link3DVelocity):
    def observe(self, state: EnvState) -> torch.Tensor:
        obs_terms = [
            self.target_lin_vel[:, :2] * self.lin_vel_obs_scale,
            self.target_ang_vel[:, [2]] * self.ang_vel_obs_scale,
        ]
        obs = torch.cat(
            obs_terms,
            dim=-1,
        )
        return obs

    def get_lin_vel_err(self, state: EnvState):
        return (
            self.target_lin_vel[:, :2] - self.get_link_local_lin_vel(state=state)[:, :2]
        ).norm(dim=-1)

    def get_ang_vel_err(self, state: EnvState):
        return (
            self.target_ang_vel[:, 2] - self.get_link_local_ang_vel(state=state)[:, 2]
        ).abs()


class Link6DVelocity(Link2DVelocity):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        z_height_range: Tuple[float, float],
        z_height_sigma: float,
        z_height_reward_scale: float,
        roll_range: Tuple[float, float],
        pitch_range: Tuple[float, float],
        gravity_sigma: float,
        gravity_reward_scale: float,
        **kwargs,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            **kwargs,
        )
        self.z_height_range = torch.tensor(z_height_range).to(self.device)
        self.roll_range = torch.tensor(roll_range).to(self.device)
        self.pitch_range = torch.tensor(pitch_range).to(self.device)
        self.z_height_sigma = z_height_sigma
        self.gravity_sigma = gravity_sigma
        self.z_height_reward_scale = z_height_reward_scale
        self.gravity_reward_scale = gravity_reward_scale
        self.target_z_height = torch.zeros((self.num_envs,), device=self.device)
        self.target_local_gravity = torch.zeros((self.num_envs, 3), device=self.device)

    def resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        super().resample_commands(env_ids)
        self.target_z_height[env_ids] = torch_rand_float(
            self.z_height_range[0],
            self.z_height_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        roll = torch_rand_float(
            self.roll_range[0],
            self.roll_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        pitch = torch_rand_float(
            self.pitch_range[0],
            self.pitch_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        # convert to gravity vector with trigonometry
        self.target_local_gravity[env_ids, 0] = torch.tan(roll)
        self.target_local_gravity[env_ids, 1] = torch.tan(pitch)
        self.target_local_gravity[env_ids, 2] = -1.0
        self.target_local_gravity /= self.target_local_gravity.norm(dim=1, keepdim=True)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        stats = super().step(state=state, control=control)
        stats["z_height_err"] = self.get_z_height_err(state=state)
        stats["gravity_err"] = self.get_gravity_err(state=state)
        return stats

    def observe(self, state: EnvState) -> torch.Tensor:
        return torch.cat(
            [
                super().observe(state=state),
                self.target_z_height[:, None],
                self.target_local_gravity,
            ],
            dim=-1,
        )

    def get_z_height_err(self, state: EnvState):
        # Penalize base height away from target
        link_height = torch.mean(
            state.rigid_body_pos[:, self.target_link_idx, [2]]
            - state.measured_terrain_heights,
            dim=1,
        )
        return torch.square(self.target_z_height - link_height)

    def get_gravity_err(self, state: EnvState):
        link_local_gravity = quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.gravity / torch.linalg.norm(state.gravity, dim=1, keepdims=True),
        )
        return torch.square(self.target_local_gravity - link_local_gravity).sum(dim=1)

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        stats = super().reward(state=state, control=control)

        stats["z_height"] = (
            torch.exp(-self.get_z_height_err(state=state) / self.z_height_sigma)
            * self.z_height_reward_scale
        )
        stats["gravity"] = (
            torch.exp(-self.get_gravity_err(state=state) / self.gravity_sigma)
            * self.gravity_reward_scale
        )
        return stats


@torch.jit.script
def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    return pt3d.quaternion_to_matrix(quat)


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    return pt3d.axis_angle_to_quaternion(axis_angle)


@torch.jit.script
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


@torch.jit.script
def quaternion_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    return pt3d.quaternion_to_axis_angle(quat)


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    return pt3d.matrix_to_quaternion(matrix)


@torch.jit.script
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


class ReachingLinkTask(Task):
    """
    World Frame Target Pose Tracking Task.

    Supports relative pose observations with pose latency, and curriculum learning for error thresholds.
    """

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        link_name: str,
        pos_obs_scale: float,
        orn_obs_scale: float,
        pos_err_sigma: float,
        orn_err_sigma: float,
        pos_reward_scale: float,
        orn_reward_scale: float,
        pose_reward_scale: float,
        target_obs_times: List[float],
        sequence_sampler: SequenceSampler,
        pose_latency: float,
        position_obs_encoding: str = "linear",
        pose_latency_variability: Optional[Tuple[float, float]] = None,
        pose_latency_warmup_steps: int = 0,
        pose_latency_warmup_start: int = 0,
        position_noise: float = 0.0,
        euler_noise: float = 0.0,
        target_relative_to_base: bool = False,
        pos_sigma_curriculum: Optional[
            Dict[float, float]
        ] = None,  # maps from error to sigma
        orn_sigma_curriculum: Optional[
            Dict[float, float]
        ] = None,  # maps from error to sigma
        init_pos_curriculum_level: int = 0,
        init_orn_curriculum_level: int = 0,
        smoothing_dt_multiplier: float = 4.0,
        storage_device: str = "cpu",
        pos_obs_clip: Optional[float] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=torch.Generator(device=storage_device),
        )
        self.storage_device = storage_device
        self.link_name = link_name
        env = gym.get_env(sim, 0)
        actor = gym.get_actor_handle(env, 0)
        self.link_index = gym.find_actor_rigid_body_handle(env, actor, link_name)
        logging.info(f"Link index: {self.link_index} ({link_name})")
        if self.link_index == -1:
            raise ValueError(
                f"Could not find {self.link_name!r} in actor {gym.get_actor_name(env, 0)!r}"
            )
        self.curr_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.curr_target_rot_mat = torch.zeros(
            (self.num_envs, 3, 3), device=self.device
        )

        self.target_pos_seq = torch.zeros(
            (self.num_envs, sequence_sampler.episode_length, 3),
            device=self.storage_device,
        )
        self.target_rot_mat_seq = torch.zeros(
            (self.num_envs, sequence_sampler.episode_length, 3, 3),
            device=self.storage_device,
        )
        self.position_obs_encoding = position_obs_encoding
        assert self.position_obs_encoding in {"linear", "log-direction"}
        self.pos_obs_scale = pos_obs_scale
        self.orn_obs_scale = orn_obs_scale
        # NOTE since we're using rotation matrix representation
        # we don't need to clip orientation observations
        self.pos_obs_clip = pos_obs_clip
        self.pos_err_sigma = pos_err_sigma
        self.orn_err_sigma = orn_err_sigma
        self.pos_reward_scale = pos_reward_scale
        self.orn_reward_scale = orn_reward_scale
        self.pose_reward_scale = pose_reward_scale
        self.sequence_sampler = sequence_sampler
        self.target_obs_times = target_obs_times
        self.target_relative_to_base = target_relative_to_base

        self.past_pos_err = torch.ones((self.num_envs,), device=self.device)
        self.past_orn_err = torch.ones((self.num_envs,), device=self.device)
        self.smoothing_dt_multiplier = smoothing_dt_multiplier
        self.pos_sigma_curriculum = pos_sigma_curriculum
        self.orn_sigma_curriculum = orn_sigma_curriculum
        self.pos_sigma_curriculum_level = (
            0 if init_pos_curriculum_level is None else init_pos_curriculum_level
        )
        self.orn_sigma_curriculum_level = (
            0 if init_orn_curriculum_level is None else init_orn_curriculum_level
        )
        if self.pos_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.pos_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.pos_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.pos_err_sigma = list(self.pos_sigma_curriculum.values())[
                self.pos_sigma_curriculum_level
            ]
            self.past_pos_err *= list(self.pos_sigma_curriculum.keys())[
                self.pos_sigma_curriculum_level
            ]
        if self.orn_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.orn_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.orn_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.orn_err_sigma = list(self.orn_sigma_curriculum.values())[
                self.orn_sigma_curriculum_level
            ]
            self.past_orn_err *= list(self.orn_sigma_curriculum.keys())[
                self.orn_sigma_curriculum_level
            ]
        self.pose_latency = pose_latency
        self.pose_latency_frames = (
            int(np.rint(pose_latency / self.sequence_sampler.dt)) + 1
        )
        self.pose_latency_frame_variability = (
            pose_latency_variability
            if pose_latency_variability is None
            else (
                int(np.rint(pose_latency_variability[0] / self.sequence_sampler.dt)),
                int(np.rint(pose_latency_variability[1] / self.sequence_sampler.dt)),
            )
        )

        self.link_pose_history = torch.zeros(
            (self.num_envs, self.pose_latency_frames, 4, 4),
            device=self.device,
        )
        self.link_pose_history[..., :, :] = torch.eye(4, device=self.device)
        self.root_pose_history = torch.zeros(
            (self.num_envs, self.pose_latency_frames, 4, 4),
            device=self.device,
        )
        self.root_pose_history[..., :, :] = torch.eye(4, device=self.device)
        self.pose_latency_warmup_steps = pose_latency_warmup_steps
        self.pose_latency_warmup_start = pose_latency_warmup_start
        self.steps = 0
        self.position_noise = position_noise
        self.euler_noise = euler_noise

    def reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        env_origins = torch.stack(
            [
                torch.tensor([env_origin.x, env_origin.y, env_origin.z])
                for env_origin in map(
                    lambda x: self.gym.get_env_origin(self.gym.get_env(self.sim, x)),
                    env_ids,
                )
            ],
            dim=0,
        ).to(
            self.storage_device
        )  # (num_envs, 3)
        # get root positions
        pos_seq, rot_mat_seq = self.sequence_sampler.sample(
            seed=int(
                torch.randint(
                    low=0,
                    high=2**32 - 1,
                    size=(1,),
                    generator=self.generator,
                    device=self.storage_device,
                )
                .cpu()
                .item()
            ),
            batch_size=len(env_ids),
        )
        assert rot_mat_seq.shape == (
            len(env_ids),
            self.sequence_sampler.episode_length,
            3,
            3,
        )

        assert pos_seq.shape == (len(env_ids), self.sequence_sampler.episode_length, 3)
        pos_seq = pos_seq.to(self.storage_device) + env_origins.unsqueeze(1)
        rot_mat_seq = rot_mat_seq.to(self.storage_device)
        device_env_ids = env_ids.to(self.storage_device)
        self.target_pos_seq[device_env_ids] = pos_seq
        self.target_rot_mat_seq[device_env_ids] = rot_mat_seq

        self.curr_target_pos[env_ids, :] = pos_seq[:, 0, :].to(self.device)
        self.curr_target_rot_mat[env_ids, :] = rot_mat_seq[:, 0, :].to(self.device)
        # update curriculum
        if self.pos_sigma_curriculum is not None:
            avg_pos_err = self.past_pos_err.mean().item()
            # find the first threshold that is greater than the average error
            for level, (threshold, sigma) in enumerate(
                self.pos_sigma_curriculum.items()
            ):
                if avg_pos_err < threshold:
                    self.pos_err_sigma = sigma
                    self.pos_sigma_curriculum_level = level
        if self.orn_sigma_curriculum is not None:
            avg_orn_err = self.past_orn_err.mean().item()
            # find the first threshold that is greater than the average error
            for level, (threshold, sigma) in enumerate(
                self.orn_sigma_curriculum.items()
            ):
                if avg_orn_err < threshold:
                    self.orn_err_sigma = sigma
                    self.orn_sigma_curriculum_level = level

        # update pose history
        self.link_pose_history[device_env_ids, :, :, :] = torch.eye(
            4, device=self.device
        )
        self.root_pose_history[device_env_ids, :, :, :] = torch.eye(
            4, device=self.device
        )

    def get_targets_at_times(
        self,
        times: torch.Tensor,
        sim_dt: float,
    ):
        episode_step = torch.clamp(
            # (torch.zeros_like(times) / sim_dt).long(),
            (times / sim_dt).long(),
            min=0,
            max=self.target_pos_seq.shape[1] - 1,
        )
        episode_step = torch.clamp(
            episode_step, min=0, max=self.target_pos_seq.shape[1] - 1
        ).to(self.storage_device)
        env_idx = torch.arange(0, self.num_envs)
        return (
            self.target_pos_seq[env_idx, episode_step].to(self.device),
            self.target_rot_mat_seq[env_idx, episode_step].to(self.device),
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        (
            self.curr_target_pos[:, :],
            self.curr_target_rot_mat[:, :],
        ) = self.get_targets_at_times(
            times=state.episode_time,
            sim_dt=state.sim_dt,
        )
        pos_err = self.get_pos_err(state=state)
        orn_err = self.get_orn_err(state=state)
        # moving average of the error
        smoothing = state.sim_dt * self.smoothing_dt_multiplier
        self.past_pos_err = (1 - smoothing) * self.past_pos_err + smoothing * pos_err
        self.past_orn_err = (1 - smoothing) * self.past_orn_err + smoothing * orn_err

        self.link_pose_history = torch.cat(
            [
                self.link_pose_history[:, 1:],
                self.get_link_pose(state=state).unsqueeze(1),
            ],
            dim=1,
        )
        self.root_pose_history = torch.cat(
            [
                self.root_pose_history[:, 1:],
                state.root_pose.clone().unsqueeze(1)[::2],
            ],
            dim=1,
        )
        self.steps += 1
        return {
            "pos_err": pos_err,
            "orn_err": orn_err,
            "smoothed_pos_err": self.past_pos_err.clone(),
            "smoothed_orn_err": self.past_orn_err.clone(),
            "pos_sigma_level": torch.ones_like(pos_err, device=self.device)
            * self.pos_sigma_curriculum_level,
            "orn_sigma_level": torch.ones_like(orn_err, device=self.device)
            * self.orn_sigma_curriculum_level,
            "pose_latency": torch.ones_like(pos_err, device=self.device)
            * self.get_latency_scheduler()
            * self.pose_latency,
        }

    def get_pos_err(self, state: EnvState) -> torch.Tensor:
        return torch.sum(
            torch.square(self.curr_target_pos - self.get_link_pos(state=state)),
            dim=1,
        ).sqrt()

    def get_orn_err(self, state: EnvState) -> torch.Tensor:
        link_rot_mat = self.get_link_rot_mat(state=state)
        # rotation from link to target
        rot_err_mat = self.curr_target_rot_mat @ link_rot_mat.transpose(1, 2)

        trace = torch.diagonal(rot_err_mat, dim1=-2, dim2=-1).sum(dim=-1)
        # to prevent numerical instability, clip the trace to [-1, 3]
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        return rotation_magnitude

    def get_target_pose(self, times: torch.Tensor, sim_dt: float):
        # returns the current target pose in the local frame of the robot
        target_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )

        pos, rot_mat = self.get_targets_at_times(times=times, sim_dt=sim_dt)
        target_pose[..., :3, 3] = pos
        target_pose[..., :3, :3] = rot_mat
        return target_pose

    def get_link_rot_mat(self, state: EnvState):
        return quaternion_to_matrix(
            state.rigid_body_xyzw_quat[:, self.link_index][:, [3, 0, 1, 2]]
        )

    def get_link_pos(self, state: EnvState):
        return state.rigid_body_pos[:, self.link_index]

    def get_link_pose(self, state: EnvState):
        # returns the current link pose in the local frame of the robot
        link_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )
        link_pose[..., :3, 3] = self.get_link_pos(state=state)
        # pt3d quaternion convention is wxyz
        link_pose[..., :3, :3] = self.get_link_rot_mat(state=state)
        return link_pose

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # compute reward using the current pose
        pos_reward = torch.exp(
            -(self.get_pos_err(state=state) ** 2) / self.pos_err_sigma
        )
        orn_reward = torch.exp(-self.get_orn_err(state=state) / self.orn_err_sigma)
        return {
            "pos": pos_reward * self.pos_reward_scale,
            "orn": orn_reward * self.orn_reward_scale,
            "pose": (pos_reward * orn_reward) * self.pose_reward_scale,
        }

    def get_latency_scheduler(self) -> float:
        return (
            min(
                max(
                    (self.steps - self.pose_latency_warmup_start)
                    / self.pose_latency_warmup_steps,
                    0.0,
                ),
                1.0,
            )
            if self.pose_latency_warmup_steps > 0
            else 1.0
        )

    def observe(self, state: EnvState) -> torch.Tensor:
        global_target_pose = torch.stack(
            [
                self.get_target_pose(
                    times=state.episode_time + t_offset,
                    sim_dt=state.sim_dt,
                )
                for t_offset in self.target_obs_times
            ],
            dim=1,
        )  # (num_envs, num_obs, 4, 4)
        # get the most outdated pose, to account for latency
        latency_idx = int(
            np.rint(self.get_latency_scheduler() * self.pose_latency_frames)
        )  # number of frames to wait
        if self.pose_latency_frame_variability is not None:
            latency_idx += int(
                torch.randint(
                    low=self.pose_latency_frame_variability[0],
                    high=self.pose_latency_frame_variability[1] + 1,
                    size=(1,),
                    device=self.storage_device,
                    generator=self.generator,
                )
                .cpu()
                .item()
            )
        # min is 1, since this index is negated to access last `latency_idx`th frame
        latency_idx = max(1, min(latency_idx, self.pose_latency_frames - 1))
        observation_link_pose = (
            self.root_pose_history[:, -latency_idx]
            if self.target_relative_to_base
            else self.link_pose_history[:, -latency_idx]
        ).clone()  # (num_envs, 4, 4), clone otherwise sim state will be modified
        if self.position_noise > 0 or self.euler_noise > 0:
            noise_transform = torch.zeros((self.num_envs, 4, 4), device=self.device)
            noise_transform[..., [0, 1, 2, 3], [0, 1, 2, 3]] = 1.0
            if self.position_noise > 0:
                noise_transform[..., :3, 3] = (
                    torch.randn((self.num_envs, 3), device=self.device)
                    * self.position_noise
                )
            if self.euler_noise > 0:
                euler_noise = (
                    torch.randn((self.num_envs, 3), device=self.device)
                    * self.euler_noise
                )
                noise_transform[..., :3, :3] = pt3d.euler_angles_to_matrix(
                    euler_noise, convention="XYZ"
                )
            observation_link_pose = noise_transform @ observation_link_pose
        local_target_pose = (
            torch.linalg.inv(observation_link_pose[:, None, :, :]) @ global_target_pose
        )
        if self.position_obs_encoding == "linear":
            pos_obs = (local_target_pose[..., :3, 3] * self.pos_obs_scale).view(
                self.num_envs, -1
            )
        elif self.position_obs_encoding == "log-direction":
            distance = (
                torch.linalg.norm(local_target_pose[..., :3, 3], dim=-1, keepdim=True)
                + 1e-8
            )
            direction = local_target_pose[..., :3, 3] / distance
            pos_obs = torch.cat(
                (
                    torch.log(distance * self.pos_obs_scale).reshape(self.num_envs, -1),
                    direction.reshape(
                        self.num_envs, -1
                    ),  # direction is already in normalized range
                ),
                dim=-1,
            )
        else:
            raise ValueError(
                f"Unknown position observation encoding: {self.position_obs_encoding!r}"
            )
        if self.pos_obs_clip is not None:
            pos_obs = torch.clamp(pos_obs, -self.pos_obs_clip, self.pos_obs_clip)
        orn_obs = (
            pt3d.matrix_to_rotation_6d(local_target_pose[..., :3, :3])
            * self.orn_obs_scale
        ).view(self.num_envs, -1)

        relative_pose_obs = torch.cat((pos_obs, orn_obs), dim=1)
        # NOTE after episode resetting, the first pose will be outdated
        # (this is a quirk of isaacgym, where state resets don't apply until the
        # next physics step), we will have to wait for `pose_latency` seconds
        # to get the first pose so just return special values for such cases
        waiting_for_pose_mask = (
            (observation_link_pose == torch.eye(4, device=self.device))
            .all(dim=-1)
            .all(dim=-1)
        )
        relative_pose_obs[waiting_for_pose_mask] = -1.0

        return relative_pose_obs

    def visualize(self, state: EnvState, viewer: gymapi.Viewer, vis_env_ids: List[int]):
        pos_err = self.get_pos_err(state=state)
        cm = plt.get_cmap("inferno")
        target_quats = (
            matrix_to_quaternion(self.curr_target_rot_mat)
            .cpu()
            .numpy()[:, [1, 2, 3, 0]]
        )
        link_pose = self.get_link_pose(state=state)
        curr_quats = (
            matrix_to_quaternion(link_pose[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
        )
        link_pose = link_pose.cpu().numpy()
        for i in vis_env_ids:
            env = self.gym.get_env(self.sim, i)
            rgb = list(cm(max(1 - pos_err[i].item(), 0)))
            target_pose = gymapi.Transform(
                p=gymapi.Vec3(*self.curr_target_pos[i].cpu().numpy().tolist()),
                r=gymapi.Quat(*target_quats[i].tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                target_pose,
            )
            curr_pose = gymapi.Transform(
                p=gymapi.Vec3(*link_pose[i, :3, 3].copy().tolist()),
                r=gymapi.Quat(*curr_quats[i].copy().tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                curr_pose,
            )
            vertices = np.array(
                [
                    *self.curr_target_pos[i].cpu().numpy().tolist(),
                    *link_pose[i, :3, 3].tolist(),
                ]
            ).astype(np.float32)
            colors = np.array(rgb).astype(np.float32)
            self.gym.add_lines(
                viewer,
                env,
                1,  # num lines
                vertices,  # vertices
                colors,  # color
            )





class CoordinatedBaseArmTask(Task):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        
        # 基座控制参数 (来自Link6DVelocity)
        base_link_name: str,
        base_lin_vel_range: List[Tuple[float, float]],
        base_ang_vel_range: List[Tuple[float, float]],
        z_height_range: Tuple[float, float],
        roll_range: Tuple[float, float],
        pitch_range: Tuple[float, float],
        base_lin_vel_obs_scale: float,
        base_ang_vel_obs_scale: float,
        base_resampling_time: float,
        min_target_base_lin_vel: float,
        base_tracking_sigma: float,
        base_lin_vel_reward_scale: float,
        base_ang_vel_reward_scale: float,
        z_height_reward_scale: float,
        z_height_sigma : float,
        gravity_reward_scale: float,
        gravity_sigma : float,
        feet_air_time_reward_scale: float,
        feet_sensor_indices: List[int],
        
        # 末端轨迹参数 (来自ReachingLinkTask)
        arm_link_name: str,
        arm_pos_obs_scale: float,
        arm_orn_obs_scale: float,
        arm_pos_err_sigma: float,
        arm_orn_err_sigma: float,
        arm_pos_reward_scale: float,
        arm_orn_reward_scale: float,
        arm_pose_reward_scale: float,
        arm_target_obs_times: List[float],
        sequence_sampler: SequenceSampler,
        arm_pose_latency: float,
        arm_target_relative_to_base: bool = False,
        arm_position_obs_encoding: str = "linear",
        arm_pos_obs_clip: Optional[float] = None,
        arm_pos_sigma_curriculum: Optional[Dict[float, float]] = None,
        arm_orn_sigma_curriculum: Optional[Dict[float, float]] = None,
        init_arm_pos_curriculum_level: int = 0,
        init_arm_orn_curriculum_level: int = 0,
        smoothing_dt_multiplier: float = 4.0,
        arm_pose_latency_warmup_steps : int = 0,
        arm_pose_latency_warmup_start : int = 0,
        
        # 协调控制参数
        coordination_reward_scale: float = 1.0,
        coordination_sigma: float = 0.5,
        storage_device: str = "cpu",
    ):
        super().__init__(gym=gym, sim=sim, device=device, generator=generator)
        self.storage_device = storage_device
        
        # === 基座控制部分初始化 ===
        self.base_link_name = base_link_name
        env = gym.get_env(sim, 0)
        self.base_link_idx = gym.find_actor_rigid_body_handle(
            env, gym.get_actor_handle(env, 0), self.base_link_name
        )
        assert self.base_link_idx != -1, f"Base link {self.base_link_name} not found"
        
        # 基座指令参数
        self.base_lin_vel_range = torch.tensor(base_lin_vel_range).to(device)
        self.base_ang_vel_range = torch.tensor(base_ang_vel_range).to(device)
        self.z_height_range = torch.tensor(z_height_range).to(device)
        self.roll_range = torch.tensor(roll_range).to(device)
        self.pitch_range = torch.tensor(pitch_range).to(device)
        
        # 基座目标状态
        self.target_base_lin_vel = torch.zeros((self.num_envs, 3), device=device)
        self.target_base_ang_vel = torch.zeros((self.num_envs, 3), device=device)
        self.target_z_height = torch.zeros((self.num_envs,), device=device)
        self.target_local_gravity = torch.zeros((self.num_envs, 3), device=device)
        
        # 基座观测和奖励参数
        self.base_lin_vel_obs_scale = base_lin_vel_obs_scale
        self.base_ang_vel_obs_scale = base_ang_vel_obs_scale
        self.base_resampling_time = base_resampling_time
        self.min_target_base_lin_vel = min_target_base_lin_vel
        self.base_tracking_sigma = base_tracking_sigma
        self.base_lin_vel_reward_scale = base_lin_vel_reward_scale
        self.base_ang_vel_reward_scale = base_ang_vel_reward_scale
        self.z_height_reward_scale = z_height_reward_scale
        self.z_height_sigma = z_height_sigma
        self.gravity_sigma = gravity_sigma
        self.gravity_reward_scale = gravity_reward_scale
        self.feet_air_time_reward_scale = feet_air_time_reward_scale
        
        # 脚部接触传感器
        self.feet_sensor_indices = torch.tensor(
            feet_sensor_indices, dtype=torch.long, device=device
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, len(feet_sensor_indices), device=device
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(feet_sensor_indices), dtype=torch.bool, device=device
        )
        
        # === 末端轨迹控制部分初始化 ===
        self.arm_link_name = arm_link_name
        self.arm_link_idx = gym.find_actor_rigid_body_handle(
            env, gym.get_actor_handle(env, 0), self.arm_link_name
        )
        assert self.arm_link_idx != -1, f"Arm link {self.arm_link_name} not found"
        
        # 末端轨迹参数
        self.arm_pos_obs_scale = arm_pos_obs_scale
        self.arm_orn_obs_scale = arm_orn_obs_scale
        self.arm_pos_err_sigma = arm_pos_err_sigma
        self.arm_orn_err_sigma = arm_orn_err_sigma
        self.arm_pos_reward_scale = arm_pos_reward_scale
        self.arm_orn_reward_scale = arm_orn_reward_scale
        self.arm_pose_reward_scale = arm_pose_reward_scale
        self.arm_target_obs_times = arm_target_obs_times
        self.arm_sequence_sampler = sequence_sampler
        self.arm_pose_latency = arm_pose_latency
        self.arm_target_relative_to_base = arm_target_relative_to_base
        self.arm_position_obs_encoding = arm_position_obs_encoding
        self.arm_pos_obs_clip = arm_pos_obs_clip
        self.arm_pose_latency_warmup_steps = arm_pose_latency_warmup_steps
        self.arm_pose_latency_warmup_start = arm_pose_latency_warmup_start
        
        # 末端轨迹目标
        self.arm_target_pos_seq = torch.zeros(
            (self.num_envs, self.arm_sequence_sampler.episode_length, 3),
            device=self.storage_device,
        )
        self.arm_target_rot_mat_seq = torch.zeros(
            (self.num_envs, self.arm_sequence_sampler.episode_length, 3, 3),
            device=self.storage_device,
        )
        self.arm_curr_target_pos = torch.zeros((self.num_envs, 3), device=device)
        self.arm_curr_target_rot_mat = torch.zeros((self.num_envs, 3, 3), device=device)
        
        # 历史状态记录
        self.arm_pose_latency_frames = int(np.rint(arm_pose_latency / self.arm_sequence_sampler.dt)) + 1
        self.arm_pose_history = torch.zeros(
            (self.num_envs, self.arm_pose_latency_frames, 4, 4),
            device=device,
        )
        self.root_pose_history = torch.zeros(
            (self.num_envs, self.arm_pose_latency_frames, 4, 4),
            device=device,
        )
        
        # === 协调控制参数 ===
        self.coordination_reward_scale = coordination_reward_scale
        self.coordination_sigma = coordination_sigma
        self.coordination_factor = torch.ones(self.num_envs, device=device) * 0.5  # 平衡权重
        self.steps = 0  # 用于延迟调度器
        
        # 初始化位姿历史
        self.arm_pose_history[..., :, :] = torch.eye(4, device=device)
        self.root_pose_history[..., :, :] = torch.eye(4, device=device)

        
        # === 课程学习参数 ===
        self.arm_pos_sigma_curriculum = arm_pos_sigma_curriculum
        self.arm_orn_sigma_curriculum = arm_orn_sigma_curriculum
        self.smoothing_dt_multiplier = smoothing_dt_multiplier
        
        # 课程学习等级
        self.arm_pos_sigma_curriculum_level = init_arm_pos_curriculum_level
        self.arm_orn_sigma_curriculum_level = init_arm_orn_curriculum_level
        
        # 当前使用的sigma值
        self.arm_pos_err_sigma = self.arm_pos_sigma_curriculum[list(self.arm_pos_sigma_curriculum.keys())[self.arm_pos_sigma_curriculum_level]]
        self.arm_orn_err_sigma = self.arm_orn_sigma_curriculum[list(self.arm_orn_sigma_curriculum.keys())[self.arm_orn_sigma_curriculum_level]]
        
        # 平滑误差变量
        self.past_arm_pos_err = torch.zeros(self.num_envs, device=device)
        self.past_arm_orn_err = torch.zeros(self.num_envs, device=device)
        
        # 确保课程学习参数排序正确
        if self.arm_pos_sigma_curriculum is not None:
            self.arm_pos_sigma_curriculum = dict(sorted(
                self.arm_pos_sigma_curriculum.items(), 
                key=lambda x: x[0], 
                reverse=True
            ))
        
        if self.arm_orn_sigma_curriculum is not None:
            self.arm_orn_sigma_curriculum = dict(sorted(
                self.arm_orn_sigma_curriculum.items(), 
                key=lambda x: x[0], 
                reverse=True
            ))
    
    def resample_base_commands(self, env_ids: torch.Tensor):
        """生成新的基座控制指令"""
        if len(env_ids) == 0:
            return
        
        # 线速度指令
        self.target_base_lin_vel[env_ids] = torch_rand_float(
            self.base_lin_vel_range[:, 0],
            self.base_lin_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)
        
        # 角速度指令
        self.target_base_ang_vel[env_ids] = torch_rand_float(
            self.base_ang_vel_range[:, 0],
            self.base_ang_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)
        
        # 高度指令
        self.target_z_height[env_ids] = torch_rand_float(
            self.z_height_range[0],
            self.z_height_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        
        # 姿态指令 (通过重力向量)
        roll = torch_rand_float(
            self.roll_range[0],
            self.roll_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        pitch = torch_rand_float(
            self.pitch_range[0],
            self.pitch_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        self.target_local_gravity[env_ids, 0] = torch.tan(roll)
        self.target_local_gravity[env_ids, 1] = torch.tan(pitch)
        self.target_local_gravity[env_ids, 2] = -1.0
        self.target_local_gravity[env_ids] /= self.target_local_gravity[env_ids].norm(dim=1, keepdim=True)
        
        # 过滤小指令值
        self.target_base_lin_vel[env_ids] *= (
            torch.norm(self.target_base_lin_vel[env_ids], dim=1) > self.min_target_base_lin_vel
        ).unsqueeze(1)
        
        # 随机化协调因子 (0.3-0.7)
        self.coordination_factor[env_ids] = torch_rand_float(
            0.3, 0.7, (len(env_ids),), device=self.device, generator=self.generator
        )
    
    def reset_idx(self, env_ids: torch.Tensor):
        """重置特定环境的指令和历史状态"""
        if len(env_ids) == 0:
            return
        
        # 重置基座指令
        self.resample_base_commands(env_ids)
        self.feet_air_time[env_ids] = 0.0
        
        # 重置末端轨迹
        env_origins = torch.stack(
            [torch.tensor([self.gym.get_env_origin(self.gym.get_env(self.sim, i)).x,
                           self.gym.get_env_origin(self.gym.get_env(self.sim, i)).y,
                           self.gym.get_env_origin(self.gym.get_env(self.sim, i)).z])
             for i in env_ids.cpu().numpy()],
            dim=0
        ).to(self.storage_device)
        
        # 采样新的末端轨迹序列
        pos_seq, rot_mat_seq = self.arm_sequence_sampler.sample(
            seed=int(
                torch.randint(0, 
                              2**32-1, 
                              (1,), 
                              device=self.device, 
                              generator=self.generator).cpu().item()),
            batch_size=len(env_ids)
        )
        self.arm_target_pos_seq[env_ids] = pos_seq.to(self.storage_device) + env_origins.unsqueeze(1)
        self.arm_target_rot_mat_seq[env_ids] = rot_mat_seq.to(self.storage_device)
        
        # 设置当前目标
        self.arm_curr_target_pos[env_ids] = pos_seq[:, 0].to(self.device)
        self.arm_curr_target_rot_mat[env_ids] = rot_mat_seq[:, 0].to(self.device)
        
        # 重置历史状态
        self.arm_pose_history[env_ids] = torch.eye(4, device=self.device)
        self.root_pose_history[env_ids] = torch.eye(4, device=self.device)


        # === 课程学习等级更新 ===
        # 基于平均误差更新位置课程等级
        if self.arm_pos_sigma_curriculum is not None:
            avg_pos_err = self.past_arm_pos_err.mean().item()
            for level, (threshold, sigma) in enumerate(self.arm_pos_sigma_curriculum.items()):
                if avg_pos_err < threshold:
                    self.arm_pos_err_sigma = sigma
                    self.arm_pos_sigma_curriculum_level = level
                    break
        
        # 基于平均误差更新姿态课程等级
        if self.arm_orn_sigma_curriculum is not None:
            avg_orn_err = self.past_arm_orn_err.mean().item()
            for level, (threshold, sigma) in enumerate(self.arm_orn_sigma_curriculum.items()):
                if avg_orn_err < threshold:
                    self.arm_orn_err_sigma = sigma
                    self.arm_orn_sigma_curriculum_level = level
                    break
                

    
    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        """每步更新逻辑，完整保留末端轨迹跟踪功能"""
        # === 基座指令重置 ===
        base_env_ids = check_should_reset(
            state.episode_time, state.sim_dt, self.base_resampling_time
        ).nonzero(as_tuple=False).flatten()
        self.resample_base_commands(base_env_ids)
        
        # === 末端轨迹更新（完整保留ReachingLinkTask功能）===
        # 更新当前末端目标
        (self.arm_curr_target_pos[:],
         self.arm_curr_target_rot_mat[:]) = self.get_arm_targets_at_times(
             times=state.episode_time,
             sim_dt=state.sim_dt
         )
        
        # 计算末端误差
        arm_pos_err = self.get_arm_pos_err(state)
        arm_orn_err = self.get_arm_orn_err(state)
        
        # 平滑误差（重要功能）
        smoothing = state.sim_dt * self.smoothing_dt_multiplier
        self.past_arm_pos_err = (1 - smoothing) * self.past_arm_pos_err + smoothing * arm_pos_err
        self.past_arm_orn_err = (1 - smoothing) * self.past_arm_orn_err + smoothing * arm_orn_err
        
        # === 历史状态更新 ===
        # 末端位姿历史
        self.arm_pose_history = torch.cat([
            self.arm_pose_history[:, 1:],
            self.get_arm_pose(state).unsqueeze(1)
        ], dim=1)
        
        # 基座位姿历史
        self.root_pose_history = torch.cat([
            self.root_pose_history[:, 1:],
            state.root_pose.clone().unsqueeze(1)
        ], dim=1)
        
        # # === 脚部接触检测 ===
        # contact = state.force_sensor_tensor[:, self.feet_sensor_indices, 2].abs() > 1.0
        # contact_filt = torch.logical_or(contact, self.last_contacts)
        # self.last_contacts = contact
        # first_contact = (self.feet_air_time > 0.0) * contact_filt
        # self.feet_air_time += state.sim_dt
        # rew_airTime = torch.sum(
        #     (self.feet_air_time - 0.5) * first_contact, dim=1
        # )
        # rew_airTime *= (
        #     torch.norm(self.target_base_lin_vel[:, :2], dim=1) > 0.1
        # )
        # self.feet_air_time *= ~contact_filt
        
        # 增加步数计数器
        self.steps += 1
        

        # === 计算末端误差和平滑 ===
        arm_pos_err = self.get_arm_pos_err(state)
        arm_orn_err = self.get_arm_orn_err(state)
        
        # 平滑误差（重要功能）
        smoothing = state.sim_dt * self.smoothing_dt_multiplier
        self.past_arm_pos_err = (1 - smoothing) * self.past_arm_pos_err + smoothing * arm_pos_err
        self.past_arm_orn_err = (1 - smoothing) * self.past_arm_orn_err + smoothing * arm_orn_err

        # === 返回完整指标 ===
        return {
            # 基座相关指标
            "base_lin_vel_err": self.get_base_lin_vel_err(state),
            "base_ang_vel_err": self.get_base_ang_vel_err(state),
            "z_height_err": self.get_z_height_err(state),
            "gravity_err": self.get_gravity_err(state),
            
            # 末端相关指标（完整保留）
            "arm_pos_err": arm_pos_err,
            "arm_orn_err": arm_orn_err,
            "smoothed_arm_pos_err": self.past_arm_pos_err.clone(),
            "smoothed_arm_orn_err": self.past_arm_orn_err.clone(),
            "arm_pos_sigma_level": torch.ones_like(arm_pos_err, device=self.device)
                * self.arm_pos_sigma_curriculum_level,
            "arm_orn_sigma_level": torch.ones_like(arm_orn_err, device=self.device)
                * self.arm_orn_sigma_curriculum_level,
            "arm_pose_latency": torch.ones_like(arm_pos_err, device=self.device)
            * self.get_arm_latency_scheduler()
            * self.arm_pose_latency,
        }
    
    # === 新增方法：末端目标获取 ===
    def get_arm_targets_at_times(
        self,
        times: torch.Tensor,
        sim_dt: float,
    ):
        """完全保留ReachingLinkTask的目标获取逻辑"""
        episode_step = torch.clamp(
            (times / sim_dt).long(),
            min=0,
            max=self.arm_sequence_sampler.episode_length - 1,
        )
        episode_step = torch.clamp(
            episode_step, min=0, max=self.arm_target_pos_seq.shape[1] - 1
        ).to(self.storage_device)
        env_idx = torch.arange(0, self.num_envs, device=self.storage_device)
        return (
            self.arm_target_pos_seq[env_idx, episode_step].to(self.device),
            self.arm_target_rot_mat_seq[env_idx, episode_step].to(self.device),
        )
    
    # === 新增方法：末端延迟调度器 ===
    def get_arm_latency_scheduler(self) -> float:
        """完全保留ReachingLinkTask的延迟调度逻辑"""
        return (
            min(
                max(
                    (self.steps - self.arm_pose_latency_warmup_start)
                    / max(1, self.arm_pose_latency_warmup_steps),
                    0.0,
                ),
                1.0,
            )
            if self.arm_pose_latency_warmup_steps > 0
            else 1.0
        )
    
    # ===== 基座控制相关方法 =====
    def get_base_local_lin_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.base_link_idx],
            state.rigid_body_lin_vel[:, self.base_link_idx],
        )
    
    def get_base_local_ang_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.base_link_idx],
            state.rigid_body_ang_vel[:, self.base_link_idx],
        )
    
    def get_base_lin_vel_err(self, state: EnvState):
        return (self.target_base_lin_vel - self.get_base_local_lin_vel(state)).norm(dim=-1)
    
    def get_base_ang_vel_err(self, state: EnvState):
        return (self.target_base_ang_vel - self.get_base_local_ang_vel(state)).norm(dim=-1)
    
    def get_z_height_err(self, state: EnvState):
        base_height = torch.mean(
            state.rigid_body_pos[:, self.base_link_idx, [2]] - state.measured_terrain_heights,
            dim=1
        )
        return torch.square(self.target_z_height - base_height)
    
    def get_gravity_err(self, state: EnvState):
        base_local_gravity = quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.base_link_idx],
            state.gravity / torch.linalg.norm(state.gravity, dim=1, keepdim=True)
        )
        return torch.square(self.target_local_gravity - base_local_gravity).sum(dim=1)
    
    # ===== 末端轨迹控制相关方法 =====
    def get_arm_rot_mat(self, state: EnvState):
        return quaternion_to_matrix(
            state.rigid_body_xyzw_quat[:, self.arm_link_idx][:, [3, 0, 1, 2]]
        )
    
    def get_arm_pos(self, state: EnvState):
        return state.rigid_body_pos[:, self.arm_link_idx]
    
    def get_arm_pose(self, state: EnvState):
        pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        pose[..., :3, 3] = self.get_arm_pos(state)
        pose[..., :3, :3] = self.get_arm_rot_mat(state)
        return pose
    
    def get_arm_pos_err(self, state: EnvState):
        return (self.arm_curr_target_pos - self.get_arm_pos(state)).norm(dim=1)
    
    def get_arm_orn_err(self, state: EnvState):
        arm_rot_mat = self.get_arm_rot_mat(state)
        rot_err_mat = self.arm_curr_target_rot_mat @ arm_rot_mat.transpose(1, 2)
        trace = rot_err_mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace = torch.clamp(trace, -1+1e-8, 3-1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        return rotation_magnitude
    
    def get_latency_scheduler(self) -> float:
        """延迟调度器，用于逐步引入延迟"""
        return min(
            max(
                (self.steps - 0) / max(1, self.arm_pose_latency_frames * 10),
                0.0,
            ),
            1.0,
        ) if self.arm_pose_latency_frames > 1 else 1.0
    
    # ===== 观测生成 =====
    def observe(self, state: EnvState) -> torch.Tensor:
        """构建包含基座指令和末端轨迹的观测向量"""
        # target_lin_vel target_ang_vel target_z_height target_local_gravity relative_pose_obs
        # 基座相关观测
        # base_obs = torch.cat([
        #     self.target_base_lin_vel * self.base_lin_vel_obs_scale,
        #     self.target_base_ang_vel * self.base_ang_vel_obs_scale,
        #     self.target_z_height.unsqueeze(-1),
        #     self.target_local_gravity,
        # ], dim=-1)
        
        # 末端轨迹相关观测
        arm_obs = self.get_arm_trajectory_obs(state)
        
        # # 相对状态观测
        # relative_arm_pos = self.get_arm_pos(state) - state.rigid_body_pos[:, self.base_link_idx]
        # relative_arm_pos = quat_rotate_inverse(
        #     state.rigid_body_xyzw_quat[:, self.base_link_idx],
        #     relative_arm_pos
        # )
        
        # 协调因子
        coordination_factor = self.coordination_factor.unsqueeze(-1)
        
        # 合并所有观测
        return torch.cat([
            self.target_base_lin_vel * self.base_lin_vel_obs_scale,
            self.target_base_ang_vel * self.base_ang_vel_obs_scale,
            self.target_z_height.unsqueeze(-1),
            self.target_local_gravity,
            arm_obs,
            coordination_factor,
        ], dim=-1)
    

    def get_arm_trajectory_obs(self, state: EnvState) -> torch.Tensor:
        """生成末端轨迹的观测（严格匹配预期形状）"""
        # 获取多个时间点的目标位姿 [2048, 4, 4, 4]
        global_target_pose = torch.stack([
            self.get_arm_target_pose(state.episode_time + t, state.sim_dt)
            for t in self.arm_target_obs_times
        ], dim=1)  # shape: [num_envs, num_obs_times=4, 4, 4]
        
        # print("global_target_pose", global_target_pose.shape)

        # 获取延迟补偿的参考位姿 [2048, 4, 4]
        latency_idx = max(1, min(
            int(np.rint(self.get_latency_scheduler() * self.arm_pose_latency_frames)),
            self.arm_pose_latency_frames - 1
        ))
        ref_pose = (self.root_pose_history if self.arm_target_relative_to_base 
                   else self.arm_pose_history)[:, -latency_idx]  # [2048, 4, 4]

        # print("latency_idx", latency_idx)
        # print("ref_pose", ref_pose.shape)

        # 转换到局部坐标系 [2048, 4, 4, 4]
        local_target_pose = torch.linalg.inv(ref_pose.unsqueeze(1)) @ global_target_pose

        # print("local_target_pose", local_target_pose.shape)


        # 位置观测处理 [2048, 12]
        if self.arm_position_obs_encoding == "linear":
            pos_obs = (local_target_pose[..., :3, 3] * self.arm_pos_obs_scale)  # [2048,4,3]
            pos_obs = pos_obs.reshape(pos_obs.shape[0], -1)  # Flatten -> [2048,12]
        else:  # log-direction
            distance = torch.norm(local_target_pose[..., :3, 3], dim=-1, keepdim=True) + 1e-8
            direction = local_target_pose[..., :3, 3] / distance
            pos_obs = torch.cat([
                torch.log(distance * self.arm_pos_obs_scale),
                direction
            ], dim=-1).reshape(distance.shape[0], -1)  # [2048,12]

        # 姿态观测处理 [2048, 24]
        orn_obs = pt3d.matrix_to_rotation_6d(local_target_pose[..., :3, :3])  # [2048,4,6]
        orn_obs = (orn_obs * self.arm_orn_obs_scale).reshape(orn_obs.shape[0], -1)  # [2048,24]

        # print("pos_obs",pos_obs.shape)
        # print("orn_obs",orn_obs.shape)


        # 合并观测 [2048, 36]
        relative_pose_obs = torch.cat([pos_obs, orn_obs], dim=1)

        # 处理重置后的无效观测
        waiting_for_pose_mask = (
            (ref_pose == torch.eye(4, device=self.device))
            .all(dim=-1)
            .all(dim=-1)
        )
        relative_pose_obs[waiting_for_pose_mask] = -1.0

        return relative_pose_obs
    

    # def get_arm_trajectory_obs(self, state: EnvState):
    #     """生成末端轨迹的观测（基于ReachingLinkTask的逻辑）"""
    #     # 获取多个时间点的目标位姿
    #     global_target_pose = torch.stack([
    #         self.get_arm_target_pose(state.episode_time + t, state.sim_dt)
    #         for t in self.arm_target_obs_times
    #     ], dim=1)
        
    #     print("global_target_pose", global_target_pose.shape)

    #     # 使用历史位姿作为参考（考虑延迟）
    #     latency_idx = int(np.rint(self.get_latency_scheduler() * self.arm_pose_latency_frames))
    #     latency_idx = max(1, min(latency_idx, self.arm_pose_latency_frames - 1))
    #     ref_pose = self.root_pose_history[:, -latency_idx] if self.arm_target_relative_to_base else self.arm_pose_history[:, -latency_idx]
        
    #     print("latency_idx", latency_idx)
    #     print("ref_pose", ref_pose.shape)


    #     # 转换到局部坐标系
    #     local_target_pose = torch.linalg.inv(ref_pose.unsqueeze(1)) @ global_target_pose
        
    #     print("local_target_pose", local_target_pose.shape)


    #     # 位置观测
    #     if self.arm_position_obs_encoding == "linear":
    #         pos_obs = local_target_pose[..., :3, 3] * self.arm_pos_obs_scale
    #     elif self.arm_position_obs_encoding == "log-direction":
    #         distance = (
    #             torch.linalg.norm(local_target_pose[..., :3, 3], dim=-1, keepdim=True)
    #             + 1e-8
    #         )
    #         direction = local_target_pose[..., :3, 3] / distance
    #         pos_obs = torch.cat(
    #             (
    #                 torch.log(distance * self.arm_pos_obs_scale),
    #                 direction,
    #             ),
    #             dim=-1,
    #         )
    #     else:
    #         raise ValueError(f"Unknown position encoding: {self.arm_position_obs_encoding}")
        
    #     # 姿态观测（使用6D旋转表示）
    #     orn_obs = pt3d.matrix_to_rotation_6d(local_target_pose[..., :3, :3]) * self.arm_orn_obs_scale
    #     print("pos_obs",pos_obs.shape)
    #     print("orn_obs",orn_obs.shape)


    #     relative_pose_obs = torch.cat((pos_obs, orn_obs), dim=1)

    #     waiting_for_pose_mask = (
    #         (ref_pose == torch.eye(4, device=self.device))
    #         .all(dim=-1)
    #         .all(dim=-1)
    #     )
    #     relative_pose_obs[waiting_for_pose_mask] = -1.0

    #     return relative_pose_obs
    
    def get_arm_target_pose(self, time: torch.Tensor, sim_dt: float):
        """获取特定时间的末端目标位姿"""
        step_idx = (time / sim_dt).long().clamp(0, self.arm_sequence_sampler.episode_length-1)
        step_idx = step_idx.to(self.storage_device)
        env_idx = torch.arange(self.num_envs, device=self.storage_device)
        
        pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        pose[..., :3, 3] = self.arm_target_pos_seq[env_idx, step_idx].to(self.device)
        pose[..., :3, :3] = self.arm_target_rot_mat_seq[env_idx, step_idx].to(self.device)
        return pose
    
    # ===== 奖励计算 =====
    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        """计算综合奖励，包括基座跟踪、末端跟踪和协调性"""
        # 基座相关奖励
        base_lin_vel_reward = torch.exp(
            -self.get_base_lin_vel_err(state) ** 2 / self.base_tracking_sigma
        )
        base_ang_vel_reward = torch.exp(
            -self.get_base_ang_vel_err(state) ** 2 / self.base_tracking_sigma
        )
        z_height_reward = torch.exp(-self.get_z_height_err(state) / self.z_height_sigma)
        gravity_reward = torch.exp(-self.get_gravity_err(state) / self.gravity_sigma)
        
        # === 脚部接触检测 ===
        contact = state.force_sensor_tensor[:, self.feet_sensor_indices, 2].abs() > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += state.sim_dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )
        rew_airTime *= (
            torch.norm(self.target_base_lin_vel[:, :2], dim=1) > 0.1
        )
        self.feet_air_time *= ~contact_filt

        # 合并基座奖励
        base_reward = (
            base_lin_vel_reward * self.base_lin_vel_reward_scale +
            base_ang_vel_reward * self.base_ang_vel_reward_scale +
            z_height_reward * self.z_height_reward_scale +
            gravity_reward * self.gravity_reward_scale +
            rew_airTime * self.feet_air_time_reward_scale
        )
        
        # 末端相关奖励
        arm_pos_reward = torch.exp(
            -self.get_arm_pos_err(state)**2 / self.arm_pos_err_sigma
        )
        arm_orn_reward = torch.exp(
            -self.get_arm_orn_err(state) / self.arm_orn_err_sigma
        )
        arm_reward = (
            arm_pos_reward * self.arm_pos_reward_scale +
            arm_orn_reward * self.arm_orn_reward_scale +
            (arm_pos_reward * arm_orn_reward) * self.arm_pose_reward_scale
        )
        
        # 协调性奖励
        arm_vel = state.rigid_body_lin_vel[:, self.arm_link_idx]
        base_vel = state.rigid_body_lin_vel[:, self.base_link_idx]
        relative_vel = arm_vel - base_vel
        
        # 协调性1: 末端相对于基座的速度稳定性
        coordination_reward1 = torch.exp(-relative_vel.norm(dim=1) / self.coordination_sigma)
        
        # 协调性2: 末端运动方向与基座运动方向的对齐
        base_vel_norm = torch.norm(base_vel, dim=1)
        valid_base_motion = base_vel_norm > 0.1
        alignment = torch.zeros_like(coordination_reward1)
        
        if torch.any(valid_base_motion):
            arm_vel_norm = torch.norm(arm_vel[valid_base_motion], dim=1, keepdim=True)
            arm_vel_norm = torch.where(arm_vel_norm > 1e-6, arm_vel_norm, torch.ones_like(arm_vel_norm))
            cos_sim = torch.sum(
                arm_vel[valid_base_motion] * base_vel[valid_base_motion], 
                dim=1
            ) / (arm_vel_norm.squeeze() * base_vel_norm[valid_base_motion])
            alignment[valid_base_motion] = torch.clamp(cos_sim, 0, 1)
        
        coordination_reward2 = alignment
        coordination_reward = (coordination_reward1 + coordination_reward2) / 2
        
        # 总奖励组合
        total_reward = (
            (1 - self.coordination_factor) * base_reward +
            self.coordination_factor * arm_reward +
            coordination_reward * self.coordination_reward_scale
        )
        
        return {
            # 基座加权后奖励
            "base_lin_vel": base_lin_vel_reward * self.base_lin_vel_reward_scale * (1 - self.coordination_factor),
            "base_ang_vel": base_ang_vel_reward * self.base_ang_vel_reward_scale * (1 - self.coordination_factor),
            "z_height": z_height_reward * self.z_height_reward_scale * (1 - self.coordination_factor),
            "gravity": gravity_reward * self.gravity_reward_scale * (1 - self.coordination_factor),
            "feet_air": rew_airTime * self.feet_air_time_reward_scale * (1 - self.coordination_factor),
            # 末端加权后奖励
            "arm_pos": arm_pos_reward * self.arm_pos_reward_scale * self.coordination_factor,
            "arm_orn": arm_orn_reward * self.arm_orn_reward_scale * self.coordination_factor,
            "arm_pose": (arm_pos_reward * arm_orn_reward) * self.arm_pose_reward_scale * self.coordination_factor,
            # 协调性奖励
            "coordination1": coordination_reward1 * self.coordination_reward_scale,
            "coordination2": coordination_reward2 * self.coordination_reward_scale,
        }
    
    
    def visualize(self, state: EnvState, viewer: gymapi.Viewer, vis_env_ids: List[int]):
        # 获取末端误差（用于颜色映射）
        arm_pos_err = self.get_arm_pos_err(state)
        cm = plt.get_cmap("inferno")
        
        # 准备末端目标姿态数据
        target_quats = (
            matrix_to_quaternion(self.arm_curr_target_rot_mat)
            .cpu()
            .numpy()[:, [1, 2, 3, 0]]  # 转换为wxyz顺序
        )
        
        # 准备当前末端姿态数据
        arm_pose = self.get_arm_pose(state)
        curr_arm_quats = (
            matrix_to_quaternion(arm_pose[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
        )
        arm_pose = arm_pose.cpu().numpy()
        
        # 准备基座姿态数据
        base_pos = state.rigid_body_pos[:, self.base_link_idx].cpu().numpy()
        base_quat = state.rigid_body_xyzw_quat[:, self.base_link_idx].cpu().numpy()
        
        for i in vis_env_ids:
            env = self.gym.get_env(self.sim, i)
            
            # 颜色映射：根据末端位置误差从红色（差）到黄色（好）
            rgb = list(cm(max(1 - arm_pos_err[i].item()/0.3, 0)))  # 误差>0.3显示红色
            
            # === 绘制基座 ===
            base_transform = gymapi.Transform(
                p=gymapi.Vec3(*base_pos[i].tolist()),
                r=gymapi.Quat(*base_quat[i].tolist())
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.3),  # 基座坐标系稍大
                self.gym,
                viewer,
                env,
                base_transform,
            )
            
            # === 绘制末端目标 ===
            target_pose = gymapi.Transform(
                p=gymapi.Vec3(*self.arm_curr_target_pos[i].cpu().numpy().tolist()),
                r=gymapi.Quat(*target_quats[i].tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                target_pose,
            )
            
            # === 绘制当前末端 ===
            curr_pose = gymapi.Transform(
                p=gymapi.Vec3(*arm_pose[i, :3, 3].tolist()),
                r=gymapi.Quat(*curr_arm_quats[i].tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                curr_pose,
            )
            
            # === 绘制目标-当前连线 ===
            vertices = np.array([
                *self.arm_curr_target_pos[i].cpu().numpy().tolist(),  # 目标位置
                *arm_pose[i, :3, 3].tolist()  # 当前位置
            ], dtype=np.float32)
            
            # 根据误差大小设置连线颜色（误差大=红，误差小=黄）
            self.gym.add_lines(
                viewer,
                env,
                1,  # num lines
                vertices,
                np.array(rgb, dtype=np.float32)  # 动态颜色
            )
            
            # === 绘制基座到末端的连线 ===
            base_to_arm_vertices = np.array([
                *base_pos[i].tolist(),  # 基座位置
                *arm_pose[i, :3, 3].tolist()  # 末端位置
            ], dtype=np.float32)
            self.gym.add_lines(
                viewer,
                env,
                1,
                base_to_arm_vertices,
                np.array([0.5, 0.5, 0.5], dtype=np.float32)  # 灰色
            )
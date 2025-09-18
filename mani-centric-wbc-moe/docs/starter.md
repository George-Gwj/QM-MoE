# Starter

## Set Up

This repository contains multiple submodules.
To clone all of them
```sh
git clone --recurse-submodules git@github.com:real-stanford/umi-on-legs-compose.git
```

Create the conda environment. 
I *highly* recommend using [Mamba](https://mamba.readthedocs.io/en/latest/installation.html), which is [faster](https://blog.hpc.qmul.ac.uk/mamba.html#:~:text=mamba%20is%20a%20re%2Dimplementation,Red%20Hat%2C%20Fedora%20and%20OpenSUSE), but if you insist on Conda, then replace `mamba` commands below with `conda`
```sh
cd mani-centric-wbc/
mamba env create -f isaac.yml
mamba activate isaac
```

Then, in the `mani-centric-wbc/`'s root, install `mani-centric-wbc`'s `legged_gym` as a pip package
```sh
pip install -e .
```

Finally, follow instructions on [NVIDIA's developer portal](https://developer.nvidia.com/isaac-gym) to install IsaacGym.
The Isaac Gym pip package should be installed in the `mamba` environment created above.

## Downloads

Download manipulation trajectories preprocessed for the whole-body controller pipeline
```sh
wget -qO- http://real.stanford.edu/umi-on-legs-compose/wbc/data.zip | bsdtar -xvf- -C ./
```
as well as the pretrained checkpoints
```sh
wget -qO- http://real.stanford.edu/umi-on-legs-compose/wbc/checkpoints.zip | bsdtar -xvf- -C ./
```

## Rollout Controller

To visualize the controller in simulation
```sh
python scripts/play.py --ckpt_path checkpoints/tossing/ours-real/model.pt --trajectory_file_path data/tossing.pkl --device cuda:0 --num_steps 1000 --num_envs 1  --visualize
```

This will also dump out the states needed for Blender visualization.
See the [visualization instructions](./visualization.md) for more details.


> 🪲 Troubleshooting IsaacGym
>
> A known issue with IsaacGym installation is that library paths aren't correctly updated, leading to the following error message
> ```
> ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
> ```
> To bypass this, add your `mamba` environment's library path to the library path environment variable by prepending commands with 
> `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/mambaforge/envs/isaac/lib/`

## Evaluation

To evaluate our controller in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl ckpt_path=checkpoints/tossing/ours/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" 
```
The summary that gets printed out at the end by weights and biases gives the metrics we care about.
 - `eval/task/reaching/pos_err/mean` and `eval/task/reaching/orn_err/mean` gives the position and orientation errors in meters and radians, respectively.
 - `eval/time_outs/sum` gives the proxy for survival rate. If the robot didn't terminate midway through the episode, then the episode should have timed out.
 - `eval/constraint/energy/sum_electrical_power/mean` gives the average power usage in Watts.

> 📈 Training Curves & Evaluation Logs
>
> Please find the training and evaluation runs in [this report](https://api.wandb.ai/links/columbia-ai-robotics/rrudtifq).

To evaluate the no-preview baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl ckpt_path=checkpoints/tossing/no-preview/model.pt env.tasks.reaching.target_obs_times="[0.0]"
```

To evaluate the body-space baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl ckpt_path=checkpoints/tossing/body-space/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" env.tasks.reaching.target_relative_to_base=true env.tasks.reaching.pos_obs_scale=1.0
```


To evaluate the random trajectories baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl ckpt_path=checkpoints/tossing/random-trajs/model.pt env.tasks.reaching.target_obs_times="[-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,1.0]" 
```

To evaluate the DeepWBC baseline in simulation
```sh
python scripts/evaluate.py env.sim_device=cuda:0 env.graphics_device_id=0 env.cfg.env.episode_length_s=17.0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl ckpt_path=checkpoints/tossing/deepwbc/model.pt env.tasks.reaching.target_obs_times="[0.0]" env.tasks.reaching.target_relative_to_base=true env.tasks.reaching.pos_obs_scale=1.0
```

## Humanoid Robot Support

For humanoid robot end-effector tasks, we provide specialized scripts and configurations:

### Humanoid End-Effector Task

To run the humanoid robot end-effector task:

```sh
python scripts/play_humanoid_ee_task.py \
    --ckpt_path checkpoints/humanoid/model.pt \
    --trajectory_file_path data/humanoid_trajectories.pkl \
    --robot_type humanoid \
    --visualize \
    --record_video
```

### Example Usage

Use the example script for easy setup:

```sh
python scripts/example_humanoid_usage.py \
    --ckpt_path checkpoints/humanoid/model.pt \
    --create_sample \
    --visualize
```

### Configuration Files

- `config/env/env_humanoid.yaml` - Humanoid robot environment configuration
- `config/env/combo_humanoid_reaching.yaml` - Combined configuration for humanoid reaching tasks
- `config/env/tasks/humanoid_reaching.yaml` - Humanoid reaching task configuration

For detailed documentation, see [Humanoid End-Effector Task Documentation](./humanoid_ee_task.md).
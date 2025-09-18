# 人型机器人末端执行器任务

本文档介绍如何使用针对人型机器人优化的末端执行器任务脚本。

## 概述

`play_humanoid_ee_task.py` 是基于原始的 `play_ee_task.py` 脚本创建的，专门针对人型机器人进行了优化。主要改进包括：

1. **关节配置适配**：支持人型机器人的27个关节（12个腿部关节 + 3个躯干关节 + 12个手臂关节）
2. **相机视角优化**：针对人型机器人的高度调整了相机位置
3. **约束条件调整**：适配人型机器人的关节限制和物理约束
4. **任务配置**：使用右手作为末端执行器进行到达任务

## 文件结构

```
mani-centric-wbc/
├── scripts/
│   └── play_humanoid_ee_task.py          # 人型机器人末端执行器任务脚本
├── config/env/
│   ├── env_humanoid.yaml                 # 人型机器人环境配置
│   ├── combo_humanoid_reaching.yaml      # 人型机器人组合配置
│   └── tasks/
│       └── humanoid_reaching.yaml        # 人型机器人任务配置
└── docs/
    └── humanoid_ee_task.md               # 本文档
```

## 使用方法

### 基本用法

```bash
python scripts/play_humanoid_ee_task.py \
    --ckpt_path /path/to/checkpoint \
    --trajectory_file_path /path/to/trajectory.pkl \
    --robot_type humanoid \
    --visualize \
    --record_video
```

### 参数说明

- `--ckpt_path`: 训练好的模型检查点路径
- `--trajectory_file_path`: 轨迹文件路径（必需）
- `--robot_type`: 机器人类型，可选 "humanoid" 或 "biped"（默认：humanoid）
- `--visualize`: 启用可视化
- `--record_video`: 录制视频
- `--device`: 计算设备（默认：cuda:0）
- `--num_envs`: 环境数量（默认：1）
- `--num_steps`: 步数（默认：1000）

### 配置选择

脚本会根据 `--robot_type` 参数自动调整配置：

#### 人型机器人 (humanoid)
- 27个关节控制（12腿 + 3躯干 + 12臂）
- 更高的初始位置（z=1.0）
- 更严格的关节限制
- 优化的相机视角

#### 双足机器人 (biped)
- 使用原始的四足机器人配置
- 保持原有的关节限制和相机设置

## 配置详解

### 环境配置 (env_humanoid.yaml)

人型机器人的环境配置包括：

1. **控制器配置**：
   - 位置控制器
   - 27个关节的PID参数
   - 关节限制和扭矩限制

2. **关节映射**：
   - 左腿：6个关节（髋关节3DOF + 膝关节 + 踝关节2DOF）
   - 右腿：6个关节
   - 躯干：3个关节（腰部）
   - 左臂：6个关节（肩关节3DOF + 肘关节 + 腕关节2DOF）
   - 右臂：6个关节

3. **约束条件**：
   - 关节限制
   - 碰撞检测
   - 能量约束
   - 平衡约束

### 任务配置 (humanoid_reaching.yaml)

末端执行器任务配置：

- **末端执行器**：使用右手 (`right_hand`)
- **目标观测**：位置和姿态误差
- **奖励函数**：基于姿态误差的奖励
- **课程学习**：逐步减小目标噪声

## 训练和评估

### 训练

使用人型机器人配置进行训练：

```bash
python scripts/train.py \
    env=combo_humanoid_reaching \
    --config-name train
```

### 评估

使用训练好的模型进行评估：

```bash
python scripts/play_humanoid_ee_task.py \
    --ckpt_path outputs/humanoid_model/checkpoint.pt \
    --trajectory_file_path data/humanoid_trajectories.pkl \
    --visualize \
    --num_steps 2000
```

## 输出文件

脚本运行后会生成以下文件：

1. **视频文件**：`video.mp4` - 任务执行过程
2. **日志文件**：
   - `logs.zarr` - 状态和动作数据
   - `logs.pkl` - 配置和日志信息
3. **导出模型**：
   - `exported/policy.pt` - TorchScript格式的策略
   - `exported/model_config.json` - 模型配置

## 注意事项

1. **URDF文件**：确保人型机器人的URDF文件路径正确
2. **关节命名**：关节名称必须与URDF文件中的名称一致
3. **轨迹文件**：轨迹文件格式必须与任务配置兼容
4. **硬件要求**：人型机器人需要更多的计算资源

## 故障排除

### 常见问题

1. **关节数量不匹配**：
   - 检查URDF文件中的关节数量
   - 确认配置文件中的关节映射

2. **相机视角问题**：
   - 调整 `update_cam_pos()` 函数中的偏移参数
   - 根据机器人实际高度调整相机位置

3. **约束违反**：
   - 检查关节限制配置
   - 调整控制器参数
   - 验证物理参数设置

### 调试建议

1. 启用可视化模式观察机器人行为
2. 检查日志文件中的约束违反情况
3. 逐步调整控制器参数
4. 验证轨迹文件的格式和内容

## 扩展功能

### 自定义任务

可以通过修改 `humanoid_reaching.yaml` 来创建自定义任务：

```yaml
_partial_: true
_target_: legged_gym.env.isaacgym.task.CustomTask
# 自定义任务参数
```

### 多末端执行器

支持同时控制多个末端执行器：

```yaml
tasks:
  left_hand_reaching: ${tasks.left_hand_reaching}
  right_hand_reaching: ${tasks.right_hand_reaching}
```

### 复杂约束

添加更复杂的约束条件：

```yaml
constraints:
  balance_constraint:
    violation_threshold: 0.1
  posture_constraint:
    target_posture: [0, 0, 0, ...]
```

## 贡献

欢迎提交问题和改进建议。请确保：

1. 测试新功能
2. 更新文档
3. 遵循代码规范
4. 添加适当的注释 
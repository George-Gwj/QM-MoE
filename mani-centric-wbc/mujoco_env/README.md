# Sim2Sim转移：IsaacGym到MuJoCo

这个目录包含了将IsaacGym环境训练的策略转移到MuJoCo环境的代码。

## 文件结构

```
mujoco_env/
├── env_velcmd_arm_swing_mujoco.py  # MuJoCo环境包装器
├── run_sim2sim.py                   # 运行脚本
└── README.md                        # 说明文档
```

## 功能特性

- **策略转移**：将IsaacGym训练的策略无缝转移到MuJoCo环境
- **课程学习**：支持与原环境相同的课程学习阶段
- **机械臂控制**：保持机械臂摆动和基座速度控制功能
- **观测兼容**：构建与原环境兼容的观测向量
- **实时渲染**：支持MuJoCo的3D可视化

## 安装依赖

```bash
pip install mujoco-py torch numpy omegaconf
```

## 使用方法

### 1. 准备文件

确保你有以下文件：
- **MuJoCo模型文件** (.xml)：机器人的URDF模型
- **训练好的策略文件** (.pt)：从IsaacGym训练得到的策略
- **配置文件** (.pkl或.yaml)：原始训练配置

### 2. 运行Sim2Sim转移

```bash
python run_sim2sim.py \
    --model_path /path/to/robot.xml \
    --policy_path /path/to/policy.pt \
    --config_path /path/to/config.pkl \
    --render \
    --num_steps 1000 \
    --vel_x 0.5 \
    --vel_y 0.0 \
    --vel_z 0.0
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | MuJoCo模型文件路径 | 必需 |
| `--policy_path` | 策略文件路径 | 必需 |
| `--config_path` | 配置文件路径 | 必需 |
| `--device` | 计算设备 (cpu/cuda) | cpu |
| `--render` | 是否渲染环境 | False |
| `--num_steps` | 仿真步数 | 1000 |
| `--dt` | 仿真时间步长 | 0.01 |
| `--vel_x` | X方向速度指令 | 0.5 |
| `--vel_y` | Y方向速度指令 | 0.0 |
| `--vel_z` | Z方向速度指令 | 0.0 |

## 课程学习阶段

环境支持5个课程学习阶段：

1. **阶段1**：仅基座速度控制，无机械臂目标
2. **阶段2**：启用机械臂随机目标（不摆动）
3. **阶段3**：小幅度机械臂摆动 (±0.3rad)
4. **阶段4**：大幅度机械臂摆动 (±0.8rad)
5. **阶段5**：超大幅度机械臂摆动 (±1.5rad)

## 观测向量结构

观测向量按以下顺序组成：

```
[setup_obs] + [state_obs] + [task_obs] + [base_lin_vel_cmd] + [additional_obs] + [action_history]
```

- **setup_obs**：控制器参数等设置信息
- **state_obs**：机器人状态（关节位置、速度等）
- **task_obs**：任务相关目标信息
- **base_lin_vel_cmd**：基座速度指令
- **additional_obs**：额外观测（位置、姿态等）
- **action_history**：上一时刻动作

## 输出结果

脚本运行完成后会生成：
- **控制台输出**：实时仿真状态和进度
- **结果文件**：`sim2sim_results/results.json`
  - 总步数和奖励
  - 仿真时间和实时因子
  - 最终课程阶段
  - 速度指令设置

## 注意事项

1. **模型兼容性**：确保MuJoCo模型与IsaacGym模型结构一致
2. **观测维度**：观测向量维度必须与训练时一致
3. **关节顺序**：机械臂关节索引需要正确映射
4. **物理参数**：可能需要调整MuJoCo的物理参数以匹配IsaacGym

## 故障排除

### 常见问题

1. **观测维度不匹配**：检查配置文件中的观测设置
2. **策略加载失败**：确保策略文件格式正确
3. **渲染问题**：检查MuJoCo安装和显示设置
4. **性能问题**：调整时间步长和渲染设置

### 调试建议

- 启用详细日志输出
- 检查观测向量的构建过程
- 验证策略输入输出的维度
- 对比IsaacGym和MuJoCo的环境状态

## 扩展功能

可以进一步扩展的功能：
- 支持更多机器人模型
- 添加更复杂的任务
- 实现多环境并行仿真
- 集成其他仿真器（如PyBullet）
- 添加性能评估指标

## 联系支持

如果遇到问题，请检查：
1. 依赖包版本兼容性
2. 文件路径和权限
3. 系统环境配置
4. 错误日志和堆栈跟踪 
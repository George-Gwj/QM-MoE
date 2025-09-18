# play_velcmd_arm_swing.py 更新说明

## 概述
根据`env_velcmd_arm_swing.py`的课程学习阶段6（三关节摆动控制）功能，对`play_velcmd_arm_swing.py`进行了相应的更新，支持三关节同时摆动控制。

## 主要修改内容

### 1. 课程阶段配置
- **默认阶段**：将默认课程阶段从1改为6，直接测试三关节摆动功能
- **配置路径**：`config["env"]["cfg"]["curriculum"]["stage"] = 6`

### 2. 机械臂关节选择逻辑
- **三关节支持**：新增`arm_joint_indices`变量，支持三关节控制
- **智能选择**：根据课程阶段自动选择单关节或三关节控制
- **环境集成**：优先使用环境在reset时确定的关节索引
- **备选方案**：提供硬编码的关节索引作为备选

```python
# 阶段6：三关节控制
if env.curriculum_stage == 6:
    arm_joint_indices = env.arm_control_joint_idx[0].cpu().numpy()
    print(f"环境选择的三关节索引: {arm_joint_indices}")
else:
    # 其他阶段：单关节控制
    arm_joint_idx = env.arm_control_joint_idx[0, 0].item()
```

### 3. 摆动控制参数
- **三关节配置**：为每个关节定义不同的摆动参数
- **差异化摆动**：
  - 关节1：小幅度摆动 ±0.5rad，基础频率
  - 关节2：中等幅度摆动 ±1.0rad，1.2倍频率，π/3相位偏移
  - 关节3：大幅度摆动 ±1.5rad，0.8倍频率，π/2相位偏移

```python
ARM_JOINT_CONFIGS = {
    0: {"amplitude": 0.5, "frequency_multiplier": 1.0, "phase_offset": 0.0},
    1: {"amplitude": 1.0, "frequency_multiplier": 1.2, "phase_offset": np.pi/3},
    2: {"amplitude": 1.5, "frequency_multiplier": 0.8, "phase_offset": np.pi/2}
}
```

### 4. 摆动控制函数
- **函数重命名**：`get_arm_swing_angle()` → `get_arm_swing_angles()`
- **多关节支持**：返回关节角度列表而不是单个角度
- **阶段适配**：根据课程阶段自动选择控制模式

```python
def get_arm_swing_angles(t):
    if env.curriculum_stage == 6 and arm_joint_indices is not None:
        # 阶段6：三关节摆动
        swing_angles = []
        for i, joint_idx in enumerate(arm_joint_indices):
            config = ARM_JOINT_CONFIGS[i]
            angle = config["amplitude"] * np.sin(
                2 * np.pi * t / ARM_SWING_PERIOD * config["frequency_multiplier"] + config["phase_offset"]
            )
            swing_angles.append(angle)
        return swing_angles
    else:
        # 其他阶段：单关节摆动
        # ... 原有逻辑
```

### 5. 动作应用逻辑
- **三关节控制**：同时控制三个关节的摆动
- **阶段判断**：根据课程阶段选择不同的控制策略
- **日志输出**：为三关节控制提供详细的日志信息

```python
if env.curriculum_stage == 6 and arm_joint_indices is not None:
    # 阶段6：三关节摆动控制
    for i, (joint_idx, swing_angle) in enumerate(zip(arm_joint_indices, swing_angles)):
        if actions.shape[1] > joint_idx:
            actions[:, joint_idx] = swing_angle
    print(f"三关节摆动 - 关节1({arm_joint_indices[0]}): {swing_angles[0]:.3f}rad, "
          f"关节2({arm_joint_indices[1]}): {swing_angles[1]:.3f}rad, "
          f"关节3({arm_joint_indices[2]}): {swing_angles[2]:.3f}rad")
```

### 6. 日志记录增强
- **保留字段**：
  - `arm_control_joint_idx`：记录环境选择的机械臂关节索引
  - `num_arm_dofs`：记录机械臂关节数量
- **移除字段**：不再记录机械臂摆动角度、位置和速度等详细数据
- **三关节支持**：为三关节控制提供基本的关节选择信息记录
- **数据精简**：保持日志记录的简洁性

### 7. 可视化功能
- **保留功能**：`plot_velocity_curves()`用于绘制速度跟踪曲线
- **移除功能**：不再生成机械臂摆动曲线图
- **数据导出**：仅保存基本的CSV数据，不包含详细的机械臂摆动数据

## 使用方法

### 基本运行
```bash
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --visualize --num_steps 1000
```

### 指定速度指令
```bash
# 指定X方向速度
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --vel_x 1.0

# 指定X和Y方向速度
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --vel_x 1.0 --vel_y 0.5

# 使用随机速度指令
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --random_vel --vel_range 2.0

# 使用时间变化的速度指令
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --time_varying_vel --vel_amplitude 1.5 --vel_frequency 0.3
```

### 指定机械臂关节和摆动参数

#### 单关节控制
```bash
# 指定关节14进行摆动，幅度±1.0rad，频率0.5Hz
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 14 --arm_swing_amplitude 1.0 --arm_swing_frequency 0.5

# 指定关节12进行摆动，幅度±0.8rad，频率0.3Hz，相位偏移π/4
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 12 --arm_swing_amplitude 0.8 --arm_swing_frequency 0.3 --arm_swing_phase 0.785
```

#### 三关节控制
```bash
# 指定关节13、15、17进行三关节摆动
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 13 15 17

# 指定三关节摆动，每个关节不同的幅度
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 13 15 17 --arm_swing_amplitude 0.5 1.0 1.5

# 指定三关节摆动，每个关节不同的频率
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 13 15 17 --arm_swing_frequency 0.3 0.5 0.7

# 指定三关节摆动，每个关节不同的相位偏移
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 13 15 17 --arm_swing_phase 0.0 1.57 3.14

# 完整的三关节摆动配置
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --arm_joint 13 15 17 --arm_swing_amplitude 0.5 1.0 1.5 --arm_swing_frequency 0.3 0.5 0.7 --arm_swing_phase 0.0 1.57 3.14
```

#### 禁用机械臂摆动
```bash
# 禁用机械臂摆动控制，只进行速度跟踪
python play_velcmd_arm_swing.py --ckpt_path /path/to/checkpoint --disable_arm_swing
```

### 课程阶段测试
- **阶段6（默认）**：三关节摆动控制
- **其他阶段**：自动降级为单关节控制
- **命令行指定**：优先使用命令行指定的关节和参数

### 输出文件
- `velocity_curves.png`：速度跟踪曲线
- `velocity_data.csv`：速度数据
- 不再生成机械臂摆动相关的图表和数据文件

## 新增功能特性

### 1. 灵活的速度控制
- **指定速度**：可以精确指定X、Y、Z方向的速度指令
- **随机速度**：支持随机速度指令，可设置速度范围
- **时间变化速度**：支持正弦波形式的时间变化速度指令

### 2. 精确的关节控制
- **关节选择**：可以指定具体的机械臂关节索引
- **单关节模式**：指定一个关节进行摆动控制
- **三关节模式**：指定三个关节进行协调摆动控制

### 3. 可调节的摆动参数
- **摆动幅度**：控制摆动的最大角度范围
- **摆动频率**：控制摆动的速度
- **相位偏移**：控制摆动的起始相位，实现协调运动

### 4. 智能参数适配
- **优先级**：命令行参数 > 环境配置 > 默认值
- **自动检测**：根据参数数量自动判断单关节或三关节模式
- **参数验证**：检查参数数量是否正确，提供警告信息

## 技术特点

1. **向后兼容**：保持对原有单关节控制的支持
2. **智能适配**：根据课程阶段和命令行参数自动选择控制模式
3. **参数灵活**：支持多种摆动参数组合，满足不同测试需求
4. **错误处理**：提供参数验证和备选方案
5. **实时控制**：支持运行时动态调整摆动参数

## 注意事项

1. **关节数量**：确保机械臂有足够的关节（≥3个）
2. **索引范围**：检查关节索引是否在动作空间范围内
3. **参数匹配**：关节数量必须与摆动参数数量匹配
4. **课程阶段**：确保环境配置正确设置课程阶段
5. **性能影响**：三关节控制可能增加计算开销

## 参数说明

### 速度控制参数
- `--vel_x, --vel_y, --vel_z`：指定X、Y、Z方向速度（m/s）
- `--random_vel`：启用随机速度指令
- `--vel_range`：随机速度范围（±vel_range m/s）
- `--time_varying_vel`：启用时间变化速度指令
- `--vel_amplitude`：时间变化速度的幅度（m/s）
- `--vel_frequency`：时间变化速度的频率（Hz）

### 机械臂控制参数
- `--arm_joint`：指定机械臂关节索引（1个或3个数字）
- `--arm_swing_amplitude`：摆动幅度（rad）
- `--arm_swing_frequency`：摆动频率（Hz）
- `--arm_swing_phase`：摆动相位偏移（rad）
- `--disable_arm_swing`：禁用机械臂摆动控制

## 未来扩展

1. **更多关节**：支持4个或更多关节的同时控制
2. **复杂轨迹**：支持更复杂的摆动轨迹（如8字形、螺旋形等）
3. **自适应控制**：根据任务需求自动调整关节选择
4. **实时监控**：提供实时的关节状态监控界面
5. **配置文件**：支持通过配置文件设置复杂的控制参数

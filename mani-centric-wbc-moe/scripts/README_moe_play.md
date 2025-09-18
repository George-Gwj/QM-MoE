# MoE模型播放脚本使用说明

## 概述

这些脚本用于加载和运行混合专家(MoE)模型进行推理，支持权重分析和可视化。

## 脚本文件

### 1. `play_moe_ee_task.py` - 完整功能版本
- 支持完整的EE任务播放
- 包含视频录制和可视化功能
- 支持权重统计和分析
- 兼容原始play_ee_task.py的所有功能

### 2. `play_moe_simple.py` - 简化版本
- 专注于MoE模型推理
- 轻量级，易于使用和调试
- 包含权重分析功能
- 适合快速测试和验证

## 使用方法

### 基本用法

```bash
# 使用完整版本
python scripts/play_moe_ee_task.py \
    --moe_ckpt_path /path/to/moe_model.pt \
    --base_ckpt_path /path/to/base_expert.pt \
    --arm_ckpt_path /path/to/arm_expert.pt \
    --trajectory_file_path /path/to/trajectory.pkl \
    --visualize \
    --show_weights

# 使用简化版本
python scripts/play_moe_simple.py \
    --moe_ckpt_path /path/to/moe_model.pt \
    --trajectory_file_path /path/to/trajectory.pkl \
    --visualize \
    --show_weights
```

### 参数说明

#### 必需参数
- `--moe_ckpt_path`: MoE模型权重文件路径
- `--trajectory_file_path`: 轨迹文件路径

#### 可选参数
- `--device`: 计算设备 (默认: cuda:0)
- `--num_envs`: 环境数量 (默认: 1)
- `--num_steps`: 运行步数 (默认: 1000)
- `--visualize`: 启用可视化
- `--show_weights`: 显示MoE权重分布
- `--record_video`: 录制视频 (仅完整版本)

### 专家模型参数 (仅完整版本)
- `--base_ckpt_path`: 基础专家模型路径
- `--arm_ckpt_path`: 手臂专家模型路径

## 功能特性

### 1. MoE权重分析
- 实时显示专家权重分布
- 计算权重熵值
- 统计权重使用情况
- 保存权重历史数据

### 2. 模型推理
- 使用MoE模型进行动作推理
- 支持批量环境推理
- 自动处理模型加载和配置

### 3. 数据记录
- 记录动作和状态数据
- 保存权重统计信息
- 支持多种数据格式输出

## 输出文件

### 权重统计文件
- `weight_stats.pkl`: 权重统计数据
- 包含base_weights, arm_weights, weight_entropy

### 日志文件
- `moe_results.pkl`: 完整的推理结果
- `logs.zarr`: 状态和动作数据
- `video.mp4`: 录制的视频 (如果启用)

## 权重分析示例

```python
# 加载权重统计
import pickle
with open('weight_stats.pkl', 'rb') as f:
    weight_stats = pickle.load(f)

# 分析权重分布
base_weights = np.array(weight_stats['base_weights'])
arm_weights = np.array(weight_stats['arm_weights'])

print(f"Base专家使用频率: {np.mean(base_weights > 0.5):.2%}")
print(f"Arm专家使用频率: {np.mean(arm_weights > 0.5):.2%}")
print(f"平均权重熵: {np.mean(weight_stats['weight_entropy']):.3f}")
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查权重文件路径是否正确
   - 确认配置文件存在
   - 验证模型类型是否匹配

2. **设备不匹配**
   - 使用`--device`参数指定正确的设备
   - 检查CUDA可用性

3. **权重显示异常**
   - 确认模型是MoE类型
   - 检查`get_moe_weights`方法是否可用

### 调试建议

1. 使用`--show_weights`参数查看权重分布
2. 检查输出日志中的错误信息
3. 验证环境配置是否正确
4. 使用简化版本进行快速测试

## 扩展功能

### 自定义权重分析
```python
# 在脚本中添加自定义分析
def analyze_weights(weight_stats):
    # 自定义分析逻辑
    pass
```

### 添加新的记录数据
```python
# 在state_logs中添加新的记录项
state_logs["custom_metric"] = []
# 在循环中记录数据
state_logs["custom_metric"].append(custom_data)
```

## 注意事项

1. 确保MoE模型已正确训练
2. 轨迹文件格式需要与任务匹配
3. 可视化需要图形界面支持
4. 大量数据记录可能占用较多存储空间
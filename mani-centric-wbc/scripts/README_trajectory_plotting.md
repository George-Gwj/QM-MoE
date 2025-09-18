# 末端执行器轨迹绘制功能

## 功能说明

`play_ee_task.py` 脚本现在支持绘制末端执行器的期望位置和实际位置曲线，帮助分析机器人的跟踪性能。

## 新增参数

- `--plot_trajectory`: 启用轨迹绘制功能
- `--plot_save_path`: 指定保存图片的路径（默认：`trajectory_plot.png`）

## 使用方法

### 基本使用
```bash
python play_ee_task.py \
    --ckpt_path /path/to/checkpoint \
    --trajectory_file_path /path/to/trajectory.pkl \
    --plot_trajectory \
    --num_steps 1000
```

### 自定义保存路径
```bash
python play_ee_task.py \
    --ckpt_path /path/to/checkpoint \
    --trajectory_file_path /path/to/trajectory.pkl \
    --plot_trajectory \
    --plot_save_path my_trajectory_analysis.png \
    --num_steps 1000
```

## 输出内容

脚本会生成一个包含4个子图的轨迹分析图：

1. **3D轨迹图**: 显示实际轨迹（蓝色实线）和期望轨迹（红色虚线）在3D空间中的对比
2. **X位置-时间图**: 显示X坐标随时间的变化
3. **Y位置-时间图**: 显示Y坐标随时间的变化  
4. **Z位置-时间图**: 显示Z坐标随时间的变化

## 统计信息

脚本还会输出以下跟踪性能统计：
- 平均位置误差 (Mean Position Error)
- 最大位置误差 (Max Position Error)  
- 均方根位置误差 (RMS Position Error)

## 注意事项

- 轨迹数据会在每个仿真步骤中记录
- 图片会保存到wandb运行目录中
- 如果启用了可视化模式，图形也会在屏幕上显示
- 确保安装了matplotlib: `pip install matplotlib`
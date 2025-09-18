#!/bin/bash

# 使用示例脚本 for combo_go2Piper_pickle_reaching_extreme_velcmd_arm_swing
# 这个脚本展示了如何使用新的播放代码

# 设置基本参数
CHECKPOINT_PATH="path/to/your/checkpoint/model_1000.pt"
TRAJECTORY_FILE="path/to/your/trajectory.pkl"
DEVICE="cuda:0"
NUM_STEPS=1000

# 基本播放（无可视化）
echo "=== 基本播放模式 ==="
python mani-centric-wbc/scripts/play_combo_go2Piper_pickle_reaching_extreme_velcmd_arm_swing.py \
    --ckpt_path $CHECKPOINT_PATH \
    --trajectory_file_path $TRAJECTORY_FILE \
    --device $DEVICE \
    --num_envs 1 \
    --num_steps $NUM_STEPS

echo ""

# 可视化播放模式
echo "=== 可视化播放模式 ==="
python mani-centric-wbc/scripts/play_combo_go2Piper_pickle_reaching_extreme_velcmd_arm_swing.py \
    --ckpt_path $CHECKPOINT_PATH \
    --trajectory_file_path $TRAJECTORY_FILE \
    --device $DEVICE \
    --num_envs 1 \
    --num_steps $NUM_STEPS \
    --visualize

echo ""

# 录制视频模式
echo "=== 录制视频模式 ==="
python mani-centric-wbc/scripts/play_combo_go2Piper_pickle_reaching_extreme_velcmd_arm_swing.py \
    --ckpt_path $CHECKPOINT_PATH \
    --trajectory_file_path $TRAJECTORY_FILE \
    --device $DEVICE \
    --num_envs 1 \
    --num_steps $NUM_STEPS \
    --visualize \
    --record_video

echo ""

# 连续播放模式（按Ctrl+C停止）
echo "=== 连续播放模式 ==="
python mani-centric-wbc/scripts/play_combo_go2Piper_pickle_reaching_extreme_velcmd_arm_swing.py \
    --ckpt_path $CHECKPOINT_PATH \
    --trajectory_file_path $TRAJECTORY_FILE \
    --device $DEVICE \
    --num_envs 1 \
    --num_steps -1 \
    --visualize

echo "所有播放模式完成！"
#!/usr/bin/env python3
"""
测试导入路径是否正确
"""

import os
import sys

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup_path import setup_correct_path
setup_correct_path()

# 测试导入
try:
    from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
    print("✓ 成功导入 OnPolicyRunner")
    print(f"  OnPolicyRunner 位置: {OnPolicyRunner.__module__}")
except ImportError as e:
    print(f"✗ 导入 OnPolicyRunner 失败: {e}")

try:
    from legged_gym.rsl_rl.modules.actor_critic_moe import ActorCriticMoE
    print("✓ 成功导入 ActorCriticMoE")
    print(f"  ActorCriticMoE 位置: {ActorCriticMoE.__module__}")
except ImportError as e:
    print(f"✗ 导入 ActorCriticMoE 失败: {e}")

try:
    from legged_gym.rsl_rl.modules.obs_moe import ObsMoE
    print("✓ 成功导入 ObsMoE")
    print(f"  ObsMoE 位置: {ObsMoE.__module__}")
except ImportError as e:
    print(f"✗ 导入 ObsMoE 失败: {e}")

# 检查模块路径
print(f"\n当前Python路径:")
for i, path in enumerate(sys.path[:5]):  # 只显示前5个路径
    print(f"  {i}: {path}")

print(f"\n当前工作目录: {os.getcwd()}")
print(f"脚本目录: {os.path.dirname(__file__)}")
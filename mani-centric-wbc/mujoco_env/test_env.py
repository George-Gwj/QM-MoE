#!/usr/bin/env python3
"""
测试MuJoCo环境是否正常工作的脚本
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_dummy_policy(obs_dim: int, action_dim: int) -> nn.Module:
    """创建一个虚拟策略用于测试"""
    class DummyPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int):
            super().__init__()
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.network = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.network(x)
    
    return DummyPolicy(obs_dim, action_dim)

def create_dummy_config(obs_dim: int, action_dim: int) -> dict:
    """创建一个虚拟配置用于测试"""
    return {
        'env': {
            'num_observations': obs_dim,
            'num_actions': action_dim,
            'cfg': {
                'curriculum': {
                    'stage': 1,
                    'transition_steps': 2000000
                }
            }
        }
    }

def test_mujoco_env():
    """测试MuJoCo环境"""
    print("开始测试MuJoCo环境...")
    
    try:
        # 检查mujoco-py是否可用
        import mujoco_py
        print("✓ mujoco-py 导入成功")
    except ImportError:
        print("✗ mujoco-py 导入失败，请安装 mujoco-py")
        return False
    
    try:
        # 检查torch是否可用
        import torch
        print("✓ PyTorch 导入成功")
    except ImportError:
        print("✗ PyTorch 导入失败，请安装 PyTorch")
        return False
    
    # 创建虚拟配置
    obs_dim = 100  # 假设的观测维度
    action_dim = 18  # 假设的动作维度
    
    config = create_dummy_config(obs_dim, action_dim)
    
    # 创建虚拟策略
    policy = create_dummy_policy(obs_dim, action_dim)
    
    # 保存虚拟策略
    policy_path = "test_policy.pt"
    torch.save(policy, policy_path)
    print(f"✓ 虚拟策略已保存到: {policy_path}")
    
    # 保存虚拟配置
    config_path = "test_config.pkl"
    import pickle
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ 虚拟配置已保存到: {config_path}")
    
    # 检查是否有可用的MuJoCo模型
    model_path = None
    possible_paths = [
        "robot.xml",
        "go2_piper.xml", 
        "go2_arx5.xml",
        "../resources/robots/go2_piper/urdf/go2_piper.xml",
        "../resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("⚠ 未找到MuJoCo模型文件，创建测试模型...")
        # 创建一个简单的测试模型
        test_model_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="test_robot">
  <compiler angle="radian" coordinate="local" meshdir="" texturedir=""/>
  <default>
    <default class="main">
      <joint armature="0.01" damping="0.1" frictionloss="0.01"/>
      <geom contype="0" conaffinity="0" condim="1" friction="0.7 0.1 0.1" rgba="0.7 0.7 0.7 1"/>
    </default>
  </default>
  <option timestep="0.01" iterations="50" solver="Newton" tolerance="1e-10" noslip_iterations="0" cone="pyramidal"/>
  <size nconmax="50" njmax="100" nstack="1000"/>
  <worldbody>
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" castshadow="false" pos="0 0 5" dir="0 0 -1"/>
    <geom name="ground" pos="0 0 0" size="0 0 .05" type="plane" conaffinity="1"/>
    <body name="base" pos="0 0 0.3">
      <joint name="base_joint" type="free"/>
      <geom name="base_geom" type="cylinder" size="0.2 0.1" rgba="0.8 0.8 0.8 1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14"/>
        <geom name="link1_geom" type="cylinder" size="0.05 0.1" rgba="0.6 0.6 0.6 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="100"/>
  </actuator>
</mujoco>"""
        
        with open("test_robot.xml", "w") as f:
            f.write(test_model_xml)
        model_path = "test_robot.xml"
        print(f"✓ 测试模型已创建: {model_path}")
    
    print(f"✓ 使用模型文件: {model_path}")
    
    try:
        # 测试环境创建
        from mujoco_env.env_velcmd_arm_swing_mujoco import create_mujoco_env
        
        env = create_mujoco_env(
            model_path=model_path,
            policy_path=policy_path,
            config_path=config_path,
            device='cpu',
            render=False  # 测试时不渲染
        )
        
        print("✓ MuJoCo环境创建成功")
        print(f"  - 观测维度: {env.obs_dim}")
        print(f"  - 动作维度: {env.action_dim}")
        print(f"  - 机械臂关节数: {env.num_arm_dofs}")
        
        # 测试环境重置
        obs = env.reset()
        print(f"✓ 环境重置成功，观测维度: {obs.shape}")
        
        # 测试环境步进
        actions = np.random.randn(env.action_dim) * 0.1
        obs, reward, done, info = env.step(actions)
        print(f"✓ 环境步进成功，新观测维度: {obs.shape}")
        
        # 测试策略推理
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = env.policy(obs_tensor)
            print(f"✓ 策略推理成功，动作维度: {action_tensor.shape}")
        
        # 清理资源
        env.close()
        
        # 清理测试文件
        if os.path.exists(policy_path):
            os.remove(policy_path)
        if os.path.exists(config_path):
            os.remove(config_path)
        if model_path == "test_robot.xml" and os.path.exists(model_path):
            os.remove(model_path)
        
        print("\n🎉 所有测试通过！MuJoCo环境工作正常。")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mujoco_env()
    if success:
        print("\n✅ 环境测试完成，可以开始使用sim2sim转移功能！")
    else:
        print("\n❌ 环境测试失败，请检查安装和配置。") 
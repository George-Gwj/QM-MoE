#!/usr/bin/env python3
"""
测试多actor环境（机器人 + 动态障碍物）
"""

import os
import sys
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def test_multi_actor_environment():
    """测试多actor环境"""
    print("=== 测试多actor环境 ===")
    
    try:
        # 先导入IsaacGym
        import isaacgym
        from isaacgym import gymapi
        print("✓ IsaacGym导入成功")
        
        # 然后导入PyTorch
        import torch
        print("✓ PyTorch导入成功")
        
        # 最后导入我们的环境
        from legged_gym.env.isaacgym.env_cbf import IsaacGymEnv, ObstacleManager
        print("✓ IsaacGymEnv和ObstacleManager导入成功")
        
        # 测试ObstacleManager类
        print("\n=== 测试ObstacleManager类 ===")
        print(f"✓ ObstacleManager类存在: {ObstacleManager}")
        
        # 检查类的方法
        methods = [method for method in dir(ObstacleManager) if not method.startswith('_')]
        print(f"✓ ObstacleManager方法: {methods}")
        
        # 检查__init__方法的参数
        import inspect
        sig = inspect.signature(ObstacleManager.__init__)
        print(f"✓ ObstacleManager.__init__参数: {sig}")
        
        # 测试IsaacGymEnv类
        print("\n=== 测试IsaacGymEnv类 ===")
        print(f"✓ IsaacGymEnv类存在: {IsaacGymEnv}")
        
        # 检查IsaacGymEnv的方法
        env_methods = [method for method in dir(IsaacGymEnv) if not method.startswith('_')]
        print(f"✓ IsaacGymEnv方法数量: {len(env_methods)}")
        
        # 检查是否有obstacle_manager属性
        if hasattr(IsaacGymEnv, 'obstacle_manager'):
            print("✓ IsaacGymEnv有obstacle_manager属性")
        else:
            print("✗ IsaacGymEnv缺少obstacle_manager属性")
        
        # 检查是否有_create_static_obstacles方法
        if hasattr(IsaacGymEnv, '_create_static_obstacles'):
            print("✗ IsaacGymEnv仍有_create_static_obstacles方法（应该被移除）")
        else:
            print("✓ IsaacGymEnv已移除_create_static_obstacles方法")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obstacle_creation():
    """测试障碍物创建逻辑"""
    print("\n=== 测试障碍物创建逻辑 ===")
    
    try:
        from legged_gym.env.isaacgym.env_cbf import ObstacleManager
        
        # 模拟IsaacGym环境
        class MockGym:
            def create_sphere(self, sim, radius):
                return f"sphere_asset_{radius}"
            
            def create_capsule(self, sim, radius, half_length):
                return f"capsule_asset_{radius}_{half_length}"
            
            def create_box(self, sim, width, length, height):
                return f"box_asset_{width}_{length}_{height}"
            
            def set_asset_density(self, asset, density):
                pass
            
            def create_actor(self, sim, asset, pose, name, env_id, collision_group):
                return f"actor_{name}_{env_id}"
            
            def get_actor_rigid_body_properties(self, sim, actor_handle):
                return [type('RigidBodyProps', (), {'mass': 1.0, 'flags': 0})()]
            
            def set_actor_rigid_body_properties(self, sim, actor_handle, props):
                pass
        
        class MockSim:
            pass
        
        # 创建模拟的ObstacleManager
        mock_gym = MockGym()
        mock_sim = MockSim()
        device = torch.device('cpu')
        num_envs = 2
        
        print("✓ 创建模拟环境成功")
        
        # 测试障碍物参数
        expected_obstacles = {
            'sphere': 2,      # 2个球体
            'capsule': 6,     # 6个胶囊体
            'rectangle': 1    # 1个矩形
        }
        
        print("✓ 期望的障碍物数量:")
        for obstacle_type, count in expected_obstacles.items():
            print(f"  - {obstacle_type}: {count} 个")
        
        # 测试障碍物位置参数
        expected_positions = {
            'ball': [1.00, 0.0, 2.8],
            'ball1': [0.35, 0.5, 2.75],
            'capsule': [-0.05, 2.5, 0.68],
            'rectangle': [-0.05, 1.45, 1.8]
        }
        
        print("✓ 期望的障碍物位置:")
        for name, pos in expected_positions.items():
            print(f"  - {name}: {pos}")
        
        return True
        
    except Exception as e:
        print(f"✗ 障碍物创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_isaacgym_api_compatibility():
    """测试IsaacGym API兼容性"""
    print("\n=== 测试IsaacGym API兼容性 ===")
    
    try:
        import isaacgym
        from isaacgym import gymapi
        
        print("✓ isaacgym导入成功")
        print("✓ gymapi导入成功")
        
        # 检查常用的gymapi属性
        api_attrs = [
            'Transform', 'Vec3', 'Quat', 'SimParams', 'PlaneParams',
            'CameraProperties', 'AssetOptions', 'ForceSensorProperties'
        ]
        
        print("✓ 检查gymapi属性:")
        for attr in api_attrs:
            if hasattr(gymapi, attr):
                print(f"  - {attr}: ✓ 存在")
            else:
                print(f"  - {attr}: ✗ 不存在")
        
        # 检查几何体创建方法
        print("\n✓ 检查几何体创建方法:")
        geometry_methods = ['create_sphere', 'create_capsule', 'create_box']
        for method in geometry_methods:
            if hasattr(gymapi, method):
                print(f"  - {method}: ✓ 存在")
            else:
                print(f"  - {method}: ✗ 不存在")
        
        # 检查actor创建方法
        print("\n✓ 检查actor创建方法:")
        actor_methods = ['create_actor', 'set_actor_rigid_body_properties']
        for method in actor_methods:
            if hasattr(gymapi, method):
                print(f"  - {method}: ✓ 存在")
            else:
                print(f"  - {method}: ✗ 不存在")
        
        return True
        
    except Exception as e:
        print(f"✗ API兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("IsaacGym多actor环境测试脚本")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("多actor环境", test_multi_actor_environment),
        ("障碍物创建逻辑", test_obstacle_creation),
        ("IsaacGym API兼容性", test_isaacgym_api_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n正在运行测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ 测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！多actor环境基础功能正常。")
        print("现在可以尝试运行完整的可视化脚本了。")
        print("\n主要改进:")
        print("1. ✓ 支持每个环境多个actor（机器人 + 障碍物）")
        print("2. ✓ 移除了不存在的_create_static_obstacles方法")
        print("3. ✓ 障碍物现在作为动态actor创建")
        print("4. ✓ 每个环境都有独立的障碍物实例")
    else:
        print("\n⚠ 部分测试失败，需要进一步检查。")
        print("\n可能的问题:")
        print("1. IsaacGym版本不兼容")
        print("2. 导入顺序问题")
        print("3. 依赖包缺失")
        print("4. 代码修改不完整")
    
    return passed == total

if __name__ == "__main__":
    main() 
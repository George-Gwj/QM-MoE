# Box Actor 集成说明

## 概述
成功在每个IsaacGym环境中添加了一个box物体作为第二个actor，使机器人可以与box进行物理交互。

## 修改的文件

### 1. `env_add_baseinfo.py`

#### 主要修改：

1. **添加box actor句柄列表**
   ```python
   self.box_handles = []  # 添加box actor句柄列表
   ```

2. **创建box asset**
   ```python
   # 创建box asset
   box_size = 0.1  # 10cm x 10cm x 10cm
   box_asset_options = gymapi.AssetOptions()
   box_asset_options.density = 1000.0  # 水的密度
   box_asset = self.gym.create_box(
       self.sim, box_size, box_size, box_size, box_asset_options
   )
   ```

3. **在环境创建循环中添加box actor**
   ```python
   # 创建box actor
   box_pose = gymapi.Transform()
   # 随机位置，在机器人附近
   box_pose.p = gymapi.Vec3(
       np.random.uniform(-0.5, 0.5),
       np.random.uniform(-0.5, 0.5),
       0.1  # 稍微高于地面
   )
   box_pose.r = gymapi.Quat.from_axis_angle(
       gymapi.Vec3(0, 0, 1), 
       np.random.uniform(0, 2 * np.pi)
   )
   
   box_handle = self.gym.create_actor(
       env_handle,
       box_asset,
       box_pose,
       f"box_{i}",
       i,
       0,  # collision group
       0,  # collision filter
   )
   
   # 设置box颜色
   color = gymapi.Vec3(
       np.random.uniform(0.5, 1.0),
       np.random.uniform(0.5, 1.0),
       np.random.uniform(0.5, 1.0)
   )
   self.gym.set_rigid_body_color(
       env_handle, box_handle, 0, 
       gymapi.MESH_VISUAL_AND_COLLISION, color
   )
   ```

4. **添加box重置方法**
   ```python
   def _reset_box_states(self, env_ids):
       """重置box物体的位置和状态"""
       for env_id in env_ids:
           if env_id < len(self.box_handles):
               # 随机位置
               box_pose = gymapi.Transform()
               box_pose.p = gymapi.Vec3(
                   np.random.uniform(-0.5, 0.5),
                   np.random.uniform(-0.5, 0.5),
                   0.1  # 稍微高于地面
               )
               box_pose.r = gymapi.Quat.from_axis_angle(
                   gymapi.Vec3(0, 0, 1), 
                   np.random.uniform(0, 2 * np.pi)
               )
               
               # 设置box位置
               self.gym.set_actor_root_state_tensor_indexed(
                   self.sim,
                   gymtorch.unwrap_tensor(
                       torch.tensor([
                           box_pose.p.x, box_pose.p.y, box_pose.p.z,
                           box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w,
                           0, 0, 0,  # 线速度
                           0, 0, 0   # 角速度
                       ], device=self.device).unsqueeze(0)
                   ),
                   gymtorch.unwrap_tensor(torch.tensor([self.box_handles[env_id]], device=self.device)),
                   1
               )
   ```

5. **在reset_idx中调用box重置**
   ```python
   def reset_idx(self, env_ids):
       # ... 其他重置代码 ...
       self._reset_box_states(env_ids)  # 重置box状态
       # ... 其他重置代码 ...
   ```

## 功能特性

### 1. Box属性
- **尺寸**: 10cm × 10cm × 10cm
- **密度**: 1000 kg/m³ (水的密度)
- **位置**: 随机分布在机器人附近(-0.5到0.5米范围内)
- **高度**: 稍微高于地面(0.1米)
- **旋转**: 随机Z轴旋转
- **颜色**: 随机颜色(0.5-1.0范围)

### 2. 物理交互
- Box与机器人可以发生碰撞
- Box受到重力影响
- Box可以被机器人推动
- Box在环境重置时会重新随机化位置

### 3. 状态管理
- Box状态在环境重置时会被重置
- Box位置和速度在重置时被清零
- Box与机器人共享同一个物理世界

## 使用方法

### 基本使用
```python
# 创建环境时，box会自动添加到每个环境中
env = IsaacGymEnv(...)

# 重置环境时，box位置会重新随机化
obs, privileged_obs = env.reset()

# 在仿真过程中，box会与机器人发生物理交互
action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
obs, privileged_obs, reward, reset_buf, info = env.step(action)
```

### 访问Box信息
```python
# 获取box actor句柄
box_handles = env.box_handles

# 每个环境都有一个box
for i, box_handle in enumerate(box_handles):
    print(f"环境 {i} 的box句柄: {box_handle}")
```

## 验证

运行验证脚本确认集成成功：
```bash
python mani-centric-wbc/verify_box_integration.py
```

## 注意事项

1. **性能影响**: 添加box会增加物理计算的复杂度，但影响很小
2. **内存使用**: 每个环境增加一个box actor，内存使用略有增加
3. **碰撞检测**: Box与机器人之间会发生碰撞，可能影响机器人的运动
4. **随机性**: Box位置是随机的，每次重置都会改变

## 扩展建议

1. **多个Box**: 可以修改代码添加多个box
2. **Box属性**: 可以调整box的尺寸、密度、颜色等
3. **Box任务**: 可以添加与box相关的任务，如推box到目标位置
4. **Box观察**: 可以将box位置添加到观察空间中

## 技术细节

### IsaacGym API使用
- `gym.create_box()`: 创建box asset
- `gym.create_actor()`: 创建box actor实例
- `gym.set_rigid_body_color()`: 设置box颜色
- `gym.set_actor_root_state_tensor_indexed()`: 设置box状态

### 坐标系
- Box位置相对于环境原点
- 机器人位置也相对于环境原点
- Box和机器人在同一个物理世界中

### 碰撞组
- Box使用碰撞组0
- 机器人与box可以发生碰撞
- 可以通过修改碰撞组来控制碰撞行为
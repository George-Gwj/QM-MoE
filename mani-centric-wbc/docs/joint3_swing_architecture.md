# Joint3摆动架构说明

## 修改概述

为了解决力矩冲突问题，我们将机械臂摆动的实现从**直接力矩控制**改为**action修改**的方式。

### 为什么需要修改env_add_baseinfo.py？

原始的架构中，callback在控制器计算力矩之前被调用，但callback无法修改action，导致：
1. **时机问题**: callback修改的action无法传递给控制器
2. **延迟问题**: 在decimation循环中，需要考虑action延迟的影响
3. **一致性问题**: 控制器使用的action与callback修改的action不一致

通过修改env_add_baseinfo.py，我们实现了：
1. **正确的时序**: callback在获取当前action后立即被调用
2. **action传递**: 修改后的action直接传递给控制器
3. **延迟处理**: 正确处理action buffer中的延迟索引

## 修改前后对比

### 修改前（有冲突）
```python
# 在 _joint3_swing_callback 中
# 1. 计算PD力矩
joint3_torques = kp * pos_error + kd * vel_error

# 2. 直接应用力矩（绕过控制器）
self.gym.set_dof_actuation_force_tensor(
    self.sim, gymtorch.unwrap_tensor(combined_torques)
)
```

**问题**: 直接调用 `set_dof_actuation_force_tensor` 绕过了控制器的力矩计算，与 `env_add_baseinfo.py` 中的控制器力矩生成冲突。

### 修改后（无冲突）
```python
# 在 _joint3_swing_callback 中
# 1. 计算目标角度
targets = safe_swing_center + safe_amplitude * torch.sin(...)

# 2. 修改action中的joint3值
modified_actions = actions.clone()
modified_actions[:, self.joint3_dof_index] = targets

# 3. 返回修改后的action
return modified_actions
```

**优势**: 通过修改action，让控制器正常生成力矩，避免了冲突。

## 新的控制流程

```
1. 用户调用 env.step(action)
2. 在父类step函数中，每个decimation步骤：
   a. 获取当前要使用的action（考虑延迟）
   b. 调用callback(env, current_action)
   c. callback修改action中joint3的值（摆动目标角度）
   d. 返回修改后的action
3. 控制器使用修改后的action计算力矩
4. 正常应用力矩到机器人
```

## 关键修改点

### 1. 函数签名变化
```python
# 修改前
def _joint3_swing_callback(self):

# 修改后  
def _joint3_swing_callback(self, actions: torch.Tensor) -> torch.Tensor:
```

### 2. 返回值变化
```python
# 修改前：无返回值，直接应用力矩
return

# 修改后：返回修改后的action
return modified_actions
```

### 3. 调用方式变化
```python
# 修改前
self._joint3_swing_callback()

# 修改后
modified_action = self._joint3_swing_callback(current_action)
```

### 4. Callback接口变化
```python
# 修改前
def combined_cb(_: IsaacGymEnv):

# 修改后
def combined_cb(env: IsaacGymEnv, current_action: torch.Tensor):
```

## 关键修改点

### 1. 函数签名变化
```python
# 修改前
def _joint3_swing_callback(self):

# 修改后  
def _joint3_swing_callback(self, actions: torch.Tensor) -> torch.Tensor:
```

### 2. 返回值变化
```python
# 修改前：无返回值，直接应用力矩
return

# 修改后：返回修改后的action
return modified_actions
```

### 3. 调用方式变化
```python
# 修改前
self._joint3_swing_callback()

# 修改后
modified_action = self._joint3_swing_callback(action)
```

## 优势

1. **无冲突**: 不再直接操作力矩，避免与控制器冲突
2. **统一控制**: 所有力矩都通过控制器生成，保持一致性
3. **易于调试**: 可以清楚地看到action的变化
4. **灵活配置**: 摆动参数可以通过配置文件调整

## 注意事项

1. **action范围**: 确保摆动目标角度在action的合理范围内
2. **控制器兼容性**: 控制器需要能够处理修改后的action值
3. **调试信息**: 可以通过 `debug_print` 参数控制调试输出

## 使用示例

```python
# 在play脚本中
actions = policy(obs)  # 获取策略输出
# action会自动通过callback修改joint3部分
obs, rews, dones, infos = env.step(actions)
```

这样的架构确保了机械臂摆动和控制器力矩生成的和谐共存！ 
import matplotlib.pyplot as plt
import numpy as np
# 加载日志
data = np.load("episode_logs.npz")

# 绘制距离奖励
plt.plot(data["distance_reward"])
plt.xlabel("Step")
plt.ylabel("Distance Reward")
plt.title("Distance Reward over Time")
plt.show()
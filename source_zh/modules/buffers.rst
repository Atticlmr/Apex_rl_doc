缓冲区
======

ApexRL 提供高效的缓冲区实现，用于存储和处理训练数据。

概述
----

可用的缓冲区类型：

1. **RolloutBuffer** - 同策略数据存储（PPO）
2. **ReplayBuffer** - 异策略数据存储（DQN、SAC - 计划中）
3. **DistillationBuffer** - 策略蒸馏数据

RolloutBuffer
-------------

``RolloutBuffer`` 存储环境交互期间收集的轨迹，用于同策略算法如 PPO。

关键特性
~~~~~~~~

- GPU 上的高效张量存储
- 支持多维观测
- 广义优势估计（GAE）
- 用于非对称 Actor-Critic 的特权观测

基本用法
~~~~~~~~

.. code-block:: python

   from apexrl.buffer.rollout_buffer import RolloutBuffer

   buffer = RolloutBuffer(
       num_envs=4096,
       num_steps=24,
       obs_shape=(48,),
       device="cuda",
       num_privileged_obs=0,
   )

   # 收集数据
   for step in range(24):
       actions, log_probs = actor.act(obs)
       next_obs, rewards, dones, extras = env.step(actions)
       values = critic.get_value(obs)
       
       buffer.add(
           observations=obs,
           privileged_observations=None,
           actions=actions,
           rewards=rewards,
           dones=dones.float(),
           values=values,
           log_probs=log_probs,
       )
       
       obs = next_obs

   # 计算优势
   last_values = critic.get_value(obs)
   buffer.compute_returns_and_advantages(
       last_values=last_values,
       gamma=0.99,
       gae_lambda=0.95,
   )

   # 获取训练数据
   data = buffer.get_all_data()

API 参考
~~~~~~~~

.. autoclass:: apexrl.buffer.rollout_buffer.RolloutBuffer
   :members:
   :undoc-members:
   :show-inheritance:

数据流
~~~~~~

Rollout 数据流：

.. code-block:: text

   环境步进 → 存储转移 → GAE 计算 → 训练
                    ↓
             ┌─────────────┐
             | observations |
             | actions      |
             | rewards      |
             | dones        |
             | values       |
             | log_probs    |
             └─────────────┘
                    ↓
             ┌─────────────┐
             | advantages  |
             | returns     |
             └─────────────┘

内存布局
~~~~~~~~

存储的张量形状为 ``(num_steps, num_envs, ...)``：

.. code-block:: python

   # 观测: (num_steps, num_envs, *obs_shape)
   self.observations  # 形状: (24, 4096, 48)
   
   # 标量: (num_steps, num_envs)
   self.rewards       # 形状: (24, 4096)
   self.dones         # 形状: (24, 4096)
   self.values        # 形状: (24, 4096)
   self.log_probs     # 形状: (24, 4096)
   self.advantages    # 形状: (24, 4096)
   self.returns       # 形状: (24, 4096)

GAE 计算
~~~~~~~~

广义优势估计反向计算：

.. math::

   \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots

其中 :math:`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)` 是 TD 误差。

.. code-block:: python

   def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
       advantages = torch.zeros_like(self.rewards)
       last_gae = torch.zeros(self.num_envs, device=self.device)
       
       for t in reversed(range(self.num_steps)):
           if t == self.num_steps - 1:
               next_values = last_values
           else:
               next_values = self.values[t + 1]
           
           delta = self.rewards[t] + gamma * next_values * (1 - self.dones[t]) - self.values[t]
           last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
           advantages[t] = last_gae
       
       self.advantages = advantages
       self.returns = advantages + self.values

ReplayBuffer
------------

用于异策略算法（DQN、SAC - 计划中）：

.. code-block:: python

   from apexrl.buffer.replay_buffer import ReplayBuffer

   buffer = ReplayBuffer(
       capacity=1_000_000,
       obs_shape=(4,),
       action_shape=(2,),
       device="cuda",
   )

   # 存储转移
   buffer.add(obs, action, reward, next_obs, done)

   # 采样批次
   batch = buffer.sample(batch_size=256)

API 参考
~~~~~~~~

.. autoclass:: apexrl.buffer.replay_buffer.ReplayBuffer
   :members:
   :undoc-members:
   :show-inheritance:

DistillationBuffer
------------------

用于策略蒸馏和模仿学习：

.. code-block:: python

   from apexrl.buffer.distillation_buffer import DistillationBuffer

   buffer = DistillationBuffer(
       capacity=100_000,
       obs_shape=(48,),
       device="cuda",
   )

   # 存储专家演示
   buffer.add(obs, action)

   # 采样用于蒸馏
   obs, expert_actions = buffer.sample(batch_size=256)

API 参考
~~~~~~~~

.. autoclass:: apexrl.buffer.distillation_buffer.DistillationBuffer
   :members:
   :undoc-members:
   :show-inheritance:

最佳实践
--------

1. **预分配**：缓冲区预先分配内存以提高效率
2. **设备放置**：保持缓冲区与模型在同一设备上
3. **清除缓冲区**：回合之间调用 ``clear()``
4. **批次大小**：确保批次大小能整除总转移数
5. **GAE Lambda**：典型值为 0.9-0.95

另请参阅
--------

- :doc:`../api/apexrl.buffer` - 完整 API 参考
- :doc:`../modules/algorithms` - 算法实现

自定义环境集成
==============

本教程展示如何将自定义环境与 ApexRL 集成，包括 Gymnasium 包装器和 GPU 加速环境。

概述
----

ApexRL 支持多种环境类型：

1. **Gymnasium 环境** - 标准单线程环境
2. **向量化环境** - 并行 CPU 环境
3. **GPU 环境** - CUDA 加速环境（Isaac Gym 等）

Gymnasium 包装器
---------------

对于标准 Gymnasium 环境，使用内置的 ``GymVecEnv``：

.. code-block:: python

   import gymnasium as gym
   from apexrl.envs.gym_wrapper import GymVecEnv

   def make_env(env_id="CartPole-v1"):
       def _init():
           env = gym.make(env_id)
           return env
       return _init

   # 创建向量化环境
   num_envs = 16
   env_fns = [make_env("CartPole-v1") for _ in range(num_envs)]
   env = GymVecEnv(env_fns, device="cpu")

自定义 VecEnv 实现
------------------

对于 GPU 加速环境（如 Isaac Gym），实现 ``VecEnv`` 接口：

.. code-block:: python

   import torch
   from typing import Dict, Tuple, Union
   from tensordict import TensorDict
   from apexrl.envs.vecenv import VecEnv

   class MyCustomVecEnv(VecEnv):
       """用于 ApexRL 的自定义向量化环境。
       
       本示例展示了基于 GPU 的机器人仿真的最小实现。
       """
       
       def __init__(self, num_envs: int = 4096, device: str = "cuda"):
           super().__init__(device=device)
           
           self.num_envs = num_envs
           self.num_obs = 48      # 观测维度
           self.num_actions = 12  # 动作维度
           self.max_episode_length = 1000
           
           # 初始化缓冲区
           self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(num_envs, device=device)
           self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
           self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=device)
           
           # 初始化仿真（此处放置您的物理引擎）
           self._init_simulation()
           
       def _init_simulation(self):
           """初始化您的物理仿真。"""
           pass
       
       def get_observations(self) -> torch.Tensor:
           """返回当前观测。"""
           return self.obs_buf
       
       def reset(self) -> torch.Tensor:
           """重置所有环境。"""
           self.obs_buf.zero_()
           self.rew_buf.zero_()
           self.reset_buf.zero_()
           self.episode_length_buf.zero_()
           
           # 重置仿真状态
           self._reset_all()
           
           return self.obs_buf
       
       def reset_idx(self, env_ids: torch.Tensor) -> None:
           """按索引重置特定环境。
           
           这对 GPU 环境至关重要，因为只有部分
           环境需要在回合终止后重置。
           """
           self.obs_buf[env_ids] = 0.0
           self.episode_length_buf[env_ids] = 0
           
           # 在仿真中重置特定环境
           self._reset_indices(env_ids)
       
       def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
           """执行一个仿真步。
           
           参数:
               actions: 要执行的动作。形状: (num_envs, num_actions)
               
           返回:
               (observations, rewards, dones, extras) 元组
           """
           # 应用动作到仿真
           self._apply_actions(actions)
           
           # 步进物理
           self._step_physics()
           
           # 计算观测和奖励
           self._compute_observations()
           self._compute_rewards()
           
           # 检查终止
           self.episode_length_buf += 1
           time_outs = self.episode_length_buf >= self.max_episode_length
           self.reset_buf = time_outs.clone()  # 或添加其他终止条件
           
           # 重置超时的环境
           if time_outs.any():
               self.reset_idx(torch.where(time_outs)[0])
           
           extras = {
               "time_outs": time_outs,
               "log": {
                   "/reward/mean": self.rew_buf.mean().item(),
                   "/episode_length/mean": self.episode_length_buf.float().mean().item(),
               }
           }
           
           return self.obs_buf, self.rew_buf, self.reset_buf, extras
       
       def _apply_actions(self, actions: torch.Tensor):
           """应用动作到仿真。"""
           pass
       
       def _step_physics(self):
           """步进物理仿真。"""
           pass
       
       def _compute_observations(self):
           """从仿真状态计算观测。"""
           pass
       
       def _compute_rewards(self):
           """基于仿真状态计算奖励。"""
           pass
       
       def _reset_all(self):
           """重置所有仿真实例。"""
           pass
       
       def _reset_indices(self, env_ids: torch.Tensor):
           """重置特定仿真实例。"""
           pass

特权观测（非对称 AC）
---------------------

对于非对称 Actor-Critic（策略和价值使用不同观测）：

.. code-block:: python

   class AsymmetricVecEnv(VecEnv):
       def __init__(self, ...):
           super().__init__(...)
           self.num_obs = 48              # 策略观测
           self.num_privileged_obs = 72   # Critic 观测（含特权信息）
           
       def get_observations(self):
           """返回策略观测。"""
           return self.obs_buf
       
       def get_privileged_observations(self):
           """返回 Critic 的特权观测。"""
           return self.privileged_obs_buf

环境包装器
----------

创建自定义包装器来修改环境行为：

.. code-block:: python

   from apexrl.envs.vecenv import VecEnvWrapper

   class RewardScalingWrapper(VecEnvWrapper):
       """按常数因子缩放奖励。"""
       
       def __init__(self, env: VecEnv, scale: float = 0.1):
           super().__init__(env)
           self.scale = scale
       
       def step(self, actions: torch.Tensor):
           obs, rewards, dones, extras = self.env.step(actions)
           rewards = rewards * self.scale
           return obs, rewards, dones, extras

   # 使用
   env = MyCustomVecEnv(num_envs=4096)
   env = RewardScalingWrapper(env, scale=0.01)

Isaac Gym 使用
--------------

Isaac Gym 集成示例：

.. code-block:: python

   from isaacgym import gymtorch, gymapi
   from apexrl.envs.vecenv import VecEnv

   class IsaacGymVecEnv(VecEnv):
       """Isaac Gym 环境的 ApexRL 包装器。"""
       
       def __init__(self, cfg, device="cuda"):
           super().__init__(device=device)
           
           # 初始化 Isaac Gym
           self.gym = gymapi.acquire_gym()
           self.sim = self._create_sim(cfg)
           
           # 创建环境
           self.num_envs = cfg.num_envs
           self._create_envs(cfg)
           
           # 设置缓冲区
           self.num_obs = cfg.num_obs
           self.num_actions = cfg.num_actions
           
           self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(self.num_envs, device=device)
           self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
       
       def step(self, actions):
           # 步进 Isaac Gym 物理
           self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(actions))
           self.gym.simulate(self.sim)
           self.gym.fetch_results(self.sim, True)
           
           # 计算观测和奖励
           self._compute_obs_rewards()
           
           # 检查重置
           self.reset_buf = self.episode_length_buf >= self.max_episode_length
           
           if self.reset_buf.any():
               self.reset_idx(torch.where(self.reset_buf)[0])
           
           extras = {"time_outs": self.reset_buf.clone(), "log": {}}
           return self.obs_buf, self.rew_buf, self.reset_buf, extras

最佳实践
--------

1. **预分配缓冲区**：在 ``__init__`` 中分配观测/奖励缓冲区
2. **使用 ``reset_idx``**：实现部分重置以提高效率
3. **处理超时**：正确设置 ``extras["time_outs"]``
4. **设备一致性**：确保所有张量在同一设备上
5. **日志记录**：添加有用的指标到 ``extras["log"]``

另请参阅
--------

- 学习 :doc:`custom_network` 架构
- 探索 :doc:`../modules/algorithms` 了解训练详情
- 查看 :doc:`../modules/environments` API 参考

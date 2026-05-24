自定义环境集成
==============

本教程展示如何把环境接入当前支持结构化观测的 ApexRL 栈。

概述
----

ApexRL 环境需要实现 ``VecEnv`` 接口，并返回：

- observations
- rewards
- done flags
- 带终止元数据的 ``extras``

对于 Gymnasium 任务，优先使用内置包装器：

- ``GymVecEnv`` 用于离散任务
- ``GymVecEnvContinuous`` 用于连续任务

标准 Gymnasium 接入
-------------------

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnv


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

结构化观测格式
--------------

对于多模态 actor 输入和 privileged critic 输入，推荐返回：

.. code-block:: python

   {
       "obs": {
           "image": image,
           "vector": vector,
       },
       "privileged_obs": {
           "state": state,
           "context": context,
       },
   }

这是当前版本推荐的格式，因为算法可以直接理解它：

- PPO actor 使用 ``obs``
- PPO 非对称 critic 使用 ``privileged_obs``
- SAC actor 使用 ``obs``
- SAC critic 在存在时使用 ``privileged_obs``

最小自定义 VecEnv
-----------------

.. code-block:: python

   import torch

   from apexrl.envs.vecenv import VecEnv


   class MyCustomVecEnv(VecEnv):
       def __init__(self, num_envs=1024, device="cuda"):
           super().__init__(device=device)
           self.num_envs = num_envs
           self.num_actions = 4
           self.max_episode_length = 200

           self.obs_buf = {
               "obs": {
                   "image": torch.zeros(num_envs, 1, 32, 32, device=device),
                   "vector": torch.zeros(num_envs, 8, device=device),
               },
               "privileged_obs": {
                   "state": torch.zeros(num_envs, 16, device=device),
               },
           }
           self.rew_buf = torch.zeros(num_envs, device=device)
           self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
           self.episode_length_buf = torch.zeros(
               num_envs,
               dtype=torch.int32,
               device=device,
           )

       def get_observations(self):
           return self.obs_buf

       def get_privileged_observations(self):
           return self.obs_buf["privileged_obs"]

       def reset(self):
           self.rew_buf.zero_()
           self.reset_buf.zero_()
           self.episode_length_buf.zero_()
           return self.obs_buf

       def reset_idx(self, env_ids):
           self.episode_length_buf[env_ids] = 0

       def step(self, actions):
           self.episode_length_buf += 1
           terminated = torch.zeros_like(self.reset_buf)
           truncated = self.episode_length_buf >= self.max_episode_length
           self.reset_buf = terminated | truncated

           final_obs = {
               "obs": {
                   "image": self.obs_buf["obs"]["image"].clone(),
                   "vector": self.obs_buf["obs"]["vector"].clone(),
               },
               "privileged_obs": {
                   "state": self.obs_buf["privileged_obs"]["state"].clone(),
               },
           }

           if self.reset_buf.any():
               self.reset_idx(torch.where(self.reset_buf)[0])

           extras = {
               "time_outs": truncated,
               "terminated": terminated,
               "truncated": truncated,
               "final_observation": final_obs,
               "log": {},
           }
           return self.obs_buf, self.rew_buf, self.reset_buf, extras

最佳实践
--------

1. 分别返回 ``terminated`` 和 ``truncated``。
2. 对截断回合始终提供 ``extras["final_observation"]``。
3. 把 actor 可见和 critic 可见观测分成不同分组。
4. 实现 ``reset_idx()`` 以支持高效部分重置。
5. 保持所有 buffer 与环境设备一致。

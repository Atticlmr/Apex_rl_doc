环境
====

ApexRL 为集成各种类型的强化学习环境提供了灵活的接口。

概述
----

ApexRL 设计用于：

1. **向量化环境** - GPU 加速并行环境
2. **Gymnasium 环境** - 标准单线程环境
3. **自定义环境** - 用户定义的仿真后端

VecEnv（向量化环境）
--------------------

``VecEnv`` 类是为 GPU 执行优化的向量化环境的基接口。

关键特性
~~~~~~~~

- 所有环境同步运行（相同的步进函数）
- 张量在 GPU 上预分配
- 支持部分重置以提高效率
- 为高通量训练而设计

接口
~~~~

.. code-block:: python

   class VecEnv(ABC):
       # 必需属性
       num_envs: int           # 并行环境数
       num_obs: int            # 观测维度
       num_actions: int        # 动作维度
       device: torch.device    # 执行设备
       
       # 必需方法
       def reset(self) -> obs
       def step(self, actions) -> (obs, rewards, dones, extras)
       def reset_idx(self, env_ids) -> None

API 参考
~~~~~~~~

.. autoclass:: apexrl.envs.vecenv.VecEnv
   :members:
   :undoc-members:
   :show-inheritance:

DummyVecEnv
~~~~~~~~~~~

简单的测试环境：

.. code-block:: python

   from apexrl.envs.vecenv import DummyVecEnv

   env = DummyVecEnv(
       num_envs=4096,
       num_obs=48,
       num_actions=12,
       device="cuda",
       max_episode_length=1000,
   )

Gymnasium 集成
--------------

ApexRL 为标准 Gymnasium 环境提供包装器。

GymVecEnv
~~~~~~~~~

包装多个 Gymnasium 环境：

.. code-block:: python

   import gymnasium as gym
   from apexrl.envs.gym_wrapper import GymVecEnv

   def make_env():
       return gym.make("Pendulum-v1")

   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

GymVecEnvContinuous
~~~~~~~~~~~~~~~~~~~

用于连续动作空间，自动动作缩放：

.. code-block:: python

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous

   env = GymVecEnvContinuous(
       [make_env for _ in range(8)],
       device="cpu",
       clip_actions=True,  # 裁剪到动作空间边界
   )

API 参考
~~~~~~~~

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnv
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnvContinuous
   :members:
   :undoc-members:
   :show-inheritance:

环境包装器
----------

VecEnvWrapper
~~~~~~~~~~~~~

创建环境包装器的基类：

.. code-block:: python

   from apexrl.envs.vecenv import VecEnvWrapper

   class NormalizeReward(VecEnvWrapper):
       def __init__(self, env):
           super().__init__(env)
           self.return_rms = RunningMeanStd()
           self.returns = torch.zeros(env.num_envs)
       
       def step(self, actions):
           obs, rewards, dones, extras = self.env.step(actions)
           
           # 更新运行统计
           self.returns = self.returns * gamma + rewards
           self.return_rms.update(self.returns)
           
           # 归一化奖励
           rewards = rewards / torch.sqrt(self.return_rms.var + 1e-8)
           
           return obs, rewards, dones, extras

API 参考
~~~~~~~~

.. autoclass:: apexrl.envs.vecenv.VecEnvWrapper
   :members:
   :undoc-members:
   :show-inheritance:

第三方集成
----------

Isaac Gym / Isaac Lab
~~~~~~~~~~~~~~~~~~~~~

集成示例模式：

.. code-block:: python

   from apexrl.envs.vecenv import VecEnv

   class IsaacLabVecEnv(VecEnv):
       def __init__(self, cfg, device="cuda"):
           from omni.isaac.lab.envs import ManagerBasedRLEnv
           
           self._env = ManagerBasedRLEnv(cfg)
           self.num_envs = self._env.num_envs
           self.num_obs = self._env.observation_space.shape[0]
           self.num_actions = self._env.action_space.shape[0]
           self.device = device
           
           # 初始化缓冲区
           self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(self.num_envs, device=device)
           self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
       
       def step(self, actions):
           obs_dict, rew, terminated, truncated, info = self._env.step(actions)
           
           self.obs_buf[:] = obs_dict["policy"]
           self.rew_buf[:] = rew
           self.reset_buf[:] = terminated | truncated
           
           extras = {
               "time_outs": truncated,
               "log": info.get("log", {}),
           }
           
           return self.obs_buf, self.rew_buf, self.reset_buf, extras
       
       def reset_idx(self, env_ids):
           self._env.reset_idx(env_ids)

Brax (JAX)
~~~~~~~~~~

用于基于 JAX 的环境：

.. code-block:: python

   import jax
   from jax import numpy as jnp

   class BraxVecEnv(VecEnv):
       def __init__(self, env_name, num_envs, device="cuda"):
           from brax import envs
           
           self._env = envs.create(env_name, batch_size=num_envs)
           self.num_envs = num_envs
           self.num_obs = self._env.observation_size
           self.num_actions = self._env.action_size
           
           # JIT 编译步进函数
           self._step = jax.jit(self._env.step)
           self._reset = jax.jit(self._env.reset)
           
           # 初始化状态
           rng = jax.random.PRNGKey(0)
           self._state = self._reset(rng)
       
       def step(self, actions):
           # PyTorch 转 JAX
           actions_jax = jax.device_put(actions.cpu().numpy())
           
           self._state = self._step(self._state, actions_jax)
           
           # 转回 PyTorch
           obs = torch.from_numpy(np.array(self._state.obs)).to(self.device)
           reward = torch.from_numpy(np.array(self._state.reward)).to(self.device)
           done = torch.from_numpy(np.array(self._state.done)).to(self.device)
           
           extras = {"time_outs": torch.zeros_like(done), "log": {}}
           
           return obs, reward, done, extras

最佳实践
--------

1. **预分配缓冲区**：在 ``__init__`` 中分配观测/奖励缓冲区
2. **使用 ``reset_idx``**：实现部分重置以提高效率
3. **处理超时**：正确设置 ``extras["time_outs"]``
4. **设备一致性**：确保所有张量在同一设备上
5. **日志记录**：添加有用的指标到 ``extras["log"]``

另请参阅
--------

- :doc:`../tutorials/custom_environment` - 详细集成教程
- :doc:`../api/apexrl.envs` - 完整 API 参考

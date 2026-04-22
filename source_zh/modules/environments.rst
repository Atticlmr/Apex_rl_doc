环境
====

ApexRL 为集成各种类型的强化学习环境提供了灵活的接口。

概述
----

ApexRL 设计用于：

1. **向量化环境** - GPU 加速并行环境
2. **Gymnasium 环境** - 标准单线程环境
3. **自定义环境** - 用户定义的仿真后端
4. **结构化观测环境** - TensorDict / 嵌套 dict 观测树

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
   :noindex:

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

ApexRL 为标准 Gymnasium 环境提供包装器。这些包装器支持普通张量观测、
结构化观测，以及显式的 timeout 元数据。

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

用于连续动作空间，自动裁剪并缩放到动作边界：

.. code-block:: python

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous

   env = GymVecEnvContinuous(
       [make_env for _ in range(8)],
       device="cpu",
       clip_actions=True,  # 裁剪到动作空间边界
   )

这是 Gymnasium ``Box`` 动作空间上运行 PPO 的推荐包装器。当前连续动作 PPO
默认使用未经过 ``tanh`` 压缩的高斯策略（``use_tanh_squash=False``），
由该包装器负责处理动作边界。

推荐的结构化观测格式：

.. code-block:: python

   {
       "obs": {
           "image": image,
           "vector": vector,
       },
       "privileged_obs": {
           "state": state,
       },
   }

API 参考
~~~~~~~~

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnv
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnvContinuous
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

第三方集成
----------

最佳实践
--------

1. **预分配缓冲区**：在 ``__init__`` 中分配观测/奖励缓冲区
2. **使用 ``reset_idx``**：实现部分重置以提高效率
3. **处理回合结束语义**：分别返回 ``terminated`` 和 ``truncated``
4. **提供最终状态**：对截断回合设置 ``extras["final_observation"]``
5. **设备一致性**：确保所有张量在同一设备上
6. **日志记录**：添加有用的指标到 ``extras["log"]``

另请参阅
--------

- :doc:`../tutorials/custom_environment` - 详细集成教程
- :doc:`../API/apexrl.envs` - 完整 API 参考

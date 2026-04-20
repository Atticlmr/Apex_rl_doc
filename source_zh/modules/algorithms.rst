算法
====

ApexRL 提供最先进的强化学习算法实现。

可用算法
--------

.. list-table::
   :header-rows: 1

   * - 算法
     - 类型
     - 状态
     - 描述
   * - PPO
     - 同策略
     - ✅ 可用
     - 近端策略优化
   * - DQN
     - 异策略
     - 🚧 计划中
     - 深度 Q 网络
   * - SAC
     - 异策略
     - 🚧 计划中
     - 软 Actor-Critic

PPO（近端策略优化）
-------------------

PPO 是 ApexRL 当前可用的主要算法。它是一种以稳定性和易用性著称的同策略算法。

关键特性
~~~~~~~~

- 用于稳定更新的裁剪替代目标
- 广义优势估计（GAE）
- 支持连续和离散动作
- 正确处理 ``terminated`` / ``truncated`` 的 timeout bootstrap
- 非对称 Actor-Critic（Critic 有特权信息）
- 策略/价值分离或联合优化器

基本用法
~~~~~~~~

.. code-block:: python

   from apexrl.algorithms.ppo import PPO, PPOConfig
   from apexrl.envs.vecenv import DummyVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   # 创建环境
   env = DummyVecEnv(num_envs=4096, num_obs=48, num_actions=12)

   # 配置 PPO
   cfg = PPOConfig(
       num_steps=24,
       num_epochs=5,
       learning_rate=3e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
   )

   # 创建智能体
   agent = PPO(
       env=env,
       cfg=cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   # 训练
   # PPO.learn() 是 OnPolicyRunner 的轻量封装。
   agent.learn(total_timesteps=10_000_000)

对新项目，建议优先把 ``OnPolicyRunner`` 作为主训练入口，
把 ``PPO`` 看作接入该 runner 的算法实现。

配置
~~~~

.. autoclass:: apexrl.algorithms.ppo.config.PPOConfig
   :members:
   :undoc-members:

API 参考
~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.ppo.PPO
   :members:
   :undoc-members:
   :show-inheritance:

算法详情
--------

PPO-Clip 目标
~~~~~~~~~~~~~

PPO-Clip 目标函数：

.. math::

   L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]

其中：

- :math:`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}` 是概率比率
- :math:`\hat{A}_t` 是估计优势
- :math:`\epsilon` 是裁剪范围（通常为 0.2）

总损失函数
~~~~~~~~~~

.. math::

   L^{TOTAL}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)

其中：

- :math:`L^{VF}` 是价值函数损失（MSE）
- :math:`S` 是熵奖励
- :math:`c_1`, :math:`c_2` 是系数

超参数调优
----------

一般准则
~~~~~~~~

.. list-table:: PPO 超参数
   :header-rows: 1

   * - 参数
     - 典型范围
     - 描述
   * - ``num_steps``
     - 2048-8192
     - 每次更新每环境的步数
   * - ``num_epochs``
     - 3-10
     - 每批数据优化轮数
   * - ``learning_rate``
     - 1e-5 到 1e-3
     - 优化步长
   * - ``gamma``
     - 0.99-0.999
     - 折扣因子
   * - ``gae_lambda``
     - 0.9-0.99
     - GAE lambda 参数
   * - ``clip_range``
     - 0.1-0.3
     - 裁剪参数
   * - ``ent_coef``
     - 0.0-0.01
     - 熵系数
   * - ``vf_coef``
     - 0.25-1.0
     - 价值函数损失系数

特定环境推荐
~~~~~~~~~~~~

**Isaac Gym（足式机器人）：**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=24,
       num_epochs=5,
       learning_rate=1e-3,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.0,
       batch_size=98304,
       minibatch_size=32768,
   )

**Gymnasium（Atari）：**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=128,
       num_epochs=4,
       learning_rate=2.5e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.1,
       ent_coef=0.01,
   )

**Gymnasium（Mujoco）：**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=2048,
       num_epochs=10,
       learning_rate=3e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.0,
   )

对于连续控制任务，默认 PPO 配置使用 ``use_tanh_squash=False``，
并通过 ``min_log_std`` / ``max_log_std`` 限制学习到的标准差范围，
以保持数值稳定性。

高级特性
--------

学习率调度
~~~~~~~~~~

ApexRL 支持多种学习率调度：

.. code-block:: python

   cfg = PPOConfig(
       learning_rate_schedule="adaptive",  # 或 "linear", "constant"
       max_learning_rate=1e-3,
       min_learning_rate=1e-5,
   )

- **constant**：固定学习率
- **linear**：从初始值线性衰减到 0
- **adaptive**：自定义衰减调度

价值函数裁剪
~~~~~~~~~~~~

启用价值函数裁剪以获得更稳定的训练：

.. code-block:: python

   cfg = PPOConfig(
       clip_range_vf=0.2,  # None 表示禁用
   )

早停
~~~~

当 KL 散度超过阈值时停止更新：

.. code-block:: python

   cfg = PPOConfig(
       target_kl=0.015,  # None 表示禁用
   )

分离优化器
~~~~~~~~~~

为策略和价值使用不同的学习率：

.. code-block:: python

   cfg = PPOConfig(
       use_policy_optimizer=True,
       policy_learning_rate=1e-4,
       value_learning_rate=3e-4,
   )

另请参阅
--------

- :doc:`../tutorials/first_agent` - 基础使用教程
- :doc:`../tutorials/custom_network` - 自定义网络架构
- :doc:`../api/apexrl.algorithms.ppo` - 完整 API 参考

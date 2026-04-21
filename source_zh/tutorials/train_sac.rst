如何训练 SAC
============

本教程展示如何在 ApexRL 中使用标准流程训练 SAC 智能体。

概览
----

当前仓库中推荐的 SAC 训练组合是：

- ``GymVecEnvContinuous`` 用于连续控制 Gymnasium 任务
- ``OffPolicyRunner`` 作为标准训练入口
- ``MLPSquashedGaussianActor`` 作为默认随机策略
- ``MLPContinuousQNetwork`` 作为默认 twin critic 基线

前置要求
--------

安装 ApexRL 和 Gymnasium：

.. code-block:: bash

   pip install -e .

环境准备
--------

对于 SAC，推荐先从 ``Pendulum-v1`` 这类连续动作任务开始。

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(2)], device="cpu")

构建 Runner
-----------

``OffPolicyRunner`` 会创建 SAC agent、填充 replay，并调度
actor / critic / temperature 更新。

.. code-block:: python

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.sac import SACConfig

   cfg = SACConfig(
       batch_size=256,
       buffer_size=100_000,
       learning_starts=5_000,
       actor_learning_rate=3e-4,
       critic_learning_rate=3e-4,
       alpha_learning_rate=3e-4,
       tau=0.005,
       log_interval=1_000,
       save_interval=10_000,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
       save_dir="./checkpoints/sac_pendulum",
   )

训练
----

.. code-block:: python

   runner.learn(total_timesteps=100_000)

评估与保存
----------

.. code-block:: python

   stats = runner.eval(num_episodes=10)
   print(f"平均奖励: {stats['eval/mean_reward']:.2f}")

   runner.save_checkpoint("sac_pendulum_final.pt")
   env.close()

完整示例
--------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.sac import SACConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(2)], device="cpu")

   cfg = SACConfig(
       batch_size=256,
       buffer_size=100_000,
       learning_starts=5_000,
       actor_learning_rate=3e-4,
       critic_learning_rate=3e-4,
       alpha_learning_rate=3e-4,
       tau=0.005,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("sac_pendulum_final.pt")
   env.close()

SAC 实际优化什么
----------------

SAC 同时更新三组目标：

.. math::

   y = r + \gamma (1-d)\left(\min(Q_1'(s', a'), Q_2'(s', a')) - \alpha \log \pi(a'|s')\right)

.. math::

   L_{Q_i} = \mathbb{E}\left[(Q_i(s, a) - y)^2\right]

.. math::

   L_{\pi} = \mathbb{E}\left[\alpha \log \pi(a|s) - \min(Q_1(s, a), Q_2(s, a))\right]

.. math::

   L_{\alpha} = -\mathbb{E}\left[\log \alpha \cdot (\log \pi(a|s) + \mathcal{H}_{target})\right]

在 ApexRL 中，这些公式已经直接写进
:doc:`../API/apexrl.algorithms.sac` 对应实现附近的注释里。

说明
----

- 当前 SAC 只支持 ``Box`` 观测和 ``Box`` 动作空间。
- 默认策略是 squashed Gaussian actor，不是 PPO 那种 unsquashed Gaussian。
- ``OffPolicyRunner`` 是推荐训练入口。
- ``SAC.learn()`` 仍可直接使用，但本质上也是同一个 runner 的轻量封装。

下一步
------

- 阅读 :doc:`train_ppo` 了解同策略训练流程
- 阅读 :doc:`train_dqn` 了解离散异策略训练流程
- 阅读 :doc:`../modules/algorithms` 查看 SAC 细节
- 阅读 :doc:`../modules/runners` 查看 runner API

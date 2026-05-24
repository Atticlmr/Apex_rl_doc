如何训练 DQN
================

本教程展示如何在 ApexRL 中使用标准流程训练 DQN 智能体。

概述
----

当前仓库中推荐的 DQN 训练组合是：

- ``GymVecEnv``：用于离散动作 Gymnasium 环境
- ``OffPolicyRunner``：标准训练入口
- ``MLPQNetwork``：默认 Q 网络基线

前置条件
--------

安装 ApexRL 和 Gymnasium：

.. code-block:: bash

   pip install -e .

环境准备
--------

DQN 推荐先从 ``CartPole-v1`` 这类离散动作任务开始。

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnv


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(2)], device="cpu")

构建 Runner
-----------

``OffPolicyRunner`` 会创建 DQN agent、填充 replay buffer，并调度参数更新。

.. code-block:: python

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.models import MLPQNetwork

   cfg = DQNConfig(
       batch_size=128,
       buffer_size=100_000,
       learning_starts=1_000,
       target_update_interval=250,
       double_dqn=True,
       dueling=True,
       log_interval=1_000,
       save_interval=10_000,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="dqn",
       q_network_class=MLPQNetwork,
       log_dir="./logs/dqn_cartpole",
       save_dir="./checkpoints/dqn_cartpole",
   )

开始训练
--------

.. code-block:: python

   runner.learn(total_timesteps=50_000)

评估与保存
----------

.. code-block:: python

   stats = runner.eval(num_episodes=10)
   print(f"平均奖励: {stats['eval/mean_reward']:.2f}")

   runner.save_checkpoint("dqn_cartpole_final.pt")
   env.close()

完整示例
--------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPQNetwork


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(2)], device="cpu")

   cfg = DQNConfig(
       batch_size=128,
       buffer_size=100_000,
       learning_starts=1_000,
       target_update_interval=250,
       double_dqn=True,
       dueling=True,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="dqn",
       q_network_class=MLPQNetwork,
       log_dir="./logs/dqn_cartpole",
   )

   runner.learn(total_timesteps=50_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("dqn_cartpole_final.pt")
   env.close()

说明
----

- ``OffPolicyRunner`` 是 DQN 的推荐训练入口。
- ``double_dqn=True`` 默认建议保持开启。
- 设置 ``dueling=True`` 可以把 ``MLPQNetwork`` 切换成 dueling 结构。

下一步
------

- 阅读 :doc:`train_ppo` 了解同策略训练流程
- 阅读 :doc:`train_sac` 了解连续控制异策略训练流程
- 阅读 :doc:`../modules/algorithms` 查看 DQN 相关配置
- 阅读 :doc:`../modules/runners` 查看 runner API 细节

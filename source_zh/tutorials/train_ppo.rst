如何训练 PPO
================

本教程展示如何在 ApexRL 中使用标准流程训练 PPO 智能体。

概述
----

当前仓库中推荐的 PPO 训练组合是：

- ``GymVecEnvContinuous``：用于连续动作 Gymnasium 环境
- ``OnPolicyRunner``：标准训练入口
- ``MLPActor`` 和 ``MLPCritic``：默认基线网络

前置条件
--------

安装 ApexRL 和 Gymnasium：

.. code-block:: bash

   pip install -e .

环境准备
--------

连续动作 PPO 推荐使用 ``GymVecEnvContinuous``，由环境包装器统一处理动作裁剪
和 timeout 元数据。

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

构建 Runner
-----------

``OnPolicyRunner`` 会创建 PPO agent，并负责外层训练循环。

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.models import MLPActor, MLPCritic

   cfg = PPOConfig(
       num_steps=256,
       num_epochs=5,
       learning_rate=3e-4,
       log_interval=10,
       save_interval=100,
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/ppo_pendulum",
       save_dir="./checkpoints/ppo_pendulum",
   )

开始训练
--------

.. code-block:: python

   runner.learn(total_timesteps=100_000)

评估与保存
----------

.. code-block:: python

   stats = runner.eval(num_episodes=10)
   print(f"平均奖励: {stats['eval/mean_reward']:.2f}")

   runner.save_checkpoint("ppo_pendulum_final.pt")
   env.close()

完整示例
--------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous
   from apexrl.models import MLPActor, MLPCritic


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

   cfg = PPOConfig(
       num_steps=256,
       num_epochs=5,
       learning_rate=3e-4,
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/ppo_pendulum",
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("ppo_pendulum_final.pt")
   env.close()

说明
----

- 连续动作 PPO 默认使用未经过 ``tanh`` 压缩的高斯策略。
- ``PPO.learn()`` 仍然可用，但更推荐用 ``OnPolicyRunner``。
- 如果需要自定义网络，只需要保持 runner 接口不变，替换
  ``MLPActor`` / ``MLPCritic`` 即可。

下一步
------

- 阅读 :doc:`train_dqn` 了解异策略训练流程
- 阅读 :doc:`custom_network` 了解自定义网络
- 阅读 :doc:`../modules/runners` 查看 runner API 细节

如何训练 SAC
============

本教程展示 ApexRL 当前版本中的 SAC 工作流。

概述
----

推荐组合：

- ``GymVecEnvContinuous`` 用于连续 Gymnasium 任务
- ``OffPolicyRunner`` 作为标准训练入口
- ``MLPSquashedGaussianActor`` 作为默认 actor
- ``MLPContinuousQNetwork`` 作为默认 twin-critic 基线

标准示例
--------

.. code-block:: python

   import gymnasium as gym
   import torch

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
       device="cpu",
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
       save_dir="./checkpoints/sac_pendulum",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.close()

结构化观测
----------

SAC 仍然要求 ``Box`` 动作空间，但观测已经不需要再是单个扁平张量。

当前实现支持结构化 actor 观测，以及可选的 critic-only privileged observations：

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

现在 SAC 内部会：

- 把 ``obs`` 传给 actor
- 当存在 ``privileged_obs`` 时，把它传给两个 critic
- 在 replay 中分别存储 actor 和 critic 的观测分支

说明
----

- SAC 支持 ``Box`` 动作空间
- 观测可以是普通张量，也可以是结构化 ``TensorDict`` / 嵌套 dict
- 默认策略是 squashed Gaussian actor，不同于 PPO 的 unsquashed Gaussian
- ``OffPolicyRunner`` 仍然是推荐入口

下一步
------

- 阅读 :doc:`custom_network` 了解自定义 actor / critic
- 阅读 :doc:`custom_environment` 了解结构化观测环境设计
- 阅读 :doc:`../modules/algorithms` 查看 SAC 细节

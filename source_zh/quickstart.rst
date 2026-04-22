快速开始
========

本指南展示 ApexRL 当前版本推荐的训练入口。

安装
----

从源码安装：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

或者使用 ``uv``：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

核心依赖：

- Python >= 3.11
- PyTorch >= 2.0
- Gymnasium >= 0.29
- TensorDict >= 0.6

训练入口
--------

- ``OnPolicyRunner`` 是 PPO 的标准入口
- ``OffPolicyRunner`` 是 DQN 和 SAC 的标准入口
- ``PPO.learn()``、``DQN.learn()``、``SAC.learn()`` 仍然可用，但只是轻量封装

第一个 PPO 智能体
-----------------

离散动作任务：

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPDiscreteActor, MLPCritic


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(device="cpu", learning_rate_schedule="constant"),
       actor_class=MLPDiscreteActor,
       critic_class=MLPCritic,
       log_dir="./logs/cartpole_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   runner.close()

连续动作任务：

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous
   from apexrl.models import MLPActor, MLPCritic


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(device="cpu"),
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/pendulum_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   runner.close()

结构化观测
----------

当前版本已经支持结构化观测贯穿环境包装器、buffer、算法和默认 MLP 模型。

推荐的环境输出格式：

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

在这种格式下：

- actor 接收 ``obs``
- PPO 在 ``use_asymmetric=True`` 时会把 ``privileged_obs`` 传给 critic
- SAC 会在 replay 中分别存储 actor 和 critic 的观测分支

下一步
------

- 阅读 :doc:`tutorials/train_ppo` 了解标准 PPO 流程
- 阅读 :doc:`tutorials/train_dqn` 了解标准 DQN 流程
- 阅读 :doc:`tutorials/train_sac` 了解标准 SAC 流程
- 阅读 :doc:`tutorials/custom_network` 了解多模态自定义 actor / critic
- 阅读 :doc:`tutorials/custom_environment` 了解 TensorDict 环境接入方式

快速开始
========

本指南将帮助您在几分钟内开始使用 ApexRL。

安装
----

从源代码安装 ApexRL：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

或者使用 uv（更快）：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

环境要求
--------

- Python >= 3.11
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0

您的第一个 RL 智能体
--------------------

以下是一个在 Gymnasium 环境上训练 PPO 智能体的最小示例：

.. code-block:: python

   import gymnasium as gym
   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   # 创建向量化环境
   def make_env():
       return gym.make("Pendulum-v1")

   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

   # 使用默认 PPO 配置创建 runner
   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs",
   )

   # 训练 100,000 步
   runner.learn(total_timesteps=100_000)

   # 保存训练好的模型
   runner.save_checkpoint("model_final.pt")

   env.close()

使用自定义网络
--------------

您可以轻松定义自定义 Actor 和 Critic 网络：

.. code-block:: python

   from apexrl.models.base import ContinuousActor, Critic
   import torch.nn as nn

   class CustomActor(ContinuousActor):
       def __init__(self, obs_space, action_space, cfg):
           super().__init__(obs_space, action_space, cfg)
           
           obs_dim = obs_space.shape[0]
           action_dim = action_space.shape[0]
           
           self.network = nn.Sequential(
               nn.Linear(obs_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
               nn.Linear(256, action_dim),
           )
           self.log_std = nn.Parameter(torch.zeros(action_dim))
       
       def get_action_dist(self, obs):
           mean = self.network(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

   # 使用您的自定义 actor
   runner = OnPolicyRunner(
       env=env,
       actor_class=CustomActor,
       critic_class=MLPCritic,
       # ... 其他参数
   )

下一步
------

- 阅读 :doc:`tutorials/first_agent` 了解详细教程
- 探索 :doc:`modules/algorithms` 了解可用算法
- 查看 :doc:`modules/environments` 了解环境集成
- 阅读 :doc:`tutorials/custom_network` 了解高级网络架构

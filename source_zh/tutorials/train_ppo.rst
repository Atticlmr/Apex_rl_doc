如何训练 PPO
============

本教程展示 ApexRL 当前版本推荐的 PPO 工作流。

概述
----

推荐组合：

- 离散 Gymnasium 任务使用 ``GymVecEnv``
- 连续 Gymnasium 任务使用 ``GymVecEnvContinuous``
- ``OnPolicyRunner`` 作为标准训练入口
- ``MLPDiscreteActor`` / ``MLPActor`` 和 ``MLPCritic`` 作为默认基线

离散 PPO 示例
-------------

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

   cfg = PPOConfig(
       num_steps=128,
       num_epochs=4,
       minibatch_size=256,
       learning_rate=3e-4,
       learning_rate_schedule="constant",
       device="cpu",
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       actor_class=MLPDiscreteActor,
       critic_class=MLPCritic,
       log_dir="./logs/cartpole_ppo",
       save_dir="./checkpoints/cartpole_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.close()

连续 PPO 示例
-------------

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

非对称 Critic 与结构化观测
--------------------------

PPO 现在已经支持结构化观测和 privileged critic observations。

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

开启非对称 critic：

.. code-block:: python

   cfg = PPOConfig(use_asymmetric=True, device="cpu")

runner 和算法会自动：

- 把 ``obs`` 传给 actor
- 把 ``privileged_obs`` 传给 critic
- 在 rollout buffer 里保持相同结构

自定义网络
----------

如果要替换默认基线，只需要保持 runner 接口不变，替换
``actor_class`` / ``critic_class``。

对于多模态 actor，actor 通常接收到的已经是拆分后的 ``obs`` 分支，
例如 ``{"image": ..., "vector": ...}``。

.. code-block:: python

   import torch
   import torch.nn as nn

   from apexrl.models.base import DiscreteActor


   class MultiModalDiscreteActor(DiscreteActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)

           image_shape = obs_space["image"].shape
           vector_dim = obs_space["vector"].shape[0]
           hidden_dim = (cfg or {}).get("hidden_dim", 256)

           self.image_encoder = nn.Sequential(
               nn.Conv2d(image_shape[0], 16, 3, stride=2, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 32, 3, stride=2, padding=1),
               nn.ReLU(),
               nn.Flatten(),
           )

           with torch.no_grad():
               dummy = torch.zeros(1, *image_shape)
               image_dim = self.image_encoder(dummy).shape[-1]

           self.vector_encoder = nn.Sequential(
               nn.Linear(vector_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 64),
               nn.ReLU(),
           )

           self.head = nn.Sequential(
               nn.Linear(image_dim + 64, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, self.num_actions),
           )

       def forward(self, obs):
           image_feat = self.image_encoder(obs["image"])
           vector_feat = self.vector_encoder(obs["vector"])
           return self.head(torch.cat([image_feat, vector_feat], dim=-1))

       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)


   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(use_asymmetric=True, device="cpu"),
       actor_class=MultiModalDiscreteActor,
       critic_class=MLPCritic,
       actor_cfg={"hidden_dim": 256},
   )

说明
----

- ``OnPolicyRunner`` 是推荐的 PPO 入口
- 连续动作 PPO 默认使用未经过 ``tanh`` 压缩的高斯策略
- ``GymVecEnvContinuous`` 负责裁剪并映射到 Gymnasium 动作边界
- ``PPO.learn()`` 仍然可用，但本质上会委托给同一个 runner

下一步
------

- 阅读 :doc:`custom_network` 了解更多网络模式
- 阅读 :doc:`custom_environment` 了解 TensorDict 环境接入
- 阅读 :doc:`../modules/runners` 查看 runner API 细节

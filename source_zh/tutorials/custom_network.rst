自定义网络架构
==============

本教程展示如何在当前支持 TensorDict 的 ApexRL 栈中实现自定义 actor 和 critic。

概述
----

主要基类：

- ``ContinuousActor`` 用于连续策略
- ``DiscreteActor`` 用于离散策略
- ``Critic`` 用于价值网络
- ``ContinuousQNetwork`` 用于 SAC 风格的 ``Q(s, a)`` critic

网络实际会接收到什么
--------------------

当前版本已经支持结构化观测。

如果环境返回：

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

那么：

- PPO actor 接收到的是 ``{"image": ..., "vector": ...}``
- PPO 非对称 critic 接收到的是 ``{"state": ..., "context": ...}``
- SAC actor 接收到的是 ``obs``
- SAC critic 在存在 ``privileged_obs`` 时会接收到它，否则接收 actor 观测

连续 Actor 示例
---------------

.. code-block:: python

   import torch
   import torch.nn as nn

   from apexrl.models.base import ContinuousActor


   class MultiModalContinuousActor(ContinuousActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           cfg = cfg or {}

           image_shape = obs_space["image"].shape
           vector_dim = obs_space["vector"].shape[0]
           hidden_dim = cfg.get("hidden_dim", 256)

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

           self.backbone = nn.Sequential(
               nn.Linear(image_dim + 64, hidden_dim),
               nn.ReLU(),
           )
           self.mean_head = nn.Linear(hidden_dim, self.action_dim)
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))

       def forward(self, obs):
           image_feat = self.image_encoder(obs["image"])
           vector_feat = self.vector_encoder(obs["vector"])
           fused = torch.cat([image_feat, vector_feat], dim=-1)
           return self.mean_head(self.backbone(fused))

       def get_action_dist(self, obs):
           mean = self.forward(obs)
           std = torch.exp(self.log_std).expand_as(mean)
           return torch.distributions.Normal(mean, std)

离散 Actor 示例
---------------

.. code-block:: python

   from apexrl.models.base import DiscreteActor


   class MultiModalDiscreteActor(DiscreteActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           image_shape = obs_space["image"].shape
           vector_dim = obs_space["vector"].shape[0]

           self.image_encoder = nn.Sequential(
               nn.Conv2d(image_shape[0], 16, 3, stride=2, padding=1),
               nn.ReLU(),
               nn.Flatten(),
           )

           with torch.no_grad():
               dummy = torch.zeros(1, *image_shape)
               image_dim = self.image_encoder(dummy).shape[-1]

           self.head = nn.Sequential(
               nn.Linear(image_dim + vector_dim, 256),
               nn.ReLU(),
               nn.Linear(256, self.num_actions),
           )

       def forward(self, obs):
           image_feat = self.image_encoder(obs["image"])
           return self.head(torch.cat([image_feat, obs["vector"]], dim=-1))

       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)

Critic 示例
-----------

对于 privileged critic：

.. code-block:: python

   import torch
   import torch.nn as nn

   from apexrl.models.base import Critic


   class PrivilegedCritic(Critic):
       def __init__(self, obs_space, cfg=None):
           super().__init__(obs_space, cfg)

           state_dim = obs_space["state"].shape[0]
           context_dim = obs_space["context"].shape[0]

           self.network = nn.Sequential(
               nn.Linear(state_dim + context_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
               nn.Linear(256, 1),
           )

       def forward(self, obs):
           x = torch.cat([obs["state"], obs["context"]], dim=-1)
           return self.network(x).squeeze(-1)

       def get_value(self, obs):
           return self.forward(obs)

如何接入自定义网络
------------------

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig

   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(use_asymmetric=True, device="cpu"),
       actor_class=MultiModalDiscreteActor,
       critic_class=PrivilegedCritic,
       actor_cfg={"hidden_dim": 256},
       log_dir="./logs",
   )

最佳实践
--------

1. 保持观测结构显式，不要一开始就把所有模态强行展平。
2. 让环境通过 ``obs`` 和 ``privileged_obs`` 暴露 actor / critic 分组。
3. 离散 PPO 使用 ``DiscreteActor``，连续 PPO 使用 ``ContinuousActor``。
4. 对 SAC，自定义 critic 时保持 ``ContinuousQNetwork`` 形式，即 ``forward(obs, actions)``。
5. 对图像 + 向量输入，优先使用分支编码器而不是单个扁平 MLP。

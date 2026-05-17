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

多智能体自定义网络
------------------

MAPPO、IPPO 和 HAPPO 使用与 PPO 相同的自定义网络模式：传入 ApexRL actor /
critic 类，并按需传入配置字典。网络内部可以包含任意 PyTorch 模块，例如 MLP、
CNN、attention、GNN 或 transformer block，同时继承 ApexRL 的 actor / critic 基类。

.. code-block:: python

   from apexrl.models.base import Critic, DiscreteActor
   from apexrl.multiagent import HAPPO, HAPPOConfig, IPPO, IPPOConfig, MAPPO, MAPPOConfig


   class EntityDiscreteActor(DiscreteActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           # 在这里构建 attention、graph、CNN 或 MLP 模块。

       def forward(self, obs):
           ...

       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)


   class EntityCritic(Critic):
       def __init__(self, obs_space, cfg=None):
           super().__init__(obs_space, cfg)
           # 在这里构建 value 网络。

       def forward(self, obs):
           ...

       def get_value(self, obs):
           return self.forward(obs)


   mappo_agent = MAPPO(
       env=multiagent_env,
       cfg=MAPPOConfig(
           centralized_critic=True,
           share_actor=True,
           share_critic=True,
       ),
       actor_class=EntityDiscreteActor,
       critic_class=EntityCritic,
       actor_cfg={"hidden_dim": 256},
       critic_cfg={"hidden_dim": 512},
   )

   ippo_agent = IPPO(
       env=multiagent_env,
       cfg=IPPOConfig(share_actor=True, share_critic=True),
       actor_class=EntityDiscreteActor,
       critic_class=EntityCritic,
   )

   happo_agent = HAPPO(
       env=multiagent_env,
       cfg=HAPPOConfig(centralized_critic=True, share_actor=False),
       actor_class=EntityDiscreteActor,
       critic_class=EntityCritic,
   )

当 MAPPO 使用 ``centralized_critic=True`` 时，critic 接收的是
``env.state_space`` 和 ``env.get_state()`` 返回的全局 state。IPPO，或设置
``centralized_critic=False`` 的 MAPPO，则让每个 critic 接收对应 agent 的局部
观测空间和局部观测。actor 始终接收每个 agent 自己的局部观测。

参数共享由 multi-agent 配置控制：

- ``share_actor=True`` 会创建一个 actor 实例并复用于所有 agent。此时要求所有
  agent 的 observation/action space 一致。
- HAPPO 使用 ``share_actor=False``，让每个 agent 拥有独立策略以进行顺序更新。
- ``share_critic=True`` 会创建一个 critic 实例并复用于所有 agent。使用分散式
  critic 时要求所有 observation space 一致；使用集中式 critic 时所有 critic
  都输入同一个 state space。
- 如果不同 agent 需要不同参数，把对应开关设为 ``False``。

也可以通过 ``models`` 字典传入已经构建好的 ApexRL actor / critic 对象。这适合
异构 agent，或者需要手动控制哪些模块共享：

.. code-block:: python

   models = {
       "agent_0": {"policy": actor_0, "value": critic_0},
       "agent_1": {"policy": actor_1, "value": critic_1},
   }

   agent = MAPPO(
       possible_agents=["agent_0", "agent_1"],
       observation_spaces=observation_spaces,
       action_spaces=action_spaces,
       state_space=state_space,
       models=models,
       cfg=MAPPOConfig(centralized_critic=True),
   )

对于更复杂的多智能体网络，建议把实体表示成固定结构的观测，例如
``spaces.Dict`` 字段、padding 后的 tensor 和 mask。这样 attention、DeepSets、
GNN、transformer encoder 可以直接写在 actor 或 critic 内部，不需要改
MAPPO/IPPO/HAPPO。

最佳实践
--------

1. 保持观测结构显式，不要一开始就把所有模态强行展平。
2. 让环境通过 ``obs`` 和 ``privileged_obs`` 暴露 actor / critic 分组。
3. 离散 PPO 使用 ``DiscreteActor``，连续 PPO 使用 ``ContinuousActor``。
4. 对 SAC，自定义 critic 时保持 ``ContinuousQNetwork`` 形式，即 ``forward(obs, actions)``。
5. 对图像 + 向量输入，优先使用分支编码器而不是单个扁平 MLP。
6. 对 MAPPO/IPPO/HAPPO，actor 使用每个 agent 的局部观测，critic 输入由算法
   配置选择集中式 state 或分散式局部观测。

自定义网络架构
==============

本教程介绍如何在 ApexRL 中创建自定义 Actor（策略）和 Critic（价值）网络。

概述
----

ApexRL 为定义自定义神经网络提供了灵活的基类：

- ``ContinuousActor``：用于连续动作空间（高斯分布）
- ``DiscreteActor``：用于离散动作空间（分类分布）
- ``Critic``：用于价值函数估计

连续 Actor
----------

用于连续控制任务：

.. code-block:: python

   import torch
   import torch.nn as nn
   from apexrl.models.base import ContinuousActor

   class MLPActor(ContinuousActor):
       """用于连续动作的简单 MLP Actor。"""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           cfg = cfg or {}
           hidden_dims = cfg.get("hidden_dims", [256, 256])
           activation = cfg.get("activation", "relu")
           
           # 构建网络
           layers = []
           input_dim = obs_space.shape[0]
           
           for hidden_dim in hidden_dims:
               layers.append(nn.Linear(input_dim, hidden_dim))
               layers.append(nn.ReLU() if activation == "relu" else nn.ELU())
               input_dim = hidden_dim
           
           layers.append(nn.Linear(input_dim, self.action_dim))
           self.network = nn.Sequential(*layers)
           
           # 可学习的对数标准差
           init_std = cfg.get("init_std", 1.0)
           self.log_std = nn.Parameter(torch.ones(self.action_dim) * torch.log(torch.tensor(init_std)))
           
           # 初始化权重
           self.apply(self._init_weights)
       
       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               nn.init.orthogonal_(module.weight, gain=1.0)
               nn.init.constant_(module.bias, 0.0)
       
       def forward(self, obs):
           """返回动作均值。"""
           return self.network(obs)
       
       def get_action_dist(self, obs):
           """返回高斯分布。"""
           mean = self.forward(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

离散 Actor
----------

用于离散动作空间：

.. code-block:: python

   from apexrl.models.base import DiscreteActor

   class DiscreteMLPActor(DiscreteActor):
       """用于离散动作的 MLP Actor。"""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           obs_dim = obs_space.shape[0]
           hidden_dim = cfg.get("hidden_dim", 256)
           
           self.network = nn.Sequential(
               nn.Linear(obs_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, self.num_actions),  # 输出 logits
           )
       
       def forward(self, obs):
           return self.network(obs)
       
       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)

Critic 网络
-----------

价值函数估计器：

.. code-block:: python

   from apexrl.models.base import Critic

   class MLPCritic(Critic):
       """简单的 MLP Critic。"""
       
       def __init__(self, obs_space, cfg=None):
           super().__init__(obs_space, cfg)
           
           cfg = cfg or {}
           hidden_dims = cfg.get("hidden_dims", [256, 256])
           
           layers = []
           input_dim = obs_space.shape[0]
           
           for hidden_dim in hidden_dims:
               layers.append(nn.Linear(input_dim, hidden_dim))
               layers.append(nn.ReLU())
               input_dim = hidden_dim
           
           layers.append(nn.Linear(input_dim, 1))  # 单个价值输出
           self.network = nn.Sequential(*layers)
           
           self.apply(self._init_weights)
       
       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               nn.init.orthogonal_(module.weight, gain=1.0)
               nn.init.constant_(module.bias, 0.0)
       
       def forward(self, obs):
           """返回状态价值。"""
           return self.network(obs).squeeze(-1)
       
       def get_value(self, obs):
           """获取价值估计（与 forward 相同）。"""
           return self.forward(obs)

图像观测的 CNN
--------------

用于基于视觉的 RL：

.. code-block:: python

   class CNNActor(ContinuousActor):
       """用于图像输入（如相机）的 CNN Actor。"""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           # obs_space.shape = (C, H, W), 如 (3, 84, 84)
           channels = obs_space.shape[0]
           
           self.encoder = nn.Sequential(
               nn.Conv2d(channels, 32, kernel_size=8, stride=4),
               nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=4, stride=2),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, stride=1),
               nn.ReLU(),
               nn.Flatten(),
           )
           
           # 计算特征维度
           with torch.no_grad():
               dummy = torch.zeros(1, *obs_space.shape)
               feature_dim = self.encoder(dummy).shape[1]
           
           self.head = nn.Sequential(
               nn.Linear(feature_dim, 512),
               nn.ReLU(),
               nn.Linear(512, self.action_dim),
           )
           
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))
       
       def forward(self, obs):
           # 如需要则归一化
           if obs.dtype == torch.uint8:
               obs = obs.float() / 255.0
           
           features = self.encoder(obs)
           return self.head(features)
       
       def get_action_dist(self, obs):
           mean = self.forward(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

多输入网络
----------

用于多种观测模态：

.. code-block:: python

   class MultiInputActor(ContinuousActor):
       """处理多种观测类型的 Actor。"""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           # 假设 obs_space 是 Dict 空间
           self.proprio_dim = obs_space["proprioception"].shape[0]
           self.vision_shape = obs_space["vision"].shape  # (C, H, W)
           
           # 视觉编码器
           self.vision_encoder = nn.Sequential(
               nn.Conv2d(self.vision_shape[0], 32, 8, stride=4),
               nn.ReLU(),
               nn.Conv2d(32, 64, 4, stride=2),
               nn.ReLU(),
               nn.Flatten(),
           )
           
           # 计算视觉特征维度
           with torch.no_grad():
               dummy = torch.zeros(1, *self.vision_shape)
               vision_feat_dim = self.vision_encoder(dummy).shape[1]
           
           # 联合处理
           combined_dim = vision_feat_dim + self.proprio_dim
           self.combined = nn.Sequential(
               nn.Linear(combined_dim, 512),
               nn.ReLU(),
               nn.Linear(512, self.action_dim),
           )
           
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))
       
       def forward(self, obs_dict):
           """处理观测字典。"""
           vision = obs_dict["vision"]
           proprio = obs_dict["proprioception"]
           
           vision_feat = self.vision_encoder(vision)
           combined = torch.cat([vision_feat, proprio], dim=-1)
           
           return self.combined(combined)
       
       def get_action_dist(self, obs_dict):
           mean = self.forward(obs_dict)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

循环网络（LSTM/GRU）
-------------------

用于部分可观测环境：

.. code-block:: python

   class RecurrentActor(ContinuousActor):
       """用于 POMDP 的基于 LSTM 的 Actor。"""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           obs_dim = obs_space.shape[0]
           hidden_size = cfg.get("hidden_size", 256)
           num_layers = cfg.get("num_layers", 1)
           
           self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers, batch_first=True)
           self.head = nn.Linear(hidden_size, self.action_dim)
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))
           
           self.hidden_size = hidden_size
           self.num_layers = num_layers
       
       def forward(self, obs, hidden=None):
           """
           参数:
               obs: (batch, seq_len, obs_dim) 或 (batch, obs_dim)
               hidden: 可选的 (h, c) 元组
           """
           if obs.dim() == 2:
               obs = obs.unsqueeze(1)  # 添加序列维度
           
           lstm_out, hidden = self.lstm(obs, hidden)
           mean = self.head(lstm_out[:, -1, :])  # 使用最后输出
           
           return mean, hidden
       
       def get_action_dist(self, obs, hidden=None):
           mean, hidden = self.forward(obs, hidden)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std), hidden

使用自定义网络
--------------

在 runner 中使用您的自定义网络：

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=CNNActor,      # 您的自定义 actor
       critic_class=MLPCritic,    # 您的自定义 critic
       actor_cfg={
           "conv_channels": [32, 64, 64],
           "hidden_dims": [512],
       },
       critic_cfg={
           "hidden_dims": [256, 256],
       },
       log_dir="./logs",
   )

最佳实践
--------

1. **权重初始化**：使用正交初始化以获得更好的训练稳定性
2. **归一化**：考虑对更深的网络使用 LayerNorm 或 BatchNorm
3. **设备处理**：确保张量移动到正确的设备
4. **动作边界**：对有界动作空间使用 ``tanh`` 压缩
5. **数值稳定性**：裁剪对数概率和标准差

高级特性
--------

**固定 vs 可学习标准差：**

.. code-block:: python

   class ActorWithFixedStd(ContinuousActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           if cfg.get("fixed_std", True):
               # 固定标准差
               std_value = cfg.get("std_value", 1.0)
               self.register_buffer("std", torch.ones(self.action_dim) * std_value)
               self.log_std = None
           else:
               # 可学习标准差
               self.log_std = nn.Parameter(torch.zeros(self.action_dim))

**共享编码器：**

.. code-block:: python

   class SharedEncoder(nn.Module):
       """Actor 和 Critic 共享的编码器。"""
       
       def __init__(self, obs_dim, hidden_dim):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Linear(obs_dim, hidden_dim),
               nn.ReLU(),
           )
       
       def forward(self, obs):
           return self.encoder(obs)

   class ActorWithSharedEncoder(ContinuousActor):
       def __init__(self, obs_space, action_space, shared_encoder, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           self.shared_encoder = shared_encoder
           self.head = nn.Linear(hidden_dim, self.action_dim)
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))

另请参阅
--------

- 探索 :doc:`../modules/networks` API 参考
- 学习 :doc:`../modules/algorithms` 训练详情
- 查看 :doc:`custom_environment` 环境集成

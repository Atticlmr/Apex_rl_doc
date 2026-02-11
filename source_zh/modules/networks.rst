网络
====

ApexRL 为定义自定义神经网络架构提供了灵活的基类。

概述
----

网络模块分为：

1. **基类** - Actor 和 Critic 的抽象接口
2. **MLP 实现** - 多层感知机网络
3. **CNN 实现** - 用于视觉的卷积网络

基类
----

Actor
~~~~~

所有策略网络的基类：

.. code-block:: python

   from apexrl.models.base import Actor

   class MyActor(Actor):
       def forward(self, obs):
           """返回动作分布参数。"""
           pass
       
       def act(self, obs, deterministic=False):
           """从策略采样动作。"""
           pass
       
       def evaluate(self, obs, actions):
           """评估动作以计算损失。"""
           pass

.. autoclass:: apexrl.models.base.Actor
   :members:
   :undoc-members:
   :show-inheritance:

ContinuousActor
~~~~~~~~~~~~~~~

用于连续动作空间，使用高斯分布：

.. code-block:: python

   from apexrl.models.base import ContinuousActor

   class MyContinuousActor(ContinuousActor):
       def get_action_dist(self, obs):
           """返回 torch.distributions.Normal。"""
           mean = self.network(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

.. autoclass:: apexrl.models.base.ContinuousActor
   :members:
   :undoc-members:
   :show-inheritance:

DiscreteActor
~~~~~~~~~~~~~

用于离散动作空间，使用分类分布：

.. code-block:: python

   from apexrl.models.base import DiscreteActor

   class MyDiscreteActor(DiscreteActor):
       def get_action_dist(self, obs):
           """返回 torch.distributions.Categorical。"""
           logits = self.network(obs)
           return torch.distributions.Categorical(logits=logits)

.. autoclass:: apexrl.models.base.DiscreteActor
   :members:
   :undoc-members:
   :show-inheritance:

Critic
~~~~~~

用于价值函数估计：

.. code-block:: python

   from apexrl.models.base import Critic

   class MyCritic(Critic):
       def forward(self, obs):
           """返回状态价值。"""
           return self.network(obs).squeeze(-1)
       
       def get_value(self, obs):
           """获取价值估计。"""
           return self.forward(obs)

.. autoclass:: apexrl.models.base.Critic
   :members:
   :undoc-members:
   :show-inheritance:

MLP 网络
--------

MLPActor
~~~~~~~~

用于连续动作的多层感知机 Actor：

.. code-block:: python

   from apexrl.models.mlp import MLPActor
   from gymnasium import spaces

   obs_space = spaces.Box(low=-1, high=1, shape=(48,))
   action_space = spaces.Box(low=-1, high=1, shape=(12,))

   actor = MLPActor(
       obs_space=obs_space,
       action_space=action_space,
       cfg={
           "hidden_dims": [256, 256, 256],
           "activation": "elu",
           "learn_std": True,
           "init_std": 1.0,
           "layer_norm": False,
       }
   )

.. autoclass:: apexrl.models.mlp.MLPActor
   :members:
   :undoc-members:
   :show-inheritance:

MLPCritic
~~~~~~~~~

多层感知机 Critic：

.. code-block:: python

   from apexrl.models.mlp import MLPCritic

   critic = MLPCritic(
       obs_space=obs_space,
       cfg={
           "hidden_dims": [256, 256, 256],
           "activation": "elu",
           "layer_norm": False,
       }
   )

.. autoclass:: apexrl.models.mlp.MLPCritic
   :members:
   :undoc-members:
   :show-inheritance:

MLPDiscreteActor
~~~~~~~~~~~~~~~~

用于离散动作的多层感知机 Actor：

.. code-block:: python

   from apexrl.models.mlp import MLPDiscreteActor

   action_space = spaces.Discrete(4)

   actor = MLPDiscreteActor(
       obs_space=obs_space,
       action_space=action_space,
       cfg={
           "hidden_dims": [256, 256],
           "activation": "relu",
       }
   )

.. autoclass:: apexrl.models.mlp.MLPDiscreteActor
   :members:
   :undoc-members:
   :show-inheritance:

CNN 网络
--------

CNNActor
~~~~~~~~

用于图像观测的卷积神经网络 Actor：

.. code-block:: python

   from apexrl.models.mlp import CNNActor

   obs_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
   action_space = spaces.Box(low=-1, high=1, shape=(4,))

   actor = CNNActor(
       obs_space=obs_space,
       action_space=action_space,
       cfg={
           "conv_channels": [32, 64, 64],
           "conv_kernels": [8, 4, 3],
           "conv_strides": [4, 2, 1],
           "hidden_dims": [512],
           "activation": "relu",
       }
   )

.. autoclass:: apexrl.models.mlp.CNNActor
   :members:
   :undoc-members:
   :show-inheritance:

CNNCritic
~~~~~~~~~

卷积神经网络 Critic：

.. code-block:: python

   from apexrl.models.mlp import CNNCritic

   critic = CNNCritic(
       obs_space=obs_space,
       cfg={
           "conv_channels": [32, 64, 64],
           "conv_kernels": [8, 4, 3],
           "conv_strides": [4, 2, 1],
           "hidden_dims": [512],
       }
   )

.. autoclass:: apexrl.models.mlp.CNNCritic
   :members:
   :undoc-members:
   :show-inheritance:

网络构建工具
------------

build_mlp
~~~~~~~~~

构建 MLP 网络的辅助函数：

.. code-block:: python

   from apexrl.models.mlp import build_mlp

   network = build_mlp(
       input_dim=48,
       hidden_dims=[256, 256],
       output_dim=12,
       activation="elu",
       layer_norm=False,
   )

.. autofunction:: apexrl.models.mlp.build_mlp

另请参阅
--------

- :doc:`../tutorials/custom_network` - 自定义网络教程
- :doc:`../api/apexrl.models` - 完整 API 参考

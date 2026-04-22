网络
====

ApexRL 为定义自定义神经网络架构提供了灵活的基类。

概述
----

网络模块分为：

1. **基类** - Actor 和 Critic 的抽象接口
2. **MLP 实现** - 多层感知机网络
3. **CNN 实现** - 用于视觉的卷积网络
4. **连续动作 Q 网络** - 面向 SAC 的 ``Q(s, a)`` critic

当前运行时同时支持扁平张量观测和结构化 ``TensorDict`` / 嵌套 dict 观测树。

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
   :noindex:

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

对于 PPO，默认推荐使用未经过 ``tanh`` 压缩的高斯策略
（``use_tanh_squash=False``）。这样策略分布、log-prob 和 entropy
语义保持一致，再由 ``GymVecEnvContinuous`` 之类的环境包装器负责裁剪和缩放。

对于 SAC，默认推荐使用 squashed Gaussian 策略。它会根据状态输出
``mean`` 和 ``log_std``，先在无界空间采样，再经过 ``tanh`` 压缩，
最后线性映射到 Gymnasium 动作边界。

.. autoclass:: apexrl.models.base.ContinuousActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

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
   :noindex:

ContinuousQNetwork
~~~~~~~~~~~~~~~~~~

用于 ``Q(s, a)`` 形式的连续动作 critic：

.. autoclass:: apexrl.models.base.ContinuousQNetwork
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
           "min_log_std": -5.0,
           "max_log_std": 2.0,
           "layer_norm": False,
       }
   )

当前默认的 MLP/CNN 初始化更贴近 PPO：隐藏层使用较大的 gain，
策略输出层使用较小的 gain，价值输出层使用 gain 1.0。

当前默认 MLP 模型也可以直接消费结构化观测。它们会在内部把递归观测树展平，
因此对于图像 + 向量这类多模态输入，可以先作为基线模型使用。

.. autoclass:: apexrl.models.mlp.MLPActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MLPSquashedGaussianActor
~~~~~~~~~~~~~~~~~~~~~~~~

SAC 默认连续动作策略：

.. code-block:: python

   from apexrl.models.mlp import MLPSquashedGaussianActor

   actor = MLPSquashedGaussianActor(
       obs_space=obs_space,
       action_space=action_space,
       cfg={
           "hidden_dims": [256, 256],
           "activation": "relu",
           "min_log_std": -20.0,
           "max_log_std": 2.0,
       }
   )

这个 actor 会输出状态相关的高斯分布参数，执行 ``tanh`` 压缩，
并把动作映射到环境动作边界。它是给 SAC 用的，不是给 PPO 用的。

.. autoclass:: apexrl.models.mlp.MLPSquashedGaussianActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

MLPContinuousQNetwork
~~~~~~~~~~~~~~~~~~~~~

SAC 使用的连续动作 critic：

.. autoclass:: apexrl.models.mlp.MLPContinuousQNetwork
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

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
   :noindex:

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
   :noindex:

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
   :noindex:

另请参阅
--------

- :doc:`../tutorials/custom_network` - 自定义网络教程
- :doc:`../API/apexrl.models` - 完整 API 参考

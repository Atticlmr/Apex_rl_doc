Networks
========

ApexRL provides flexible base classes for defining custom neural network architectures.

Overview
--------

The network module is organized into:

1. **Base Classes** - Abstract interfaces for Actor and Critic
2. **MLP Implementations** - Multi-layer perceptron networks
3. **CNN Implementations** - Convolutional networks for vision
4. **Continuous Q Networks** - ``Q(s, a)`` critics for SAC-style algorithms

The current runtime supports both flat tensor observations and structured
``TensorDict`` / nested dict observation trees.

Base Classes
------------

Actor
~~~~~

The base class for all policy networks:

.. code-block:: python

   from apexrl.models.base import Actor

   class MyActor(Actor):
       def forward(self, obs):
           """Return action distribution parameters."""
           pass
       
       def act(self, obs, deterministic=False):
           """Sample actions from the policy."""
           pass
       
       def evaluate(self, obs, actions):
           """Evaluate actions for loss computation."""
           pass

.. autoclass:: apexrl.models.base.Actor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ContinuousActor
~~~~~~~~~~~~~~~

For continuous action spaces using Gaussian distributions:

.. code-block:: python

   from apexrl.models.base import ContinuousActor

   class MyContinuousActor(ContinuousActor):
       def get_action_dist(self, obs):
           """Return torch.distributions.Normal."""
           mean = self.network(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

For PPO, the default recommendation is an unsquashed Gaussian policy
(``use_tanh_squash=False``). This keeps the log-probability and entropy terms
aligned with the policy distribution, while environment wrappers such as
``GymVecEnvContinuous`` handle clipping and scaling to action bounds.

For SAC, the default recommendation is a squashed Gaussian policy with
state-dependent ``mean`` and ``log_std`` heads. Actions are sampled in the
unconstrained space, pushed through ``tanh``, and then affine-scaled to the
Gymnasium action bounds.

.. autoclass:: apexrl.models.base.ContinuousActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

DiscreteActor
~~~~~~~~~~~~~

For discrete action spaces using Categorical distributions:

.. code-block:: python

   from apexrl.models.base import DiscreteActor

   class MyDiscreteActor(DiscreteActor):
       def get_action_dist(self, obs):
           """Return torch.distributions.Categorical."""
           logits = self.network(obs)
           return torch.distributions.Categorical(logits=logits)

.. autoclass:: apexrl.models.base.DiscreteActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Critic
~~~~~~

For value function estimation:

.. code-block:: python

   from apexrl.models.base import Critic

   class MyCritic(Critic):
       def forward(self, obs):
           """Return state values."""
           return self.network(obs).squeeze(-1)
       
       def get_value(self, obs):
           """Get value estimates."""
           return self.forward(obs)

.. autoclass:: apexrl.models.base.Critic
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ContinuousQNetwork
~~~~~~~~~~~~~~~~~~

For continuous-control critics of the form ``Q(s, a)``:

.. autoclass:: apexrl.models.base.ContinuousQNetwork
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MLP Networks
------------

MLPActor
~~~~~~~~

Multi-layer perceptron actor for continuous actions:

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

The default MLP/CNN initializers are PPO-oriented: hidden layers use a larger
gain for stable optimization, policy output layers use a small gain, and value
output layers use gain 1.0.

Current default MLP models can also consume structured observations. They flatten
recursive observation trees internally, which makes them a practical baseline for
multimodal inputs such as image + vector observations.

.. autoclass:: apexrl.models.mlp.MLPActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MLPSquashedGaussianActor
~~~~~~~~~~~~~~~~~~~~~~~~

SAC's default continuous-control actor:

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

This actor predicts state-dependent Gaussian parameters, applies ``tanh``
squashing, and rescales actions to the environment bounds. It is intended for
SAC rather than PPO.

.. autoclass:: apexrl.models.mlp.MLPSquashedGaussianActor
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MLPCritic
~~~~~~~~~

Multi-layer perceptron critic:

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

Continuous-action critic used by SAC:

.. autoclass:: apexrl.models.mlp.MLPContinuousQNetwork
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MLPDiscreteActor
~~~~~~~~~~~~~~~~

Multi-layer perceptron actor for discrete actions:

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

CNN Networks
------------

CNNActor
~~~~~~~~

Convolutional neural network actor for image observations:

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

Convolutional neural network critic:

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

Network Construction Utilities
------------------------------

build_mlp
~~~~~~~~~

Helper function to build MLP networks:

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

See Also
--------

- :doc:`../tutorials/custom_network` - Custom network tutorial
- :doc:`../API/apexrl.models` - Full API reference

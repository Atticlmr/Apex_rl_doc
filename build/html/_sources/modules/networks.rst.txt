Networks
========

ApexRL provides flexible base classes for defining custom neural network architectures.

Overview
--------

The network module is organized into:

1. **Base Classes** - Abstract interfaces for Actor and Critic
2. **MLP Implementations** - Multi-layer perceptron networks
3. **CNN Implementations** - Convolutional networks for vision

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

.. autoclass:: apexrl.models.base.ContinuousActor
   :members:
   :undoc-members:
   :show-inheritance:

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
           "layer_norm": False,
       }
   )

.. autoclass:: apexrl.models.mlp.MLPActor
   :members:
   :undoc-members:
   :show-inheritance:

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

See Also
--------

- :doc:`../tutorials/custom_network` - Custom network tutorial
- :doc:`../api/apexrl.models` - Full API reference

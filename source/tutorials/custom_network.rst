Custom Network Architectures
============================

This tutorial covers creating custom Actor (policy) and Critic (value) networks in ApexRL.

Overview
--------

ApexRL provides flexible base classes for defining custom neural networks:

- ``ContinuousActor``: For continuous action spaces (Gaussian distribution)
- ``DiscreteActor``: For discrete action spaces (Categorical distribution)
- ``Critic``: For value function estimation

Continuous Actor
----------------

For continuous control tasks:

.. code-block:: python

   import torch
   import torch.nn as nn
   from apexrl.models.base import ContinuousActor

   class MLPActor(ContinuousActor):
       """Simple MLP actor for continuous actions."""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           cfg = cfg or {}
           hidden_dims = cfg.get("hidden_dims", [256, 256])
           activation = cfg.get("activation", "relu")
           
           # Build network
           layers = []
           input_dim = obs_space.shape[0]
           
           for hidden_dim in hidden_dims:
               layers.append(nn.Linear(input_dim, hidden_dim))
               layers.append(nn.ReLU() if activation == "relu" else nn.ELU())
               input_dim = hidden_dim
           
           layers.append(nn.Linear(input_dim, self.action_dim))
           self.network = nn.Sequential(*layers)
           
           # Learnable log standard deviation
           init_std = cfg.get("init_std", 1.0)
           self.log_std = nn.Parameter(torch.ones(self.action_dim) * torch.log(torch.tensor(init_std)))
           
           # Initialize weights
           self.apply(self._init_weights)
       
       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               nn.init.orthogonal_(module.weight, gain=1.0)
               nn.init.constant_(module.bias, 0.0)
       
       def forward(self, obs):
           """Return action means."""
           return self.network(obs)
       
       def get_action_dist(self, obs):
           """Return Gaussian distribution."""
           mean = self.forward(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

Discrete Actor
--------------

For discrete action spaces:

.. code-block:: python

   from apexrl.models.base import DiscreteActor

   class DiscreteMLPActor(DiscreteActor):
       """MLP actor for discrete actions."""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           obs_dim = obs_space.shape[0]
           hidden_dim = cfg.get("hidden_dim", 256)
           
           self.network = nn.Sequential(
               nn.Linear(obs_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, self.num_actions),  # Output logits
           )
       
       def forward(self, obs):
           return self.network(obs)
       
       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)

Critic Network
--------------

Value function estimator:

.. code-block:: python

   from apexrl.models.base import Critic

   class MLPCritic(Critic):
       """Simple MLP critic."""
       
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
           
           layers.append(nn.Linear(input_dim, 1))  # Single value output
           self.network = nn.Sequential(*layers)
           
           self.apply(self._init_weights)
       
       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               nn.init.orthogonal_(module.weight, gain=1.0)
               nn.init.constant_(module.bias, 0.0)
       
       def forward(self, obs):
           """Return state values."""
           return self.network(obs).squeeze(-1)
       
       def get_value(self, obs):
           """Get value estimates (same as forward)."""
           return self.forward(obs)

CNN for Image Observations
--------------------------

For vision-based RL:

.. code-block:: python

   class CNNActor(ContinuousActor):
       """CNN actor for image inputs (e.g., from camera)."""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           # obs_space.shape = (C, H, W), e.g., (3, 84, 84)
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
           
           # Compute feature dimension
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
           # Normalize if needed
           if obs.dtype == torch.uint8:
               obs = obs.float() / 255.0
           
           features = self.encoder(obs)
           return self.head(features)
       
       def get_action_dist(self, obs):
           mean = self.forward(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

Multi-Input Networks
--------------------

For multiple observation modalities:

.. code-block:: python

   class MultiInputActor(ContinuousActor):
       """Actor that processes multiple observation types."""
       
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           # Assuming obs_space is a Dict space
           self.proprio_dim = obs_space["proprioception"].shape[0]
           self.vision_shape = obs_space["vision"].shape  # (C, H, W)
           
           # Vision encoder
           self.vision_encoder = nn.Sequential(
               nn.Conv2d(self.vision_shape[0], 32, 8, stride=4),
               nn.ReLU(),
               nn.Conv2d(32, 64, 4, stride=2),
               nn.ReLU(),
               nn.Flatten(),
           )
           
           # Compute vision feature dim
           with torch.no_grad():
               dummy = torch.zeros(1, *self.vision_shape)
               vision_feat_dim = self.vision_encoder(dummy).shape[1]
           
           # Combined processing
           combined_dim = vision_feat_dim + self.proprio_dim
           self.combined = nn.Sequential(
               nn.Linear(combined_dim, 512),
               nn.ReLU(),
               nn.Linear(512, self.action_dim),
           )
           
           self.log_std = nn.Parameter(torch.zeros(self.action_dim))
       
       def forward(self, obs_dict):
           """Process dictionary of observations."""
           vision = obs_dict["vision"]
           proprio = obs_dict["proprioception"]
           
           vision_feat = self.vision_encoder(vision)
           combined = torch.cat([vision_feat, proprio], dim=-1)
           
           return self.combined(combined)
       
       def get_action_dist(self, obs_dict):
           mean = self.forward(obs_dict)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

Recurrent Networks (LSTM/GRU)
-----------------------------

For partially observable environments:

.. code-block:: python

   class RecurrentActor(ContinuousActor):
       """LSTM-based actor for POMDPs."""
       
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
           Args:
               obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
               hidden: Optional (h, c) tuple
           """
           if obs.dim() == 2:
               obs = obs.unsqueeze(1)  # Add sequence dimension
           
           lstm_out, hidden = self.lstm(obs, hidden)
           mean = self.head(lstm_out[:, -1, :])  # Use last output
           
           return mean, hidden
       
       def get_action_dist(self, obs, hidden=None):
           mean, hidden = self.forward(obs, hidden)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std), hidden

Using Custom Networks
---------------------

Use your custom networks with the runner:

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=CNNActor,      # Your custom actor
       critic_class=MLPCritic,    # Your custom critic
       actor_cfg={
           "conv_channels": [32, 64, 64],
           "hidden_dims": [512],
       },
       critic_cfg={
           "hidden_dims": [256, 256],
       },
       log_dir="./logs",
   )

Best Practices
--------------

1. **Weight Initialization**: Use orthogonal initialization for better training stability
2. **Normalization**: Consider LayerNorm or BatchNorm for deeper networks
3. **Device Handling**: Ensure tensors are moved to the correct device
4. **Action Bounds**: Use ``tanh`` squashing for bounded action spaces
5. **Numerical Stability**: Clamp log probabilities and standard deviations

Advanced Features
-----------------

**Fixed vs Learnable Std:**

.. code-block:: python

   class ActorWithFixedStd(ContinuousActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           
           if cfg.get("fixed_std", True):
               # Fixed standard deviation
               std_value = cfg.get("std_value", 1.0)
               self.register_buffer("std", torch.ones(self.action_dim) * std_value)
               self.log_std = None
           else:
               # Learnable standard deviation
               self.log_std = nn.Parameter(torch.zeros(self.action_dim))

**Shared Encoders:**

.. code-block:: python

   class SharedEncoder(nn.Module):
       """Shared encoder for actor and critic."""
       
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

Next Steps
----------

- Explore :doc:`../modules/networks` for API reference
- Learn about :doc:`../modules/algorithms` training details
- Check :doc:`custom_environment` for environment integration

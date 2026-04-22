Custom Network Architectures
============================

This tutorial shows how to implement custom actors and critics for the current
TensorDict-capable ApexRL stack.

Overview
--------

Base classes:

- ``ContinuousActor`` for continuous policies
- ``DiscreteActor`` for discrete policies
- ``Critic`` for value networks
- ``ContinuousQNetwork`` for SAC-style ``Q(s, a)`` critics

What Your Network Receives
--------------------------

The current repository version supports structured observations.

If your environment returns:

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

then:

- PPO actor receives ``{"image": ..., "vector": ...}``
- PPO asymmetric critic receives ``{"state": ..., "context": ...}``
- SAC actor receives ``obs``
- SAC critics receive ``privileged_obs`` when present, otherwise the actor observation

Continuous Actor Example
------------------------

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

Discrete Actor Example
----------------------

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

Critic Example
--------------

For a privileged critic:

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

Using Custom Networks
---------------------

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

Best Practices
--------------

1. Keep the observation structure explicit instead of flattening everything immediately.
2. Let the environment expose actor and critic groups as ``obs`` and ``privileged_obs``.
3. Use ``DiscreteActor`` for discrete PPO and ``ContinuousActor`` for continuous PPO.
4. For SAC, keep custom critics in ``ContinuousQNetwork`` form, i.e. ``forward(obs, actions)``.
5. Prefer explicit branch encoders for image + vector inputs instead of a single flat MLP.

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

Recurrent PPO Networks
----------------------

``RecurrentPPO`` uses the same ``actor_class`` / ``critic_class`` injection
pattern. Custom recurrent networks should follow the RecurrentPPO actor and
critic interfaces. ApexRL includes ``GRUActor``, ``GRUDiscreteActor`` and
``GRUCritic`` as reference implementations.

.. code-block:: python

   from apexrl.agent import OnPolicyRunner
   from apexrl.algorithms.ppo import RecurrentPPOConfig
   from apexrl.models import GRUCritic, GRUDiscreteActor

   runner = OnPolicyRunner(
       env=env,
       algorithm="recurrent_ppo",
       cfg=RecurrentPPOConfig(
           num_steps=64,
           sequence_length=16,
           recurrent_minibatch_size=256,
       ),
       actor_class=GRUDiscreteActor,
       critic_class=GRUCritic,
       actor_cfg={"hidden_dims": [128], "rnn_hidden_size": 128},
       critic_cfg={"hidden_dims": [128], "rnn_hidden_size": 128},
   )

Multi-Agent Custom Networks
---------------------------

MAPPO, IPPO and HAPPO use the same custom network pattern as PPO: pass ApexRL
actor and critic classes, plus optional configuration dictionaries. The network
classes can contain arbitrary PyTorch modules internally while inheriting
ApexRL's actor or critic base interfaces.

.. code-block:: python

   from apexrl.models.base import Critic, DiscreteActor
   from apexrl.multiagent import HAPPO, HAPPOConfig, IPPO, IPPOConfig, MAPPO, MAPPOConfig


   class EntityDiscreteActor(DiscreteActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)
           # Build attention, graph, CNN or MLP blocks here.

       def forward(self, obs):
           ...

       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)


   class EntityCritic(Critic):
       def __init__(self, obs_space, cfg=None):
           super().__init__(obs_space, cfg)
           # Build the value network here.

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

For MAPPO with ``centralized_critic=True``, the critic receives
``env.state_space`` and ``env.get_state()`` outputs. For IPPO, or MAPPO with
``centralized_critic=False``, each critic receives that agent's local
observation space and local observations. Actors always receive per-agent local
observations.

Parameter sharing is controlled by the multi-agent config:

- ``share_actor=True`` creates one actor instance and reuses it for all agents.
  This requires identical observation and action spaces across agents.
- HAPPO uses ``share_actor=False`` so each agent has a separate policy for its
  sequential update.
- ``share_critic=True`` creates one critic instance and reuses it for all
  agents. With decentralized critics, this requires identical observation
  spaces; with centralized critics, all critics consume the shared state space.
- Set either flag to ``False`` when agents need separate model parameters.

You can also pass already constructed ApexRL actor/critic objects through the
``models`` dictionary. This is useful for heterogeneous agents or for manually
sharing selected modules:

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

For more complex multi-agent networks, prefer representing entities with fixed
structured observations, such as ``spaces.Dict`` entries, padded tensors and
masks. Attention, DeepSets, GNN and transformer-style encoders can then be
implemented inside the actor or critic without changing MAPPO/IPPO/HAPPO.

Best Practices
--------------

1. Keep the observation structure explicit instead of flattening everything immediately.
2. Let the environment expose actor and critic groups as ``obs`` and ``privileged_obs``.
3. Use ``DiscreteActor`` for discrete PPO and ``ContinuousActor`` for continuous PPO.
4. For SAC, keep custom critics in ``ContinuousQNetwork`` form, i.e. ``forward(obs, actions)``.
5. Prefer explicit branch encoders for image + vector inputs instead of a single flat MLP.
6. For MAPPO/IPPO/HAPPO, use local per-agent observations in actors and choose
   centralized or decentralized critic observations through the algorithm config.

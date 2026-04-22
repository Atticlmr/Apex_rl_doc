Train PPO
=========

This tutorial shows the current recommended PPO workflow in ApexRL.

Overview
--------

Recommended stack:

- ``GymVecEnv`` for discrete Gymnasium tasks
- ``GymVecEnvContinuous`` for continuous Gymnasium tasks
- ``OnPolicyRunner`` as the canonical training entrypoint
- ``MLPDiscreteActor`` / ``MLPActor`` and ``MLPCritic`` as default baselines

Discrete PPO Example
--------------------

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPDiscreteActor, MLPCritic


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

   cfg = PPOConfig(
       num_steps=128,
       num_epochs=4,
       minibatch_size=256,
       learning_rate=3e-4,
       learning_rate_schedule="constant",
       device="cpu",
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       actor_class=MLPDiscreteActor,
       critic_class=MLPCritic,
       log_dir="./logs/cartpole_ppo",
       save_dir="./checkpoints/cartpole_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.close()

Continuous PPO Example
----------------------

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous
   from apexrl.models import MLPActor, MLPCritic


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(device="cpu"),
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/pendulum_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   runner.close()

Asymmetric Critic and Structured Observations
---------------------------------------------

PPO now supports structured observations and privileged critic observations.

Recommended environment output format:

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

Enable the asymmetric critic path with:

.. code-block:: python

   cfg = PPOConfig(use_asymmetric=True, device="cpu")

The runner and algorithm automatically:

- send ``obs`` to the actor
- send ``privileged_obs`` to the critic
- keep the same structure in rollout storage

Custom Networks
---------------

To replace the default baselines, keep the same runner interface and swap
``actor_class`` / ``critic_class``.

For multimodal actors, your actor usually receives the already-split ``obs`` branch,
for example ``{"image": ..., "vector": ...}``.

.. code-block:: python

   import torch
   import torch.nn as nn

   from apexrl.models.base import DiscreteActor


   class MultiModalDiscreteActor(DiscreteActor):
       def __init__(self, obs_space, action_space, cfg=None):
           super().__init__(obs_space, action_space, cfg)

           image_shape = obs_space["image"].shape
           vector_dim = obs_space["vector"].shape[0]
           hidden_dim = (cfg or {}).get("hidden_dim", 256)

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

           self.head = nn.Sequential(
               nn.Linear(image_dim + 64, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, self.num_actions),
           )

       def forward(self, obs):
           image_feat = self.image_encoder(obs["image"])
           vector_feat = self.vector_encoder(obs["vector"])
           return self.head(torch.cat([image_feat, vector_feat], dim=-1))

       def get_action_dist(self, obs):
           logits = self.forward(obs)
           return torch.distributions.Categorical(logits=logits)


   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(use_asymmetric=True, device="cpu"),
       actor_class=MultiModalDiscreteActor,
       critic_class=MLPCritic,
       actor_cfg={"hidden_dim": 256},
   )

Notes
-----

- ``OnPolicyRunner`` is the preferred PPO entrypoint
- continuous-action PPO defaults to an unsquashed Gaussian policy
- ``GymVecEnvContinuous`` handles clipping and scaling to Gymnasium action bounds
- ``PPO.learn()`` remains available, but delegates to the same runner

Next Steps
----------

- Read :doc:`custom_network` for more network patterns
- Read :doc:`custom_environment` for TensorDict environment integration
- Read :doc:`../modules/runners` for runner API details

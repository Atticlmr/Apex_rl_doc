Quick Start
===========

This guide will help you get started with ApexRL in just a few minutes.

Installation
------------

Install ApexRL from source:

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

Or using uv (faster):

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

Requirements
------------

- Python >= 3.11
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0

Your First RL Agent
-------------------

Here's a minimal example to train a PPO agent on a Gymnasium environment:

.. code-block:: python

   import gymnasium as gym
   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   # Create vectorized environment
   def make_env():
       return gym.make("Pendulum-v1")

   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

   # Create runner with default PPO configuration
   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs",
   )

   # Train for 100,000 timesteps
   runner.learn(total_timesteps=100_000)

   # Save the trained model
   runner.save_checkpoint("model_final.pt")

   env.close()

Using Custom Networks
---------------------

You can easily define custom Actor and Critic networks:

.. code-block:: python

   from apexrl.models.base import ContinuousActor, Critic
   import torch.nn as nn

   class CustomActor(ContinuousActor):
       def __init__(self, obs_space, action_space, cfg):
           super().__init__(obs_space, action_space, cfg)
           
           obs_dim = obs_space.shape[0]
           action_dim = action_space.shape[0]
           
           self.network = nn.Sequential(
               nn.Linear(obs_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
               nn.Linear(256, action_dim),
           )
           self.log_std = nn.Parameter(torch.zeros(action_dim))
       
       def get_action_dist(self, obs):
           mean = self.network(obs)
           std = torch.exp(self.log_std)
           return torch.distributions.Normal(mean, std)

   # Use your custom actor
   runner = OnPolicyRunner(
       env=env,
       actor_class=CustomActor,
       critic_class=MLPCritic,
       # ... other args
   )

Next Steps
----------

- Learn about :doc:`tutorials/first_agent` for a detailed walkthrough
- Explore :doc:`modules/algorithms` to understand available algorithms
- Check :doc:`modules/environments` for environment integration
- Read :doc:`tutorials/custom_network` for advanced network architectures

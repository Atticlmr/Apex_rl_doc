Quick Start
===========

This guide shows the current recommended training entrypoints in ApexRL.

Installation
------------

Install from source:

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

or with ``uv``:

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

Core requirements:

- Python >= 3.11
- PyTorch >= 2.0
- Gymnasium >= 0.29
- TensorDict >= 0.6

Training Entry Points
---------------------

- ``OnPolicyRunner`` is the canonical entrypoint for PPO
- ``OffPolicyRunner`` is the canonical entrypoint for DQN and SAC
- ``PPO.learn()``, ``DQN.learn()``, and ``SAC.learn()`` remain available as thin wrappers

First PPO Agent
---------------

Discrete control:

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

   runner = OnPolicyRunner(
       env=env,
       cfg=PPOConfig(device="cpu", learning_rate_schedule="constant"),
       actor_class=MLPDiscreteActor,
       critic_class=MLPCritic,
       log_dir="./logs/cartpole_ppo",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   runner.close()

Continuous control:

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

Structured Observations
-----------------------

The current repository version supports structured observations all the way through
environment wrappers, buffers, algorithms, and default MLP models.

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

In this format:

- the actor receives ``obs``
- PPO with ``use_asymmetric=True`` sends ``privileged_obs`` to the critic
- SAC stores actor and critic branches separately in replay

Next Steps
----------

- Read :doc:`tutorials/train_ppo` for the standard PPO flow
- Read :doc:`tutorials/train_dqn` for the standard DQN flow
- Read :doc:`tutorials/train_sac` for the standard SAC flow
- Read :doc:`tutorials/custom_network` for multimodal custom actors and critics
- Read :doc:`tutorials/custom_environment` for TensorDict-based environment integration

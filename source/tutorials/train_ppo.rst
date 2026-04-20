Train PPO
=========

This tutorial shows the standard way to train a PPO agent in ApexRL.

Overview
--------

The recommended PPO stack in this repository is:

- ``GymVecEnvContinuous`` for continuous-control Gymnasium tasks
- ``OnPolicyRunner`` as the canonical training entrypoint
- ``MLPActor`` and ``MLPCritic`` as the default baseline networks

Prerequisites
-------------

Install ApexRL and Gymnasium:

.. code-block:: bash

   pip install -e .

Environment Setup
-----------------

For continuous-control PPO, use ``GymVecEnvContinuous`` so the environment
wrapper handles action clipping and timeout metadata consistently.

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

Build the Runner
----------------

``OnPolicyRunner`` creates the PPO agent and owns the outer training loop.

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.models import MLPActor, MLPCritic

   cfg = PPOConfig(
       num_steps=256,
       num_epochs=5,
       learning_rate=3e-4,
       log_interval=10,
       save_interval=100,
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/ppo_pendulum",
       save_dir="./checkpoints/ppo_pendulum",
   )

Train
-----

.. code-block:: python

   runner.learn(total_timesteps=100_000)

Evaluate and Save
-----------------

.. code-block:: python

   stats = runner.eval(num_episodes=10)
   print(f"Mean reward: {stats['eval/mean_reward']:.2f}")

   runner.save_checkpoint("ppo_pendulum_final.pt")
   env.close()

Complete Example
----------------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.algorithms.ppo import PPOConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous
   from apexrl.models import MLPActor, MLPCritic


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(8)], device="cpu")

   cfg = PPOConfig(
       num_steps=256,
       num_epochs=5,
       learning_rate=3e-4,
   )

   runner = OnPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs/ppo_pendulum",
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("ppo_pendulum_final.pt")
   env.close()

Notes
-----

- Continuous-action PPO defaults to an unsquashed Gaussian policy.
- ``PPO.learn()`` still exists, but ``OnPolicyRunner`` is the preferred entrypoint.
- If you need custom networks, keep the same runner interface and replace
  ``MLPActor`` / ``MLPCritic``.

Next Steps
----------

- Read :doc:`train_dqn` for the off-policy training flow
- Read :doc:`custom_network` for custom model architectures
- Read :doc:`../modules/runners` for the runner API details

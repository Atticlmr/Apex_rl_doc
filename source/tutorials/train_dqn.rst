Train DQN
=========

This tutorial shows the standard way to train a DQN agent in ApexRL.

Overview
--------

The recommended DQN stack in this repository is:

- ``GymVecEnv`` for discrete-action Gymnasium tasks
- ``OffPolicyRunner`` as the canonical training entrypoint
- ``MLPQNetwork`` as the default Q-network baseline

Prerequisites
-------------

Install ApexRL and Gymnasium:

.. code-block:: bash

   pip install -e .

Environment Setup
-----------------

For DQN, start with a discrete-control environment such as ``CartPole-v1``.

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnv


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(2)], device="cpu")

Build the Runner
----------------

``OffPolicyRunner`` creates the DQN agent, fills replay, and schedules updates.

.. code-block:: python

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.models import MLPQNetwork

   cfg = DQNConfig(
       batch_size=128,
       buffer_size=100_000,
       learning_starts=1_000,
       target_update_interval=250,
       double_dqn=True,
       dueling=True,
       log_interval=1_000,
       save_interval=10_000,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="dqn",
       q_network_class=MLPQNetwork,
       log_dir="./logs/dqn_cartpole",
       save_dir="./checkpoints/dqn_cartpole",
   )

Train
-----

.. code-block:: python

   runner.learn(total_timesteps=50_000)

Evaluate and Save
-----------------

.. code-block:: python

   stats = runner.eval(num_episodes=10)
   print(f"Mean reward: {stats['eval/mean_reward']:.2f}")

   runner.save_checkpoint("dqn_cartpole_final.pt")
   env.close()

Complete Example
----------------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPQNetwork


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(2)], device="cpu")

   cfg = DQNConfig(
       batch_size=128,
       buffer_size=100_000,
       learning_starts=1_000,
       target_update_interval=250,
       double_dqn=True,
       dueling=True,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="dqn",
       q_network_class=MLPQNetwork,
       log_dir="./logs/dqn_cartpole",
   )

   runner.learn(total_timesteps=50_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("dqn_cartpole_final.pt")
   env.close()

Notes
-----

- ``OffPolicyRunner`` is the preferred training entrypoint for DQN.
- ``double_dqn=True`` is enabled by default and should usually stay on.
- Set ``dueling=True`` to switch ``MLPQNetwork`` to the dueling architecture.

Next Steps
----------

- Read :doc:`train_ppo` for the on-policy training flow
- Read :doc:`../modules/algorithms` for DQN-specific options
- Read :doc:`../modules/runners` for runner API details

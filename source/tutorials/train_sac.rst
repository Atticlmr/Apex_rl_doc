Train SAC
=========

This tutorial shows the standard way to train a SAC agent in ApexRL.

Overview
--------

The recommended SAC stack in this repository is:

- ``GymVecEnvContinuous`` for continuous-control Gymnasium tasks
- ``OffPolicyRunner`` as the canonical training entrypoint
- ``MLPSquashedGaussianActor`` as the default stochastic policy
- ``MLPContinuousQNetwork`` as the default twin-critic baseline

Prerequisites
-------------

Install ApexRL and Gymnasium:

.. code-block:: bash

   pip install -e .

Environment Setup
-----------------

For SAC, start with a continuous-control task such as ``Pendulum-v1``.

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(2)], device="cpu")

Build the Runner
----------------

``OffPolicyRunner`` creates the SAC agent, fills replay, and schedules
actor/critic/temperature updates.

.. code-block:: python

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.sac import SACConfig

   cfg = SACConfig(
       batch_size=256,
       buffer_size=100_000,
       learning_starts=5_000,
       actor_learning_rate=3e-4,
       critic_learning_rate=3e-4,
       alpha_learning_rate=3e-4,
       tau=0.005,
       log_interval=1_000,
       save_interval=10_000,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
       save_dir="./checkpoints/sac_pendulum",
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

   runner.save_checkpoint("sac_pendulum_final.pt")
   env.close()

Complete Example
----------------

.. code-block:: python

   import gymnasium as gym

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.sac import SACConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous


   def make_env():
       return gym.make("Pendulum-v1")


   env = GymVecEnvContinuous([make_env for _ in range(2)], device="cpu")

   cfg = SACConfig(
       batch_size=256,
       buffer_size=100_000,
       learning_starts=5_000,
       actor_learning_rate=3e-4,
       critic_learning_rate=3e-4,
       alpha_learning_rate=3e-4,
       tau=0.005,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.save_checkpoint("sac_pendulum_final.pt")
   env.close()

What SAC Optimizes
------------------

SAC updates three coupled objectives:

.. math::

   y = r + \gamma (1-d)\left(\min(Q_1'(s', a'), Q_2'(s', a')) - \alpha \log \pi(a'|s')\right)

.. math::

   L_{Q_i} = \mathbb{E}\left[(Q_i(s, a) - y)^2\right]

.. math::

   L_{\pi} = \mathbb{E}\left[\alpha \log \pi(a|s) - \min(Q_1(s, a), Q_2(s, a))\right]

.. math::

   L_{\alpha} = -\mathbb{E}\left[\log \alpha \cdot (\log \pi(a|s) + \mathcal{H}_{target})\right]

In ApexRL these map directly to the comments in
:doc:`../API/apexrl.algorithms.sac`.

Notes
-----

- SAC is currently implemented for ``Box`` observations and ``Box`` actions.
- The default policy is a squashed Gaussian actor, not PPO's unsquashed Gaussian.
- ``OffPolicyRunner`` remains the preferred training entrypoint.
- ``SAC.learn()`` exists as a thin wrapper around the same runner.

Next Steps
----------

- Read :doc:`train_ppo` for the on-policy training flow
- Read :doc:`train_dqn` for the discrete off-policy training flow
- Read :doc:`../modules/algorithms` for SAC-specific details
- Read :doc:`../modules/runners` for the runner API details

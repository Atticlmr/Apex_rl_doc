Train SAC
=========

This tutorial shows the current SAC workflow in ApexRL.

Overview
--------

Recommended stack:

- ``GymVecEnvContinuous`` for continuous Gymnasium tasks
- ``OffPolicyRunner`` as the canonical training entrypoint
- ``MLPSquashedGaussianActor`` as the default actor
- ``MLPContinuousQNetwork`` as the default twin-critic baseline

Standard Example
----------------

.. code-block:: python

   import gymnasium as gym
   import torch

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
       device="cpu",
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       algorithm="sac",
       log_dir="./logs/sac_pendulum",
       save_dir="./checkpoints/sac_pendulum",
       device=torch.device("cpu"),
   )

   runner.learn(total_timesteps=100_000)
   print(runner.eval(num_episodes=10))
   runner.close()

Structured Observations
-----------------------

SAC still requires ``Box`` actions, but observations no longer need to be a single flat tensor.

The current implementation supports structured actor observations and optional
critic-only privileged observations:

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

Internally, SAC now:

- sends ``obs`` to the actor
- sends ``privileged_obs`` to both critics when present
- stores actor and critic branches separately in replay

Notes
-----

- SAC supports ``Box`` action spaces
- observations can be plain tensors or structured ``TensorDict`` / nested dict trees
- the default policy is a squashed Gaussian actor, unlike PPO's unsquashed Gaussian
- ``OffPolicyRunner`` remains the preferred entrypoint

Next Steps
----------

- Read :doc:`custom_network` for custom actor / critic implementations
- Read :doc:`custom_environment` for structured observation environment design
- Read :doc:`../modules/algorithms` for SAC-specific details

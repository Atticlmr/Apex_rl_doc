Custom Environment Integration
==============================

This tutorial shows how to integrate environments with the current structured-observation ApexRL stack.

Overview
--------

ApexRL environments should implement the ``VecEnv`` interface and return:

- observations
- rewards
- done flags
- ``extras`` with termination metadata

For Gymnasium tasks, prefer the built-in wrappers:

- ``GymVecEnv`` for discrete tasks
- ``GymVecEnvContinuous`` for continuous tasks

Standard Gymnasium Integration
------------------------------

.. code-block:: python

   import gymnasium as gym

   from apexrl.envs.gym_wrapper import GymVecEnv


   def make_env():
       return gym.make("CartPole-v1")


   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

Structured Observation Format
-----------------------------

For multimodal actor inputs and privileged critic inputs, return:

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

This is the recommended format because the current algorithms understand it directly:

- PPO actor uses ``obs``
- PPO asymmetric critic uses ``privileged_obs``
- SAC actor uses ``obs``
- SAC critics use ``privileged_obs`` when present

Minimal Custom VecEnv
---------------------

.. code-block:: python

   import torch

   from apexrl.envs.vecenv import VecEnv


   class MyCustomVecEnv(VecEnv):
       def __init__(self, num_envs=1024, device="cuda"):
           super().__init__(device=device)
           self.num_envs = num_envs
           self.num_actions = 4
           self.max_episode_length = 200

           self.obs_buf = {
               "obs": {
                   "image": torch.zeros(num_envs, 1, 32, 32, device=device),
                   "vector": torch.zeros(num_envs, 8, device=device),
               },
               "privileged_obs": {
                   "state": torch.zeros(num_envs, 16, device=device),
               },
           }
           self.rew_buf = torch.zeros(num_envs, device=device)
           self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
           self.episode_length_buf = torch.zeros(
               num_envs,
               dtype=torch.int32,
               device=device,
           )

       def get_observations(self):
           return self.obs_buf

       def get_privileged_observations(self):
           return self.obs_buf["privileged_obs"]

       def reset(self):
           self.rew_buf.zero_()
           self.reset_buf.zero_()
           self.episode_length_buf.zero_()
           return self.obs_buf

       def reset_idx(self, env_ids):
           self.episode_length_buf[env_ids] = 0

       def step(self, actions):
           self.episode_length_buf += 1
           terminated = torch.zeros_like(self.reset_buf)
           truncated = self.episode_length_buf >= self.max_episode_length
           self.reset_buf = terminated | truncated

           final_obs = {
               "obs": {
                   "image": self.obs_buf["obs"]["image"].clone(),
                   "vector": self.obs_buf["obs"]["vector"].clone(),
               },
               "privileged_obs": {
                   "state": self.obs_buf["privileged_obs"]["state"].clone(),
               },
           }

           if self.reset_buf.any():
               self.reset_idx(torch.where(self.reset_buf)[0])

           extras = {
               "time_outs": truncated,
               "terminated": terminated,
               "truncated": truncated,
               "final_observation": final_obs,
               "log": {},
           }
           return self.obs_buf, self.rew_buf, self.reset_buf, extras

Best Practices
--------------

1. Return ``terminated`` and ``truncated`` separately.
2. Always provide ``extras["final_observation"]`` for truncated episodes.
3. Keep actor-visible and critic-visible observations in separate groups.
4. Implement ``reset_idx()`` for efficient partial resets.
5. Keep all buffers on the same device as the environment.

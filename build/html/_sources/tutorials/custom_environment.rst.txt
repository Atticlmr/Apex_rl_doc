Custom Environment Integration
==============================

This tutorial shows how to integrate custom environments with ApexRL, including Gymnasium wrappers and GPU-accelerated environments.

Overview
--------

ApexRL supports multiple environment types:

1. **Gymnasium environments** - Standard single-threaded environments
2. **Vectorized environments** - Parallel CPU environments
3. **GPU environments** - CUDA-accelerated environments (Isaac Gym, etc.)

Gymnasium Wrapper
-----------------

For standard Gymnasium environments, use the built-in ``GymVecEnv``:

.. code-block:: python

   import gymnasium as gym
   from apexrl.envs.gym_wrapper import GymVecEnv

   def make_env(env_id="CartPole-v1"):
       def _init():
           env = gym.make(env_id)
           return env
       return _init

   # Create vectorized environment
   num_envs = 16
   env_fns = [make_env("CartPole-v1") for _ in range(num_envs)]
   env = GymVecEnv(env_fns, device="cpu")

Custom VecEnv Implementation
----------------------------

For GPU-accelerated environments (like Isaac Gym), implement the ``VecEnv`` interface:

.. code-block:: python

   import torch
   from typing import Dict, Tuple, Union
   from tensordict import TensorDict
   from apexrl.envs.vecenv import VecEnv

   class MyCustomVecEnv(VecEnv):
       """Custom vectorized environment for ApexRL.
       
       This example shows a minimal implementation for a GPU-based
       robotics simulation.
       """
       
       def __init__(self, num_envs: int = 4096, device: str = "cuda"):
           super().__init__(device=device)
           
           self.num_envs = num_envs
           self.num_obs = 48      # Observation dimension
           self.num_actions = 12  # Action dimension
           self.max_episode_length = 1000
           
           # Initialize buffers
           self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(num_envs, device=device)
           self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
           self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=device)
           
           # Initialize simulation (your physics engine here)
           self._init_simulation()
           
       def _init_simulation(self):
           """Initialize your physics simulation."""
           pass
       
       def get_observations(self) -> torch.Tensor:
           """Return current observations."""
           return self.obs_buf
       
       def reset(self) -> torch.Tensor:
           """Reset all environments."""
           self.obs_buf.zero_()
           self.rew_buf.zero_()
           self.reset_buf.zero_()
           self.episode_length_buf.zero_()
           
           # Reset simulation states
           self._reset_all()
           
           return self.obs_buf
       
       def reset_idx(self, env_ids: torch.Tensor) -> None:
           """Reset specific environments by index.
           
           This is crucial for GPU environments where only some
           environments need to be reset after episode termination.
           """
           self.obs_buf[env_ids] = 0.0
           self.episode_length_buf[env_ids] = 0
           
           # Reset specific environments in simulation
           self._reset_indices(env_ids)
       
       def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
           """Execute one simulation step.
           
           Args:
               actions: Actions to apply. Shape: (num_envs, num_actions)
               
           Returns:
               Tuple of (observations, rewards, dones, extras)
           """
           # Apply actions to simulation
           self._apply_actions(actions)
           
           # Step physics
           self._step_physics()
           
           # Compute observations and rewards
           self._compute_observations()
           self._compute_rewards()
           
           # Check terminations
           self.episode_length_buf += 1
           time_outs = self.episode_length_buf >= self.max_episode_length
           self.reset_buf = time_outs.clone()  # or add other termination conditions
           
           # Reset environments that timed out
           if time_outs.any():
               self.reset_idx(torch.where(time_outs)[0])
           
           extras = {
               "time_outs": time_outs,
               "log": {
                   "/reward/mean": self.rew_buf.mean().item(),
                   "/episode_length/mean": self.episode_length_buf.float().mean().item(),
               }
           }
           
           return self.obs_buf, self.rew_buf, self.reset_buf, extras
       
       def _apply_actions(self, actions: torch.Tensor):
           """Apply actions to the simulation."""
           pass
       
       def _step_physics(self):
           """Step the physics simulation."""
           pass
       
       def _compute_observations(self):
           """Compute observations from simulation state."""
           pass
       
       def _compute_rewards(self):
           """Compute rewards based on simulation state."""
           pass
       
       def _reset_all(self):
           """Reset all simulation instances."""
           pass
       
       def _reset_indices(self, env_ids: torch.Tensor):
           """Reset specific simulation instances."""
           pass

Privileged Observations (Asymmetric AC)
---------------------------------------

For asymmetric actor-critic (different observations for policy and value):

.. code-block:: python

   class AsymmetricVecEnv(VecEnv):
       def __init__(self, ...):
           super().__init__(...)
           self.num_obs = 48              # Policy observations
           self.num_privileged_obs = 72   # Critic observations (with privileged info)
           
       def get_observations(self):
           """Return policy observations."""
           return self.obs_buf
       
       def get_privileged_observations(self):
           """Return privileged observations for critic."""
           return self.privileged_obs_buf

Environment Wrapper
-------------------

Create custom wrappers to modify environment behavior:

.. code-block:: python

   from apexrl.envs.vecenv import VecEnvWrapper

   class RewardScalingWrapper(VecEnvWrapper):
       """Scale rewards by a constant factor."""
       
       def __init__(self, env: VecEnv, scale: float = 0.1):
           super().__init__(env)
           self.scale = scale
       
       def step(self, actions: torch.Tensor):
           obs, rewards, dones, extras = self.env.step(actions)
           rewards = rewards * self.scale
           return obs, rewards, dones, extras

   # Usage
   env = MyCustomVecEnv(num_envs=4096)
   env = RewardScalingWrapper(env, scale=0.01)

Using with Isaac Gym
--------------------

Example integration with Isaac Gym:

.. code-block:: python

   from isaacgym import gymtorch, gymapi
   from apexrl.envs.vecenv import VecEnv

   class IsaacGymVecEnv(VecEnv):
       """ApexRL wrapper for Isaac Gym environments."""
       
       def __init__(self, cfg, device="cuda"):
           super().__init__(device=device)
           
           # Initialize Isaac Gym
           self.gym = gymapi.acquire_gym()
           self.sim = self._create_sim(cfg)
           
           # Create environments
           self.num_envs = cfg.num_envs
           self._create_envs(cfg)
           
           # Setup buffers
           self.num_obs = cfg.num_obs
           self.num_actions = cfg.num_actions
           
           self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(self.num_envs, device=device)
           self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
       
       def step(self, actions):
           # Step Isaac Gym physics
           self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(actions))
           self.gym.simulate(self.sim)
           self.gym.fetch_results(self.sim, True)
           
           # Compute observations and rewards
           self._compute_obs_rewards()
           
           # Check resets
           self.reset_buf = self.episode_length_buf >= self.max_episode_length
           
           if self.reset_buf.any():
               self.reset_idx(torch.where(self.reset_buf)[0])
           
           extras = {"time_outs": self.reset_buf.clone(), "log": {}}
           return self.obs_buf, self.rew_buf, self.reset_buf, extras

Best Practices
--------------

1. **Device Consistency**: Ensure all tensors are on the same device
2. **Partial Reset**: Always implement ``reset_idx()`` for efficiency
3. **Time Outs**: Use ``extras["time_outs"]`` to distinguish timeout from failure
4. **Logging**: Add useful metrics to ``extras["log"]`` for monitoring
5. **Buffer Pre-allocation**: Pre-allocate buffers to avoid memory allocation during step

Next Steps
----------

- Learn about :doc:`custom_network` architectures
- Explore :doc:`../modules/algorithms` for training details
- Check :doc:`../modules/environments` for API reference

Environments
============

ApexRL provides flexible interfaces for integrating various types of reinforcement learning environments.

Overview
--------

ApexRL is designed to work with:

1. **Vectorized Environments** - GPU-accelerated parallel environments
2. **Gymnasium Environments** - Standard single-threaded environments
3. **Custom Environments** - User-defined simulation backends

VecEnv (Vectorized Environment)
-------------------------------

The ``VecEnv`` class is the base interface for vectorized environments optimized for GPU execution.

Key Characteristics
~~~~~~~~~~~~~~~~~~~

- All environments run synchronously (same step function)
- Tensors are pre-allocated on GPU
- Supports partial resets for efficiency
- Designed for high-throughput training

Interface
~~~~~~~~~

.. code-block:: python

   class VecEnv(ABC):
       # Required attributes
       num_envs: int           # Number of parallel environments
       num_obs: int            # Observation dimension
       num_actions: int        # Action dimension
       device: torch.device    # Execution device
       
       # Required methods
       def reset(self) -> obs
       def step(self, actions) -> (obs, rewards, dones, extras)
       def reset_idx(self, env_ids) -> None

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.envs.vecenv.VecEnv
   :members:
   :undoc-members:
   :show-inheritance:

DummyVecEnv
~~~~~~~~~~~

A simple test environment:

.. code-block:: python

   from apexrl.envs.vecenv import DummyVecEnv

   env = DummyVecEnv(
       num_envs=4096,
       num_obs=48,
       num_actions=12,
       device="cuda",
       max_episode_length=1000,
   )

Gymnasium Integration
---------------------

ApexRL provides wrappers for standard Gymnasium environments.

GymVecEnv
~~~~~~~~~

Wraps multiple Gymnasium environments:

.. code-block:: python

   import gymnasium as gym
   from apexrl.envs.gym_wrapper import GymVecEnv

   def make_env():
       return gym.make("Pendulum-v1")

   env = GymVecEnv([make_env for _ in range(8)], device="cpu")

GymVecEnvContinuous
~~~~~~~~~~~~~~~~~~~

For continuous action spaces with automatic action scaling:

.. code-block:: python

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous

   env = GymVecEnvContinuous(
       [make_env for _ in range(8)],
       device="cpu",
       clip_actions=True,  # Clip to action space bounds
   )

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnv
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnvContinuous
   :members:
   :undoc-members:
   :show-inheritance:

Environment Wrappers
--------------------

VecEnvWrapper
~~~~~~~~~~~~~

Base class for creating environment wrappers:

.. code-block:: python

   from apexrl.envs.vecenv import VecEnvWrapper

   class NormalizeReward(VecEnvWrapper):
       def __init__(self, env):
           super().__init__(env)
           self.return_rms = RunningMeanStd()
           self.returns = torch.zeros(env.num_envs)
       
       def step(self, actions):
           obs, rewards, dones, extras = self.env.step(actions)
           
           # Update running statistics
           self.returns = self.returns * gamma + rewards
           self.return_rms.update(self.returns)
           
           # Normalize rewards
           rewards = rewards / torch.sqrt(self.return_rms.var + 1e-8)
           
           return obs, rewards, dones, extras

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.envs.vecenv.VecEnvWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Third-Party Integrations
------------------------

Isaac Gym / Isaac Lab
~~~~~~~~~~~~~~~~~~~~~

Example integration pattern:

.. code-block:: python

   from apexrl.envs.vecenv import VecEnv

   class IsaacLabVecEnv(VecEnv):
       def __init__(self, cfg, device="cuda"):
           from omni.isaac.lab.envs import ManagerBasedRLEnv
           
           self._env = ManagerBasedRLEnv(cfg)
           self.num_envs = self._env.num_envs
           self.num_obs = self._env.observation_space.shape[0]
           self.num_actions = self._env.action_space.shape[0]
           self.device = device
           
           # Initialize buffers
           self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
           self.rew_buf = torch.zeros(self.num_envs, device=device)
           self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
       
       def step(self, actions):
           obs_dict, rew, terminated, truncated, info = self._env.step(actions)
           
           self.obs_buf[:] = obs_dict["policy"]
           self.rew_buf[:] = rew
           self.reset_buf[:] = terminated | truncated
           
           extras = {
               "time_outs": truncated,
               "log": info.get("log", {}),
           }
           
           return self.obs_buf, self.rew_buf, self.reset_buf, extras
       
       def reset_idx(self, env_ids):
           self._env.reset_idx(env_ids)

Brax (JAX)
~~~~~~~~~~

For JAX-based environments:

.. code-block:: python

   import jax
   from jax import numpy as jnp

   class BraxVecEnv(VecEnv):
       def __init__(self, env_name, num_envs, device="cuda"):
           from brax import envs
           
           self._env = envs.create(env_name, batch_size=num_envs)
           self.num_envs = num_envs
           self.num_obs = self._env.observation_size
           self.num_actions = self._env.action_size
           
           # JIT compile step function
           self._step = jax.jit(self._env.step)
           self._reset = jax.jit(self._env.reset)
           
           # Initialize state
           rng = jax.random.PRNGKey(0)
           self._state = self._reset(rng)
       
       def step(self, actions):
           # Convert PyTorch to JAX
           actions_jax = jax.device_put(actions.cpu().numpy())
           
           self._state = self._step(self._state, actions_jax)
           
           # Convert back to PyTorch
           obs = torch.from_numpy(np.array(self._state.obs)).to(self.device)
           reward = torch.from_numpy(np.array(self._state.reward)).to(self.device)
           done = torch.from_numpy(np.array(self._state.done)).to(self.device)
           
           extras = {"time_outs": torch.zeros_like(done), "log": {}}
           
           return obs, reward, done, extras

Best Practices
--------------

1. **Pre-allocate Buffers**: Allocate observation/reward buffers in ``__init__``
2. **Use ``reset_idx``**: Implement partial reset for efficiency
3. **Handle Timeouts**: Set ``extras["time_outs"]`` correctly
4. **Device Consistency**: Ensure all tensors on same device
5. **Logging**: Add useful metrics to ``extras["log"]``

See Also
--------

- :doc:`../tutorials/custom_environment` - Detailed integration tutorial
- :doc:`../api/apexrl.envs` - Full API reference

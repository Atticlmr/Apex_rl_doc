Environments
============

ApexRL provides flexible interfaces for integrating various types of reinforcement learning environments.

Overview
--------

ApexRL is designed to work with:

1. **Vectorized Environments** - GPU-accelerated parallel environments
2. **Gymnasium Environments** - Standard single-threaded environments
3. **Custom Environments** - User-defined simulation backends
4. **Structured Observation Environments** - TensorDict / nested dict observation trees

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
   :noindex:

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

ApexRL provides wrappers for standard Gymnasium environments. The wrappers support
plain tensor observations, structured observations, and explicit timeout metadata.

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

For continuous action spaces with automatic clipping and action-space scaling:

.. code-block:: python

   from apexrl.envs.gym_wrapper import GymVecEnvContinuous

   env = GymVecEnvContinuous(
       [make_env for _ in range(8)],
       device="cpu",
       clip_actions=True,  # Clip to action space bounds
   )

This is the recommended wrapper for PPO on Gymnasium ``Box`` action spaces.
The default continuous PPO policy is an unsquashed Gaussian
(``use_tanh_squash=False``), and the wrapper handles action bounds.

Recommended structured observation format:

.. code-block:: python

   {
       "obs": {
           "image": image,
           "vector": vector,
       },
       "privileged_obs": {
           "state": state,
       },
   }

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnv
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: apexrl.envs.gym_wrapper.GymVecEnvContinuous
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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
   :noindex:

Third-Party Integrations
------------------------

Best Practices
--------------

1. **Pre-allocate Buffers**: Allocate observation/reward buffers in ``__init__``
2. **Use ``reset_idx``**: Implement partial reset for efficiency
3. **Handle Episode End Semantics**: Return ``terminated`` and ``truncated``
4. **Provide Final State**: Set ``extras["final_observation"]`` for truncated episodes
5. **Device Consistency**: Ensure all tensors on same device
6. **Logging**: Add useful metrics to ``extras["log"]``

See Also
--------

- :doc:`../tutorials/custom_environment` - Detailed integration tutorial
- :doc:`../API/apexrl.envs` - Full API reference

Buffers
=======

ApexRL provides efficient buffer implementations for storing and processing training data.

Overview
--------

Available buffer types:

1. **RolloutBuffer** - On-policy data storage (PPO)
2. **ReplayBuffer** - Off-policy data storage (DQN, SAC - planned)
3. **DistillationBuffer** - Policy distillation data

RolloutBuffer
-------------

The ``RolloutBuffer`` stores trajectories collected during environment interaction for on-policy algorithms like PPO.

Key Features
~~~~~~~~~~~~

- Efficient tensor storage on GPU
- Support for multi-dimensional observations
- Generalized Advantage Estimation (GAE)
- Privileged observations for asymmetric actor-critic

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from apexrl.buffer.rollout_buffer import RolloutBuffer

   buffer = RolloutBuffer(
       num_envs=4096,
       num_steps=24,
       obs_shape=(48,),
       device="cuda",
       num_privileged_obs=0,
   )

   # Collect data
   for step in range(24):
       actions, log_probs = actor.act(obs)
       next_obs, rewards, dones, extras = env.step(actions)
       values = critic.get_value(obs)
       
       buffer.add(
           observations=obs,
           privileged_observations=None,
           actions=actions,
           rewards=rewards,
           dones=dones.float(),
           values=values,
           log_probs=log_probs,
       )
       
       obs = next_obs

   # Compute advantages
   last_values = critic.get_value(obs)
   buffer.compute_returns_and_advantages(
       last_values=last_values,
       gamma=0.99,
       gae_lambda=0.95,
   )

   # Get training data
   data = buffer.get_all_data()

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.buffer.rollout_buffer.RolloutBuffer
   :members:
   :undoc-members:
   :show-inheritance:

Data Flow
~~~~~~~~~

The rollout data flow:

.. code-block:: text

   Environment Step → Store Transition → GAE Computation → Training
                          ↓
                   ┌─────────────┐
                   | observations |
                   | actions      |
                   | rewards      |
                   | dones        |
                   | values       |
                   | log_probs    |
                   └─────────────┘
                          ↓
                   ┌─────────────┐
                   | advantages  |
                   | returns     |
                   └─────────────┘

Memory Layout
~~~~~~~~~~~~~

Stored tensors have shape ``(num_steps, num_envs, ...)``:

.. code-block:: python

   # Observations: (num_steps, num_envs, *obs_shape)
   self.observations  # Shape: (24, 4096, 48)
   
   # Scalars: (num_steps, num_envs)
   self.rewards       # Shape: (24, 4096)
   self.dones         # Shape: (24, 4096)
   self.values        # Shape: (24, 4096)
   self.log_probs     # Shape: (24, 4096)
   self.advantages    # Shape: (24, 4096)
   self.returns       # Shape: (24, 4096)

GAE Computation
~~~~~~~~~~~~~~~

Generalized Advantage Estimation is computed backwards:

.. math::

   \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots

where :math:`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)` is the TD error.

.. code-block:: python

   def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
       advantages = torch.zeros_like(self.rewards)
       last_gae = torch.zeros(self.num_envs, device=self.device)
       
       for t in reversed(range(self.num_steps)):
           if t == self.num_steps - 1:
               next_values = last_values
           else:
               next_values = self.values[t + 1]
           
           delta = self.rewards[t] + gamma * next_values * (1 - self.dones[t]) - self.values[t]
           last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
           advantages[t] = last_gae
       
       self.advantages = advantages
       self.returns = advantages + self.values

ReplayBuffer
------------

For off-policy algorithms (DQN, SAC - planned):

.. code-block:: python

   from apexrl.buffer.replay_buffer import ReplayBuffer

   buffer = ReplayBuffer(
       capacity=1_000_000,
       obs_shape=(4,),
       action_shape=(2,),
       device="cuda",
   )

   # Store transition
   buffer.add(obs, action, reward, next_obs, done)

   # Sample batch
   batch = buffer.sample(batch_size=256)

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.buffer.replay_buffer.ReplayBuffer
   :members:
   :undoc-members:
   :show-inheritance:

DistillationBuffer
------------------

For policy distillation and imitation learning:

.. code-block:: python

   from apexrl.buffer.distillation_buffer import DistillationBuffer

   buffer = DistillationBuffer(
       capacity=100_000,
       obs_shape=(48,),
       device="cuda",
   )

   # Store expert demonstration
   buffer.add(obs, action)

   # Sample for distillation
   obs, expert_actions = buffer.sample(batch_size=256)

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.buffer.distillation_buffer.DistillationBuffer
   :members:
   :undoc-members:
   :show-inheritance:

Best Practices
--------------

1. **Pre-allocate**: Buffers pre-allocate memory for efficiency
2. **Device Placement**: Keep buffers on the same device as models
3. **Clear Buffers**: Call ``clear()`` between rollouts
4. **Batch Size**: Ensure batch size divides total transitions evenly
5. **GAE Lambda**: Typical values are 0.9-0.95

See Also
--------

- :doc:`../api/apexrl.buffer` - Full API reference
- :doc:`../modules/algorithms` - Algorithm implementations

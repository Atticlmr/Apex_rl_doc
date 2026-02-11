Algorithms
==========

ApexRL provides implementations of state-of-the-art reinforcement learning algorithms.

Available Algorithms
--------------------

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Type
     - Status
     - Description
   * - PPO
     - On-policy
     - ✅ Available
     - Proximal Policy Optimization
   * - DQN
     - Off-policy
     - 🚧 Planned
     - Deep Q-Network
   * - SAC
     - Off-policy
     - 🚧 Planned
     - Soft Actor-Critic

PPO (Proximal Policy Optimization)
----------------------------------

PPO is the primary algorithm currently available in ApexRL. It's an on-policy algorithm known for its stability and ease of use.

Key Features
~~~~~~~~~~~~

- Clipped surrogate objective for stable updates
- Generalized Advantage Estimation (GAE)
- Support for both continuous and discrete actions
- Asymmetric actor-critic (privileged information for critic)
- Separate or joint policy/value optimizers

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from apexrl.algorithms.ppo import PPO, PPOConfig
   from apexrl.envs.vecenv import DummyVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   # Create environment
   env = DummyVecEnv(num_envs=4096, num_obs=48, num_actions=12)

   # Configure PPO
   cfg = PPOConfig(
       num_steps=24,
       num_epochs=5,
       learning_rate=3e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
   )

   # Create agent
   agent = PPO(
       env=env,
       cfg=cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   # Train
   agent.learn(total_timesteps=10_000_000)

Configuration
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.config.PPOConfig
   :members:
   :undoc-members:

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.ppo.PPO
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Details
-----------------

PPO-Clip Objective
~~~~~~~~~~~~~~~~~~

The PPO-Clip objective function:

.. math::

   L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]

where:

- :math:`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}` is the probability ratio
- :math:`\hat{A}_t` is the estimated advantage
- :math:`\epsilon` is the clip range (typically 0.2)

Total Loss Function
~~~~~~~~~~~~~~~~~~~

.. math::

   L^{TOTAL}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)

where:

- :math:`L^{VF}` is the value function loss (MSE)
- :math:`S` is the entropy bonus
- :math:`c_1`, :math:`c_2` are coefficients

Hyperparameter Tuning
---------------------

General Guidelines
~~~~~~~~~~~~~~~~~~

.. list-table:: PPO Hyperparameters
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Description
   * - ``num_steps``
     - 2048-8192
     - Steps per environment per update
   * - ``num_epochs``
     - 3-10
     - Optimization epochs per batch
   * - ``learning_rate``
     - 1e-5 to 1e-3
     - Step size for optimization
   * - ``gamma``
     - 0.99-0.999
     - Discount factor
   * - ``gae_lambda``
     - 0.9-0.99
     - GAE lambda parameter
   * - ``clip_range``
     - 0.1-0.3
     - Clipping parameter
   * - ``ent_coef``
     - 0.0-0.01
     - Entropy coefficient
   * - ``vf_coef``
     - 0.25-1.0
     - Value function loss coefficient

Environment-Specific Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Isaac Gym (Legged Robots):**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=24,
       num_epochs=5,
       learning_rate=1e-3,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.0,
       batch_size=98304,
       minibatch_size=32768,
   )

**Gymnasium (Atari):**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=128,
       num_epochs=4,
       learning_rate=2.5e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.1,
       ent_coef=0.01,
   )

**Gymnasium (Mujoco):**

.. code-block:: python

   cfg = PPOConfig(
       num_steps=2048,
       num_epochs=10,
       learning_rate=3e-4,
       gamma=0.99,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.0,
   )

Advanced Features
-----------------

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

ApexRL supports multiple learning rate schedules:

.. code-block:: python

   cfg = PPOConfig(
       learning_rate_schedule="adaptive",  # or "linear", "constant"
       max_learning_rate=1e-3,
       min_learning_rate=1e-5,
   )

- **constant**: Fixed learning rate
- **linear**: Linear decay from initial to 0
- **adaptive**: Custom decay schedule

Value Function Clipping
~~~~~~~~~~~~~~~~~~~~~~~

Enable value function clipping for more stable training:

.. code-block:: python

   cfg = PPOConfig(
       clip_range_vf=0.2,  # None to disable
   )

Early Stopping
~~~~~~~~~~~~~~

Stop updates when KL divergence exceeds threshold:

.. code-block:: python

   cfg = PPOConfig(
       target_kl=0.015,  # None to disable
   )

Separate Optimizers
~~~~~~~~~~~~~~~~~~~

Use different learning rates for policy and value:

.. code-block:: python

   cfg = PPOConfig(
       use_policy_optimizer=True,
       policy_learning_rate=1e-4,
       value_learning_rate=3e-4,
   )

See Also
--------

- :doc:`../tutorials/first_agent` - Basic usage tutorial
- :doc:`../tutorials/custom_network` - Custom network architectures
- :doc:`../api/apexrl.algorithms.ppo` - Full API reference

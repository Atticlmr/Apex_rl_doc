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
     - ✅ Available
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
- Correct timeout bootstrapping with ``terminated`` / ``truncated`` semantics
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
   # PPO.learn() is a thin convenience wrapper around OnPolicyRunner.
   agent.learn(total_timesteps=10_000_000)

For new projects, prefer ``OnPolicyRunner`` as the primary training entrypoint
and treat ``PPO`` as the algorithm implementation plugged into that runner.

Configuration
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.config.PPOConfig
   :members:
   :undoc-members:
   :noindex:

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.ppo.PPO
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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

For continuous control tasks, the default PPO configuration uses
``use_tanh_squash=False`` and clamps learned log standard deviations with
``min_log_std`` / ``max_log_std`` to keep the policy numerically stable.

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
- :doc:`../API/apexrl.algorithms.ppo` - Full API reference

DQN (Deep Q-Network)
--------------------

DQN is available for discrete-action environments through ``ReplayBuffer``,
``OffPolicyRunner``, and MLP-based Q networks. The current implementation
supports standard DQN, Double DQN, and Dueling DQN.

Key Features
~~~~~~~~~~~~

- Experience replay with device-resident sampling
- Target network updates with hard or soft synchronization
- ``double_dqn`` target computation
- ``dueling`` Q-network architecture
- Epsilon-greedy exploration

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from gymnasium import make

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPQNetwork

   env = GymVecEnv([lambda: make("CartPole-v1") for _ in range(4)], device="cpu")

   cfg = DQNConfig(
       double_dqn=True,
       dueling=True,
       learning_starts=1_000,
       batch_size=128,
   )

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       q_network_class=MLPQNetwork,
       device=torch.device("cpu"),
   )
   runner.learn(total_timesteps=200_000)

``DQN.learn()`` is also available as a convenience wrapper, but
``OffPolicyRunner`` is the canonical training entrypoint for off-policy methods.

Configuration
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.dqn.config.DQNConfig
   :members:
   :undoc-members:
   :noindex:

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.dqn.dqn.DQN
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- Set ``double_dqn=True`` to reduce overestimation bias.
- Set ``dueling=True`` to split value and advantage estimation in the Q network.
- ``MLPQNetwork`` supports both standard and dueling layouts through config only.

Smoke Benchmarks
~~~~~~~~~~~~~~~~

The benchmark script includes lightweight DQN and Dueling DQN smoke tasks:

.. code-block:: bash

   python benchmarks/run_smoke_benchmarks.py --iterations 1 --num-envs 1

Included off-policy smoke tasks:

- ``CartPole-v1 (DQN)``
- ``CartPole-v1 (Dueling DQN)``
- ``Acrobot-v1 (DQN)``
- ``Acrobot-v1 (Dueling DQN)``

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
   * - RecurrentPPO
     - On-policy
     - ✅ Available
     - PPO with sequence minibatches and recurrent actor/critic state
   * - DQN
     - Off-policy
     - ✅ Available
     - Deep Q-Network
   * - SAC
     - Off-policy
     - ✅ Available
     - Soft Actor-Critic
   * - MAPPO
     - Multi-agent on-policy
     - ✅ Available
     - Multi-Agent PPO with centralized critic support
   * - IPPO
     - Multi-agent on-policy
     - ✅ Available
     - Independent PPO with decentralized critics
   * - HAPPO
     - Multi-agent on-policy
     - ✅ Available
     - Heterogeneous-Agent PPO with sequential policy updates

PPO (Proximal Policy Optimization)
----------------------------------

PPO is an on-policy algorithm known for its stability and ease of use.

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

Recurrent PPO
~~~~~~~~~~~~~

``RecurrentPPO`` keeps actor and critic hidden state during rollout collection
and trains on contiguous sequence minibatches instead of shuffled single-step
transitions. It accepts custom recurrent ``actor_class`` and ``critic_class``
arguments, matching the normal PPO construction pattern.

Multi-Agent PPO Algorithms
--------------------------

MAPPO, IPPO and HAPPO share the same multi-agent runner and rollout storage.
MAPPO uses centralized training with decentralized execution: each actor
consumes local agent observations, while critics can consume a centralized
environment state. IPPO keeps the same per-agent actor interface but uses local
observations for each critic by setting ``centralized_critic=False``. HAPPO uses
separate actors and sequential policy updates with correction factors from
agents updated earlier in the current update order.

.. code-block:: python

   from apexrl.models import MLPActor, MLPCritic
   from apexrl.multiagent import HAPPO, HAPPOConfig, IPPO, IPPOConfig, MAPPO, MAPPOConfig

   mappo_cfg = MAPPOConfig(centralized_critic=True, share_actor=True)
   mappo_agent = MAPPO(
       env=multiagent_env,
       cfg=mappo_cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   ippo_cfg = IPPOConfig(share_actor=True)
   ippo_agent = IPPO(
       env=multiagent_env,
       cfg=ippo_cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   happo_cfg = HAPPOConfig(centralized_critic=True, share_actor=False)
   happo_agent = HAPPO(
       env=multiagent_env,
       cfg=happo_cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

Paper References
----------------

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Reference
     - Link
   * - PPO
     - Proximal Policy Optimization Algorithms
     - https://arxiv.org/abs/1707.06347
   * - DQN
     - Playing Atari with Deep Reinforcement Learning
     - https://arxiv.org/abs/1312.5602
   * - SAC
     - Soft Actor-Critic Algorithms and Applications
     - https://arxiv.org/abs/1812.05905
   * - MAPPO
     - The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games
     - https://arxiv.org/abs/2103.01955
   * - IPPO
     - Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?
     - https://arxiv.org/abs/2011.09533
   * - HAPPO
     - Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning
     - https://arxiv.org/abs/2109.11251

Configuration
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.ppo.config.PPOConfig
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: apexrl.algorithms.ppo.config_rnn.RecurrentPPOConfig
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

.. autoclass:: apexrl.algorithms.ppo.ppo_rnn.RecurrentPPO
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
- ``Pendulum-v1 (SAC)``
- ``MountainCarContinuous-v0 (SAC)``

SAC (Soft Actor-Critic)
-----------------------

SAC is available for continuous-control environments through
``ReplayBuffer``, ``OffPolicyRunner``, a squashed Gaussian actor, and
twin ``Q(s, a)`` critics.

Key Features
~~~~~~~~~~~~

- Off-policy continuous control with replay reuse
- Squashed Gaussian actor with action-bound rescaling
- Twin critics and target critics
- Automatic entropy-temperature tuning
- Shared ``OffPolicyRunner`` training entrypoint

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from gymnasium import make

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.sac import SACConfig
   from apexrl.envs.gym_wrapper import GymVecEnvContinuous

   env = GymVecEnvContinuous(
       [lambda: make("Pendulum-v1") for _ in range(2)],
       device="cpu",
   )

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
       device=torch.device("cpu"),
   )
   runner.learn(total_timesteps=200_000)

``SAC.learn()`` is also available as a convenience wrapper, but
``OffPolicyRunner`` remains the canonical training entrypoint.

Configuration
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.sac.config.SACConfig
   :members:
   :undoc-members:
   :noindex:

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.algorithms.sac.sac.SAC
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Algorithm Details
~~~~~~~~~~~~~~~~~

SAC critic target:

.. math::

   y = r + \gamma (1-d)\left(\min(Q_1'(s', a'), Q_2'(s', a')) - \alpha \log \pi(a'|s')\right)

Twin critic losses:

.. math::

   L_{Q_i} = \mathbb{E}\left[(Q_i(s, a) - y)^2\right]

Actor loss:

.. math::

   L_{\pi} = \mathbb{E}\left[\alpha \log \pi(a|s) - \min(Q_1(s, a), Q_2(s, a))\right]

Temperature loss:

.. math::

   L_{\alpha} = -\mathbb{E}\left[\log \alpha \cdot (\log \pi(a|s) + \mathcal{H}_{target})\right]

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- The default actor is ``MLPSquashedGaussianActor``.
- The default critics are twin ``MLPContinuousQNetwork`` instances.
- ``ReplayBuffer`` stores continuous vector actions by setting
  ``action_shape=env.action_space.shape``.
- Bootstrap masking follows Gymnasium semantics: true terminals stop
  bootstrapping; truncation should preserve the final observation for
  value estimation.

Smoke Benchmarks
~~~~~~~~~~~~~~~~

The benchmark script includes lightweight SAC smoke tasks:

.. code-block:: bash

   python benchmarks/run_smoke_benchmarks.py --iterations 1 --num-envs 1

Included SAC smoke tasks:

- ``Pendulum-v1 (SAC)``
- ``MountainCarContinuous-v0 (SAC)``

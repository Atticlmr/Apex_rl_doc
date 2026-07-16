Changelog
=========

All notable changes to ApexRL will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.3.1] - 2026-07-16
--------------------

Added
~~~~~

- Reworked FlashSAC around the reference algorithm's residual BatchNorm/RMSNorm
  actor and critic architecture with unit-row weight projection.
- Added twin categorical critics and entropy-regularized Bellman projection.
- Added discounted-return reward normalization, target-sigma entropy tuning,
  delayed actor updates, and truncated-zeta exploration noise repetition.
- Added FlashSAC checkpoint coverage for reward-normalizer and exploration
  state, plus multimodal and privileged-observation training tests.

Fixed
~~~~~

- PPO and multi-agent runners now round timestep targets up to complete
  rollouts instead of silently undershooting the requested target.
- Checkpoint resume now continues global iteration counters, learning-rate
  schedules, logs, and checkpoint filenames instead of restarting at zero.

Changed
~~~~~~~

- Updated the documented algorithm matrix for Recurrent PPO, TD3, FlashSAC,
  MAPPO, IPPO, and HAPPO.
- Added Python 3.10 through 3.13 test CI and a dedicated Ruff lint job.
- Added independent FlashSAC update, runner, and checkpoint coverage.

[0.3.0] - 2026-05-24
--------------------

Added
~~~~~

single-agent:

- add support for Recurrent-PPO
- Added TD3 with delayed actor updates, target policy smoothing, and bounded
  deterministic actions.
- Added FlashSAC with large-batch defaults, reward scaling, target clipping,
  and optional critic/actor norm controls.
- Added structured and multimodal observations throughout environment wrappers,
  buffers, default models, and single-agent algorithms.

multi-agent:

- Full multi-agent reinforcement learning support with ``MAPPO``, ``IPPO``,
  and ``HAPPO`` algorithms, including dedicated configs and training loops.
- ``MultiAgentRunner`` for unified multi-agent training orchestration with
  logging, checkpointing, and callback support on par with single-agent
  runners.
- ``MultiAgentVecEnv`` base class and cooperative environment wrappers for
  batched multi-agent episode collection.
- ``MultiAgentRolloutBuffer`` for structured storage of multi-agent
  observations, actions, rewards, and terminal flags.


TODO 
~~~~

- Recurrent-Network support for multi-agent RL.
- JAX support.

[0.2.2] - 2026-05-13
--------------------

Changed
~~~~~~~

- Improved runner logging so environment extras are recorded only from
  user-selected ``extra_log_keys`` instead of hard-coded extras names.
- Refined the logging documentation to describe configurable extras logging
  more clearly.

[0.2.1] - 2026-04-22
--------------------

Added
~~~~~

- Official ``Muon`` optimizer support across PPO, DQN, and SAC using the
  bundled mixed Muon-plus-AuxAdam implementation.
- Smoke-test coverage for PPO, DQN, and SAC training with
  ``optimizer="muon"``.

Changed
~~~~~~~

- Optimizer construction now routes ``Muon`` through parameter grouping so
  matrix-like hidden weights use Muon while scalar, bias, and output-head
  parameters stay on the auxiliary Adam path.

Fixed
~~~~~

- Structured observation tensors now preserve leaf dtypes end to end, so
  multimodal environments keep ``uint8`` image leaves and other non-float
  modalities intact through wrappers, buffers, and algorithm input paths.
- API reference now includes the missing DQN pages, removes duplicate SAC
  entries, and restores the missing OffPolicyRunner documentation entry.
- Environment documentation drops the unverified Brax, Isaac Gym, and Isaac
  Lab examples until tested integration guides are added back.

[0.2.0] - 2026-04-22
--------------------

Added
~~~~~

- End-to-end ``TensorDict`` and nested-dict observation support across PPO,
  DQN, SAC, vectorized Gymnasium wrappers, replay buffers, and rollout
  buffers.
- Multimodal observation support for default training stacks, including common
  combinations such as image plus vector inputs.
- Privileged critic observation support for asymmetric actor-critic training,
  with separate actor and critic observation branches in PPO and SAC.
- Structured-observation smoke and regression coverage for PPO, DQN, SAC, and
  buffer behavior.

Changed
~~~~~~~

- Default MLP actor, critic, and Q-network implementations now flatten nested
  observation leaves recursively, so they can be used directly with structured
  inputs.
- README and bilingual documentation now describe the structured observation
  format, PPO training flow, SAC critic branches, and multimodal custom-network
  authoring.

Fixed
~~~~~

- Off-policy runners and Gymnasium wrappers now preserve structured final
  observations consistently when episodes terminate or truncate.

[0.0.3] - 2026-04-20
--------------------

Fixed
~~~~~

- Release workflow now skips PyPI publishing when ``PYPI_API_TOKEN`` is not configured.

Documentation
~~~~~~~~~~~~~

- Updated English and Chinese docs to reflect the new PPO training flow,
  continuous-action defaults, and timeout semantics.

[0.0.2] - 2026-04-20
--------------------

Changed
~~~~~~~

- ``PPO.learn()`` now delegates to ``OnPolicyRunner`` so there is a single
  on-policy training loop for logging, checkpointing, and callbacks.
- Continuous-action PPO defaults now use an unsquashed Gaussian policy with
  bounded log standard deviation.

Fixed
~~~~~

- ``RolloutBuffer`` now stores multi-dimensional continuous actions correctly.
- Gymnasium wrappers now expose ``terminated``, ``truncated``, and
  ``final_observation`` so PPO can bootstrap truncated episodes correctly.
- Added smoke coverage for ``CartPole-v1``, ``Pendulum-v1``, and
  ``MountainCarContinuous-v0``.

[0.0.1] - 2026-02-11
--------------------

Initial release of ApexRL.

Added
~~~~~

Core Features
^^^^^^^^^^^^^

- PPO (Proximal Policy Optimization) algorithm implementation
- OnPolicyRunner for managing training loops
- Vectorized environment interface (VecEnv)
- Gymnasium environment wrappers (GymVecEnv, GymVecEnvContinuous)

Networks
^^^^^^^^

- Base classes: Actor, ContinuousActor, DiscreteActor, Critic
- MLP implementations: MLPActor, MLPCritic, MLPDiscreteActor
- CNN implementations: CNNActor, CNNCritic
- Network construction utilities (build_mlp)

Buffers
^^^^^^^

- RolloutBuffer for on-policy algorithms
- ReplayBuffer for off-policy algorithms (planned)
- DistillationBuffer for policy distillation (planned)

Optimizers
^^^^^^^^^^

- Support for Adam, AdamW optimizers
- Experimental Muon optimizer support

Configuration
^^^^^^^^^^^^^

- PPOConfig dataclass with comprehensive hyperparameters
- Learning rate scheduling (constant, linear, adaptive)

Documentation
^^^^^^^^^^^^^

- Sphinx documentation with Furo theme
- API reference documentation
- Tutorial guides
- English and Chinese documentation

Planned
~~~~~~~

Algorithms
^^^^^^^^^^

- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

Features
^^^^^^^^

- Observation normalization
- Reward normalization
- Multi-GPU training support
- Distributed training

[Unreleased]
------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

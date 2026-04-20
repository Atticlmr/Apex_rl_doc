Changelog
=========

All notable changes to ApexRL will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

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

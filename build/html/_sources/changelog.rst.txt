Changelog
=========

All notable changes to ApexRL will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

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

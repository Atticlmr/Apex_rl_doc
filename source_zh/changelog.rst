更新日志
========

ApexRL 的所有显著变更都将记录在此文件中。

格式基于 `Keep a Changelog <https://keepachangelog.com/zh-CN/1.0.0/>`_，
本项目遵循 `语义化版本控制 <https://semver.org/lang/zh-CN/>`_。

[0.2.1] - 2026-04-22
--------------------

新增
~~~~

- 正式在 PPO、DQN、SAC 中接入仓库内置的 ``Muon`` 优化器实现，并沿用
  Muon 加辅助 Adam 的混合优化路径。
- 新增 ``optimizer="muon"`` 下 PPO、DQN、SAC 的 smoke 测试覆盖。

变更
~~~~

- 优化器构建逻辑现在会为 ``Muon`` 自动做参数分组，让隐藏层矩阵权重走
  Muon，bias、标量参数和输出头相关参数继续走辅助 Adam。

修复
~~~~

- 结构化观测链路现在会端到端保留各叶子节点的原始 dtype，使多模态环境中的
  ``uint8`` 图像和其他非浮点模态能正确穿过包装器、缓冲区和算法输入链路。
- API Reference 补齐了缺失的 DQN 页面，去掉了重复的 SAC 条目，并恢复了
  OffPolicyRunner 的文档入口。
- 在集成验证补齐前，临时移除了 Brax、Isaac Gym 和 Isaac Lab 的环境文档示例。

[0.2.0] - 2026-04-22
--------------------

新增
~~~~

- 在 PPO、DQN、SAC、Gymnasium 向量环境包装器、ReplayBuffer 和
  RolloutBuffer 中打通 ``TensorDict`` / 嵌套字典观测支持。
- 默认训练链路支持多模态观测，例如图像加向量的组合输入。
- 支持 asymmetric actor-critic 的 privileged critic 观测，并在 PPO 与
  SAC 中拆分 actor / critic 观测分支。
- 新增 PPO、DQN、SAC 以及缓冲区在结构化观测下的 smoke 与回归测试覆盖。

变更
~~~~

- 默认 MLP actor、critic 和 Q 网络现在会递归展平嵌套观测叶子节点，可直接
  处理结构化输入。
- README 与中英文文档补充了结构化观测格式、PPO 训练流程、SAC critic 分支
  以及多模态自定义网络的写法。

修复
~~~~

- 异策略 runner 与 Gymnasium 包装器现在会在回合终止或截断时一致地保留
  结构化 ``final_observation``。

[0.0.3] - 2026-04-20
--------------------

修复
~~~~

- 当未配置 ``PYPI_API_TOKEN`` 时，发布工作流会跳过 PyPI 发布而不是直接失败。

文档
~~~~

- 更新中英文文档，同步新的 PPO 训练入口、连续动作默认行为和 timeout 语义。

[0.0.2] - 2026-04-20
--------------------

变更
~~~~

- ``PPO.learn()`` 现在委托给 ``OnPolicyRunner``，统一同策略训练循环、日志和检查点行为。
- 连续动作 PPO 默认改为未经过 ``tanh`` 压缩的高斯策略，并限制可学习标准差范围。

修复
~~~~

- ``RolloutBuffer`` 现在能正确存储多维连续动作。
- Gymnasium 包装器现在返回 ``terminated``、``truncated`` 和
  ``final_observation``，使 PPO 能正确处理截断回合的 bootstrap。
- 新增 ``CartPole-v1``、``Pendulum-v1`` 和
  ``MountainCarContinuous-v0`` 的 smoke 覆盖。

[0.0.1] - 2026-02-11
--------------------

ApexRL 初始版本。

新增
~~~~

核心特性
^^^^^^^^

- PPO（近端策略优化）算法实现
- OnPolicyRunner 用于管理训练循环
- 向量化环境接口（VecEnv）
- Gymnasium 环境包装器（GymVecEnv、GymVecEnvContinuous）

网络
^^^^

- 基类：Actor、ContinuousActor、DiscreteActor、Critic
- MLP 实现：MLPActor、MLPCritic、MLPDiscreteActor
- CNN 实现：CNNActor、CNNCritic
- 网络构建工具（build_mlp）

缓冲区
^^^^^^

- RolloutBuffer 用于同策略算法
- ReplayBuffer 用于异策略算法（计划中）
- DistillationBuffer 用于策略蒸馏（计划中）

优化器
^^^^^^

- 支持 Adam、AdamW 优化器
- 实验性 Muon 优化器支持

配置
^^^^

- PPOConfig 数据类，含完整超参数
- 学习率调度（constant、linear、adaptive）

文档
^^^^

- Sphinx 文档，使用 Furo 主题
- API 参考文档
- 教程指南
- 中英文文档

计划中
~~~~~~

算法
^^^^

- DQN（深度 Q 网络）
- SAC（软 Actor-Critic）
- TD3（双延迟 DDPG）

特性
^^^^

- 观测归一化
- 奖励归一化
- 多 GPU 训练支持
- 分布式训练

[未发布]
--------

新增
~~~~

变更
~~~~

弃用
~~~~

移除
~~~~

修复
~~~~

安全
~~~~

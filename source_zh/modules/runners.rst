Runners
=======

Runners 管理 RL 智能体的训练循环、日志记录、检查点保存和评估。

概述
----

Runner 模块为以下功能提供高级接口：

1. **训练管理** - 自动化训练循环
2. **日志记录** - TensorBoard、wandb 和 SwanLab 集成
3. **检查点保存** - 模型保存和加载
4. **评估** - 定期智能体评估
5. **同策略 / 异策略入口** - PPO、DQN 和 SAC 风格训练

OnPolicyRunner
--------------

``OnPolicyRunner`` 是 PPO 等同策略算法的标准训练入口。它负责外层训练循环，
而算法对象只负责 rollout 解释、损失计算和优化步骤。

关键特性
~~~~~~~~

- 带回调的自动化训练循环
- TensorBoard / wandb / SwanLab 指标记录
- 定期检查点保存
- 奖励组件跟踪
- 环境指标记录
- 统一处理 ``terminated`` / ``truncated`` 超时语义

基本用法
~~~~~~~~

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs",
       save_dir="./checkpoints",
       log_interval=10,
       save_interval=100,
   )

   # 训练
   runner.learn(total_timesteps=10_000_000)

   # 评估
   stats = runner.eval(num_episodes=100)

   # 保存
   runner.save_checkpoint("final_model.pt")

配置
~~~~

.. code-block:: python

   runner = OnPolicyRunner(
       env=env,                          # 向量化环境
       algorithm="ppo",                  # 算法名称
       actor_class=MLPActor,             # Actor 网络类
       critic_class=MLPCritic,           # Critic 网络类
       actor_cfg={"hidden_dims": [256]}, # Actor 网络配置
       critic_cfg={"hidden_dims": [256]}, # Critic 网络配置
       log_dir="./logs",                 # 所选日志后端使用的日志目录
       save_dir="./checkpoints",         # 检查点目录
       device=torch.device("cuda"),      # 训练设备
       log_interval=10,                  # 每 N 次迭代记录
       save_interval=100,                # 每 N 次迭代保存
       log_reward_components=True,       # 记录奖励组件
   )

如果您直接实例化 ``PPO``，``PPO.learn()`` 也会委托给这个 runner，
从而只维护一套同策略训练循环。

API 参考
~~~~~~~~

.. autoclass:: apexrl.agent.on_policy_runner.OnPolicyRunner
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

训练循环
~~~~~~~~

训练循环结构：

.. code-block:: text

   for iteration in range(total_iterations):
       # 1. 收集 rollout
       rollout_stats = collect_rollout()
       
       # 2. 更新策略
       update_stats = update()
       
       # 3. 调整学习率
       adjust_learning_rate()
       
       # 4. 记录指标
       if iteration % log_interval == 0:
           log_metrics()
       
       # 5. 保存检查点
       if iteration % save_interval == 0:
           save_checkpoint()

回调
~~~~

为训练事件添加自定义回调：

.. code-block:: python

   def on_iteration_start(runner):
       print(f"开始迭代 {runner.iteration}")

   def on_iteration_end(runner, stats):
       if stats["rollout/mean_reward"] > threshold:
           print("达到奖励阈值！")

   runner.add_callback("pre_iteration", on_iteration_start)
   runner.add_callback("post_iteration", on_iteration_end)

可用事件：

- ``pre_iteration`` - 每次训练迭代前
- ``post_iteration`` - 每次训练迭代后
- ``pre_rollout`` - 收集 rollout 前
- ``post_rollout`` - 收集 rollout 后
- ``pre_update`` - 策略更新前
- ``post_update`` - 策略更新后

日志记录
~~~~~~~~

Runner 会把指标记录到当前配置的日志后端：

**训练指标：**

- ``train/policy_loss`` - 策略损失
- ``train/value_loss`` - 价值函数损失
- ``train/entropy_loss`` - 熵损失
- ``train/approx_kl`` - 近似 KL 散度
- ``train/learning_rate`` - 当前学习率

**回合指标：**

- ``episode/mean_reward`` - 平均回合奖励
- ``episode/mean_length`` - 平均回合长度

**Rollout 指标：**

- ``rollout/mean_reward`` - 平均步奖励
- ``rollout/mean_value`` - 平均价值估计

**环境指标：**

通用环境 extras 默认不会记录。可以在 ``cfg.extra_log_keys`` 中配置要记录的
顶层 extras key，runner 会递归展开其中的标量值，并写到 ``extra/<key>/...``：

.. code-block:: python

   cfg = PPOConfig(
       extra_log_keys=["log", "time_outs", "terminated", "truncated"],
   )

``PPOConfig``、``DQNConfig`` 和 ``SACConfig`` 都支持这个字段。展开 extras
用于日志时，会跳过 ``final_observation`` 这类较大的观测载荷。

后端配置示例：

.. code-block:: python

   cfg = PPOConfig(
       logger_backend="wandb",
       logger_kwargs={
           "project": "apexrl",
           "tags": ["ppo"],
       },
   )

``tensorboard`` 默认包含在基础安装中；``wandb`` 和 ``swanlab`` 需要先安装对应的可选依赖。

奖励组件
~~~~~~~~

跟踪各个奖励组件：

.. code-block:: python

   # 在环境 step() 中
   extras = {
       "time_outs": truncated,  # 向后兼容别名
       "terminated": terminated,
       "truncated": truncated,
       "final_observation": final_obs,
       "reward_components": {
           "velocity": velocity_reward,
           "energy": -energy_penalty,
           "stability": stability_reward,
       },
       "log": {
           "/robot/height_mean": robot_height.mean().item(),
       },
   }

Runner 自动：

1. 每回合累加奖励组件
2. 回合结束时记录平均值
3. 当设置 ``extra_log_keys`` 时，把配置的 extras key 记录到 ``extra/...``

``reward_components`` 在 ``log_reward_components=True`` 时仍会按回合累计并记录。
如果同时把 ``"reward_components"`` 放进 ``extra_log_keys``，这些值还会作为
step-level extras 记录到 ``extra/reward_components/...``。

超时语义遵循 Gymnasium：``terminated`` 表示真实终止，
``truncated`` 表示时间限制或外部截断，``final_observation``
用于在截断回合上做 value bootstrap。

检查点保存
~~~~~~~~~~

保存和加载训练检查点：

.. code-block:: python

   # 保存检查点
   runner.save_checkpoint("model.pt")
   
   # 带迭代号保存
   runner.save_checkpoint(f"checkpoint_{runner.iteration}.pt")

   # 加载检查点
   runner.load_checkpoint("model.pt")

检查点包括：

- Actor 网络状态
- Critic 网络状态
- 优化器状态
- 训练迭代数
- 总步数
- 配置

评估
~~~~

评估训练好的智能体：

.. code-block:: python

   # 运行评估
   stats = runner.eval(num_episodes=100)
   
   print(f"平均奖励: {stats['eval/mean_reward']:.2f}")
   print(f"奖励标准差: {stats['eval/std_reward']:.2f}")
   print(f"最小奖励: {stats['eval/min_reward']:.2f}")
   print(f"最大奖励: {stats['eval/max_reward']:.2f}")

评估以确定性模式运行（``deterministic=True``）。

预配置智能体
~~~~~~~~~~~~

使用预配置智能体而非自动创建：

.. code-block:: python

   from apexrl.algorithms.ppo import PPO, PPOConfig

   cfg = PPOConfig(learning_rate=3e-4)
   agent = PPO(
       env=env,
       cfg=cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   runner = OnPolicyRunner(
       agent=agent,
       env=env,
       cfg=cfg,
       log_dir="./logs",
   )

最佳实践
--------

1. **日志目录**：始终指定 ``log_dir`` 以跟踪实验
2. **保存间隔**：根据训练时长设置 ``save_interval``
3. **回调**：使用回调添加自定义逻辑而无需修改 runner
4. **设备**：让 runner 自动检测设备，或显式指定
5. **评估**：定期运行评估以监控进度

另请参阅
--------

- :doc:`../tutorials/first_agent` - 详细使用教程
- :doc:`../API/apexrl.agent` - 完整 API 参考

OffPolicyRunner
---------------

``OffPolicyRunner`` 是 DQN、SAC 等异策略算法的标准训练入口。它负责环境交互、
经验回放写入和定期梯度更新，具体探索语义由算法自身决定。

关键特性
~~~~~~~~

- 基于 replay buffer 的训练循环
- 算法自定义的探索处理
- 由算法对象控制 target network 更新
- 统一的日志、检查点和评估流程

基本用法
~~~~~~~~

.. code-block:: python

   import torch
   from gymnasium import make

   from apexrl.agent.off_policy_runner import OffPolicyRunner
   from apexrl.algorithms.dqn import DQNConfig
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models import MLPQNetwork

   env = GymVecEnv([lambda: make("CartPole-v1") for _ in range(4)], device="cpu")
   cfg = DQNConfig(double_dqn=True, dueling=True)

   runner = OffPolicyRunner(
       env=env,
       cfg=cfg,
       q_network_class=MLPQNetwork,
       device=torch.device("cpu"),
   )
   runner.learn(total_timesteps=200_000)

API 参考
~~~~~~~~

.. autoclass:: apexrl.agent.off_policy_runner.OffPolicyRunner
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

训练循环
~~~~~~~~

异策略训练循环结构：

.. code-block:: text

   for step in range(total_timesteps):
       action = epsilon_greedy(q_network, obs)
       next_obs, reward, done, extras = env.step(action)
       replay_buffer.add(obs, action, reward, next_obs, done)

       if step >= learning_starts and step % train_freq == 0:
           for _ in range(gradient_steps):
               update()

       if step % save_interval == 0:
           save_checkpoint()

日志记录
~~~~~~~~

常见 DQN 指标：

- ``train/q_loss`` - TD 损失
- ``train/mean_q`` - 被选动作的平均 Q 值
- ``train/td_target_mean`` - TD target 平均值
- ``exploration/epsilon`` - 当前 epsilon
- ``buffer/size`` - replay buffer 大小

SAC 还会额外记录：

- ``train/actor_loss`` - 策略目标
- ``train/critic1_loss`` / ``train/critic2_loss`` - twin critic 损失
- ``train/alpha`` - 当前熵温度
- ``train/entropy`` - 策略熵代理指标

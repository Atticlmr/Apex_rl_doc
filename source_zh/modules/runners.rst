Runners
=======

Runners 管理 RL 智能体的训练循环、日志记录、检查点保存和评估。

概述
----

Runner 模块为以下功能提供高级接口：

1. **训练管理** - 自动化训练循环
2. **日志记录** - TensorBoard 集成
3. **检查点保存** - 模型保存和加载
4. **评估** - 定期智能体评估

OnPolicyRunner
--------------

``OnPolicyRunner`` 管理同策略算法如 PPO 的训练。

关键特性
~~~~~~~~

- 带回调的自动化训练循环
- TensorBoard 指标记录
- 定期检查点保存
- 奖励组件跟踪
- 环境指标记录

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
       log_dir="./logs",                 # TensorBoard 日志目录
       save_dir="./checkpoints",         # 检查点目录
       device=torch.device("cuda"),      # 训练设备
       log_interval=10,                  # 每 N 次迭代记录
       save_interval=100,                # 每 N 次迭代保存
       log_reward_components=True,       # 记录奖励组件
   )

API 参考
~~~~~~~~

.. autoclass:: apexrl.agent.on_policy_runner.OnPolicyRunner
   :members:
   :undoc-members:
   :show-inheritance:

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

Runner 自动记录指标到 TensorBoard：

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

环境 ``extras["log"]`` 中的自定义指标会自动记录。

奖励组件
~~~~~~~~

跟踪各个奖励组件：

.. code-block:: python

   # 在环境 step() 中
   extras = {
       "time_outs": time_outs,
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
3. 从 ``extras["log"]`` 记录自定义指标

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
- :doc:`../api/apexrl.agent` - 完整 API 参考

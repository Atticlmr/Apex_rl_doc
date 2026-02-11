您的第一个 RL 智能体
===================

本教程将指导您使用 ApexRL 创建和训练您的第一个强化学习智能体。

概述
----

完成本教程后，您将：

- 了解 ApexRL 的基本组件
- 创建向量化环境
- 配置和训练 PPO 智能体
- 评估和保存训练好的模型

前置条件
--------

确保您已安装 ApexRL：

.. code-block:: bash

   pip install -e .

第一步：导入库
-------------

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

第二步：创建环境
---------------

ApexRL 使用向量化环境进行并行训练。让我们创建 8 个并行的 Pendulum-v1 实例：

.. code-block:: python

   def make_env():
       """创建单个环境的工厂函数。"""
       return gym.make("Pendulum-v1")

   # 创建 8 个并行环境
   num_envs = 8
   env = GymVecEnv([make_env for _ in range(num_envs)], device="cpu")

   print(f"环境数量: {env.num_envs}")
   print(f"观测维度: {env.num_obs}")
   print(f"动作维度: {env.num_actions}")

第三步：配置 Runner
-------------------

``OnPolicyRunner`` 处理训练循环、日志记录和检查点保存：

.. code-block:: python

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",           # 使用的算法
       actor_class=MLPActor,      # Actor 网络类
       critic_class=MLPCritic,    # Critic 网络类
       log_dir="./logs",          # TensorBoard 日志目录
       save_dir="./checkpoints",  # 检查点保存目录
       log_interval=10,           # 每 10 次迭代记录一次
       save_interval=100,         # 每 100 次迭代保存一次
   )

第四步：训练智能体
-----------------

训练智能体指定的步数：

.. code-block:: python

   # 训练 100,000 步
   runner.learn(total_timesteps=100_000)

训练过程中，您将看到类似输出：

.. code-block:: text

   Training for 520 iterations (104,000 steps)
   Iter 0/520 | Steps 0 | FPS 0 | Policy Loss -0.0012 | Value Loss 0.0234 | KL 0.0012
   Iter 10/520 | Steps 1,920 | FPS 3421 | Policy Loss -0.0023 | Value Loss 0.0187 | KL 0.0008 | Reward -456.23
   ...

第五步：评估智能体
-----------------

评估训练好的智能体：

.. code-block:: python

   eval_stats = runner.eval(num_episodes=10)
   
   print(f"平均奖励: {eval_stats['eval/mean_reward']:.2f}")
   print(f"奖励标准差: {eval_stats['eval/std_reward']:.2f}")
   print(f"最小奖励: {eval_stats['eval/min_reward']:.2f}")
   print(f"最大奖励: {eval_stats['eval/max_reward']:.2f}")

第六步：保存和加载
-----------------

保存训练好的模型：

.. code-block:: python

   runner.save_checkpoint("final_model.pt")

加载已保存的模型：

.. code-block:: python

   runner.load_checkpoint("final_model.pt")

完整代码
--------

以下是完整的训练脚本：

.. code-block:: python

   import gymnasium as gym
   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   def main():
       # 创建环境
       def make_env():
           return gym.make("Pendulum-v1")
       
       env = GymVecEnv([make_env for _ in range(8)], device="cpu")
       
       # 创建 runner
       runner = OnPolicyRunner(
           env=env,
           algorithm="ppo",
           actor_class=MLPActor,
           critic_class=MLPCritic,
           log_dir="./logs",
       )
       
       # 训练
       runner.learn(total_timesteps=100_000)
       
       # 评估
       stats = runner.eval(num_episodes=10)
       print(f"最终平均奖励: {stats['eval/mean_reward']:.2f}")
       
       # 保存
       runner.save_checkpoint("pendulum_model.pt")
       
       env.close()

   if __name__ == "__main__":
       main()

可视化训练
----------

使用 TensorBoard 查看训练指标：

.. code-block:: bash

   tensorboard --logdir=./logs

在浏览器中打开 ``http://localhost:6006`` 查看：

- 回合奖励
- 策略和价值损失
- KL 散度
- 梯度范数
- 学习率调度

下一步
------

- 学习创建 :doc:`custom_environment`
- 探索 :doc:`custom_network` 架构
- 阅读 :doc:`../modules/algorithms` 了解高级功能

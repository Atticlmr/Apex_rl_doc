实践示例
========

配套仓库 `apexrl_example <https://github.com/Atticlmr/apexrl_example>`_
提供了可以直接运行的实践示例，用于把 ApexRL 接入 Genesis 和 Atari 任务。

当前示例
--------

- ``go2_example``：Genesis Unitree Go2 行走，包含 PPO 和 SAC 脚本。
- ``drone_example``：Genesis Crazyflie 悬停，使用 PPO。
- ``breakout_dqn_example``：Atari Breakout，使用 DQN。

可以把示例仓库 clone 到 ApexRL 旁边：

.. code-block:: bash

   git clone https://github.com/Atticlmr/apexrl_example.git
   cd apexrl_example

在当前 Python 环境中安装 ApexRL 和对应环境依赖后，按示例仓库 README
中的命令运行训练和播放脚本。


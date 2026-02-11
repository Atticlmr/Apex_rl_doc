贡献指南
========

感谢您有兴趣为 ApexRL 做出贡献！本指南将帮助您开始。

开发环境设置
------------

1. Fork 并克隆仓库：

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/Apex_rl.git
   cd Apex_rl

2. 创建虚拟环境：

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

3. 以开发模式安装：

.. code-block:: bash

   pip install -e .

代码风格
--------

我们使用多种工具维护代码质量：

- **Black**：代码格式化
- **Ruff**：代码检查
- **MyPy**：类型检查

提交前格式化代码：

.. code-block:: bash

   black src/
   ruff check src/
   mypy src/

已配置预提交钩子：

.. code-block:: bash

   pre-commit install
   pre-commit run --all-files

提交更改
--------

1. **创建分支**：

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **进行更改**并添加适当的测试

3. **运行测试**：

.. code-block:: bash

   pytest

4. **如需要更新文档**

5. **提交更改**：

.. code-block:: bash

   git add .
   git commit -m "feat: 添加您的功能"

提交信息格式
~~~~~~~~~~~~

我们遵循约定式提交：

- ``feat:`` 新功能
- ``fix:`` 错误修复
- ``docs:`` 文档更改
- ``style:`` 代码风格更改
- ``refactor:`` 代码重构
- ``test:`` 测试更改
- ``chore:`` 维护任务

Pull Request 流程
-----------------

1. **更新文档**以添加新功能
2. **添加测试**以测试新功能
3. **确保所有测试通过**
4. **更新 CHANGELOG.md**
5. **提交 PR**并附清晰描述

贡献领域
--------

算法
~~~~

- [ ] DQN（深度 Q 网络）
- [ ] SAC（软 Actor-Critic）
- [ ] TD3（双延迟 DDPG）
- [ ] A3C（异步优势 Actor-Critic）

网络
~~~~

- [ ] 基于 Transformer 的 Actor/Critic
- [ ] 图神经网络
- [ ] 更多 CNN 架构

特性
~~~~

- [ ] 观测归一化
- [ ] 奖励归一化
- [ ] 自动超参数调优
- [ ] 分布式训练支持

文档
~~~~

- [ ] 更多教程
- [ ] 视频教程
- [ ] 示例项目
- [ ] API 示例

报告问题
--------

报告问题时请包含：

1. **问题描述**
2. **复现步骤**
3. **预期与实际行为**
4. **环境信息**（Python 版本、操作系统、GPU）
5. **错误信息**或堆栈跟踪

行为准则
--------

- 相互尊重，包容差异
- 欢迎新人
- 专注于建设性反馈
- 尊重不同观点

许可证
------

通过贡献，您同意您的贡献将在 Apache 2.0 许可证下授权。

联系方式
--------

- GitHub Issues：https://github.com/Atticlmr/Apex_rl/issues
- 邮箱：Atticlmr@gmail.com

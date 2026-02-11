.. ApexRL 中文文档

===================================
ApexRL 文档
===================================

**ApexRL** 是一个模块化的强化学习库，专为高性能 RL 研究和应用而设计。它提供了简洁、可扩展的架构，针对 GPU 加速环境（如 Isaac Gym 和 Isaac Sim）进行了优化。

.. image:: https://img.shields.io/badge/version-0.0.1-blue.svg
   :target: https://github.com/Atticlmr/Apex_rl
   :alt: 版本

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://github.com/Atticlmr/Apex_rl/blob/main/LICENSE
   :alt: 许可证

.. image:: https://img.shields.io/badge/python-3.11%2B-blue.svg
   :target: https://www.python.org/
   :alt: Python

核心特性
--------

- **GPU 原生设计**：针对 CUDA 加速并行环境优化
- **模块化架构**：易于扩展自定义算法、网络和环境
- **向量化环境**：内置对高性能向量化环境的支持
- **灵活的网络设计**：提供基类用于自定义 Actor/Critic 架构
- **生产就绪**：完善的日志记录、检查点保存和评估工具

快速链接
--------

* :doc:`quickstart` - 5 分钟快速入门
* :doc:`tutorials/first_agent` - 创建您的第一个 RL 智能体
* :doc:`api/modules` - API 参考

.. toctree::
   :hidden:
   :caption: 入门指南

   quickstart
   installation

.. toctree::
   :hidden:
   :caption: 教程

   tutorials/first_agent
   tutorials/custom_environment
   tutorials/custom_network

.. toctree::
   :hidden:
   :caption: 模块

   modules/algorithms
   modules/environments
   modules/networks
   modules/buffers
   modules/runners

.. toctree::
   :hidden:
   :caption: API 参考

   API/modules

.. toctree::
   :hidden:
   :caption: 开发

   contributing
   changelog

语言 / Languages
----------------

* `English <../>`_ (英文)
* `中文 <./>`_ (当前)

索引和表格
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

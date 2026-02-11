安装指南
========

本指南介绍安装 ApexRL 及其依赖的各种方法。

前置条件
--------

- **Python**：版本 3.11 或更高
- **PyTorch**：版本 2.0.0 或更高（推荐支持 CUDA）
- **操作系统**：Linux（推荐）、macOS、Windows

从源代码安装（推荐）
--------------------

可直接从 GitHub 安装最新开发版本：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

使用 uv（更快）
~~~~~~~~~~~~~~~

如果您已安装 `uv <https://github.com/astral-sh/uv>`_：

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

验证安装
--------

验证您的安装：

.. code-block:: python

   import apexrl
   print(apexrl.hello())  # 输出: Hello from ApexRl!

   import torch
   print(f"PyTorch 版本: {torch.__version__}")
   print(f"CUDA 可用: {torch.cuda.is_available()}")

GPU 支持
--------

如需 GPU 加速，请确保安装了适当版本的 CUDA：

**PyTorch with CUDA 12.1：**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu121

**PyTorch with CUDA 11.8：**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu118

**仅 CPU：**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

可选依赖
--------

针对特定用例，您可能想安装额外的包：

**Isaac Gym 支持：**

.. code-block:: bash

   pip install isaacgym  # 请参考 Isaac Gym 安装指南

**TensorBoard 日志：**

.. code-block:: bash

   pip install tensorboard

故障排除
--------

导入错误
~~~~~~~~

如果遇到 ``ModuleNotFoundError``，请确保：

1. 您在正确的 Python 环境中
2. 包已安装：``pip list | grep apexrl``
3. PYTHONPATH 包含源代码目录

CUDA 错误
~~~~~~~~~

如果看到 CUDA 相关错误：

1. 检查 CUDA 安装：``nvidia-smi``
2. 验证 PyTorch CUDA：``torch.cuda.is_available()``
3. 安装匹配的 PyTorch CUDA 版本

版本冲突
~~~~~~~~

使用虚拟环境避免冲突：

.. code-block:: bash

   python -m venv apexrl_env
   source apexrl_env/bin/activate  # Windows: apexrl_env\Scripts\activate
   pip install -e .

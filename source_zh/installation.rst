安装指南
========

本指南介绍 ApexRL 当前版本的安装方式。

环境要求
--------

- Python >= 3.10
- PyTorch >= 2.0
- Gymnasium >= 0.29
- NumPy >= 1.24
- TensorDict >= 0.6

从源码安装
----------

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

使用 uv
-------

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

可选日志依赖
------------

如果要使用托管日志后端，可安装对应的可选依赖：

.. code-block:: bash

   pip install -e ".[wandb]"
   pip install -e ".[swanlab]"

使用 ``uv``：

.. code-block:: bash

   uv pip install -e ".[wandb]"
   uv pip install -e ".[swanlab]"

安装验证
--------

.. code-block:: python

   import apexrl
   import torch

   print(apexrl.hello())
   print(torch.__version__)

运行建议
--------

- ``tensorboard`` 依赖默认已包含，可直接使用 TensorBoard 日志。
- ``wandb`` 和 ``swanlab`` 作为可选依赖提供，按需安装即可。
- 结构化观测支持依赖 ``tensordict``，默认安装已经开启。
- 本地开发建议使用单独虚拟环境，避免与 PyTorch 相关依赖冲突。

安装指南
========

本指南介绍 ApexRL 当前版本的安装方式。

环境要求
--------

- Python >= 3.11
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
- 结构化观测支持依赖 ``tensordict``，默认安装已经开启。
- 本地开发建议使用单独虚拟环境，避免与 PyTorch 相关依赖冲突。

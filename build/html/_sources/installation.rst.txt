Installation Guide
==================

This guide covers various ways to install ApexRL and its dependencies.

Prerequisites
-------------

- **Python**: Version 3.11 or higher
- **PyTorch**: Version 2.0.0 or higher (with CUDA support recommended)
- **Operating System**: Linux (recommended), macOS, Windows

From Source (Recommended)
-------------------------

The latest development version can be installed directly from GitHub:

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

Using uv (Faster)
~~~~~~~~~~~~~~~~~

If you have `uv <https://github.com/astral-sh/uv>`_ installed:

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

Verification
------------

Verify your installation:

.. code-block:: python

   import apexrl
   print(apexrl.hello())  # Output: Hello from ApexRl!

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")

GPU Support
-----------

For GPU acceleration, ensure you have the appropriate CUDA version installed:

**PyTorch with CUDA 12.1:**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu121

**PyTorch with CUDA 11.8:**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu118

**CPU-only:**

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

Optional Dependencies
---------------------

For specific use cases, you may want to install additional packages:

**Isaac Gym Support:**

.. code-block:: bash

   pip install isaacgym  # Follow Isaac Gym installation guide

**TensorBoard Logging:**

.. code-block:: bash

   pip install tensorboard

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter ``ModuleNotFoundError``, ensure:

1. You're in the correct Python environment
2. The package is installed: ``pip list | grep apexrl``
3. Your PYTHONPATH includes the source directory

CUDA Errors
~~~~~~~~~~~

If you see CUDA-related errors:

1. Check CUDA installation: ``nvidia-smi``
2. Verify PyTorch CUDA: ``torch.cuda.is_available()``
3. Install matching PyTorch CUDA version

Version Conflicts
~~~~~~~~~~~~~~~~~

Use a virtual environment to avoid conflicts:

.. code-block:: bash

   python -m venv apexrl_env
   source apexrl_env/bin/activate  # On Windows: apexrl_env\Scripts\activate
   pip install -e .

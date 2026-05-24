Installation Guide
==================

This guide covers the current ApexRL installation path.

Requirements
------------

- Python >= 3.10
- PyTorch >= 2.0
- Gymnasium >= 0.29
- NumPy >= 1.24
- TensorDict >= 0.6

From Source
-----------

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   pip install -e .

Using uv
--------

.. code-block:: bash

   git clone https://github.com/Atticlmr/Apex_rl.git
   cd Apex_rl
   uv pip install -e .

Optional Logging Extras
-----------------------

Install optional SDKs when using hosted logging backends:

.. code-block:: bash

   pip install -e ".[wandb]"
   pip install -e ".[swanlab]"

With ``uv``:

.. code-block:: bash

   uv pip install -e ".[wandb]"
   uv pip install -e ".[swanlab]"

Verification
------------

.. code-block:: python

   import apexrl
   import torch

   print(apexrl.hello())
   print(torch.__version__)

Recommended Runtime Notes
-------------------------

- TensorBoard logging works out of the box through the ``tensorboard`` dependency.
- ``wandb`` and ``swanlab`` are optional extras because not every installation needs hosted experiment tracking.
- Structured observation support relies on ``tensordict`` and is enabled in the default install.
- For local development, use a dedicated virtual environment to avoid PyTorch dependency conflicts.

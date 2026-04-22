Installation Guide
==================

This guide covers the current ApexRL installation path.

Requirements
------------

- Python >= 3.11
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
- Structured observation support relies on ``tensordict`` and is enabled in the default install.
- For local development, use a dedicated virtual environment to avoid PyTorch dependency conflicts.

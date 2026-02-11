.. ApexRL documentation master file

===================================
ApexRL Documentation
===================================

**ApexRL** is a modular reinforcement learning library designed for high-performance RL research and applications. It provides a clean, extensible architecture optimized for GPU-accelerated environments like Isaac Gym and Isaac Sim.

.. image:: https://img.shields.io/badge/version-0.0.1-blue.svg
   :target: https://github.com/Atticlmr/Apex_rl
   :alt: Version

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://github.com/Atticlmr/Apex_rl/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/python-3.11%2B-blue.svg
   :target: https://www.python.org/
   :alt: Python

Key Features
------------

- **GPU-Native Design**: Optimized for CUDA-accelerated parallel environments
- **Modular Architecture**: Easy to extend with custom algorithms, networks, and environments
- **Vectorized Environments**: Built-in support for high-performance vectorized environments
- **Flexible Network Design**: Custom Actor/Critic architectures with base classes
- **Production Ready**: Comprehensive logging, checkpointing, and evaluation tools

Quick Links
-----------

* :doc:`quickstart` - Get started in 5 minutes
* :doc:`tutorials/first_agent` - Create your first RL agent
* :doc:`api/modules` - API reference

.. toctree::
   :hidden:
   :caption: Getting Started

   quickstart
   installation

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/first_agent
   tutorials/custom_environment
   tutorials/custom_network

.. toctree::
   :hidden:
   :caption: Modules

   modules/algorithms
   modules/environments
   modules/networks
   modules/buffers
   modules/runners

.. toctree::
   :hidden:
   :caption: API Reference

   API/modules

.. toctree::
   :hidden:
   :caption: Development

   contributing
   changelog

Languages / 语言
----------------

* `English <./>`_ (current)
* `中文 <./zh/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

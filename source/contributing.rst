Contributing
============

Thank you for your interest in contributing to ApexRL! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/Apex_rl.git
   cd Apex_rl

2. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

.. code-block:: bash

   pip install -e .

Code Style
----------

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Format your code before committing:

.. code-block:: bash

   black src/
   ruff check src/
   mypy src/

Pre-commit hooks are configured:

.. code-block:: bash

   pre-commit install
   pre-commit run --all-files

Making Changes
--------------

1. **Create a branch**:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make your changes** with appropriate tests

3. **Run tests**:

.. code-block:: bash

   pytest

4. **Update documentation** if needed

5. **Commit your changes**:

.. code-block:: bash

   git add .
   git commit -m "feat: add your feature"

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

We follow conventional commits:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation changes
- ``style:`` Code style changes
- ``refactor:`` Code refactoring
- ``test:`` Test changes
- ``chore:`` Maintenance tasks

Pull Request Process
--------------------

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update CHANGELOG.md**
5. **Submit PR** with clear description

Areas for Contribution
----------------------

Algorithms
~~~~~~~~~~

- [ ] DQN (Deep Q-Network)
- [ ] SAC (Soft Actor-Critic)
- [ ] TD3 (Twin Delayed DDPG)
- [ ] A3C (Asynchronous Advantage Actor-Critic)

Networks
~~~~~~~~

- [ ] Transformer-based actors/critics
- [ ] Graph Neural Networks
- [ ] More CNN architectures

Features
~~~~~~~~

- [ ] Observation normalization
- [ ] Reward normalization
- [ ] Automatic hyperparameter tuning
- [ ] Distributed training support

Documentation
~~~~~~~~~~~~~

- [ ] More tutorials
- [ ] Video tutorials
- [ ] Example projects
- [ ] API examples

Reporting Issues
----------------

When reporting issues, please include:

1. **Description** of the issue
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Environment info** (Python version, OS, GPU)
5. **Error messages** or stack traces

Code of Conduct
---------------

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect differing viewpoints

License
-------

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

Contact
-------

- GitHub Issues: https://github.com/Atticlmr/Apex_rl/issues
- Email: Atticlmr@gmail.com

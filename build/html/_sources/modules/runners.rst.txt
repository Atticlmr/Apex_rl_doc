Runners
=======

Runners manage the training loop, logging, checkpointing, and evaluation for RL agents.

Overview
--------

The runner module provides high-level interfaces for:

1. **Training Management** - Automated training loops
2. **Logging** - TensorBoard integration
3. **Checkpointing** - Model saving and loading
4. **Evaluation** - Periodic agent evaluation

OnPolicyRunner
--------------

The ``OnPolicyRunner`` manages training for on-policy algorithms like PPO.

Key Features
~~~~~~~~~~~~

- Automated training loop with callbacks
- TensorBoard logging of metrics
- Periodic checkpoint saving
- Reward component tracking
- Environment metrics logging

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from apexrl.agent.on_policy_runner import OnPolicyRunner

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",
       actor_class=MLPActor,
       critic_class=MLPCritic,
       log_dir="./logs",
       save_dir="./checkpoints",
       log_interval=10,
       save_interval=100,
   )

   # Train
   runner.learn(total_timesteps=10_000_000)

   # Evaluate
   stats = runner.eval(num_episodes=100)

   # Save
   runner.save_checkpoint("final_model.pt")

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   runner = OnPolicyRunner(
       env=env,                          # Vectorized environment
       algorithm="ppo",                  # Algorithm name
       actor_class=MLPActor,             # Actor network class
       critic_class=MLPCritic,           # Critic network class
       actor_cfg={"hidden_dims": [256]}, # Actor network config
       critic_cfg={"hidden_dims": [256]}, # Critic network config
       log_dir="./logs",                 # TensorBoard log directory
       save_dir="./checkpoints",         # Checkpoint directory
       device=torch.device("cuda"),      # Training device
       log_interval=10,                  # Log every N iterations
       save_interval=100,                # Save every N iterations
       log_reward_components=True,       # Log reward components
   )

API Reference
~~~~~~~~~~~~~

.. autoclass:: apexrl.agent.on_policy_runner.OnPolicyRunner
   :members:
   :undoc-members:
   :show-inheritance:

Training Loop
~~~~~~~~~~~~~

The training loop structure:

.. code-block:: text

   for iteration in range(total_iterations):
       # 1. Collect rollout
       rollout_stats = collect_rollout()
       
       # 2. Update policy
       update_stats = update()
       
       # 3. Adjust learning rate
       adjust_learning_rate()
       
       # 4. Log metrics
       if iteration % log_interval == 0:
           log_metrics()
       
       # 5. Save checkpoint
       if iteration % save_interval == 0:
           save_checkpoint()

Callbacks
~~~~~~~~~

Add custom callbacks for training events:

.. code-block:: python

   def on_iteration_start(runner):
       print(f"Starting iteration {runner.iteration}")

   def on_iteration_end(runner, stats):
       if stats["rollout/mean_reward"] > threshold:
           print("Reward threshold reached!")

   runner.add_callback("pre_iteration", on_iteration_start)
   runner.add_callback("post_iteration", on_iteration_end)

Available events:

- ``pre_iteration`` - Before each training iteration
- ``post_iteration`` - After each training iteration
- ``pre_rollout`` - Before collecting rollout
- ``post_rollout`` - After collecting rollout
- ``pre_update`` - Before policy update
- ``post_update`` - After policy update

Logging
~~~~~~~

The runner automatically logs metrics to TensorBoard:

**Training Metrics:**

- ``train/policy_loss`` - Policy loss
- ``train/value_loss`` - Value function loss
- ``train/entropy_loss`` - Entropy loss
- ``train/approx_kl`` - Approximate KL divergence
- ``train/learning_rate`` - Current learning rate

**Episode Metrics:**

- ``episode/mean_reward`` - Mean episode reward
- ``episode/mean_length`` - Mean episode length

**Rollout Metrics:**

- ``rollout/mean_reward`` - Mean step reward
- ``rollout/mean_value`` - Mean value estimate

**Environment Metrics:**

Custom metrics from environment ``extras["log"]`` are automatically logged.

Reward Components
~~~~~~~~~~~~~~~~~

Track individual reward components:

.. code-block:: python

   # In environment step()
   extras = {
       "time_outs": time_outs,
       "reward_components": {
           "velocity": velocity_reward,
           "energy": -energy_penalty,
           "stability": stability_reward,
       },
       "log": {
           "/robot/height_mean": robot_height.mean().item(),
       },
   }

The runner automatically:

1. Accumulates reward components per episode
2. Logs mean values at episode end
3. Logs custom metrics from ``extras["log"]``

Checkpointing
~~~~~~~~~~~~~

Save and load training checkpoints:

.. code-block:: python

   # Save checkpoint
   runner.save_checkpoint("model.pt")
   
   # Save with iteration number
   runner.save_checkpoint(f"checkpoint_{runner.iteration}.pt")

   # Load checkpoint
   runner.load_checkpoint("model.pt")

Checkpoint includes:

- Actor network state
- Critic network state
- Optimizer states
- Training iteration
- Total timesteps
- Configuration

Evaluation
~~~~~~~~~~

Evaluate the trained agent:

.. code-block:: python

   # Run evaluation
   stats = runner.eval(num_episodes=100)
   
   print(f"Mean reward: {stats['eval/mean_reward']:.2f}")
   print(f"Std reward: {stats['eval/std_reward']:.2f}")
   print(f"Min reward: {stats['eval/min_reward']:.2f}")
   print(f"Max reward: {stats['eval/max_reward']:.2f}")

Evaluation runs the agent deterministically (``deterministic=True``).

Pre-configured Agent
~~~~~~~~~~~~~~~~~~~~

Use a pre-configured agent instead of auto-creation:

.. code-block:: python

   from apexrl.algorithms.ppo import PPO, PPOConfig

   cfg = PPOConfig(learning_rate=3e-4)
   agent = PPO(
       env=env,
       cfg=cfg,
       actor_class=MLPActor,
       critic_class=MLPCritic,
   )

   runner = OnPolicyRunner(
       agent=agent,
       env=env,
       cfg=cfg,
       log_dir="./logs",
   )

Best Practices
--------------

1. **Log Directory**: Always specify ``log_dir`` for experiment tracking
2. **Save Interval**: Set ``save_interval`` based on training duration
3. **Callbacks**: Use callbacks for custom logic without modifying runner
4. **Device**: Let runner auto-detect device, or specify explicitly
5. **Evaluation**: Run evaluation periodically to monitor progress

See Also
--------

- :doc:`../tutorials/first_agent` - Detailed usage tutorial
- :doc:`../api/apexrl.agent` - Full API reference

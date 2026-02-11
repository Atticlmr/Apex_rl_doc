Your First RL Agent
===================

This tutorial walks you through creating and training your first reinforcement learning agent using ApexRL.

Overview
--------

By the end of this tutorial, you will:

- Understand the basic components of ApexRL
- Create a vectorized environment
- Configure and train a PPO agent
- Evaluate and save your trained model

Prerequisites
-------------

Ensure you have ApexRL installed:

.. code-block:: bash

   pip install -e .

Step 1: Import Libraries
------------------------

.. code-block:: python

   import gymnasium as gym
   import torch

   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

Step 2: Create the Environment
------------------------------

ApexRL uses vectorized environments for parallel training. Let's create 8 parallel instances of Pendulum-v1:

.. code-block:: python

   def make_env():
       """Factory function to create a single environment."""
       return gym.make("Pendulum-v1")

   # Create 8 parallel environments
   num_envs = 8
   env = GymVecEnv([make_env for _ in range(num_envs)], device="cpu")

   print(f"Number of environments: {env.num_envs}")
   print(f"Observation dimension: {env.num_obs}")
   print(f"Action dimension: {env.num_actions}")

Step 3: Configure the Runner
----------------------------

The ``OnPolicyRunner`` handles the training loop, logging, and checkpointing:

.. code-block:: python

   runner = OnPolicyRunner(
       env=env,
       algorithm="ppo",           # Algorithm to use
       actor_class=MLPActor,      # Actor network class
       critic_class=MLPCritic,    # Critic network class
       log_dir="./logs",          # TensorBoard log directory
       save_dir="./checkpoints",  # Checkpoint save directory
       log_interval=10,           # Log every 10 iterations
       save_interval=100,         # Save every 100 iterations
   )

Step 4: Train the Agent
-----------------------

Train the agent for a specified number of timesteps:

.. code-block:: python

   # Train for 100,000 timesteps
   runner.learn(total_timesteps=100_000)

During training, you'll see output like:

.. code-block:: text

   Training for 520 iterations (104,000 steps)
   Iter 0/520 | Steps 0 | FPS 0 | Policy Loss -0.0012 | Value Loss 0.0234 | KL 0.0012
   Iter 10/520 | Steps 1,920 | FPS 3421 | Policy Loss -0.0023 | Value Loss 0.0187 | KL 0.0008 | Reward -456.23
   ...

Step 5: Evaluate the Agent
--------------------------

Evaluate the trained agent:

.. code-block:: python

   eval_stats = runner.eval(num_episodes=10)
   
   print(f"Mean reward: {eval_stats['eval/mean_reward']:.2f}")
   print(f"Std reward: {eval_stats['eval/std_reward']:.2f}")
   print(f"Min reward: {eval_stats['eval/min_reward']:.2f}")
   print(f"Max reward: {eval_stats['eval/max_reward']:.2f}")

Step 6: Save and Load
---------------------

Save the trained model:

.. code-block:: python

   runner.save_checkpoint("final_model.pt")

Load a saved model:

.. code-block:: python

   runner.load_checkpoint("final_model.pt")

Complete Code
-------------

Here's the complete training script:

.. code-block:: python

   import gymnasium as gym
   from apexrl.agent.on_policy_runner import OnPolicyRunner
   from apexrl.envs.gym_wrapper import GymVecEnv
   from apexrl.models.mlp import MLPActor, MLPCritic

   def main():
       # Create environment
       def make_env():
           return gym.make("Pendulum-v1")
       
       env = GymVecEnv([make_env for _ in range(8)], device="cpu")
       
       # Create runner
       runner = OnPolicyRunner(
           env=env,
           algorithm="ppo",
           actor_class=MLPActor,
           critic_class=MLPCritic,
           log_dir="./logs",
       )
       
       # Train
       runner.learn(total_timesteps=100_000)
       
       # Evaluate
       stats = runner.eval(num_episodes=10)
       print(f"Final mean reward: {stats['eval/mean_reward']:.2f}")
       
       # Save
       runner.save_checkpoint("pendulum_model.pt")
       
       env.close()

   if __name__ == "__main__":
       main()

Visualizing Training
--------------------

View training metrics with TensorBoard:

.. code-block:: bash

   tensorboard --logdir=./logs

Open your browser at ``http://localhost:6006`` to see:

- Episode rewards
- Policy and value losses
- KL divergence
- Gradient norms
- Learning rate schedule

Next Steps
----------

- Learn to create :doc:`custom_environment`
- Explore :doc:`custom_network` architectures
- Read about advanced :doc:`../modules/algorithms` features

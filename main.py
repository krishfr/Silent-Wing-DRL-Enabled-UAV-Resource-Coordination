import time
import numpy as np
import torch
import shutil
import os
from typing import Tuple, List
from env import UAVEnv
from agent import TD3
from log import ExperimentManager
from visual import DynamicTrajectoryVisualizer

# Global Configuration Dictionary
# Visualization specific settings are in visual.py; all other parameters are adjusted here.
CONFIG = {
    # Training Parameters
    'TRAIN': {
        'num_episodes': 500,           # Total training episodes
        'min_buffer_size': 1500,       # Minimum transitions before training starts
        'noise_decay_factor': 1.5,     # Factor for exploration noise reduction
        'min_noise': 0.08,             # Lower bound for exploration noise
        'seed': int(time.time()),      # Random seed for reproducibility
    },
    
    # Testing Parameters
    'TEST': {
        'num_tests': 10,               # Number of evaluation runs
    },
    
    # TD3 Algorithm Hyperparameters
    'TD3': {
        'state_dim': 2,
        'action_dim': 2,
        'batch_size': 32,               # Mini-batch size for gradient descent
        'gamma': 0.99,                  # Discount factor for future rewards
        'tau': 1e-5,                    # Soft update coefficient for target networks
        'policy_noise': 1.0,            # Noise added to target policy during smoothing
        'exploration_noise': 1.0,       # Initial noise for action exploration
        'noise_clip': 0.5,              # Limit for policy noise
        'policy_freq': 10,              # Frequency of policy (Actor) updates relative to Critic
        'max_action': 1.0,
        'replay_buffer_size': 1000000,  # Capacity of experience replay
        'lr_actor': 1e-4,               # Learning rate for the Actor network
        'lr_critic': 1e-3,              # Learning rate for the Critic network
        
        # Multi-step Learning
        'n_step': 6,                    # Steps for N-step return calculation
        'multi_step_gamma': 0.99,       # Discount factor for multi-step returns

        # Prioritized Experience Replay (PER)
        'per_alpha': 0.6,               # Priority exponent
        'per_beta': 0.4,                # Initial Importance Sampling weight
        'per_beta_increment': 0.001,    # Growth rate for beta
    },
    
    # Environment Settings
    'ENV': {
        'max_steps': 100,                   # Max steps per episode
        'velocity_magnitude': 50.0,         # Constant speed in m/s
        'time_slot': 0.5,                   # Duration of one time step in seconds
        'map_size': 1000.0,                 # Map dimension (Square area)

        'start_pos': [60.0, 200.0],         # Initial UAV coordinates
        'goal_pos': [940.0, 600.0],         # Target coordinates
        'warden_count_range': [1, 1],       # Number of wardens (detectors)
        'warden_pos_range': [490, 510],     # Range to place wardens
        'legitimate_nodes': [               # Ground communication nodes
            [200.0, 300.0],
            [300.0, 800.0], 
            [700.0, 200.0],
            [800.0, 700.0]
        ],

        # ITU Building Distribution Parameters
        'itu_alpha': 0.2,
        'itu_beta': 40, 
        'itu_gamma': 25, 
        'density_factor': 1.0, 
        
        # Building Generation Settings
        'height_range': [10, 70], 
        'width_range': [40, 100], 
        'length_range': [40, 100],               # Dimensions for generated obstacles
        'min_distance': 5.0,                     # Minimum gap between buildings
        'max_attempts': 10,
        'building_generation_range': [60, 940],  # Coordinate bounds for obstacles
        'warden_exclusion_size': 50,             # Protected area around wardens to prevent occlusion issues

        # Covert Hole (Shadowing) visualization
        'num_rays': 180,                # Ray-casting resolution
        'max_ray_length': 300.0,        # Maximum radius for visibility
        'min_building_height': 30,      # Height threshold for blocking line-of-sight
    },
    
    # Reward Function Weights
    'REWARD': {
        'transmission_weight': 5.0,            # Benefit of high throughput
        'base_penalty': -1.0,                  # Constant penalty for time spent
        'covertness_threshold': 0.99,          # Minimum DEP to be considered "covert"
        'covert_penalty_weight': -500.0,       # Penalty for entering warden visibility
        
        'goal_reward': 4000.0,                 # Large bonus for reaching the target
        'goal_threshold': 20.0,                # Distance to goal to trigger completion

        'distance_guidance_weight': 2,         # Reward for reducing distance to goal
        'proximity_reward_weight': 1,          # Absolute proximity reward
        'step_penalty_weight': 0.5,            # Penalty to encourage path efficiency
        
        # Boundary and Collision
        'boundary_penalty': -10000.0,          # Massive penalty for hitting map edges
        'boundary_penalty_threshold': 40.0,    # Distance at which boundary force begins
        'boundary_penalty_strength': 0.3,      # Scaling factor for boundary force
        
        # Anti-oscillation measures
        'oscillation_penalty': -100.0,         # Penalty for repetitive movement
        'oscillation_history_length': 12,      # Steps to keep in memory
        'oscillation_check_steps': 6,          # Steps compared for movement check
        'oscillation_distance_threshold': 12,  # Displacement threshold for oscillation
    },
    
    # Radio Channel Model
    'CHANNEL': {
        'fc': 2e9,                  # Carrier frequency 2GHz
        'uav_height': 100,          # Flying altitude
        'tx_power': 30,             # Transmit power in dBm
        'noise_power': -90,         # Background noise in dBm
        'noise_uncertainty': 3,     # Variation in noise environment
        'nominal_noise': 1e-6,      # Linear scale noise power

        'location_error_var': 0.025,
        'num_evaluations': 1000,
        
        # Path Loss Exponents (LoS and NLoS)
        'los_a': 22.0,
        'los_b': 28.0,
        'los_c': 20.0,
        'nlos_a': 36.7,
        'nlos_b': 22.7, 
        'nlos_c': 26.0, 
        'nlos_d': 0.3, 
    },
    
    # Logging and Output
    'LOG': {
        'output_dir': 'figures', 
        'save_data': True,                 # Export results to CSV
        'show_plots': False,               # Display matplotlib windows
        'show_dynamic_training': True,     # Enable real-time trajectory updates
        'dpi': 300,
        'figsize': (12, 8),
    }
}

# Training Module
def train() -> Tuple[TD3, List[float], 'DynamicTrajectoryVisualizer', 'ExperimentManager', UAVEnv]:

    # Initialization
    env = UAVEnv(env_params=CONFIG['ENV'], reward_params=CONFIG['REWARD'], channel_params=CONFIG['CHANNEL'])
    agent = TD3(state_dim=CONFIG['TD3']['state_dim'], action_dim=CONFIG['TD3']['action_dim'], td3_params=CONFIG['TD3'])

    np.random.seed(CONFIG['TRAIN']['seed'])
    torch.manual_seed(CONFIG['TRAIN']['seed'])

    exp_manager = ExperimentManager(log_params=CONFIG['LOG'])
    dynamic_visualizer = None

    if CONFIG['LOG']['show_dynamic_training']:
        dynamic_visualizer = DynamicTrajectoryVisualizer(CONFIG, CONFIG['LOG']['output_dir'])
        dynamic_visualizer.initialize_plot(env)

    total_steps = 0 

    # Initial exploration phase to fill replay buffer
    while len(agent.replay_buffer) < CONFIG['TRAIN']['min_buffer_size']:
        s = env.reset()
        done = False
        while not done:
            # Random exploration actions
            a = np.random.uniform(-1.0, 1.0, size=agent.action_dim)
            s_, r, done, info = env.step(a)
            agent.store_transition(s, a, r, s_, done)
            s = s_
            total_steps += 1

    # Main training loop
    for ep in range(CONFIG['TRAIN']['num_episodes']):

        s = env.reset()
        done = False
        ep_reward, ep_deps, ep_throughputs = 0.0, [], []
        ep_trajectory = [env.state.copy()]

        # Calculate noise with exponential decay
        base_noise = np.sqrt(CONFIG['TD3']['exploration_noise'])
        noise = max(base_noise * np.exp(-CONFIG['TRAIN']['noise_decay_factor'] * ep / CONFIG['TRAIN']['num_episodes']),
                    CONFIG['TRAIN']['min_noise'])

        # Step-by-step episode execution
        while not done:
            a = agent.select_action(s, noise)
            s_, r, done, info = env.step(a)
            agent.store_transition(s, a, r, s_, done)
            s = s_

            ep_trajectory.append(env.state.copy())
            ep_reward += r
            ep_deps.append(info['DEP'])
            ep_throughputs.append(info['throughput'])

            exp_manager.logger.log_training_step(ep, env.current_step, env.state, r, info)

            total_steps += 1
            agent.update(total_steps)

        # Log episode metrics
        avg_dep = np.mean(ep_deps) if ep_deps else 0.0
        avg_throughput = np.mean(ep_throughputs) if ep_throughputs else 0.0
        exp_manager.log_training_episode(ep, ep_reward, avg_dep, avg_throughput)
        
        if CONFIG['LOG']['show_dynamic_training'] and dynamic_visualizer is not None:
            dynamic_visualizer.update_trajectory(np.asarray(ep_trajectory), ep_reward, ep + 1)

        # Display progress in console
        print(f"Episode {ep + 1}: Reward={ep_reward:.2f}, DEP={avg_dep:.4f}, Throughput={avg_throughput:.4f}, Noise={noise:.3f}")

    exp_manager.visualize_training()
    return agent, exp_manager.logger.episode_rewards, dynamic_visualizer, exp_manager, env


# Testing Module
def test(agent: TD3, exp_manager: ExperimentManager, env: UAVEnv, train_dynamic_visualizer=None):

    rewards_list, deps_list, thrpts_list = [], [], []
    all_trajectories = []
    all_deps, all_thrpts = [], []

    test_dynamic_visualizer = train_dynamic_visualizer

    for test_run in range(CONFIG['TEST']['num_tests']):

        s = env.reset()
        done = False
        traj, deps, thrpts, rs = [env.state.copy()], [], [], []
        step = 0

        while not done:
            a = agent.select_action(s, noise=0)  # Use deterministic policy (no noise) for evaluation
            s_, r, done, info = env.step(a)

            exp_manager.log_test_step(test_run + 1, step, env.state, r, info)

            traj.append(env.state.copy())
            deps.append(info['DEP'])
            thrpts.append(info['throughput'])
            rs.append(r)
            s = s_
            step += 1

        # Calculate metrics for the test run
        total_reward = sum(rs)
        rewards_list.append(total_reward)
        deps_list.append(np.mean(deps))
        thrpts_list.append(sum(thrpts))

        trajectory_array = np.asarray(traj)
        all_trajectories.append((trajectory_array, total_reward))
        all_deps.append(deps)
        all_thrpts.append(thrpts)

        exp_manager.log_test_trajectory(trajectory_array, total_reward, deps, thrpts)

        print(f"Test {test_run + 1}: Reward={total_reward:.2f}, DEP={np.mean(deps):.4f}, Throughput={sum(thrpts):.4f}")

        if test_dynamic_visualizer is not None:
            test_dynamic_visualizer.update_trajectory(trajectory_array, total_reward, test_run + 1)
    
    exp_manager.visualize_testing(env)

    if test_dynamic_visualizer is not None:
        test_dynamic_visualizer._save_test_trajectory_image(CONFIG['TEST']['num_tests'])

    return rewards_list, deps_list, thrpts_list, all_trajectories, env


# Main Entry Point
def main():

    # Clean up previous experiment directories
    figures_dir = "figures"
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    
    log_dir = "log"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print("Cleared previous output directories: figures and log")

    # Run training
    print("\nStarting training phase...")
    agent, rewards, train_dynamic_visualizer, exp_manager, env = train()

    # Run evaluation
    print("\nStarting evaluation phase...")
    test(agent, exp_manager, env, train_dynamic_visualizer)

    print("Training and evaluation completed successfully")


if __name__ == "__main__":
    main()
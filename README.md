🚁 Silent-Wing: DRL-Enabled-UAV-Resource-Coordination

📌 Project Overview
Silent-Wing is a research-oriented project based on Deep Reinforcement Learning (DRL) that focuses on resource coordination in UAV-Aided IoT environments.
The project aims to balance covert communication and energy efficiency by optimizing UAV trajectories in complex and dynamic scenarios.

A DRL-based framework is implemented to evaluate the effectiveness of intelligent UAV trajectory optimization under realistic constraints such as obstacles, legitimate users, and potential eavesdroppers.

✨ Key Features
🧠 TD3-Based Trajectory Optimization
Uses Twin Delayed Deep Deterministic Policy Gradient (TD3) for continuous action spaces.
Enables intelligent UAV movement and decision-making.

🌆 Complex Environment Modeling
Simulates realistic environments with:
Dynamically generated buildings
Legitimate ground communication nodes
Potential eavesdroppers (wardens)
Designed to resemble urban or complex terrains.

🔐 Covertness vs Transmission Efficiency
Balances two critical objectives using a custom reward function:
Covertness: Maximizes Detection Error Probability (DEP).
Efficiency: Maximizes data transmission throughput.

⚙️ Advanced Reinforcement Learning Techniques
Improves training stability and convergence using:
Prioritized Experience Replay (PER)
Multi-step Learning

📊 Visualization and Logging
Visualizes:
UAV trajectories
Training progress
Environment layout
Reward heatmaps
Logs key metrics such as rewards, DEP, and throughput.

🚀 Quick Start Guide

🛠 Environment Setup
1. Create a virtual environment
```bash
python -m venv .venv

Activate the virtual environment

macOS / Linux

source .venv/bin/activate


Windows

.venv\Scripts\activate


Install dependencies

pip install -r requirements.txt

▶️ Run the Project
python main.py

Project Structure
├── main.py        # Main entry point
├── agent.py      # TD3 algorithm implementation
├── env.py        # Environment and map modeling
├── uav.py        # UAV behavior and interactions
├── log.py        # Logging utilities
├── visual.py     # Visualization tools
├── figures/      # Generated visualization images
└── log/          # Training and testing logs

🔄 Typical Execution Flow

Run the main program
python main.py

Generate the initial environment map
initial_env.png

Start training
Training trajectory generated every 50 epochs
Example: train_traj_ep_200.png

Testing phase
10 test episodes executed
Output: test_traj_ep_10.png

Performance metrics saved
performance_metrics.png

🧰 Technologies Used
Python
Deep Reinforcement Learning (DRL)
TD3 Algorithm
Simulation-Based Environment Modeling
Data Visualization
Git & GitHub

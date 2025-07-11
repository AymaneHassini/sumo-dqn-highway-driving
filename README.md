# DQN for Autonomous Highway Driving in SUMO

This repository contains the implementation of a Deep Q-Network (DQN) agent trained for autonomous driving on a simulated highway. Using the [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) traffic simulator, the agent learns to navigate a multi-lane environment by balancing the objectives of maintaining a target speed and avoiding collisions.

This project covers an end-to-end reinforcement learning workflow, from environment design and agent training to quantitative performance evaluation.

---

## Project Overview

- **Environment**: [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/)
- **Algorithm**: Deep Q-Network (DQN) with Experience Replay and a Target Network
- **Framework**: [PyTorch](https://pytorch.org/)

---

## Key Features

### Custom SUMO Highway Environment
A 1000-meter, two-lane highway dynamically populated with human-driven vehicles and a single controllable ego agent.

### Rich State Representation
The agent observes a 14-dimensional state vector capturing its own kinematics (speed, lane) and its spatial relationship to the nearest vehicles (front, rear, and adjacent lanes).

### Discrete Action Space
The agent selects from five distinct driving actions:
- Maintain Speed
- Accelerate
- Decelerate
- Change Lane Left
- Change Lane Right

### ✅ Engineered Reward Function
A carefully designed reward function encourages target-speed driving (30 m/s) while heavily penalizing unsafe behavior:

- **Collision**: `-10`
- **Near-collision (Time-to-Collision < 1s)**: `-1`
- **Speed Deviation**: A continuous penalty proportional to the deviation from the target speed.

---

## Results

### Training Performance
The agent was trained for 1,000 episodes, successfully learning a stable driving policy. The learning curve below shows the total reward per episode, with a 100-episode moving average (orange) clearly illustrating convergence as the reward plateaus.

![Training Rewards Plot](training_rewards_plot.png)

### Inference Performance
The final trained agent was evaluated over 10 episodes in deterministic inference mode (no random actions). The agent demonstrated robust performance, successfully completing all episodes without collisions. The bar chart shows the time taken to complete each run.

![Inference Performance Plot](inference_performance.png)

The trained model weights (`dqn_highway_model.pth`) are available in the **[v1.0.0 Release](https://github.com/AymaneHassini/sumo-dqn-highway-driving/releases/tag/v1.0.0)**.

---




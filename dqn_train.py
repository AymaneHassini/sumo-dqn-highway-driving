import os
import sys
traci_path = '/opt/homebrew/opt/sumo/share/sumo/tools'
if traci_path not in sys.path:
    sys.path.append(traci_path)
import traci
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'.")

SUMO_BINARY = "sumo-gui" # use "sumo" for faster training
SUMO_CMD = [SUMO_BINARY, "-c", "highway.sumocfg", "--step-length", "0.1", "--no-warnings", "true", "--collision.action", "remove", "--start", "--quit-on-end"]
EGO_ID = 'ego_flow.0' 
TARGET_SPEED = 30.0
MAX_GAP = 1000.0
STATE_SIZE = 14
ACTION_SIZE = 5

# --- DQN HYPERPARAMETERS ---
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 5

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x): return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_, d): self.buffer.append((s, a, r, s_, d))
    def sample(self, n): return random.sample(self.buffer, n)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
    def choose_action(self, state):
        if random.random() < self.epsilon: return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
        return torch.argmax(q_values).item()
    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        next_q_values[dones] = 0.0
        expected_q = rewards + (GAMMA * next_q_values)
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def update_epsilon(self):
        if self.epsilon > EPSILON_END: self.epsilon *= EPSILON_DECAY
    def update_target_net(self): self.target_net.load_state_dict(self.policy_net.state_dict())

def get_full_state():
    try:
        ego_speed = traci.vehicle.getSpeed(EGO_ID)
        ego_lane = traci.vehicle.getLaneIndex(EGO_ID)
        ego_pos_x = traci.vehicle.getPosition(EGO_ID)[0]
        d_f, dv_f, d_r, dv_r, d_fL, dv_fL, d_rL, dv_rL, d_fR, dv_fR, d_rR, dv_rR = [MAX_GAP, 0] * 6
        leader = traci.vehicle.getLeader(EGO_ID)
        if leader: d_f, dv_f = leader[1], ego_speed - traci.vehicle.getSpeed(leader[0])
        all_vehicles = traci.vehicle.getIDList()
        min_dist_r, min_dist_fL, min_dist_rL, min_dist_fR, min_dist_rR = [MAX_GAP] * 5
        left_lane, right_lane = ego_lane - 1, ego_lane + 1
        for veh_id in all_vehicles:
            if veh_id == EGO_ID: continue
            veh_lane, veh_pos_x = traci.vehicle.getLaneIndex(veh_id), traci.vehicle.getPosition(veh_id)[0]
            dist = veh_pos_x - ego_pos_x
            if veh_lane == ego_lane and dist < 0 and -dist < min_dist_r:
                min_dist_r, d_r, dv_r = -dist, -dist, ego_speed - traci.vehicle.getSpeed(veh_id)
            elif veh_lane == left_lane:
                if dist > 0 and dist < min_dist_fL: min_dist_fL, d_fL, dv_fL = dist, dist, ego_speed - traci.vehicle.getSpeed(veh_id)
                elif dist < 0 and -dist < min_dist_rL: min_dist_rL, d_rL, dv_rL = -dist, -dist, ego_speed - traci.vehicle.getSpeed(veh_id)
            elif veh_lane == right_lane:
                if dist > 0 and dist < min_dist_fR: min_dist_fR, d_fR, dv_fR = dist, dist, ego_speed - traci.vehicle.getSpeed(veh_id)
                elif dist < 0 and -dist < min_dist_rR: min_dist_rR, d_rR, dv_rR = -dist, -dist, ego_speed - traci.vehicle.getSpeed(veh_id)
        state = [ego_speed, d_f, dv_f, d_r, dv_r, ego_lane, d_fL, dv_fL, d_rL, dv_rL, d_fR, dv_fR, d_rR, dv_rR]
        norm_factors = np.array([35, MAX_GAP, 40, MAX_GAP, 40, 1, MAX_GAP, 40, MAX_GAP, 40, MAX_GAP, 40, MAX_GAP, 40])
        return np.array(state) / norm_factors
    except traci.TraCIException: return np.zeros(STATE_SIZE)

def perform_action(action):
    if EGO_ID not in traci.vehicle.getIDList(): return
    current_speed = traci.vehicle.getSpeed(EGO_ID)
    if action == 1: traci.vehicle.setSpeed(EGO_ID, current_speed + 0.1)
    elif action == 2: traci.vehicle.setSpeed(EGO_ID, max(0, current_speed - 0.1))
    elif action == 3 and traci.vehicle.getLaneIndex(EGO_ID) == 1: traci.vehicle.changeLane(EGO_ID, 0, 2.0)
    elif action == 4 and traci.vehicle.getLaneIndex(EGO_ID) == 0: traci.vehicle.changeLane(EGO_ID, 1, 2.0)

def calculate_reward(done):
    if done: return -10.0
    reward = -abs(traci.vehicle.getSpeed(EGO_ID) - TARGET_SPEED) / TARGET_SPEED
    leader = traci.vehicle.getLeader(EGO_ID)
    if leader and (traci.vehicle.getSpeed(EGO_ID) > traci.vehicle.getSpeed(leader[0])):
        ttc = leader[1] / (traci.vehicle.getSpeed(EGO_ID) - traci.vehicle.getSpeed(leader[0]))
        if ttc < 1.0: reward -= 1.0
    return reward


episode_rewards = []
def plot_rewards():
    plt.figure(2); plt.clf(); plt.title('Training...'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.plot(episode_rewards)
    if len(episode_rewards) >= 100:
        means = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards)), means)
    plt.pause(0.001)

def run():
    agent = DQNAgent()
    for e in range(1000):
        traci.start(SUMO_CMD)
        total_reward = 0
        while EGO_ID not in traci.vehicle.getIDList():
            traci.simulationStep()
        
        state = get_full_state()
        for time_step in range(1000):
            action = agent.choose_action(state)
            perform_action(action)
            traci.simulationStep()
            
            done = EGO_ID not in traci.vehicle.getIDList()
            reward = calculate_reward(done)
            next_state = get_full_state()
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.learn()
            if done: break
        
        agent.update_epsilon()
        if e % TARGET_UPDATE == 0: agent.update_target_net()
        episode_rewards.append(total_reward)
        plot_rewards()
        print(f"Episode: {e+1}/1000, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        traci.close()
    print("Training finished.")
    torch.save(agent.policy_net.state_dict(), 'dqn_highway_model.pth')
    plt.ioff(); plt.show()

if __name__ == "__main__":
    run()
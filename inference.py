import os
import sys
import random
import matplotlib.pyplot as plt
traci_path = '/opt/homebrew/opt/sumo/share/sumo/tools'
if traci_path not in sys.path:
    sys.path.append(traci_path)
import traci
import numpy as np
import torch
import torch.nn as nn

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'.")

EGO_ID = 'ego_flow.0'
STATE_SIZE = 14
ACTION_SIZE = 5
MODEL_PATH = 'dqn_highway_model.pth'
MAX_GAP = 1000.0

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x): return self.network(x)

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

def run_inference(num_episodes=10):
    print(f"Loading trained model from: {MODEL_PATH}")
    model = DQN(STATE_SIZE, ACTION_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully. Starting inference...")

    completion_times = []
    success_count = 0

    for e in range(num_episodes):
        sumo_cmd = [
            "sumo-gui", "-c", "highway.sumocfg", "--step-length", "0.1",
            "--no-warnings", "true", "--collision.action", "remove",
            "--start", "--quit-on-end", "--seed", str(random.randint(0, 10000))
        ]
        traci.start(sumo_cmd)
        
        while EGO_ID not in traci.vehicle.getIDList():
            traci.simulationStep()
        
        state = get_full_state()
        episode_ended = False
        
        for time_step in range(1500):
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()

            perform_action(action)
            traci.simulationStep()
            
            if EGO_ID not in traci.vehicle.getIDList():
                collisions = traci.simulation.getEndingTeleportNumber()
                if collisions > 0:
                    print(f"Episode {e+1}/{num_episodes}: Ended due to COLLISION at step {time_step}.")
                else:
                    print(f"Episode {e+1}/{num_episodes}: Finished SUCCESSFULLY at step {time_step}.")
                    success_count += 1
                    completion_times.append(time_step * 0.1) # Convert steps to seconds
                episode_ended = True
                break
            
            state = get_full_state()
        
        if not episode_ended:
            print(f"Episode {e+1}/{num_episodes}: Finished SUCCESSFULLY by reaching max steps.")
            success_count += 1
            completion_times.append(1500 * 0.1)
        
        traci.close()

    print("\n--- INFERENCE SUMMARY ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Successful Episodes: {success_count} ({success_count/num_episodes*100:.1f}%)")
    
    if completion_times:
        print(f"Average Completion Time: {np.mean(completion_times):.2f} seconds")
        print(f"Fastest Time: {np.min(completion_times):.2f}s, Slowest Time: {np.max(completion_times):.2f}s")

        plt.figure(figsize=(10, 6))
        episodes = range(1, len(completion_times) + 1)
        plt.bar(episodes, completion_times, color='deepskyblue', label='Completion Time')
        
        avg_time = np.mean(completion_times)
        plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Average: {avg_time:.2f}s')
        
        plt.xlabel('Successful Episode Number')
        plt.ylabel('Completion Time (seconds)')
        plt.title('Agent Performance Across Inference Episodes')
        plt.xticks(episodes)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig('inference_performance.png')
        print("\nSaved inference performance plot to 'inference_performance.png'")
        plt.show()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    else:
        run_inference()
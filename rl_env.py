import torch
import random
from vanet_dataset import VanetGraphDataset

class VanetRLEnv:
    def __init__(self, csv_file="small_gnn_dqn_dataset.csv"): # make sure it points to your small CSV
        print("Initializing RL Environment...")
        self.dataset = VanetGraphDataset(csv_file)
        self.current_step = 0
        self.current_graph = None

    def reset(self):
        self.current_step = 0
        self.current_graph = self.dataset[self.current_step]
        return self.current_graph

    def calculate_reward(self, action, target_idx):
        """
        Implements the Autonomous Reward Function.
        Action: 0 = Accept (Normal), 1 = Reject (Attack)
        """
        graph = self.current_graph
        
        # --- NEW SAFETY CLAMP ---
        # Ensures target_idx never exceeds the available nodes in the tensor
        max_idx = graph.x.shape[0] - 1
        safe_target_idx = min(target_idx, max_idx)
        
        node_features = graph.x[safe_target_idx]
        
        rel_spd_x, rel_spd_y = node_features[2].item(), node_features[3].item()
        dist_diff = node_features[6].item()
        sender_speed = ((rel_spd_x**2 + rel_spd_y**2)**0.5) + graph.baseline_speed_mean
        
        mu = graph.baseline_speed_mean
        sig = graph.baseline_speed_std
        
        # Plausibility check: Is the speed within Mu +/- Sigma? AND is distance_diff ~ 0?
        is_plausible = (mu - sig <= sender_speed <= mu + sig) and (abs(dist_diff) < 2.0)
        
        # Assign Rewards
        if is_plausible:
            if action == 0: # Correctly Accepted
                return 1.0
            else:           # Wrongly Rejected
                return -1.0
        else:
            if action == 1: # Correctly Rejected
                return 1.0
            else:           # Wrongly Accepted
                return -1.0

    def step(self, action, target_idx):
        reward = self.calculate_reward(action, target_idx)
        
        self.current_step += 1
        done = self.current_step >= len(self.dataset)
        
        if not done:
            self.current_graph = self.dataset[self.current_step]
            next_state = self.current_graph
        else:
            next_state = None
            
        return next_state, reward, done
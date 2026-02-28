import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch_geometric.data import Batch
from hybrid_model import HybridDQN
from rl_env import VanetRLEnv

# FIXED HYPERPARAMETERS
LR = 0.01              # 10x higher (was 0.001)
GAMMA = 0.9           # 900x higher (was 0.001) - THIS IS THE KEY FIX!
EPSILON_START = 0.9
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
BATCH_SIZE = 128
MEMORY_CAPACITY = 50000
EPISODES = 300

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, target_idx, action, reward, next_state):
        self.memory.append((state, target_idx, action, reward, next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

def train_dqn_fixed():
    print("🚀 STARTING FIXED DQN TRAINING (GAMMA=0.9)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading environment...")
    env = VanetRLEnv("small_gnn_dqn_dataset.csv")
    
    print("Initializing networks...")
    policy_net = HybridDQN(gnn_weights_path="pretrained_sybil_gnn.pth").to(device)
    target_net = HybridDQN(gnn_weights_path="pretrained_sybil_gnn.pth").to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayMemory(MEMORY_CAPACITY)
    epsilon = EPSILON_START
    
    print("Beginning training loop...")
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            num_senders = state.x.shape[0] - 1
            if num_senders < 1:
                state, _, done = env.step(0, 0)
                steps += 1
                continue
            
            target_idx = random.randint(1, num_senders)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    batch = torch.zeros(state.x.shape[0], dtype=torch.long, device=device)
                    q_values = policy_net(state.x.to(device), state.edge_index.to(device), 
                                        batch, target_idx)
                    action = q_values.argmax(dim=1).item()
                     # ← ADD DEBUG PRINT HERE
                    if episode % 2 == 0 and random.random() < 0.1:
                        print(f"Ep{episode+1} Q_accept={q_values[0,0]:.2f}, Q_reject={q_values[0,1]:.2f}, action={action}")

            next_state, reward, done = env.step(action, target_idx)
            total_reward += reward
            
            memory.push(state, target_idx, action, reward, next_state)
            state = next_state if next_state is not None else state
            steps += 1
            
            # Batch training
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                
                states = [t[0] for t in transitions]
                actions = torch.tensor([int(t[2]) for t in transitions], dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor([float(t[3]) for t in transitions], dtype=torch.float32, device=device)
                
                batch_states = Batch.from_data_list(states).to(device)
                target_idxs = [batch_states.ptr[i].item() + t[1] for i, t in enumerate(transitions)]
                target_idxs = torch.tensor(target_idxs, dtype=torch.long, device=device)
                
                q_values = policy_net(batch_states.x, batch_states.edge_index, batch_states.batch, target_idxs)
                q_vals = q_values.gather(1, actions).squeeze(1)
                
                target_q = rewards.clone()
                non_final_mask = torch.tensor([t[4] is not None for t in transitions], dtype=torch.bool, device=device)
                non_final_next_states = [t[4] for t in transitions if t[4] is not None]
                
                if len(non_final_next_states) > 0:
                    batch_next_states = Batch.from_data_list(non_final_next_states).to(device)
                    next_target_idxs = []
                    for i, t in enumerate([t for t in transitions if t[4] is not None]):
                        local_idx = min(t[1], t[4].x.shape[0] - 1)
                        global_idx = batch_next_states.ptr[i].item() + local_idx
                        next_target_idxs.append(global_idx)
                    next_target_idxs = torch.tensor(next_target_idxs, dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        next_q_values = target_net(batch_next_states.x, batch_next_states.edge_index, 
                                                 batch_next_states.batch, next_target_idxs)
                        max_next_q = next_q_values.max(1)[0]
                        target_q[non_final_mask] += GAMMA * max_next_q
                
                loss = criterion(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if episode % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if episode % 10 == 0:
            print(f"Ep {episode+1}/{EPISODES}: Reward={total_reward:.1f} ε={epsilon:.3f} Steps={steps}")
    
    torch.save(policy_net.state_dict(), "fixed_dqn3.pth")
    print("✅ SAVED fixed_dqn2.pth")

if __name__ == "__main__":
    train_dqn_fixed()
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch_geometric.data import Batch
from hybrid_model import HybridDQN
from rl_env import VanetRLEnv

# Hyperparameters
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
EPOCHS = 10

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, target_idx, action, reward, next_state):
        self.memory.append((state, target_idx, action, reward, next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

def train_dqn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running DQN on {device}")
    
    # IMPORTANT: Use the smaller subset CSV for testing in Colab to save time
    env = VanetRLEnv("small_gnn_dqn_dataset.csv") 
    
    policy_net = HybridDQN(gnn_weights_path="pretrained_sybil_gnn.pth").to(device)
    target_net = HybridDQN(gnn_weights_path="pretrained_sybil_gnn.pth").to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayMemory(MEMORY_CAPACITY)
    
    epsilon = EPSILON_START
    
    for epoch in range(EPOCHS):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            num_senders = state.x.shape[0] - 1
            if num_senders < 1:
                state, _, done = env.step(0, 0)
                continue
                
            target_idx = random.randint(1, num_senders)
            
            # 1. Select Action (Epsilon-Greedy)
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    batch_zeros = torch.zeros(state.x.shape[0], dtype=torch.long, device=device)
                    q_values = policy_net(state.x.to(device), state.edge_index.to(device), batch_zeros, target_idx)
                    action = q_values.argmax(dim=1).item()
            
            # 2. Take Action in Environment
            next_state, reward, done = env.step(action, target_idx)
            total_reward += reward
            
            # 3. Store in Memory
            memory.push(state, target_idx, action, reward, next_state)
            
            # 4. Optimize / Train Neural Network (NEW BATCHED LOGIC)
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                
                # Unpack transitions
                states = [t[0] for t in transitions]
                actions = torch.tensor([int(t[2]) for t in transitions], dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor([float(t[3]) for t in transitions], dtype=torch.float32, device=device)
                
                # Group all 64 graphs into one massive graph for GPU processing
                batch_states = Batch.from_data_list(states).to(device)
                
                target_idxs = []
                for i, t in enumerate(transitions):
                    local_idx = t[1]
                    global_idx = batch_states.ptr[i].item() + local_idx
                    target_idxs.append(global_idx)
                target_idxs = torch.tensor(target_idxs, dtype=torch.long, device=device)
                
                # Fast forward pass for the whole batch
                q_values = policy_net(batch_states.x, batch_states.edge_index, batch_states.batch, target_idxs)
                q_vals = q_values.gather(1, actions).squeeze(1)
                
                # Calculate targets cleanly (no tensor warnings)
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
                        next_q_values = target_net(batch_next_states.x, batch_next_states.edge_index, batch_next_states.batch, next_target_idxs)
                        max_next_q = next_q_values.max(1)[0]
                        target_q[non_final_mask] += GAMMA * max_next_q
                
                loss_value = criterion(q_vals, target_q)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
        
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Reward: {total_reward} | Epsilon: {epsilon:.4f}")
        
    torch.save(policy_net.state_dict(), "final_hybrid_dqn.pth")
    print("DQN Training Complete. Weights Saved.")

if __name__ == "__main__":
    train_dqn()
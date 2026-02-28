import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_model import SybilGNN

class HybridDQN(nn.Module):
    def __init__(self, gnn_weights_path, num_features=7, gnn_hidden_dim=64, gnn_out_dim=32, dqn_hidden_dim=64):
        super(HybridDQN, self).__init__()
        
        # 1. Initialize the GNN and load the pre-trained weights
        self.gnn = SybilGNN(num_features=num_features, hidden_dim=gnn_hidden_dim, out_dim=gnn_out_dim)
        self.gnn.load_state_dict(torch.load(gnn_weights_path, map_location=torch.device('cpu')))
        
        # Freeze the GNN weights so it acts as a stable feature extractor
        for param in self.gnn.parameters():
            param.requires_grad = False
            
        # 2. Define the DQN specific layers
        # Input = Graph Context Vector (gnn_out_dim) + Target Node's Features (num_features)
        dqn_input_dim = gnn_out_dim + num_features
        
        self.fc1 = nn.Linear(dqn_input_dim, dqn_hidden_dim)
        self.fc2 = nn.Linear(dqn_hidden_dim, 32)
        
        # Output is 2 Q-values: [Q(Accept), Q(Reject)]
        self.q_out = nn.Linear(32, 2) 

    def forward(self, x, edge_index, batch, target_node_idx):
        # A. Get the Graph-level Context Vector from the pre-trained GNN
        context_vector, _, _ = self.gnn(x, edge_index, batch)
        
        # Ensure target_node_idx is a tensor (handles both single item and batched items)
        if isinstance(target_node_idx, int):
            target_node_idx = torch.tensor([target_node_idx], device=x.device)
            
        # B. Extract the specific sender's features that the agent is evaluating
        target_features = x[target_node_idx]
        if target_features.dim() == 1:
            target_features = target_features.unsqueeze(0)
        
        # C. Concatenate Context + Target Features
        combined = torch.cat([context_vector, target_features], dim=1)
        
        # D. Predict Q-Values
        q_vals = F.relu(self.fc1(combined))
        q_vals = F.relu(self.fc2(q_vals))
        q_vals = self.q_out(q_vals)
        
        return q_vals
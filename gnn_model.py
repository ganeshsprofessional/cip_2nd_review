import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SybilGNN(nn.Module):
    def __init__(self, num_features=7, hidden_dim=64, out_dim=32):
        super(SybilGNN, self).__init__()
        
        # Message Passing Layers
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=3, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 3, hidden_dim, heads=1, concat=False) # Outputs dimension: hidden_dim (64)
        
        # Output layer for the Graph-level Context Vector (used by DQN)
        self.lin = nn.Linear(hidden_dim, out_dim) 
        
        # Output layer for Node-level predictions (used ONLY for pre-training)
        # FIX: Changed 'out_dim' to 'hidden_dim' to match the output of conv2
        self.classifier = nn.Linear(hidden_dim, 2) 

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Message Passing
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        
        # Graph-level embedding (The "Context Vector" for the DQN)
        context_vector = global_mean_pool(h, batch)
        context_vector = self.lin(context_vector)
        
        # Node-level predictions (Used for pre-training)
        # 'h' has shape [num_nodes, hidden_dim], so classifier must accept hidden_dim
        node_logits = self.classifier(h)
        
        return context_vector, node_logits, h
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class VanetGraphDataset(Dataset):
    def __init__(self, csv_file, time_window=1.0):
        super().__init__()
        print("Loading and grouping data...")
        self.df = pd.read_csv(csv_file)
        
        # Create discrete time bins to group messages into "snapshots"
        self.df['time_bin'] = (self.df['rcvTime'] // time_window) * time_window
        
        # Group by receiver and time bin
        self.grouped = list(self.df.groupby(['receiver', 'time_bin']))
        print(f"Created {len(self.grouped)} graph snapshots.")

        # Define which features go into the GNN nodes
        self.feature_cols = [
            "rel_pos_x", "rel_pos_y", "rel_spd_x", "rel_spd_y", 
            "acceleration", "beacon_rate", "distance_diff"
        ]

    def len(self):
        return len(self.grouped)

    def get(self, idx):
        (receiver_id, time_bin), group = self.grouped[idx]
        
        # 1. Construct Node Features (X)
        # Node 0 is the Ego Vehicle (Receiver). Since features are relative to Ego, 
        # its relative position/speed is 0. We'll use its actual acceleration.
        ego_features = [0.0, 0.0, 0.0, 0.0, 
                        group['acceleration'].mean(), # Approximate Ego Accel
                        0.0, 0.0] 
        
        node_features = [ego_features]
        labels = [-1] # Node 0 (Ego) doesn't have an attack label
        
        # Nodes 1..N are the Senders
        for _, row in group.iterrows():
            node_features.append(row[self.feature_cols].values.tolist())
            labels.append(row['is_attack'])
            
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        
        # 2. Construct Edge Index (Star Graph: All senders point to Ego)
        num_senders = len(group)
        source_nodes = torch.arange(1, num_senders + 1)
        target_nodes = torch.zeros(num_senders, dtype=torch.long)
        
        # Add reverse edges so Ego can broadcast state to senders in GNN message passing
        edge_index = torch.stack([
            torch.cat([source_nodes, target_nodes]), 
            torch.cat([target_nodes, source_nodes])
        ], dim=0)

        # 3. Create PyG Data Object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Attach context for DQN reward calculation later
        data.baseline_speed_mean = group['avg_speed_1s'].iloc[0]
        data.baseline_speed_std = group['stddev_speed_1s'].iloc[0]
        data.receiver_id = receiver_id
        
        return data

# Test the loader
if __name__ == "__main__":
    dataset = VanetGraphDataset("gnn_dqn_ready_dataset.csv", time_window=1.0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch node feature shape: {batch.x.shape}")
    print(f"Batch edge index shape: {batch.edge_index.shape}")
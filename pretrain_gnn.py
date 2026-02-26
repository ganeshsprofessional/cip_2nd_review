import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from vanet_dataset import VanetGraphDataset
from gnn_model import SybilGNN

def pretrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    dataset = VanetGraphDataset("gnn_dqn_ready_dataset.csv")
    
    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SybilGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # We use CrossEntropyLoss and ignore Node 0 (Ego vehicle, label = -1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            _, node_logits, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Calculate loss only on sender nodes
            loss = criterion(node_logits, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    # Save the pre-trained weights
    torch.save(model.state_dict(), "pretrained_sybil_gnn.pth")
    print("Pre-training complete. Weights saved.")

if __name__ == "__main__":
    pretrain()
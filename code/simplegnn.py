#%%
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, adjacency_matrix):
        # x: input feature matrix (batch_size x num_nodes x input_dim)
        # adjacency_matrix: binary adjacency matrix (batch_size x num_nodes x num_nodes)
        
        # Compute the weighted sum of neighbors' features
        neighbors_sum = torch.matmul(adjacency_matrix, x)
        
        # Normalize by the degree of each node
        degree = adjacency_matrix.sum(dim=2, keepdim=True)
        normalized_neighbors_sum = neighbors_sum / degree
        
        # Apply linear transformation
        linear_output = self.linear(normalized_neighbors_sum)
        
        # Apply ReLU activation
        output = F.relu(linear_output)
        
        return output

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.gnn_layer = SimpleGNNLayer(input_dim, hidden_dim)
        
    def forward(self, x, adjacency_matrix, num_layers):
        # x: input feature matrix (batch_size x num_nodes x input_dim)
        # adjacency_matrix: binary adjacency matrix (batch_size x num_nodes x num_nodes)
        # num_layers: number of GNN layers
        
        # Iterate through GNN layers
        for _ in range(num_layers):
            x = self.gnn_layer(x, adjacency_matrix)
        
        return x

# Example usage
input_dim = 2
hidden_dim = 2
num_nodes = 3
batch_size = 1
num_layers = 1

# Sample input feature matrix (batch_size x num_nodes x input_dim)
x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)

# Sample adjacency matrix (batch_size x num_nodes x num_nodes)
adjacency_matrix = torch.tensor([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]], dtype=torch.float32)

# Create SimpleGNN model
gnn_model = SimpleGNN(input_dim, hidden_dim)

# Forward pass
output = gnn_model(x, adjacency_matrix, num_layers)
print("Output after GNN layers:")
print(output)

# %%

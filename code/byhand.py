#%%
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from tqdm.notebook import tqdm
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

pd.set_option('display.max_colwidth', None)

print(f"torch_geometric version: {torch_geometric.__version__}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = [[0,0,4],
        [0,1,3],
        [1,1,4],
        [2,1,5],
        [2,2,4]]

df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
print(f"Initial DataFrame:\n{df.head()}\n")

train, test = train_test_split(df.values, test_size=0.2, random_state=3)
print(f"Train size: {len(train)}, Test size: {len(test)}")

train_df = pd.DataFrame(train, columns=df.columns)
test_df = pd.DataFrame(test, columns=df.columns)
print(f"Train DataFrame size: {len(train_df)}, Test DataFrame size: {len(test_df)}")

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()

train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)
print(f"Encoded 'user_id_idx':\n{train_df['user_id_idx'].head()}\n")
print(f"Encoded 'item_id_idx':\n{train_df['item_id_idx'].head()}\n")

train_user_ids = train_df['user_id'].unique()
train_item_ids = train_df['item_id'].unique()
print(f"Unique users in training set: {len(train_user_ids)}, Unique items in training set: {len(train_item_ids)}")

# Filter test set to only include users/items seen during training
test_df = test_df[
    (test_df['user_id'].isin(train_user_ids)) & 
    (test_df['item_id'].isin(train_item_ids))
]
print(f"Filtered Test DataFrame size: {len(test_df)}")

test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)

n_users = train_df['user_id_idx'].nunique()
n_items = train_df['item_id_idx'].nunique()
print(f"Number of unique users: {n_users}, Number of unique items: {n_items}")

#%%
def data_loader(data, batch_size, n_usr, n_itm):
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    print("Interacted Items DataFrame:\n", interected_items_df.head())  # Shows a sample of user-item interactions
    
    indices = [x for x in range(n_usr)]
    
    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()  # Sort the user indices for consistent output
    print("Selected User Indices for Batch:", users)
    
    users_df = pd.DataFrame(users, columns=['users'])
    
    # Merge to filter only selected users' interactions
    interected_items_df = pd.merge(interected_items_df, users_df, how='right', left_on='user_id_idx', right_on='users')
    print("Filtered Interacted Items DataFrame:\n", interected_items_df)
    
    # Select positive items randomly from interacted items
    pos_items = interected_items_df['item_id_idx'].apply(lambda x: random.choice(x)).values
    print("Positive Items for Batch:", pos_items)
    
    # Sample negative items ensuring they are not in the interacted items
    neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values
    print("Negative Items for Batch:", neg_items)
    
    # Convert user, positive item, and negative item indices to tensors and offset item indices
    users_tensor = torch.LongTensor(list(users)).to(device)
    pos_items_tensor = torch.LongTensor(list(pos_items)).to(device) + n_usr
    neg_items_tensor = torch.LongTensor(list(neg_items)).to(device) + n_usr
    
    print("Users Tensor:", users_tensor)
    print("Positive Items Tensor:", pos_items_tensor)
    print("Negative Items Tensor:", neg_items_tensor)
    
    return users_tensor, pos_items_tensor, neg_items_tensor

# Example usage of the modified data_loader function
users_tensor, pos_items_tensor, neg_items_tensor = data_loader(train_df, 1, n_users, n_items)

# %%
# Convert user IDs in the training DataFrame to a PyTorch LongTensor
u_t = torch.LongTensor(train_df.user_id_idx)
print("User Tensor:", u_t)

# Convert item IDs in the training DataFrame to a PyTorch LongTensor and offset them by the number of users
# This ensures that user and item indices are in separate numerical spaces
i_t = torch.LongTensor(train_df.item_id_idx) + n_users
print("Item Tensor with Offset:", i_t)

# Create edge indices for a bipartite graph by concatenating user and item tensors
# This represents connections (edges) between users and items in the graph
# The graph is undirected, so edges are added in both directions: user to item and item to user
train_edge_index = torch.stack((
  torch.cat([u_t, i_t]),  # Concatenate user tensor and item tensor for one direction
  torch.cat([i_t, u_t])   # Concatenate item tensor and user tensor for the opposite direction
)).to(device)

print("Train Edge Index Tensor:\n", train_edge_index)

# %%


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')  # Use sum aggregation.
        print("Initialized LightGCNConv with sum aggregation.")

    def forward(self, x, edge_index):
        # edge_index: [2, E] where E is the number of edges.
        # x: [N, F] where N is the number of nodes and F is the number of features.
        print("Forward pass:")
        print(f"Input feature matrix shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")

        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        print(f"Degree of nodes (computed from 'to' indices): {deg}")

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Avoid division by zero.
        print(f"Inverse square root of degrees: {deg_inv_sqrt}")

        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        print(f"Normalization factors for edges: {norm}")

        # Propagate messages using the defined message and aggregate functions.
        result = self.propagate(edge_index, x=x, norm=norm)
        print(f"Resulting feature matrix shape after message passing: {result.shape}")
        return result

    def message(self, x_j, norm):
        # x_j: [E, F] where E is the number of edges.
        # This function computes the message to be sent along each edge.
        message = norm.view(-1, 1) * x_j  # Element-wise multiplication by the normalization factor.
        print(f"Message shape: {message.shape}")
        return message



LightGCNConv()(torch.Tensor(np.eye(6)), train_edge_index)


#%%

class NGCFConv(MessagePassing):
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)

        self.dropout = dropout

        # Linear transformations for embedding updates
        self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

        self.init_parameters()
        print(f"NGCFConv initialized with latent_dim={latent_dim}, dropout={dropout}")

    def init_parameters(self):
        # Initialize weights using Xavier uniform distribution
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)
        print("Weights initialized using Xavier uniform distribution.")

    def forward(self, x, edge_index):
        print("Forward pass started.")
        print(f"Input feature matrix shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")

        # Compute normalization based on node degrees
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        print(f"Normalization factors for edges: {norm}")

        # Propagate messages using the defined message function
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # Update node features post message aggregation
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        print(f"Output feature matrix shape after dropout: {out.shape}")

        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        # Message function combines features of connected nodes
        message = norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))
        print(f"Message shape: {message.shape}")
        return message



NGCFConv(6, 0.2)(torch.Tensor(np.eye(6)), train_edge_index)


#%%
class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
        model,  # 'NGCF' or 'LightGCN'
        dropout=0.1  # Only used in NGCF
    ):
        super(RecSysGNN, self).__init__()

        assert model in ['NGCF', 'LightGCN'], 'Model must be NGCF or LightGCN'
        self.model = model
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        if self.model == 'NGCF':
            self.convs = nn.ModuleList(
                NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
            )
        else:
            self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

        self.init_parameters()
        print(f"Initialized {self.model} model with {num_layers} layers, "
              f"latent_dim={latent_dim}, dropout={dropout} (if NGCF)")

    def init_parameters(self):
        if self.model == 'NGCF':
            nn.init.xavier_uniform_(self.embedding.weight, gain=1)
            print("Initialized embeddings with Xavier uniform for NGCF.")
        else:
            nn.init.normal_(self.embedding.weight, std=0.1)
            print("Initialized embeddings with normal distribution for LightGCN.")

    def forward(self, edge_index):
        print("Forward pass started.")
        emb0 = self.embedding.weight
        print(f"Initial embeddings shape: {emb0.shape}")

        embs = [emb0]

        emb = emb0  # Initialize emb with emb0 before the loop
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)  # Now emb is correctly initialized
            embs.append(emb)
            print(f"Updated embeddings shape after layer: {emb.shape}")


        if self.model == 'NGCF':
            out = torch.cat(embs, dim=-1)
            print("Concatenated embeddings for NGCF.")
        else:
            out = torch.mean(torch.stack(embs, dim=0), dim=0)
            print("Averaged embeddings for LightGCN.")

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        print("Encoding minibatch.")
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )


# %%
def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    # Compute regularization loss from initial embeddings
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) +
        pos_emb0.norm().pow(2) +
        neg_emb0.norm().pow(2)
    ) / float(len(users))
    print(f"Regularization loss: {reg_loss}")

    # Compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
    print(f"BPR loss: {bpr_loss}")

    return bpr_loss, reg_loss

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K):
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())
    print(f"Unique test user IDs: {test_user_ids}")

    # Compute the score of all user-item pairs
    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))
    print(f"Relevance scores shape: {relevance_score.shape}")

    # Create dense tensor of all user-item interactions
    i = torch.stack((
        torch.LongTensor(train_data['user_id_idx'].values),
        torch.LongTensor(train_data['item_id_idx'].values)
    ))
    v = torch.ones((len(train_data)), dtype=torch.float64)
    interactions_t = torch.sparse.FloatTensor(i, v, (n_users, n_items)).to_dense().to(device)
    print(f"Interactions tensor shape: {interactions_t.shape}")

    # Mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))
    print("Masked relevance scores to exclude training interactions.")

    # Compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    print(f"Top-{K} relevance indices shape: {topk_relevance_indices.shape}")

    # Prepare DataFrame for metrics computation
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(), columns=['top_indx_' + str(x + 1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_id_idx', right_on=['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    print(f"Average recall: {metrics_df['recall'].mean()}, Average precision: {metrics_df['precision'].mean()}")

    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df



#%%

latent_dim = 64
n_layers = 3

EPOCHS = 50
BATCH_SIZE = 1
DECAY = 0.0001
LR = 0.1
K = 1

def train_and_eval(model, optimizer, train_df):
    loss_list_epoch = []
    bpr_loss_list_epoch = []
    reg_loss_list_epoch = []

    recall_list = []
    precision_list = []

    for epoch in tqdm(range(EPOCHS)):
        print(f"Starting epoch {epoch + 1}/{EPOCHS}")
        n_batch = int(len(train_df) / BATCH_SIZE)
        print(f"Number of batches: {n_batch}")

        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []

        model.train()
        for batch_idx in range(n_batch):
            optimizer.zero_grad()

            # Load batch data
            users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
            # Encode minibatch
            users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

            # Compute losses
            bpr_loss, reg_loss = compute_bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0)
            reg_loss *= DECAY
            final_loss = bpr_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

            print(f"Batch {batch_idx + 1}/{n_batch}, BPR Loss: {bpr_loss.item()}, Reg Loss: {reg_loss.item()}, Total Loss: {final_loss.item()}")

        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            test_topK_recall, test_topK_precision, metrics = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train_df, test_df, K)

        # Store average losses and metrics for the epoch
        loss_list_epoch.append(np.mean(final_loss_list))
        bpr_loss_list_epoch.append(np.mean(bpr_loss_list))
        reg_loss_list_epoch.append(np.mean(reg_loss_list))
        recall_list.append(test_topK_recall)
        precision_list.append(test_topK_precision)

        print(f"Epoch {epoch + 1} completed. Avg Total Loss: {loss_list_epoch[-1]}, Avg BPR Loss: {bpr_loss_list_epoch[-1]}, Avg Reg Loss: {reg_loss_list_epoch[-1]}, Recall: {recall_list[-1]}, Precision: {precision_list[-1]}")

    return loss_list_epoch, bpr_loss_list_epoch, reg_loss_list_epoch, recall_list, precision_list, metrics


"""### Train and eval LightGCN"""

# Initialize the LightGCN model with the specified parameters
lightgcn = RecSysGNN(
    latent_dim=latent_dim,
    num_layers=n_layers,
    num_users=n_users,
    num_items=n_items,
    model='LightGCN'
)
lightgcn.to(device)  # Move the model to the appropriate device (GPU or CPU)

# Initialize the optimizer for the LightGCN model
optimizer = torch.optim.Adam(lightgcn.parameters(), lr=LR)
print("Initialized LightGCN model with the following parameters:")
print("Latent dimension:", latent_dim)
print("Number of layers:", n_layers)
print("Learning rate:", LR)
print("Size of learnable embeddings and parameters:", [x.shape for x in list(lightgcn.parameters())])

# Train and evaluate the LightGCN model
light_loss, light_bpr, light_reg, light_recall, light_precision, metrics = train_and_eval(lightgcn, optimizer, train_df)
print("Training and evaluation completed.")

# Prepare data for plotting
epoch_list = [(i + 1) for i in range(EPOCHS)]

# Plotting training loss over epochs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_list, light_loss, label='Total Training Loss')
plt.plot(epoch_list, light_bpr, label='BPR Training Loss')
plt.plot(epoch_list, light_reg, label='Reg Training Loss')
plt.title('LightGCN Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting recall and precision metrics over epochs
plt.subplot(1, 2, 2)
plt.plot(epoch_list, light_recall, label='Recall')
plt.plot(epoch_list, light_precision, label='Precision')
plt.title('LightGCN Recall and Precision')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

plt.tight_layout()
plt.show()


"""### Train and eval NGCF"""
# Initialize the NGCF model with specified parameters
ngcf = RecSysGNN(
    latent_dim=latent_dim,
    num_layers=n_layers,
    num_users=n_users,
    num_items=n_items,
    model='NGCF'
)
ngcf.to(device)  # Move the model to the appropriate device (GPU or CPU)

# Initialize the optimizer for the NGCF model
optimizer = torch.optim.Adam(ngcf.parameters(), lr=LR)
print("Initialized NGCF model with the following parameters:")
print("Latent dimension:", latent_dim)
print("Number of layers:", n_layers)
print("Learning rate:", LR)
print("Size of learnable embeddings and parameters:", [x.shape for x in list(ngcf.parameters())])

# Train and evaluate the NGCF model
ngcf_loss, ngcf_bpr, ngcf_reg, ngcf_recall, ngcf_precision, metrics = train_and_eval(ngcf, optimizer, train_df)
print("Training and evaluation completed for NGCF.")

# Prepare data for plotting
epoch_list = [(i + 1) for i in range(EPOCHS)]

# Plotting training loss over epochs for NGCF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_list, ngcf_loss, label='Total Training Loss')
plt.plot(epoch_list, ngcf_bpr, label='BPR Training Loss')
plt.plot(epoch_list, ngcf_reg, label='Reg Training Loss')
plt.title('NGCF Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting recall and precision metrics over epochs for NGCF
plt.subplot(1, 2, 2)
plt.plot(epoch_list, ngcf_recall, label='Recall')
plt.plot(epoch_list, ngcf_precision, label='Precision')
plt.title('NGCF Recall and Precision')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

plt.tight_layout()
plt.show()

# Compare model performance
max_light_precision, max_light_recall = max(light_precision), max(light_recall)
max_ngcf_precision, max_ngcf_recall = max(ngcf_precision), max(ngcf_recall)

print("Maximum Precision and Recall for LightGCN:")
print("Precision:", max_light_precision, "Recall:", max_light_recall)

print("Maximum Precision and Recall for NGCF:")
print("Precision:", max_ngcf_precision, "Recall:", max_ngcf_recall)



# %%

# %%

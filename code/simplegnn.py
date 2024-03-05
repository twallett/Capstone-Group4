#%%
import pandas as pd
movie_path = 'https://raw.githubusercontent.com/twallett/Capstone-Group4/main/data/MovieLens/raw/ml-latest-small/movies.csv'
rating_path = 'https://raw.githubusercontent.com/twallett/Capstone-Group4/main/data/MovieLens/raw/ml-latest-small/ratings.csv'

movie = pd.read_csv(movie_path)
rating = pd.read_csv(rating_path)
print (movie.shape,rating.shape)


#%%
## get genre index
## input the list, return the dict {"genre":idx}
def genres2index(genres):
    genre_2_idx={}
    idx=0
    for x in genres:
        for xi in x.split("|"):
            if xi not in genre_2_idx.keys():
                genre_2_idx[xi]=idx
                idx+=1
    return genre_2_idx
genre_index=genres2index(movie.genres.tolist())

## use mid instead of movieid, since the movieid is not continous
movie["mid"] = movie.index
movie_2_genre = []
for mid,genres in movie[["mid","genres"]].values:
    for gx in genres.split("|"):
        movie_2_genre.append([mid,genre_index[gx]])

## prepare genre_x, onehot encode 20x20
genre_x = []
for k,v in genre_index.items():
    x = [ 0 for i in range(len(genre_index))]
    x[v]=1
    genre_x.append(x)


# #%%
# import torchtext
# import pandas as pd

# def title2vector(x):
#     x=x.split("(")[0]
#     x2v=glove840b.get_vecs_by_tokens([ xi.lower() for xi in x.split(" ")])
#     if len(x2v.size())==2:
#         x2v=x2v.mean(dim=0)
#     return x2v.view(1,300)

# glove840b=torchtext.vocab.GloVe("840B")
# import torch
# titles=[]
# for title in movie.title:
#     titles.append(title2vector(title))
# titles_tensor=torch.cat(titles)


# #%%
# rating_m = rating.merge(movie,left_on="movieId",right_on="movieId")
# n_users = len(rating.userId.unique())
# user_rates_movie = torch.from_numpy(rating_m[["userId","mid"]].transpose().values-1)
# user_rates_movie_attr = torch.from_numpy(rating_m["rating"].values).float().view(len(rating_m),1)


# user_x = []
# for i in range(610):
#     v=[ 0 for j in range(610)]
#     v[i]=1
#     user_x.append(v)

# #%%
# from torch_geometric.data import HeteroData

# data = HeteroData()

# #data['user'].num_nodes = n_users  # Users do not have any features.
# data['user'].x = user_x
# data['movie'].x = movie_x
# data['genre'].x = genre_x


# data['user', 'rates', 'movie'].edge_index = user_rates_movie
# data['user', 'rates', 'movie'].train_mask=train_flag
# data['user', 'rates', 'movie'].test_mask=test_flag

# data['user', 'rates', 'movie'].edge_label = user_rates_movie_attr
# data['movie', 'belongto', 'genre'].edge_index = movie_2_genre

# print(data)




#%%===========================================================
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric import EdgeIndex

def preprocess(df, user_col, item_col, features, time, latent_dim = 64):
    data = HeteroData()

    user_mapping = {idx:enum for enum, idx in enumerate(df[user_col].unique())}
    item_mapping = {idx:enum for enum, idx in enumerate(df[item_col].unique())}

    src = [user_mapping[idx] for idx in df[user_col]]
    dst = [item_mapping[idx] for idx in df[item_col]]
    edge_index = torch.tensor([src, dst])
    
    data['user'].x = torch.eye(len(user_mapping))
    data['item'].x = torch.randn(len(item_mapping), latent_dim).detach().numpy()
    data['item'].num_nodes = len(item_mapping)
    
    time = torch.from_numpy(df[time].values).to(torch.long)
    
    for enum, feature in enumerate(features):
        feature_x = torch.from_numpy(df[feature].values).to(torch.long)
        data['user', f'feature{enum}', 'item'].edge_index = edge_index
        data['user', f'feature{enum}', 'item'].edge_label = feature_x
        data['user', f'feature{enum}', 'item'].time = time
        
    data = T.ToUndirected()(data)
        
    return data


data = preprocess(df = rating,
                  user_col = 'userId',
                  item_col = 'movieId',
                  features = ['rating'],
                  time = 'timestamp',
                  latent_dim = 64)



#%%
def split(data, train_size = 0.8):
    
    edge_label_index = data['user', 'item'].edge_index
    time = data['user', 'item'].time
    
    perm = time.argsort()
    train_index = perm[:int(train_size * perm.numel())]
    test_index = perm[int(train_size * perm.numel()):]
    
    return edge_label_index, train_index, test_index 
    3
def dataloader(data, edge_label_index, train_index, test_index, device, batch_size = 256):
    
    kwargs = dict(  # Shared data loader arguments:
        data=data,
        num_neighbors=[5, 5, 5],
        batch_size=batch_size,
    )

    train_loader = LinkNeighborLoader(
        edge_label_index= (('user', 'item'), edge_label_index[:, train_index]),
        neg_sampling= dict(mode='binary', amount=2),
        shuffle=True,
        **kwargs
        )

    src_loader = NeighborLoader(
        input_nodes='user',
        **kwargs
        )

    dst_loader = NeighborLoader(
        input_nodes='item',
        **kwargs
        )

    sparse_size = (data['user'].num_nodes, data['item'].num_nodes)

    test_edge_label_index = EdgeIndex(
        edge_label_index[:, test_index].to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]

    test_exclude_links = EdgeIndex(
        edge_label_index[:, train_index].to(device),
        sparse_size=sparse_size,
    ).sort_by('row')[0]
    
    return train_loader, src_loader, dst_loader, test_edge_label_index, test_exclude_links

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset

########## Movie Lens ##########

# columns_name=['user_id','item_id','rating','timestamp']
# df = pd.read_csv("MovieLens100K.txt", 
#                  sep="\t", 
#                  names=columns_name)

# df = df[df['rating'] >= 4]

data = preprocess(df = rating,
                  user_col = 'userID',
                  item_col = 'movieId',
                  features = ['rating'],
                  time = 'timestamp',
                  latent_dim = 64)




#%%===========================================================
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

#%%
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

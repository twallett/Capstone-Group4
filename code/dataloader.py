import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

def detect_feature_type(series):
    if series.dtype in ['int64', 'float64']:
        return 'numerical'
    elif any(series.str.contains('|')):
        return 'multi_categorical'
    else:
        return 'text'

def apply_encoder(data, encoder):
    if encoder == 'numerical':
        return torch.tensor(data.values).float().unsqueeze(1)
    elif encoder == 'multi_categorical':
        categories = set(cat for row in data for cat in row.split('|'))
        category_map = {c: i for i, c in enumerate(categories)}
        encoded = torch.zeros(len(data), len(categories))
        for i, row in enumerate(data):
            for cat in row.split('|'):
                encoded[i, category_map[cat]] = 1
        return encoded
    elif encoder == 'text':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with torch.no_grad():
            embeddings = model.encode(data.tolist(), convert_to_tensor=True)
        return embeddings

def load_node_csv(path, node_type, feature_cols=None):
    df = pd.read_csv(path)
    data = HeteroData()
    for col in feature_cols or []:
        encoder = detect_feature_type(df[col])
        data[node_type][col] = apply_encoder(df[col], encoder)
    data[node_type].num_nodes = len(df)
    return data

def load_edge_csv(path, src_type, dst_type, src_col, dst_col, edge_attr_cols=None):
    df = pd.read_csv(path)
    edge_index = torch.tensor([df[src_col].values, df[dst_col].values])
    data = HeteroData()
    data[src_type, dst_type].edge_index = edge_index
    for col in edge_attr_cols or []:
        encoder = detect_feature_type(df[col])
        data[src_type, dst_type][col] = apply_encoder(df[col], encoder)
    return data

# Example usage
# data = HeteroData()
# node_data = load_node_csv('path_to_nodes.csv', 'node_type', ['feature1', 'feature2'])
# edge_data = load_edge_csv('path_to_edges.csv', 'src_type', 'dst_type', 'src_col', 'dst_col', ['edge_feature'])
# data.update(node_data)
# data.update(edge_data)

# print(data)

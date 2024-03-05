#%%
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall
    )
from torch_geometric.nn import MIPSKNNIndex, GATConv, to_hetero

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x
    
class InnerProductDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict['user'][edge_label_index[0]]
        x_dst = x_dict['item'][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.encoder = GAT(hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x_dict, edge_label_index)

class GAT_Model():
    
    def __init__(self, train_loader, src_loader, dst_loader, test_edge_label_index, test_exclude_links, K, device):
        self.train_loader = train_loader
        self.dst_loader = dst_loader
        self.src_loader = src_loader
        self.test_edge_label_index = test_edge_label_index
        self.test_exclude_links = test_exclude_links
        self.K = K
        self.device = device
    
    def train(self, model, optimizer):
                
        model.train()

        total_loss = total_examples = 0
        for batch in tqdm(self.train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()

            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'item'].edge_label_index,
            )
            y = batch['user', 'item'].edge_label

            loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * y.numel()
            total_examples += y.numel()

        return total_loss / total_examples

    @torch.no_grad()
    def test(self, model, edge_label_index, exclude_links):
        model.eval()

        dst_embs = []
        for batch in self.dst_loader:  # Collect destination node/item embeddings:
            batch = batch.to(self.device)
            emb = model.encoder(batch.x_dict, batch.edge_index_dict)['item']
            emb = emb[:batch['item'].batch_size]
            dst_embs.append(emb)
        dst_emb = torch.cat(dst_embs, dim=0)
        del dst_embs

        # Instantiate k-NN index based on maximum inner product search (MIPS):
        mips = MIPSKNNIndex(dst_emb)

        # Initialize metrics:
        map_metric = LinkPredMAP(k=self.K).to(self.device)
        precision_metric = LinkPredPrecision(k=self.K).to(self.device)
        recall_metric = LinkPredRecall(k=self.K).to(self.device)

        num_processed = 0
        for batch in self.src_loader:  # Collect source node/user embeddings:
            batch = batch.to(self.device)

            # Compute user embeddings:
            emb = model.encoder(batch.x_dict, batch.edge_index_dict)['user']
            emb = emb[:batch['user'].batch_size]

            # Filter labels/exclusion by current batch:
            _edge_label_index = edge_label_index.sparse_narrow(
                dim=0,
                start=num_processed,
                length=emb.size(0),
            )
            _exclude_links = exclude_links.sparse_narrow(
                dim=0,
                start=num_processed,
                length=emb.size(0),
            )
            num_processed += emb.size(0)

            # Perform MIPS search:
            _, pred_index_mat = mips.search(emb, self.K, _exclude_links)

            # Update retrieval metrics:
            map_metric.update(pred_index_mat, _edge_label_index)
            precision_metric.update(pred_index_mat, _edge_label_index)
            recall_metric.update(pred_index_mat, _edge_label_index)

        return (
            pred_index_mat,
            float(map_metric.compute()),
            float(precision_metric.compute()),
            float(recall_metric.compute()),
        )
    
    # @torch.no_grad()
    # def test(self,  model, edge_label_index, exclude_links):
    #     model.eval()
    #     preds, targets = [], []
    #     for batch in self.src_loader:
    #         batch = batch.to(self.device)
    #         pred = model(
    #             batch.x_dict,
    #             batch.edge_index_dict,
    #             batch['user', 'item'].edge_label_index,
    #         ).clamp(min=0, max=5)
    #         preds.append(pred)
    #         targets.append(batch['user', 'item'].edge_label.float())

    #     pred = torch.cat(preds, dim=0)
    #     target = torch.cat(targets, dim=0)
    #     rmse = (pred - target).pow(2).mean().sqrt()
    #     return float(rmse)
        
    def forward(self, data, epochs = 100, hidden_channels = 64):
        
        model = Model(data=data,
                      hidden_channels = hidden_channels).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
         
        for epoch in range(0, epochs):
            train_loss = self.train(model, optimizer)
            print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
            pred, val_map, val_precision, val_recall = self.test(
                model,
                self.test_edge_label_index,
                self.test_exclude_links,
            )
            print(f'Test MAP@{self.K}: {val_map:.4f}, '
                  f'Test Precision@{self.K}: {val_precision:.4f}, '
                  f'Test Recall@{self.K}: {val_recall:.4f}')
            
        return pred
            
    # def recommend(self, user_id):
        
    #     return item_ids
        

# %%

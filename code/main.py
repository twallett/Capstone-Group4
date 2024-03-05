#%%
import pandas as pd
from utils import *
from models.GAT import GAT_Model
from models.SAGE import SAGE_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset

########## Movie Lens ##########

columns_name=['user_id','item_id','rating','timestamp']
df = pd.read_csv("MovieLens100K.txt", 
                 sep="\t", 
                 names=columns_name)

df = df[df['rating'] >= 4]

########## Movie Lens ##########

########## Food ##########

# df = pd.read_csv("food.csv")
# df = df.iloc[:,:-2]
# df['timestamp'] = pd.to_datetime(df.date).astype(int) // 10**9
# df.drop(columns='date')
# df = df[df['rating'] >= 4]

########## Food ##########

# Preprocessing 

########## Preprocessing ##########

data = preprocess(df = df,
                  user_col = 'user_id',
                  item_col = 'item_id',
                  features = ['rating'],
                  time = 'timestamp',
                  latent_dim = 64)

# data = preprocess(df = df,
#                   user_col = 'user_id',
#                   item_col = 'recipe_id',
#                   features = ['rating'],
#                   time = 'timestamp',
#                   latent_dim = 64)

edge_label_index, train_index, test_index = split(data = data, 
                                                  train_size = 0.8)

train_loader, src_loader, dst_loader, test_edge_label_index, test_exclude_links = dataloader(data, 
                                                                                             edge_label_index, 
                                                                                             train_index, 
                                                                                             test_index, 
                                                                                             device,
                                                                                             batch_size=256)

########## Preprocessing ##########

# Models 

# put if main file

########## GAT Model ##########

model = GAT_Model(train_loader, src_loader, dst_loader, test_edge_label_index, test_exclude_links, K = 10, device = device)

emb = model.forward(data=data,
                    epochs= 1)

########## GAT Model ##########

########## SAGE Model ##########

# model = SAGE_Model(train_loader, src_loader, dst_loader, test_edge_label_index, test_exclude_links, K = 10, device=device)

# model.forward(data=data, epochs= 5)

########## SAGE Model ##########

#%%

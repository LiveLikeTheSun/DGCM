import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class Model(nn.Module):
    def __init__(self, args, input_feature, hidden_feature, out_feature, num_classes=1):
        super(GNNModel, self).__init__()
        self.args = args
        torch.manual_seed(12345)
        self.query_embedding = nn.Embedding(6338, 160)
        self.doc_embedding = nn.Embedding(56477, 160)
        self.title_embedding = nn.Embedding(56363, 160)
        self.position_embedding = nn.Embedding(10, 1)
        self.click_embedding = nn.Embedding(2, 1)
        self.conv1 = GATConv(input_feature, hidden_feature)
        self.conv2 = GATConv(hidden_feature, out_feature)
        self.linear = nn.Linear(out_feature, num_classes)

    def forward(self, i, x, edge_index, doc_id, click, query, docu, title_id):
        query_embed = self.query_embedding(query)
        doc_embed = self.doc_embedding(docu)
        title_embed = self.title_embedding(title_id)
        position_embedding = self.position_embedding(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda())
        click_embedding = self.click_embedding(click)
        
        if i == 0:
            idx = 0
            for docs in doc_id.tolist():
                x[docs[0]-1][:160] = query_embed[idx]  # query
                ind = 0
                for doc in docs:
                    x[doc][:160] = doc_embed[idx][ind]
                    x[doc][160] = position_embedding[docs.index(doc)][0]
                    x[doc][161+i] = click_embedding[idx][0]
                    x[doc+10][:160] = title_embed[idx][ind]
                    ind += 1
                idx += 1
        else:
            idx = 0
            for docs in doc_id.tolist():
                ind = 0
                for doc in docs:
                    x[doc][161+i] = click_embedding[idx][0]
                    ind += 1
                idx += 1
        
        y = self.conv1(x, edge_index)
        y = y.relu()
        y = F.dropout(y, p=0.5, training=self.training)
        hidden_state = self.conv2(y, edge_index)
        y = hidden_state.relu()
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.linear(y)
        y = torch.sigmoid(y)
        return hidden_state, y, x[:, :171]

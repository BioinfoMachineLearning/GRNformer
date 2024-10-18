import torch
import lightning as pl

import torch_geometric as pyg
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer




class GRNFormerLayerBlock(TransformerConv):
    def __init__(self, in_channels,out_channels,heads,**kwargs):
        super(GRNFormerLayerBlock, self).__init__(in_channels=in_channels,out_channels=out_channels)
        self.transformer_conv = TransformerConv(in_channels=in_channels,out_channels=out_channels,heads=heads,**kwargs)
        self.transf1 = Linear(out_channels*heads,out_channels)
        self.b1 = BatchNorm(out_channels)
        self.gelu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        
        out, attention_weights = self.transformer_conv(x, edge_index, edge_attr,return_attention_weights=True)
        out = self.transf1(out)
        out = self.gelu(out)
        #out = self.b1(out)
        out = self.dropout(out)
        return out, attention_weights
    

class GATFormerLayerBlock(TransformerConv):
    def __init__(self, in_channels,out_channels,heads,**kwargs):
        super(GATFormerLayerBlock, self).__init__(in_channels=in_channels,out_channels=out_channels)
        self.transformer_conv = GATConv(in_channels=in_channels,out_channels=out_channels,heads=heads,**kwargs)
        self.transf1 = Linear(out_channels*heads,out_channels)
        self.b1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        
        out, attention_weights = self.transformer_conv(x, edge_index, edge_attr,return_attention_weights=True)
        out = self.relu(self.transf1(out))
        out = self.b1(out)
        return out, attention_weights
    
class GATv2FormerLayerBlock(TransformerConv):
    def __init__(self, in_channels,out_channels,heads,**kwargs):
        super(GATv2FormerLayerBlock, self).__init__(in_channels=in_channels,out_channels=out_channels)
        self.transformer_conv = GATv2Conv(in_channels=in_channels,out_channels=out_channels,heads=heads,**kwargs)
        self.transf1 = Linear(out_channels*heads,out_channels)
        self.b1 = nn.BatchNorm1d(out_channels)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        
        out, attention_weights = self.transformer_conv(x, edge_index, edge_attr,return_attention_weights=True)
        out = self.transf1(out)
        out = self.b1(out)
        return out, attention_weights
    

class Embedder(nn.Module):
    def __init__(self, node_dim, edge_dim,out_edge_feats, out_node_feats):
        super().__init__()
        self.node_encoder = nn.Sequential(
            Linear(node_dim, out_node_feats // 2),
            nn.ReLU(),
            Linear(out_node_feats // 2, out_node_feats),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            Linear(edge_dim, out_edge_feats // 2),
            nn.ReLU(),
            Linear(out_edge_feats // 2, out_edge_feats),
            nn.ReLU(),
        )

    def forward(self, x, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr



class SingleHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleHeadAttention, self).__init__()
        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        attention_probs = self.softmax(attention_scores)
        
        context = torch.matmul(attention_probs, value)
        return self.out_proj(context)


class FullyConnectedGT_UGformerV2(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size,
                 num_self_att_layers, dropout, edge_dim, nhead,num_GNN_layers):
        super(FullyConnectedGT_UGformerV2, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.bcn = nn.BatchNorm1d(self.feature_dim_size)
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        self.num_GNN_layers = num_GNN_layers
        self.nhead = nhead
        #self.lst_gnn = torch.nn.ModuleList()
        #
        self.ugformer_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=self.nhead, dim_feedforward=self.ff_hidden_size, dropout=0.5,batch_first=True) # Default batch_first=False (seq, batch, feature)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            #self.lst_gnn.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))
        self.edge_encoder = nn.Sequential(
            Linear(edge_dim, feature_dim_size // 2),
            nn.ReLU(),
            Linear(feature_dim_size // 2, feature_dim_size),
            nn.ReLU(),
        )

    def forward(self, node_features,edge_attr):
        
        input_Tr = node_features
        for layer_idx in range(self.num_GNN_layers):
            # self-attention over all nodes
            #input_Tr = torch.unsqueeze(input_Tr, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
            graph_embedding = self.ugformer_layers[layer_idx](input_Tr)
            graph_embedding = self.bcn(graph_embedding)
            #input_Tr = torch.squeeze(input_Tr, 1)
            # take a sum over neighbors followed by a linear transformation and an activation function --> similar to GCN
            #input_Tr = self.lst_gnn[layer_idx](input_Tr, Adj_block)
            # take a sum over all node representations to get graph representations
            #graph_embedding = torch.sum(input_Tr, dim=0)
            #graph_embedding = self.dropouts[layer_idx](graph_embedding)
        edge_attr = self.edge_encoder(edge_attr)
        return graph_embedding, edge_attr

class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist
############ PROGRESS OF GRNFORMER MODEL BY MODEL #############################################################
'''
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    

class VariationalTransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_of_layers):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, 2 * out_channels,heads=num_head,beta=True,dropout=0.1)
        self.transf1 = Linear(2*out_channels*num_head, 2*out_channels)
        self.b1 = nn.BatchNorm1d(2*out_channels)
        self.conv2 = TransformerConv(2*out_channels, 2 * out_channels,heads=num_head)
        self.transf2 = Linear(2*out_channels*num_head, 2*out_channels)
        self.relu = nn.ReLU(inplace=True)    
        self.b2 =nn.BatchNorm1d(2*out_channels)
        
        
        self.conv_mu = TransformerConv(2 * out_channels, out_channels,heads=num_head)
        self.conv_logstd = TransformerConv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.relu(self.transf1(x))
        x = self.b1(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_of_layers):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, 2 * out_channels,heads=num_head,beta=True,dropout=0.1)
        self.transf1 = Linear(2*out_channels*num_head, 2*out_channels)
        self.b1 = nn.BatchNorm1d(2*out_channels)
        self.conv2 = TransformerConv(2*out_channels, 2 * out_channels,heads=num_head)
        self.transf2 = Linear(2*out_channels*num_head, 2*out_channels)
        self.relu = nn.ReLU(inplace=True)    
        self.b2 =nn.BatchNorm1d(2*out_channels)
        
        
        self.conv_mu = TransformerConv(2 * out_channels, out_channels,heads=num_head)
        #self.conv_logstd = TransformerConv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.relu(self.transf1(x))
        x = self.b1(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x = self.conv2(x,edge_index)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        return self.conv_mu(x, edge_index)
    

class EdgeTransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_layers):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, 2 * out_channels,heads=num_head,edge_dim=edge_dim,beta=True,dropout=0.1)
        self.transf1 = Linear(2*out_channels*num_head, 2*out_channels)
        self.b1 = nn.BatchNorm1d(2*out_channels)
        self.conv2 = TransformerConv(2*out_channels, 2 * out_channels,heads=num_head,edge_dim=edge_dim,dropout=0.1,beta=True)
        self.transf2 = Linear(2*out_channels*num_head, 2*out_channels)
        self.relu = nn.ReLU(inplace=True)    
        self.b2 =nn.BatchNorm1d(2*out_channels)
        
        
        self.conv_mu = TransformerConv(2 * out_channels, out_channels,heads=num_head)
        self.conv_logstd = TransformerConv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index,edge_attr):
        x = x.float()
        
        x,edge_index = self.conv1(x, edge_index,edge_attr,return_attention_weights=True)
        
        x = self.relu(self.transf1(x))
        x = self.b1(x)
        x,edge_index = self.conv2(x,edge_index[0],edge_attr,return_attention_weights=True)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x,edge_index= self.conv2(x,edge_index[0],edge_attr,return_attention_weights=True)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        x ,edge_index= self.conv2(x,edge_index[0],edge_attr,return_attention_weights=True)
        x = self.relu(self.transf2(x))
        x = self.b2(x)
        return self.conv_mu(x, edge_index[0]), self.conv_logstd(x, edge_index[0])
        '''
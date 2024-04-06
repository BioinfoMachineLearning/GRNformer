import torch
import lightning as pl

import torch_geometric as pyg
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn.models import InnerProductDecoder
from GRNModelblocks import GATFormerLayerBlock

# Definition of Encoder

class Encoder(nn.Module):

    def __init__(self,in_channels,out_channels,heads,**kwargs):
        super().__init__()
        self.encoder1 = GATFormerLayerBlock(in_channels=in_channels,out_channels=out_channels,heads=heads)
        self.encoder2 = GATFormerLayerBlock(in_channels=out_channels,out_channels=4,heads=heads)

    def forward(self,x,edge_index,edge_attr):
        #x, edge_index = x.x, x.train_pos_edge_index
        x= x.float()
        print(x.shape,edge_index.shape,edge_attr.shape)
        x,edge_index = self.encoder1(x,edge_index,edge_attr)
      
        x,edge_index = self.encoder2(x,edge_index[0],edge_index[1])
        
        return x, edge_index[0],edge_index[1]

# Definition of Encoder
class Decoder(nn.Module):

    def __init__(self,in_channels,out_channels,heads,**kwargs):
        super().__init__()
        self.decoder1 = GATFormerLayerBlock(in_channels=4,out_channels=out_channels,heads=heads)
        self.decoder2 = GATFormerLayerBlock(in_channels=out_channels,out_channels=in_channels,heads=heads)

    def forward(self,x,edge_index,edge_attr):
        x,edge_index = self.decoder1(x,edge_index,edge_attr)
        x,edge_index = self.decoder2(x,edge_index[0],edge_index[1])
        adj_pred = InnerProductDecoder().forward_all(x, sigmoid= True)
        print(adj_pred.shape)
        return adj_pred, edge_index[0]

# Definition of GNF
class GNF(nn.Module):

    def __init__(self):
        super().__init__()
        self.F1 = GATConv(2,2)
        self.F2 = GATConv(2,2)
        self.G1 = GATConv(2,2)
        self.G2 = GATConv(2,2)

    def forward(self,x,edge_index):
        x1,x2 = x[:,:2], x[:,2:]
        x1,x2,s1 = self.f_a(x1,x2,edge_index)
        x2,x1,s2 = self.f_b(x2,x1,edge_index)
        # Calculate Jacobian
        log_det_xz = torch.sum(s1,axis=1) + torch.sum(s2,axis=1)
        return x1, x2, log_det_xz

    def f_a(self,x1,x2,edge_index):
        s_x = self.F1(x1,edge_index)
        t_x = self.F2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def f_b(self,x1,x2,edge_index):
        s_x = self.G1(x1,edge_index)
        t_x = self.G2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def inverse_b(self,z1,z2,edge_index):
        s_x = self.G1(z1,edge_index)
        t_x = self.G2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse_a(self,z1,z2,edge_index):
        s_x = self.F1(z1,edge_index)
        t_x = self.F2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse(self,z,edge_index):
        z1,z2 = z[:,:2], z[:,2:]
        z1,z2,s1 = self.inverse_b(z1,z2,edge_index)
        z2,z1,s2 = self.inverse_a(z2,z1,edge_index)
        # Calculate Jacobian
        log_det_zx = torch.sum(-s1,axis=1) + torch.sum(-s2,axis=1)
        return z1, z2, log_det_zx
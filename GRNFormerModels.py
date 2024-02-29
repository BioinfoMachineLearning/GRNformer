






from GRNModelblocks import GRNFormerLayerBlock,GATFormerLayerBlock,GATv2FormerLayerBlock,Embedder,FullyConnectedGT_UGformerV2
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv
from torch import nn
from torch.nn import Module


class EdgeTransformerEncoder_GAT(nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_layers):
        super().__init__()
        self.conv1 = GATFormerLayerBlock(in_channels=in_channels, out_channels=num_layers * out_channels,heads=num_head,edge_dim=edge_dim,beta=True,dropout=0.1)
        #self.transf1 = Linear(num_layers*out_channels*num_head, num_layers*out_channels)
        #self.b1 = nn.BatchNorm1d(num_layers*out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers,2,-1):
            print(i)
            #self.conv_layers.append(nn.Sequential(*[TransformerConv(i*out_channels, (i-1) * out_channels,heads=num_head,edge_dim=edge_dim,dropout=0.1,beta=True),
             #                   Linear((i-1)*out_channels*num_head, (i-1)*out_channels),
             ##                   nn.ReLU(inplace=True), 
              #                  nn.BatchNorm1d((i-1)*out_channels)]))
            self.conv_layers.append(GATFormerLayerBlock(in_channels=i*out_channels, out_channels=(i-1) * out_channels,heads=num_head,edge_dim=edge_dim,beta=True,dropout=0.1))
        
        self.conv_mu = GATConv(2 * out_channels, out_channels,heads=num_head)
        self.conv_logstd = GATConv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index,edge_attr):
        x = x.float()
        
        x,edge_index = self.conv1(x, edge_index,edge_attr)
        
     
        for encoder_layer in self.conv_layers:
            
            x,edge_index = encoder_layer(x,edge_index[0],edge_attr)
       
        
        return self.conv_mu(x, edge_index[0]), self.conv_logstd(x, edge_index[0])
    
class EdgeTransformerEncoder_new(nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_layers):
        super().__init__()
        #self.conv1 = GCNConv(in_channels, out_channels*2)
        self.bc1 = BatchNorm(out_channels*2)
        #self.conem = GCNConv(out_channels*2,out_channels)

        #self.conv1 = Embedder(node_dim=in_channels,edge_dim=edge_dim,out_edge_feats=out_channels,out_node_feats=out_channels)
        self.conv2 = GRNFormerLayerBlock(in_channels=out_channels, out_channels=num_layers * out_channels,heads=num_head,edge_dim=edge_dim,beta=True,dropout=0.5)
        #self.transf1 = Linear(num_layers*out_channels*num_head, num_layers*out_channels)
        #self.b1 = nn.BatchNorm1d(num_layers*out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers,2,-1):
            print(i)
            #self.conv_layers.append(nn.Sequential(*[TransformerConv(i*out_channels, (i-1) * out_channels,heads=num_head,edge_dim=edge_dim,dropout=0.1,beta=True),
             #                   Linear((i-1)*out_channels*num_head, (i-1)*out_channels),
             ##                   nn.ReLU(inplace=True), 
              #                  nn.BatchNorm1d((i-1)*out_channels)]))
            self.conv_layers.append(GRNFormerLayerBlock(in_channels=i*out_channels, out_channels=(i-1) * out_channels,heads=num_head,edge_dim=num_head,beta=True,dropout=0.5))
        
        self.conv_mu = TransformerConv(2 * out_channels, out_channels,heads=num_head)
        self.conv_logstd = TransformerConv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index,edge_attr):
        x = x.float()
        
        x = self.conv1(x,edge_index)
        x = self.bc1(x)
        x = self.conem(x,edge_index)
        x,edge_index = self.conv2(x,edge_index,edge_attr)
       
     
        for encoder_layer in self.conv_layers:
            
            x,edge_index = encoder_layer(x,edge_index[0],edge_index[1])
            
       
        
        return self.conv_mu(x, edge_index[0]), self.conv_logstd(x, edge_index[0])
    

class DeepVGAE(VGAE):
    def __init__(self,encoder:Module,decoder:Module):
        super().__init__(encoder, decoder)

    def forward(self, x, edge_index,edge_attr):
        z = self.encode(x, edge_index,edge_attr)
        adj_pred = self.decoder.forward_all(z,sigmoid=True)
        adj_pred = adj_pred.view(-1)
        return adj_pred



class EdgeTransformerEncoder_GATv2(nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_layers):
        super().__init__()
        self.conv2 = GATv2FormerLayerBlock(in_channels=in_channels, out_channels=num_layers * out_channels,heads=num_head,edge_dim=in_channels,beta=True,dropout=0.1)
        #self.transf1 = Linear(num_layers*out_channels*num_head, num_layers*out_channels)
        #self.b1 = nn.BatchNorm1d(num_layers*out_channels)
        #self.relu = nn.ReLU(inplace=True)
        #self.conv1 = Embedder(node_dim=in_channels,edge_dim=edge_dim,out_edge_feats=out_channels,out_node_feats=out_channels)
        self.emb = FullyConnectedGT_UGformerV2(feature_dim_size=in_channels,ff_hidden_size=out_channels,num_self_att_layers=3,num_GNN_layers=3,edge_dim=edge_dim,nhead=1,dropout=0.5)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers,2,-1):
            print(i)
            #self.conv_layers.append(nn.Sequential(*[TransformerConv(i*out_channels, (i-1) * out_channels,heads=num_head,edge_dim=edge_dim,dropout=0.1,beta=True),
             #                   Linear((i-1)*out_channels*num_head, (i-1)*out_channels),
             ##                   nn.ReLU(inplace=True), 
              #                  nn.BatchNorm1d((i-1)*out_channels)]))
            self.conv_layers.append(GATv2FormerLayerBlock(in_channels=i*out_channels, out_channels=(i-1) * out_channels,heads=num_head,edge_dim=num_head,beta=True,dropout=0.1))
        
        self.conv_mu = GATv2Conv(2 * out_channels, out_channels,heads=num_head)
        self.conv_logstd = GATv2Conv(2 * out_channels, out_channels,heads=num_head)

    def forward(self, x, edge_index,edge_attr):
        x = x.float()
        
        x,edge_attr = self.emb(x,edge_attr)
       
        x,edge_index = self.conv2(x,edge_index,edge_attr)
        
        for encoder_layer in self.conv_layers:
            
            x,edge_index = encoder_layer(x,edge_index[0],edge_index[1])
            
        
        return self.conv_mu(x, edge_index[0]), self.conv_logstd(x, edge_index[0])

#class discriminator(nn.Module):
#    def __init__(self, *args, **kwargs) -> None:
#        super().__init__(*args, **kwargs)(self,in_channels, out_channels,)
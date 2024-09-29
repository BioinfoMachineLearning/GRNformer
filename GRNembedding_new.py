
from GRNModelblocks import GRNFormerLayerBlock,GATFormerLayerBlock,GATv2FormerLayerBlock,Embedder,FullyConnectedGT_UGformerV2
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv
from torch import nn
from torch.nn import Module

class Embeddings(nn.Module):
    def __init__(self, exp_channel,bert_channel, out_channels):
        super().__init__()
        
        self.output_dim = out_channels
        self.projection =nn.Linear(exp_channel, self.output_dim)
        self.projection1=nn.Linear(bert_channel, self.output_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=2,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        self.linear = nn.Linear(out_channels, out_channels)
        self.linearb = nn.Linear(out_channels, out_channels)

    def forward(self,exp_input,bert_input):
        exp_input=exp_input.float()
        
        bert_input=bert_input.float()
        #print(bert_input.shape)
        # Transpose x to shape (sequence_length, batch_size, embedding_dim)
        _,n, m = exp_input.shape
        
        # Initialize projection layer based on input dimension
        #self.update_projection(m)
        #if self.projection is None or self.projection.in_features != m:
        #    self.projection = nn.Linear(m, self.output_dim).cuda()
        
        # Project input to output dimension
        x = self.projection(exp_input)
        
        x = x.permute(0, 1, 2)
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        #x = x.permute(0, 2, 1)
        # Apply linear layer to reduce sequence length to 128
        x = self.linear(x)
        # Transpose x back to shape (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 1, 2)
                # Transpose x to shape (sequence_length, batch_size, embedding_dim)
        _,n, m1 = bert_input.shape
        # Initialize projection layer based on input dimension
        #if self.projection is None or self.projection.in_features != m1:
        #    self.projection = nn.Linear(m1, self.output_dim)
        #self.update_projection(m1)
        x1 = self.projection1(bert_input).cuda()
        x1 = x1.permute(0, 1, 2)
        # Apply transformer encoder
        x1 = self.transformer_encoder(x1)
        #x1 = x1.permute(0, 2, 1)
        # Apply linear layer to reduce sequence length to 128
        x1 = self.linearb(x1)
        # Transpose x back to shape (batch_size, sequence_length, embedding_dim)
        x1 = x1.permute(0, 1, 2)

        x2=x1+x
        
        return x2

    def update_projection(self, m1):
        if self.projection is None or self.projection.in_features != m1:
            self.projection = nn.Linear(m1, self.output_dim) 
    def update_projection1(self, m1):
        if self.projection1 is None or self.projection1.in_features != m1:
            self.projection1 = nn.Linear(m1, self.output_dim)
        
  


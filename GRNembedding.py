
from GRNModelblocks import GRNFormerLayerBlock,GATFormerLayerBlock,GATv2FormerLayerBlock,Embedder,FullyConnectedGT_UGformerV2
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv
from torch import nn
from torch.nn import Module

class Embeddings(nn.Module):
    def __init__(self, exp_channel,bert_channel, out_channels):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=426, nhead=2,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        self.linear = nn.Linear(exp_channel, out_channels)
        self.linearb = nn.Linear(bert_channel, out_channels)

    def forward(self,exp_input,bert_input):
        exp_input=exp_input.float()
        
        bert_input=bert_input.float()
        #print(bert_input.shape)
        # Transpose x to shape (sequence_length, batch_size, embedding_dim)
        x = exp_input.permute(0, 1, 2)
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        #x = x.permute(0, 2, 1)
        # Apply linear layer to reduce sequence length to 128
        x = self.linear(x)
        # Transpose x back to shape (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 1, 2)
                # Transpose x to shape (sequence_length, batch_size, embedding_dim)
        x1 = bert_input.permute(0, 1, 2)
        # Apply transformer encoder
        x1 = self.transformer_encoder(x1)
        #x1 = x1.permute(0, 2, 1)
        # Apply linear layer to reduce sequence length to 128
        x1 = self.linearb(x1)
        # Transpose x back to shape (batch_size, sequence_length, embedding_dim)
        x1 = x1.permute(0, 1, 2)

        x2=x1+x
        
        return x2

        
  


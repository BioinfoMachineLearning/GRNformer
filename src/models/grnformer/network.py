

from GRNModelblocks import GRNFormerLayerBlock
from torch_geometric.nn import GAE,BatchNorm
from torch import nn
from GRNembedding import Embeddings
from torch.nn import Module
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling,to_dense_adj
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv

EPS = 1e-15
MAX_LOGSTD = 10

class GRNFormerLayerBlock(TransformerConv):
    def __init__(self, in_channels,out_channels,heads,**kwargs):
        super(GRNFormerLayerBlock, self).__init__(in_channels=in_channels,out_channels=out_channels)
        self.transformer_conv = TransformerConv(in_channels=in_channels,out_channels=out_channels,heads=heads,**kwargs)
        self.transf1 = Linear(out_channels*heads,out_channels)
        self.b1 = BatchNorm(out_channels)
        self.gelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        
        out, attention_weights = self.transformer_conv(x, edge_index, edge_attr,return_attention_weights=True)
        out = self.transf1(out)
        out = self.gelu(out)
        #out = self.b1(out)
        out = self.dropout(out)
        return out, attention_weights

class TransformerAutoencoder(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, num_layers=6):
        super(TransformerAutoencoder, self).__init__()

        # Encoder: Conv2d to project input feature dimension n to embed_dim, keeping the shape [1, m, n] unchanged
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1, padding=0)
        self.gelu = nn.LeakyReLU()
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,batch_first=True),
            num_layers=num_layers
        )

        # Decoder: Conv2d transpose to project embed_dim back to original feature dimension n
        self.deconv2d = nn.ConvTranspose1d(in_channels=embed_dim, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        # Input shape: [batch_size, m, n]
        batch_size, seq_len, feature_dim = x.shape
        
        # Reshape input to add a channel dimension for Conv2d: [batch_size, 1, m, n]
        x = x.permute(1,0,2).float()

        # 1. Encoder
        x = self.conv1d(x)  # Apply Conv2d, project feature dimension n to embed_dim, keep shape [batch_size, embed_dim, m, n]
        #x = self.gelu(x)
        #print(x.shape)
        #x = x.squeeze(1)  # Remove the channel dimension, now shape is [batch_size, embed_dim, m]
        
        # Permute to [m, batch_size, embed_dim] for Transformer
        x = x.permute(2, 0, 1)
        latent = self.transformer_encoder(x)  # Transformer encoder
        latent = latent.permute(1, 2, 0)  # Shape back to [batch_size, embed_dim, m]
        #print(latent.shape)
        latent1 =  torch.mean(latent, dim=-1, keepdim=True) 
        #print(latent1.shape)
        # 2. Decoder
        #decoded = latent  # Add channel dimension back
        #decoded = self.deconv2d(decoded)  # Apply Transposed Conv2d, project embed_dim back to original feature dimension n
        #decoded =  torch.mean(decoded, dim=1, keepdim=True)
        #decoded = decoded.permute(1,0,2)
           # Shape: [batch_size, m, n]
        #print(decoded.shape)
        return  latent1.permute(2, 0, 1)  # Return reconstructed output and latent space

class EdgeTransformerEncoder_tcn(nn.Module):
    def __init__(self, in_channels, out_channels,num_head,edge_dim,num_layers):
        super().__init__()
        #self.conv1 = GCNConv(in_channels, out_channels*2)

        #self.bc1 = BatchNorm(in_channels)
       #self.conem = GCNConv(out_channels*2,out_channels)

        #self.conv1 = Embedder(node_dim=in_channels,edge_dim=edge_dim,out_edge_feats=out_channels,out_node_feats=out_channels)
        self.conv2 = GRNFormerLayerBlock(in_channels=in_channels, out_channels=num_layers * out_channels,heads=num_head,edge_dim=edge_dim,beta=True,dropout=0.5)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers,2,-1):
            print(i)

            self.conv_layers.append(GRNFormerLayerBlock(in_channels=i*out_channels, out_channels=(i-1) * out_channels,heads=num_head,edge_dim=num_head,beta=True,dropout=0.5))
        
        self.conv_mu = GRNFormerLayerBlock(2 * out_channels, out_channels,heads=num_head,edge_dim=num_head,beta=True,dropout=0.5)
        self.conv_logstd = GRNFormerLayerBlock(2 * out_channels, out_channels,heads=num_head,edge_dim=num_head,beta=True,dropout=0.5)

    def forward(self, x, edge_index,edge_attr):
        x = x.float()
        
        #x = self.conv1(x,edge_index)
        #x = self.bc1(x)
        #x = self.conem(x,edge_index)
        
        x,edge_index1 = self.conv2(x,edge_index,edge_attr)
       
     
        for encoder_layer in self.conv_layers:
            
            x,edge_index1 = encoder_layer(x,edge_index1[0],edge_index1[1])
        x_mu, edge_index1_mu = self.conv_mu(x, edge_index1[0],edge_index1[1])
        x_std ,edge_index1_std= self.conv_logstd(x, edge_index1[0],edge_index1[1])
        #print(x.shape)
        return x_mu, x_std , edge_index1_mu,edge_index1_std

class TransformerDecoder_tcn(nn.Module):
    def __init__(self, latent_dim, out_channels, num_head, edge_dim, num_layers):
        super().__init__()
        self.conv2 = GRNFormerLayerBlock(in_channels=latent_dim, out_channels=out_channels, heads=num_head, edge_dim=num_head, beta=True, dropout=0.5)
        
        self.conv_layers = nn.ModuleList()
        for i in range(2, num_layers + 1):
            self.conv_layers.append(GRNFormerLayerBlock(in_channels=(i - 1) * out_channels, out_channels=i * out_channels, heads=num_head, edge_dim=num_head, beta=True, dropout=0.5))

        self.fc1 = nn.Linear(i*out_channels, out_channels)
        #self.fc2 = nn.Linear(out_channels, latent_dim)

        self.norm = BatchNorm(out_channels*i)
        self.dropout = nn.Dropout(0.1)

    def forward(self, z, edge_index, edge_attr):
        z = z.float()
        
        z, edge_index1 = self.conv2(z, edge_index, edge_attr)
        
        for decoder_layer in self.conv_layers:
            z, edge_index1 = decoder_layer(z, edge_index1[0], edge_index1[1])
        edge_ind = edge_index1[0]
        #edge_att1 = torch.mean(edge_index1[1], dim=1)
        #z = self.norm(z)
        z = self.dropout(z)

        z = self.fc1(z)
        z = torch.relu(z)
        #z = self.fc2(z)

        return z, edge_ind,edge_index1[1]

class EdgePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear( latent_dim, 1)

    
    
    def forward(self, z, edge_index,edge_attr):
        num_nodes = z.size(0)
        # Predict edge probabilities for specified edges
        
        row, col = edge_index
        #edge_features = torch.cat([z[row], z[col]], dim=1)
        edge_features = torch.mean(edge_attr,dim=1)
        #specified_edge_probs = (self.fc(edge_features))
        #specified_edge_probs = 
        # Initialize full adjacency matrix with zeros
        full_edge_probs = torch.zeros((num_nodes, num_nodes), device=z.device)

        # Fill in the specified edge probabilities
        full_edge_probs[row, col] = edge_features
        
        # Predict probabilities for all pairs of nodes
        row, col = torch.meshgrid(torch.arange(num_nodes, device=z.device), torch.arange(num_nodes, device=z.device), indexing='ij')
        #all_edge_features = torch.cat([z[row].view(-1, z.size(1)), z[col].view(-1, z.size(1))], dim=1)
        #all_edge_probs = (self.fc(all_edge_features)).view(num_nodes, num_nodes)
        all_edge_probs =  torch.matmul(z, z.t())
        # Combine specified edge probabilities with all edge probabilities
        #full_edge_probs = torch.where(full_edge_probs == 0, all_edge_probs, full_edge_probs)
        #print(full_edge_probs)
        #adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr,max_num_nodes=num_nodes)
        #print(adj)
        #adj = adj.sum(dim=3)
        #print(adj.shape)
        adj = all_edge_probs+full_edge_probs
        #print(adj)
        return torch.sigmoid(adj.squeeze(0))
    
class Reconstruct(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0],edge_index[1]])
        #return torch.sigmoid(value) if sigmoid else value
        return value

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""
        self.__mu__, self.__logstd__,edge_ind,edge_attr = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z,edge_ind,self.__mu__,self.__logstd__

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

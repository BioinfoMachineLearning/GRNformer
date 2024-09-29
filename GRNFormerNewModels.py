

from GRNModelblocks import GRNFormerLayerBlock,GATFormerLayerBlock,GATv2FormerLayerBlock,Embedder,FullyConnectedGT_UGformerV2
from torch_geometric.nn import GAE, VGAE, GCNConv,TransformerConv,Linear,BatchNorm,GATConv,GATv2Conv
from torch import nn
from GRNembedding import Embeddings
from torch.nn import Module
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling,to_dense_adj

EPS = 1e-15
MAX_LOGSTD = 10
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
        self.fc2 = nn.Linear(out_channels, latent_dim)

        self.norm = BatchNorm(out_channels*i)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z, edge_index, edge_attr):
        z = z.float()
        
        z, edge_index1 = self.conv2(z, edge_index, edge_attr)
        
        for decoder_layer in self.conv_layers:
            z, edge_index1 = decoder_layer(z, edge_index1[0], edge_index1[1])
        edge_ind = edge_index1[0]
        #edge_att1 = torch.mean(edge_index1[1], dim=1)
        z = self.norm(z)
        z = self.dropout(z)

        z = self.fc1(z)
        z = torch.relu(z)
        z = self.fc2(z)

        return z, edge_ind

class EdgePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(2 * latent_dim, 1)

    
    
    def forward(self, z, edge_index,edge_attr):
        num_nodes = z.size(0)
        # Predict edge probabilities for specified edges
        
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        specified_edge_probs = (self.fc(edge_features))
        
        # Initialize full adjacency matrix with zeros
        full_edge_probs = torch.zeros((num_nodes, num_nodes), device=z.device)

        # Fill in the specified edge probabilities
        full_edge_probs[row, col] = specified_edge_probs.squeeze(1)
        '''
        # Predict probabilities for all pairs of nodes
        row, col = torch.meshgrid(torch.arange(num_nodes, device=z.device), torch.arange(num_nodes, device=z.device), indexing='ij')
        all_edge_features = torch.cat([z[row].view(-1, z.size(1)), z[col].view(-1, z.size(1))], dim=1)
        all_edge_probs = (self.fc(all_edge_features)).view(num_nodes, num_nodes)

        # Combine specified edge probabilities with all edge probabilities
        full_edge_probs = torch.where(full_edge_probs == 0, all_edge_probs, full_edge_probs)
        '''
        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr,max_num_nodes=num_nodes)
        adj = adj.sum(dim=3)
        adj = adj+full_edge_probs
        
        return adj.squeeze(0)
    
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
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
        #return value
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
        self.__mu__, self.__logstd__,edge_ind,_ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z,edge_ind

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














class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

MAX_LOGSTD = 10
class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        edge_dim=1
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.linear = torch.nn.ModuleList()
        #
        # self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        self.down_convs.append(TransformerConv(in_channels, channels, heads=8,edge_dim=edge_dim,beta=True,dropout=0.5))
        self.linear.append(Linear(channels*8,channels))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(TransformerConv(channels, channels,  heads=8,edge_dim=1,beta=True,dropout=0.5))
            self.linear.append(Linear(channels*8,channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        self.linear_up = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(TransformerConv(channels, channels,  heads=8,edge_dim=1))
            self.linear_up.append(Linear(channels*8,channels))
        self.up_convs.append(TransformerConv(channels, out_channels,  heads=8,edge_dim=1))
        self.linear_up.append(Linear(channels*8,out_channels))

        self.lin_mu = TransformerConv(hidden_channels, channels,heads=8)
        self.transf1 = Linear(out_channels*8,out_channels)
        self.lin_logvar = TransformerConv(hidden_channels, channels,heads=8)
        self.transf2 = Linear(out_channels*8,out_channels)
        self.reset_parameters()
        

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(logstd.mul(0.5))
            eps = torch.randn_like(logstd)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
        
    
    def forward(self, x: Tensor, edge_index: Tensor,edge_attr:Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #edge_weight = x.new_ones(edge_index.size(1))
        edge_weight=edge_attr
        print("x start",x.shape)
        x,edge = self.down_convs[0](x, edge_index, edge_weight,return_attention_weights=True)
        x = self.act(x)
        x=self.linear[0](x)
        print("xdown",x.shape)
        edge_index=edge[0]
        edge_weight=torch.sum(edge[1], dim=1)
        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        
        perms = []
        print(edge_index.shape,edge_weight.shape)
        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)
            print("augx",x.shape, edge_index.shape,edge_weight.shape)
            x,edge = self.down_convs[i](x, edge_index, edge_weight.unsqueeze(1),return_attention_weights=True)
            x = self.act(x)
            x=self.linear[i](x)

            edge_index=edge[0]
            edge_weight=torch.sum(edge[1], dim=1)
            
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
        
        #self.__mu__ = self.lin_mu(xs,edge_index)
        ##self.__mu__ = self.transf1(self.__mu__)
        #self.__logstd__ = self.lin_logvar(xs,edge_index)
        ##self.__logstd__ = self.transf2(self.__logstd__)
        ##self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        #z = self.reparametrize(self.__mu__, self.__logstd__)

        #print("z:",z.shape)
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            print("res:",res.shape)
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]
            print("perm:",perm.shape)
            up = torch.zeros_like(res)
            print("up:",up.shape,"x:",x.shape,edge_index.shape)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            print("up_x: ",x.shape,edge_weight.shape)
            
            x,edge = self.up_convs[i](x, edge_index, edge_weight.unsqueeze(1),return_attention_weights=True)
            x = self.act(x) if i < self.depth - 1 else x
            #x=self.linear_up[i](x)
            edge_index=edge[0]
            edge_weight=torch.sum(edge[1], dim=1)
        

        print(x.shape)
        return x
        #mu = self.lin_mu(x)
        #logvar = self.lin_logvar(x)
        
        #return mu, logvar
        #return x


    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        #adj = (adj @ adj)#.to_sparse_coo()
        #adj_dense = adj.to_dense()
        print(adj.shape)
        # Square the adjacency matrix
        adj_sq = torch.sparse.mm(adj, adj)
        # Ensure that resulting edge_index and edge_weight are not empty
        if adj_sq._nnz() == 0:
            # If the resulting adjacency matrix is empty, add a small epsilon to the diagonal
            adj = adj + torch.eye(num_nodes)
            adj = torch.sparse.mm(adj, adj).to_sparse_coo()
        else:
            adj = torch.sparse.mm(adj, adj).to_sparse_coo()
            #adj = mm(adj,adj).to_sparse_coo()
        #adj = torch.matmul(adj.to_dense().squeeze(), adj.to_dense().squeeze()).to_sparse_coo()

        # Convert back to sparse tensor
        

        edge_index, edge_weight = adj.indices(), adj.values()
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
    
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

import simple_dense_net as sp


class GRNFormer(nn.Module):
    def __init__(self, node_dimesion, edge_dimension, n_trans_layers):
        super(GRNFormer, self).__init__()

        self.graphtrans = sp.GraphTransformer(
            dim = node_dimesion,
            depth = n_trans_layers,
            edge_dim = edge_dimension,             # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual = True,      # to use the gated residual to prevent over-smoothing
            rel_pos_emb = False       # set to True if the nodes are ordered, default to False
            #accept_adjacency_matrix = True
            )

    
    def forward(self, data, edges ):
        
        

        # Initializing hidden state for first input using method defined below
        
        x= data
        print(edges.dtype)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out, edges = self.graphtrans(x, edges)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        #out = self.fc(out[:, -1, :])
        return out, edges
    
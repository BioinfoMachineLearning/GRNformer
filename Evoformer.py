import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        nn.init.constant_(self.net[-1].weight, 0.)
        if exists(self.net[-1].bias):
            nn.init.constant_(self.net[-1].bias, 0.)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.to_out.weight, 0.)

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        gates = self.gating(x).sigmoid()
        out = out * gates
        return self.to_out(out)

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        global_query_attn = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return out

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix='ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = torch.einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

class OuterProductMean(nn.Module):
    def __init__(self, node_dim, edge_dim, eps=1e-5):
        super(OuterProductMean, self).__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(node_dim)
        hidden_dim = edge_dim if edge_dim is not None else node_dim

        self.left_proj = nn.Linear(node_dim, hidden_dim)
        self.right_proj = nn.Linear(node_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, edge_dim)

    def forward(self, x):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = torch.einsum('b i d, b j d -> b i j d', left, right)

        outer = outer.mean(dim=1)
        return self.proj_out(outer)

class PointwiseOperations(nn.Module):
    def __init__(self, dim):
        super(PointwiseOperations, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class PairwiseAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.,global_column_attn = False):
        super().__init__()
        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, hidden_dim=None, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, hidden_dim=None, mix='ingoing')
        self.feed_forward = FeedForward(dim, dropout=dropout)

    def forward(self, x, pairwise_repr=None):
        if exists(pairwise_repr):
            x = x + pairwise_repr

        x = self.triangle_multiply_outgoing(x) + x
        x = self.triangle_multiply_ingoing(x) + x
        x = self.triangle_attention_outgoing(x,edges = x) + x
        x = self.triangle_attention_ingoing(x,edges = x) + x
        x = self.feed_forward(x) + x
        return x

class MsaAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)
        self.feed_forward = FeedForward(dim, dropout=dropout)

    def forward(self, x, pairwise_repr=None):
        x = self.row_attn(x, edges = pairwise_repr) + x
        x = self.col_attn(x) + x
        x = self.feed_forward(x) + x
        return x

class EvoformerBlock(nn.Module):
    def __init__(self, dim,edge_dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.pairwise_attention = PairwiseAttentionBlock(edge_dim, heads, dim_head, dropout)
        self.msa_attention = MsaAttentionBlock(dim, heads, dim_head, dropout)
        self.outer_product_mean = OuterProductMean(dim, dim)

    def forward(self, node_features, edge_features):
        node_features = self.msa_attention(node_features, pairwise_repr=edge_features)
        print(node_features.shape)
        edge_features = self.pairwise_attention(edge_features, pairwise_repr=None)
        outer_product_mean = self.outer_product_mean(node_features)
        edge_probabilities = torch.sigmoid(edge_features + outer_product_mean)
        return edge_probabilities

class Evoformer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, num_layers, dim_head=64, dropout=0.):
        super(Evoformer, self).__init__()
        self.layers = nn.ModuleList([
            EvoformerBlock(node_dim,edge_dim=edge_dim, heads=num_heads, dim_head=dim_head, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, node_features, edge_features):
        for layer in self.layers:
            edge_probabilities = layer(node_features, edge_features)
            edge_features = edge_probabilities  # Use the output probabilities as input to the next layer

        return edge_probabilities

# Example usage:
node_dim = 128
edge_dim = 64
num_heads = 8
num_layers = 6
model = Evoformer(node_dim, edge_dim, num_heads, num_layers)

# Dummy input tensors for node features and edge features
node_features = torch.rand((1, 100, node_dim))  # (batch_size, num_nodes, node_dim)
edge_features = torch.rand((1, 100, 100, edge_dim))  # (batch_size, num_nodes, num_nodes, edge_dim)

edge_probabilities = model(node_features, edge_features)
print(edge_probabilities.shape)  # Output shape: (batch_size, num_nodes, num_nodes, 1)

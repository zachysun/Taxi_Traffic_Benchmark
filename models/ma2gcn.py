import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_scaled_laplacian_torch, get_Tk


# ------------------------ Graph convolution ------------------------

class GraphConv(nn.Module):
    """
     - GraphConv:
        - Params: in_channels*, out_channels*, device, K
        - Input: x*(b, t_in, n, c_in), Tks*
        - Output: x(b, t_in, n, c_out)
    """

    def __init__(self, K, in_channels, out_channels, device):
        super(GraphConv, self).__init__()
        self.K = K
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)).to(device) for _ in range(K)]
        )

    def forward(self, x, Tks):
        batch_size, seq_length, num_of_vertices, in_channels = x.shape
        output = torch.zeros(batch_size, seq_length, num_of_vertices, self.out_channels).to(self.device)

        for k in range(self.K):
            T_k = Tks[k]
            theta_k = self.Theta[k]

            temp = torch.einsum('vv,btvc->btvc', (T_k, x))
            output = output + torch.einsum('btvc,co->btvo', (temp, theta_k))

        return torch.Tensor(output)


# ------------------------ Adjacency attention mechanism------------------------

class AdjAttention(nn.Module):
    """
     - AdjAttention:
        - Params: adj_input_dim, adj_hidden_dim(For weights)
        - Input: adj_list*:[origin_adj(n, n), origin_adj_2(n, n), dynamic_adj(n, n), cur_io_adj(n, n)]
        - Output: adj_aggregated(n, n)
    """

    def __init__(self, adj_input_dim, adj_hidden_dim):
        super(AdjAttention, self).__init__()

        self.W = nn.Linear(adj_input_dim, adj_hidden_dim)
        self.V = nn.Linear(adj_hidden_dim, 1)

    def forward(self, adj_list):
        # adj_list: a list of adjacency matrices with shape (num_nodes, num_nodes)
        num_nodes, _ = adj_list[0].shape

        # Compute weights for each adjacency matrix
        weights = []
        for adj in adj_list:
            x = F.relu(self.W(adj))
            x = self.V(x)
            x = x.view(num_nodes, 1)
            alpha = x.mean()
            weights.append(alpha)

        weights = F.softmax(torch.Tensor(weights), dim=0)
        # Compute weighted sum of adjacency matrices
        adj_aggregated = torch.zeros_like(adj_list[0])
        for i in range(len(weights)):
            adj_aggregated += adj_list[i] * weights[i]

        return adj_aggregated


# ------------------------ Gated TCN module ------------------------

class GatedTCN(nn.Module):
    """
     - GatedTCN:
        - Params: in_tcn_dim, out_tcn_dim, cur_dilation, kernel_size
        - Input: x*(b, c, t_in, n)
        - Output: x(b, c, t_out, n)
    """

    def __init__(self, in_tcn_dim, out_tcn_dim, cur_dilation, kernel_size):
        super(GatedTCN, self).__init__()
        self.filter_convs = nn.Conv2d(in_channels=in_tcn_dim,
                                      out_channels=out_tcn_dim,
                                      kernel_size=kernel_size, dilation=cur_dilation)

        self.gated_convs = nn.Conv2d(in_channels=in_tcn_dim,
                                     out_channels=out_tcn_dim,
                                     kernel_size=kernel_size, dilation=cur_dilation)

    def forward(self, x):
        x = torch.einsum('btvf->bfvt', x)
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gated_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = torch.einsum('bfvt->btvf', x)
        return x


# ------------------------ Dynamic adjacency generator ------------------------

class DynamicAdjGen(nn.Module):
    """
     - DynamicAdjGen:
        - Params: nodes_num, all_feature_dim*, device
        - Input: x*(b, t, n, c), origin_adj(n, n), cur_io_adj*(n, n)
        - Output: adj_list:[origin_adj(n, n), origin_adj_2(n, n), dynamic_adj(n, n), cur_io_adj(n, n)]
    """

    def __init__(self, nodes_num, all_feature_dim, device):
        super(DynamicAdjGen, self).__init__()
        self.all_feature_dim = all_feature_dim
        self.nodes_num = nodes_num
        self.node_vec1 = nn.Parameter(torch.FloatTensor(self.all_feature_dim, self.nodes_num // 10)).to(device)
        self.node_vec2 = nn.Parameter(torch.FloatTensor(self.nodes_num, self.nodes_num)).to(device)
        nn.init.uniform_(self.node_vec1)
        nn.init.uniform_(self.node_vec2)

    def forward(self, x, origin_adj, cur_io_adj):
        b, t, n, c = x.shape
        x = x.reshape(b, n, t * c)
        x_emb = x @ self.node_vec1
        x_emb_T = torch.einsum('bnf->bfn', x_emb)
        x_selfdot = torch.mean(torch.einsum('bnf,bfm->bnm', (x_emb, x_emb_T)), dim=0)
        x_selfdot = x_selfdot.reshape(n, n)
        dynamic_adj = x_selfdot @ self.node_vec2
        dynamic_adj = F.softmax(dynamic_adj, dim=1)
        origin_adj = F.softmax(origin_adj, dim=1)
        origin_adj_2 = origin_adj @ origin_adj.T
        cur_io_adj = F.softmax(cur_io_adj, dim=1)

        return [origin_adj, origin_adj_2, dynamic_adj, cur_io_adj]
        # return [origin_adj, origin_adj_2, cur_io_adj] # without dynamic adjacency matrix


# ------------------------ Multi adjacency attention convolution Block ------------------------


class MA2ConvBlock(nn.Module):
    """
     - MA2ConvBlock:
        - Params: adj_input_dim/adj_hidden_dim/origin_adj(n, n)(for AdjAttention layer);
                  device(for GraphConv layer and DynamicAdjGen layer);
                  K/in_channel/out_channel(for GraphConv layer);
                  in_tcn_dim/out_tcn_dim/cur_dilation/kernel_size(for GatedTCN layer);
                  nodes_num/all_feature_dim(for DynamicAdjGen layer);
        - Input: x*(b, t_in, n, c_in), cur_io_adj*(for DynamicAdjGen layer);
        - Output: x(b, t_out, n, c_out)
    """

    def __init__(self, adj_input_dim, adj_hidden_dim, K, in_channel, out_channel,
                 in_tcn_dim, out_tcn_dim, origin_adj, nodes_num, all_feature_dim,
                 cur_dilation, kernel_size, droprate, device):
        super(MA2ConvBlock, self).__init__()

        self.adj_gen = DynamicAdjGen(nodes_num, all_feature_dim, device)
        self.adj_att = AdjAttention(adj_input_dim, adj_hidden_dim)
        self.gated_tcn = GatedTCN(in_tcn_dim, out_tcn_dim, cur_dilation, kernel_size)
        self.cheb_gcn = GraphConv(K, in_channel, out_channel, device)
        self.residual = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                  kernel_size=(1, 1))
        self.origin_adj = origin_adj
        self.K = K
        self.device = device
        self.dropout = nn.Dropout(droprate)

    def forward(self, x_with_io):
        x = x_with_io[0]
        residual = x
        cur_io_adj = x_with_io[1]
        x_t = self.gated_tcn(x)
        adj_list = self.adj_gen(x, self.origin_adj, cur_io_adj)
        adj_agg = self.adj_att(adj_list)

        # # [start] without attention module
        # adj_mean = torch.zeros_like(adj_list[0])
        # ratio = 1/len(adj_list)
        # for adj in adj_list:
        #     adj_mean = adj_mean + ratio * adj
        # L_tilde = calculate_scaled_laplacian_torch(adj_mean)
        # # [end] without attention module

        L_tilde = calculate_scaled_laplacian_torch(adj_agg)

        # # [start]without attention module and dynamic adjacency matrix
        # L_tilde = calculate_scaled_laplacian_torch(self.origin_adj)
        # # [end]without attention module and dynamic adjacency matrix

        Tks = torch.Tensor(np.array(get_Tk(L_tilde, self.K))).to(self.device)
        x_t_g = self.cheb_gcn(x_t, Tks)
        x_t_g = self.dropout(x_t_g)

        residual = residual.permute(0, 3, 1, 2)
        x_residual = self.residual(residual)
        x_residual = x_residual.permute(0, 2, 3, 1)
        x_t_g = x_t_g + x_residual[:, -x_t_g.size(1):, :, :]

        return (x_t_g, cur_io_adj)


# ------------------------ Main architecture[Multi adjacency attention GCN] ------------------------

class MA2GCN(nn.Module):
    """
     - MA2GCN:
        - Params: blocks, adj_input_dim/adj_hidden_dim/origin_adj(n, n)(for AdjAttention layer);
                  device(for GraphConv layer and DynamicAdjGen layer);
                  K/in_channel_list/out_channel_list(for GraphConv layer);
                  in_tcn_dim_list/out_tcn_dim_list/cur_dilation/kernel_size(for GatedTCN layer);
                  nodes_num/all_feature_dim(for DynamicAdjGen layer);
        - Input: x*(b, t_in, n, c), cur_io_adj*(for DynamicAdjGen layer);
        - Output: y(b, t_out, n, c)
    """

    def __init__(self, blocks_num, K, adj_input_dim, adj_hidden_dim, origin_adj,
                 in_channel_list, out_channel_list, in_tcn_dim_list,
                 out_tcn_dim_list, cur_dilation_list, kernel_size,
                 nodes_num, all_feature_dim, receptive_field, droprate, device):
        super(MA2GCN, self).__init__()
        modules = []
        for i in range(blocks_num):
            modules.append(MA2ConvBlock(adj_input_dim, adj_hidden_dim, K, in_channel_list[i], out_channel_list[i],
                                        in_tcn_dim_list[i], out_tcn_dim_list[i], origin_adj, nodes_num,
                                        all_feature_dim[i], cur_dilation_list[i], kernel_size, droprate, device))
        self.blocks = nn.Sequential(*modules)
        self.receptive_field = receptive_field

    def forward(self, x_with_io):
        x = x_with_io[0]
        input_seq = x.shape[1]
        # b,t,n,c -> b,c,n,t
        x = x.permute(0, 3, 2, 1)
        if input_seq < self.receptive_field:
            x_pad = F.pad(x, (self.receptive_field - input_seq, 0, 0, 0))
        else:
            x_pad = input
        x_pad = x_pad.permute(0, 3, 2, 1)
        x_with_io = (x_pad, x_with_io[1])
        return self.blocks(x_with_io)

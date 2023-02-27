import torch_geometric as ptg
from .utils import *


def create_sequential_gnn(input_size=8, output_size=8, hidden_size=32, n_heads=3, dropout=0.1,
                          layers=1, activation='relu', alpha=0.2, gnn_layer="graphconv", edge_dim=None):
    act_params = {"inplace": True}
    if activation == "lrelu":
        act_params["negative_slope"] = alpha
    act = activations[activation](**act_params)

    module_lst = []

    if gnn_layer in ("gat", "transformer", "natt"):
        # Attention layers with higher output dim
        hidden_input_size = n_heads * hidden_size
    else:
        hidden_input_size = hidden_size

    if layers == 1:
        module_lst.append((create_gnn_layer(gnn_layer, input_size, output_size,
                                            bias=False, att_heads=n_heads, att_concat=False, edge_dim=edge_dim),
                           'x, edge_index, edge_attr -> x'))
    else:
        module_lst.append((create_gnn_layer(gnn_layer, hidden_input_size, output_size,
                                            bias=False, att_heads=n_heads, att_concat=False, edge_dim=edge_dim),
                           'x, edge_index, edge_attr -> x'))
        ith_layer = layers - 1
        while ith_layer > 1:
            module_lst.insert(0, act)
            module_lst.insert(0, (create_gnn_layer(gnn_layer, hidden_input_size,
                                                   hidden_size, bias=True, att_heads=n_heads, edge_dim=edge_dim),
                                  'x, edge_index, edge_attr -> x'))
            ith_layer -= 1
        else:
            module_lst.insert(0, act)
            module_lst.insert(0, (create_gnn_layer(gnn_layer, input_size,
                                                   hidden_size, bias=True, att_heads=n_heads, edge_dim=edge_dim),
                                  'x, edge_index, edge_attr -> x'))
    return ptg.nn.Sequential('x, edge_index, edge_attr', module_lst)


def create_gnn_layer(layer_type, in_channels, out_channels, bias, att_heads=1,
                     att_concat=True, edge_dim=None):
    # Create a GNN-layer based on given description
    if layer_type == "gcn":
        gc_layer = ptg.nn.GCNConv(in_channels=in_channels, out_channels=out_channels, bias=bias)
    elif layer_type == "graphconv":
        gc_layer = ptg.nn.GraphConv(in_channels=in_channels, out_channels=out_channels,
                                    bias=bias, aggr="mean")
    elif layer_type == "natt":
        gc_layer = NeighborAttConv(in_channels=in_channels, out_channels=out_channels,
                                   bias=bias, heads=att_heads, concat=att_concat, edge_dim=edge_dim)

    elif layer_type == "gat":
        gc_layer = ptg.nn.GATv2Conv(in_channels=in_channels, out_channels=out_channels,
                                    bias=bias, heads=att_heads, concat=att_concat, add_self_loops=False,
                                    edge_dim=edge_dim)

    elif layer_type == "transformer":
        gc_layer = ptg.nn.TransformerConv(in_channels=in_channels,
                                          out_channels=out_channels, bias=bias, heads=att_heads,
                                          concat=att_concat, edge_dim=edge_dim)
    else:
        assert False, f"Unknown graph layer: {layer_type}"

    if edge_dim is not None:
        return GaussianWeightWrapper(gc_layer)
    else:
        return gc_layer


# Linear mapping of node itself, attention over neighbors (and center node)
class NeighborAttConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias, heads, concat, edge_dim):
        super().__init__()

        if concat:
            lin_out_dim = out_channels * heads
        else:
            lin_out_dim = out_channels

        self.linear = nn.Linear(in_channels, lin_out_dim, bias=bias)
        self.gat = ptg.nn.GATv2Conv(in_channels, out_channels,
                                    add_self_loops=False, bias=False, heads=heads, concat=concat,
                                    edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        return self.linear(x) + self.gat(x, edge_index, edge_attr)


class IgnoreWeightsWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class GaussianWeightWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.log_edge_bw = nn.Parameter(torch.log(20 * torch.ones(1)))

    def forward(self, x, edge_index, edge_attr):
        bw = torch.exp(self.log_edge_bw)
        edge_weight = torch.exp(-(edge_attr / bw) ** 2)
        return self.layer(x, edge_index, edge_weight)

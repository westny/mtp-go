import math
import torch.nn.functional as F
from .gnn_layers import *
from .motion_models import *


class GRUGNNCell(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3,
                 n_layers=1, dropout=0.1, gnn_layer="graphconv", edge_dim=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size  # Need 3 hidden states for all GRU gates

        self.Wx = create_sequential_gnn(input_size=input_size,
                                        output_size=gate_size,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='elu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim)

        self.Wh = create_sequential_gnn(input_size=hidden_size,
                                        output_size=gate_size,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='elu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim)

        self.bias = nn.Parameter(torch.empty(gate_size).uniform_(-1e-2, 1e-2))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (math.sqrt(self.hidden_size))

        # Exclude edge bws from initialization
        init_params = (p for name, p in self.named_parameters() if
                       not str.endswith(name, "log_edge_bw"))
        for p in init_params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)

    def forward(self, x, edge_index, h=None, edge_attr=None):
        #  Implements GRUCell update:
        #  https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        #  Using GNNs as learnable functions
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        wrx, wzx, wqx = torch.split(self.Wx(x, edge_index, edge_attr),
                                    self.hidden_size, dim=1)
        wrh, wzh, wqh = torch.split(self.Wh(h, edge_index, edge_attr),
                                    self.hidden_size, dim=1)
        br, bz, bq = torch.split(self.bias, self.hidden_size, dim=0)

        r = torch.sigmoid(wrx + wrh + br)
        z = torch.sigmoid(wzx + wzh + bz)
        q = torch.tanh(wqx + r * wqh + bq)
        h = (1 - z) * q + z * h
        return h


class GRUGNNEncoder(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3, n_layers=1,
                 n_mixtures=7, static_f_dim=6, dropout=0.1, gnn_layer="graphconv",
                 init_static=False, use_edge_features=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_static = init_static
        self.dropout = nn.Dropout(p=dropout)
        self.static_feature_dim = static_f_dim

        if self.init_static:
            # GNN model to learn init states
            self.init_gnn = create_sequential_gnn(input_size=self.static_feature_dim,
                                                  output_size=hidden_size,
                                                  hidden_size=hidden_size,
                                                  n_heads=n_heads,
                                                  dropout=dropout,
                                                  layers=n_layers,
                                                  activation='elu',
                                                  gnn_layer=gnn_layer)
        else:
            init_std = 1.0 / (math.sqrt(hidden_size))
            self.init_state_param = nn.Parameter(torch.empty(hidden_size).uniform_(
                -1e-2, 1e-2))

        edge_dim = 1 if use_edge_features else None
        self.gru_cell = GRUGNNCell(input_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer, edge_dim=edge_dim)

        self.mixture = nn.Linear(hidden_size, n_mixtures)  # <-- mixture weights

    def init_hidden(self, data, batch_size):
        if self.init_static:
            init_gnn_input = extract_static_features(data)
            # No edge features
            return self.init_gnn(init_gnn_input, data.edge_index[0], None)
        else:
            return self.init_state_param.repeat(batch_size, 1)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        batch_size, _, _ = x.size()
        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden(data, batch_size)
        output = [hidden]

        for x_i, ei_i, ef_i in zip(x.transpose(0, 1), edge_index, edge_features):
            hidden = self.gru_cell(x_i, ei_i, hidden, edge_attr=ef_i.to(torch.float32))
            output.append(hidden)
        output = torch.stack(output, dim=1)
        mixture_w = self.mixture(self.dropout(F.elu(hidden)))
        return output, mixture_w


class GRUGNNDecoder(nn.Module):
    def __init__(self, motion_model, max_length=10, hidden_size=64,
                 n_heads=3, n_layers=1, static_f_dim=6, alpha=0.2, dropout=0.1,
                 gnn_layer="graphconv", init_static=False):
        super().__init__()
        self.gru_cell = GRUGNNCell(hidden_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer)
        self.alpha = alpha
        self.init_static = init_static
        self.static_feature_dim = static_f_dim

        if init_static:
            self.init_combiner = nn.Linear(hidden_size, hidden_size, bias=True)
            self.init_gnn = create_sequential_gnn(input_size=self.static_feature_dim,
                                                  output_size=hidden_size,
                                                  hidden_size=hidden_size,
                                                  n_heads=n_heads,
                                                  dropout=dropout,
                                                  layers=1,
                                                  activation='elu',
                                                  gnn_layer=gnn_layer)

        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = motion_model.n_inputs
        self.mixtures = motion_model.mixtures
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout)

        # Temporal attention weight calculations
        self.embedding = nn.Linear(self.input_size * self.mixtures, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Scale GRU outputs
        self.generator = nn.Linear(hidden_size, hidden_size * 4)

        # Motion model inputs
        self.controller = nn.Linear(hidden_size, self.output_size * self.mixtures)

        # To generate process noise
        self.sig_1 = nn.Linear(hidden_size, self.mixtures)
        self.sig_2 = nn.Linear(hidden_size, self.mixtures)
        self.rho = nn.Linear(hidden_size, self.mixtures)

    def process_noise_matrix(self, x1, x2, x3, batch_size):
        sig1 = F.softplus(self.sig_1(x1))
        sig2 = F.softplus(self.sig_1(x2))
        rho = F.softsign(self.rho(x3))

        q_t = torch.zeros(batch_size, self.mixtures, self.output_size,
                          self.output_size, device=x1.device)

        q_t[..., 0, 0] = torch.pow(sig1, 2)
        q_t[..., 1, 1] = torch.pow(sig2, 2)
        q_t[..., 0, 1] = q_t[..., 1, 0] = sig1 * sig2 * rho
        return q_t

    def forward(self, data):
        x, hidden, encoder_out, edge_index, past_state, static_features = data
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_out)

        output = torch.cat((embedded, attn_applied[:, 0]), 1)
        output = self.attn_combine(output)
        output = F.leaky_relu(output, self.alpha)
        hidden = self.gru_cell(output, edge_index, hidden)

        output = F.elu(self.generator(F.elu(hidden)))
        output = self.dropout(output)

        x1, x2, x3, x4 = torch.split(output, self.hidden_size, dim=-1)
        process_noise = self.process_noise_matrix(x1, x2, x3, batch_size)
        model_input = self.controller(x4).view(batch_size,
                                               self.mixtures,
                                               self.output_size)

        next_state, model_input = self.motion_model(past_state, model_input,
                                                    static_features)

        return next_state, model_input, process_noise, hidden

    def get_initial_state(self, last_enc_state, data):
        if self.init_static:
            static_features = extract_static_features(data)  # (B, 6)

            edge_index = data.edge_index[-1]  # Use last graph from encoder
            graph_combined = self.init_gnn(static_features, edge_index, None)  # No edge features
            combined_repr = nn.functional.elu(graph_combined)
            return last_enc_state + self.init_combiner(combined_repr)
        else:
            return last_enc_state

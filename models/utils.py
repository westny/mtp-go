import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x


def extract_static_features(data):
    # Use only one-hot encoded vehicle type as static feature
    return data.v_type  # (B, n_vehicle_types)


activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
}

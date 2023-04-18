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


def extract_static_features(data, motion_model=None):
    if motion_model == 'singletrack':
        wheelbase = get_wheelbase(data.dim)
        return wheelbase
    else:
        # Use only one-hot encoded vehicle type as static feature
        return data.v_type  # (B, n_vehicle_types)


def get_wheelbase(vehicle_dim):
    """roughly 60% of vehicle length is wheelbase
       38 % of the wheelbase is from CoG to front axle
       leaving 62% of the wheelbase from CoG to rear axle
    """
    L = vehicle_dim[:, 0] * 0.6
    lf = L * 0.38
    lr = L - lf
    return torch.stack((lf, lr), dim=-1)


activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
}

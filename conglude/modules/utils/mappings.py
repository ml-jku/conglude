import torch.nn as nn
from torch_geometric.nn import (
    MeanAggregation,
    SumAggregation,
)


# Maps a string identifier to a PyG aggregation operator
aggregation_mapping = {
    "sum": SumAggregation(),
    "mean": MeanAggregation(),
}

# Maps a string identifier to an activation module instance
activation_mapping = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
}

# Maps activation name to the appropriate dropout type
dropout_mapping = {
    "relu": nn.Dropout,
    "leaky_relu": nn.Dropout,
    "selu": nn.AlphaDropout,
    "sigmoid": nn.Dropout,
    "silu": nn.Dropout,
    "gelu": nn.Dropout,
}
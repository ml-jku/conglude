import torch.nn as nn
import torch
import pytorch_lightning as pl
from conglude.modules.utils.mappings import activation_mapping, dropout_mapping


class MLPEncoder(pl.LightningModule):
    """
    Multi-layer perceptron (MLP) encoder with configurable depth, activation function, dropout, and optional batch normalization.

    Parameters
    ----------
    act: str
        Activation function identifier. Must exist in `activation_mapping` and `dropout_mapping`. Examples: "relu", "gelu", "selu".
    input_dim: int
        Dimensionality of the input feature vector.
    hidden_dim: int
        Dimensionality of the hidden layers.
    output_dim: int
        Dimensionality of the output embedding.
    num_layers : int
        Total number of linear layers (including the output layer).
    input_dropout : float
        Dropout rate applied before the first linear layer.
    dropout : float
        Dropout rate applied after each hidden layer.
    batch_norm: bool
        If True, applies BatchNorm1d after each hidden linear layer.
    """

    def __init__(
        self, 
        act: str = "gelu", 
        input_dim: int = 2258, 
        hidden_dim: int = 1024, 
        output_dim: int = 256, 
        num_layers: int = 3, 
        input_dropout: float = 0.1, 
        dropout: float = 0.5, 
        batch_norm: bool = False
    ):

        super().__init__()
        self.save_hyperparameters()
        
        # Build layers of dropout, linear, batch norm (optional), and activation
        layers = []
        layers.append(dropout_mapping[act](input_dropout))
        prev_dim = input_dim
        for _ in range(num_layers-1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_mapping[act])
            layers.append(dropout_mapping[act](dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Wrap layers into a single sequential module
        self.encoder = nn.Sequential(*layers)

        # Special initialization for SELU
        if act == "selu":
            self.apply(self.init_)
    

    def init_(
        self, 
        module: nn.Module
    ):
        """
        Custom weight initialization for SELU activation.
        """
        if type(module) in {nn.Linear}:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
            nn.init.zeros_(module.bias)


    def forward(
        self, 
        x
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Encoded embedding of shape (batch_size, output_dim)
        """

        return self.encoder(x)
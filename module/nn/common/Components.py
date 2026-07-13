# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import torch.nn as nn
import torch.nn.init as init
from .act import get_activation


class FFN(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation='relu', dropout=0.0, norm_mode='post', use_residual=True):
        """
        Feed-Forward Networks module
        Args:
            d_model (int): Input and output feature dimension
            dim_feedforward (int): Hidden dimension of the feedforward network
            activation (str): Activation function type, default is 'relu'
            dropout (float): Dropout probability, default is 0.0
            norm_mode (str): One of {'post', 'pre', 'none'}.
                - post: LN(x + FF(x)) (default, backward-compatible)
                - pre:  x + FF(LN(x))
                - none: x + FF(x) (or FF(x) if use_residual=False)
            use_residual (bool): Whether to add residual connection.
        """
        super(FFN, self).__init__()
        if norm_mode not in ('post', 'pre', 'none'):
            raise ValueError(f"norm_mode must be one of ('post', 'pre', 'none'), got {norm_mode}")
        self.use_residual = use_residual
        self.activation = get_activation(activation)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_mode = norm_mode
        self.norm = nn.LayerNorm(d_model) if norm_mode in ('post', 'pre') else nn.Identity()

        self._reset_parameters()

    def forward(self, tgt, residual=None):
        if residual is None:
            residual = tgt

        if self.norm_mode == 'pre':
            x = self.norm(tgt)
        else:
            x = tgt

        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        out = self.dropout2(tgt2)
        if self.use_residual:
            out = residual + out

        if self.norm_mode == 'post':
            out = self.norm(out)
        return out

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        """
        Multi-Layer Perceptron module
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output feature dimension
            num_layers (int): Number of layers in the MLP
            act (str): Activation function type, default is 'relu'
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

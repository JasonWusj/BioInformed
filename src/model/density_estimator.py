import torch
import torch.nn as nn
import torch.nn.functional as F


class SirenLayer(nn.Module):
    """SIREN layer with sine activation (Sitzmann et al., NeurIPS 2020)."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -torch.sqrt(torch.tensor(6.0 / in_features)) / omega_0,
                    torch.sqrt(torch.tensor(6.0 / in_features)) / omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class DensityEstimator(nn.Module):
    """
    Tumour cell density estimator using SIREN MLP.

    Takes intermediate feature maps from the segmentation backbone,
    concatenates with temporal information, and outputs estimated
    tumour cell density u_hat for PDE loss computation.
    """

    def __init__(self, in_channels, hidden_dim=256, num_layers=3, feature_size=(16, 16)):
        super().__init__()
        self.feature_size = feature_size
        self.in_channels = in_channels

        # Adaptive pool to fixed spatial size
        self.adapt_pool = nn.AdaptiveAvgPool2d(feature_size)

        # Input: flattened features + temporal dimension
        # Feature dim per spatial location: in_channels + 1 (time)
        input_dim = in_channels + 1

        layers = []
        layers.append(SirenLayer(input_dim, hidden_dim, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))

        self.siren_net = nn.Sequential(*layers)
        # Final linear layer (no sine activation)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features, t=1.0):
        """
        Args:
            features: (B, C, H, W) intermediate feature maps from backbone
            t: temporal step (scalar or tensor)
        Returns:
            u_hat: (B, 1, H_feat, W_feat) estimated tumour cell density
        """
        B = features.shape[0]
        H, W = self.feature_size

        # Pool to fixed spatial size
        x = self.adapt_pool(features)  # (B, C, H, W)

        # Reshape: (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Create temporal input
        t_tensor = torch.full((B, H * W, 1), t, device=x.device, dtype=x.dtype)

        # Concatenate features with time
        y = torch.cat([x, t_tensor], dim=-1)  # (B, H*W, C+1)

        # Pass through SIREN
        y = self.siren_net(y)  # (B, H*W, hidden_dim)
        u_hat = self.output_layer(y)  # (B, H*W, 1)

        # Sigmoid to constrain density to [0, 1]
        u_hat = torch.sigmoid(u_hat)

        # Reshape back to spatial
        u_hat = u_hat.permute(0, 2, 1).reshape(B, 1, H, W)

        return u_hat

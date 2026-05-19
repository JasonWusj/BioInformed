import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], num_classes, 1)

        # Store the bottleneck feature size for density estimator access
        self.bottleneck_channels = features[-1] * 2

    def forward(self, x, return_features=False):
        skip_connections = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        bottleneck_features = x

        skip_connections = skip_connections[::-1]

        for upconv, dec, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = upconv(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        logits = self.final_conv(x)

        if return_features:
            return logits, bottleneck_features
        return logits

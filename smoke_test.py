"""
Smoke test: verify model forward pass works with random data.
Run this to check the environment is set up correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.model.unet2d import UNet2D
from src.model.density_estimator import DensityEstimator
from src.losses import BiophysicsInformedLoss


def test_unet():
    print("Testing UNet2D...")
    model = UNet2D(in_channels=4, num_classes=4, features=[32, 64, 128, 256, 512])
    x = torch.randn(2, 4, 128, 128)

    # Without features
    out = model(x, return_features=False)
    assert out.shape == (2, 4, 128, 128), f"Expected (2,4,128,128), got {out.shape}"

    # With features
    out, feat = model(x, return_features=True)
    assert out.shape == (2, 4, 128, 128), f"Expected (2,4,128,128), got {out.shape}"
    print(f"  Output shape: {out.shape}")
    print(f"  Bottleneck features shape: {feat.shape}")
    print("  PASSED")


def test_density_estimator():
    print("Testing DensityEstimator...")
    in_channels = 1024  # bottleneck channels for features=[32,64,128,256,512]
    estimator = DensityEstimator(
        in_channels=in_channels,
        hidden_dim=256,
        num_layers=3,
        feature_size=(16, 16),
    )
    features = torch.randn(2, in_channels, 4, 4)  # bottleneck is small
    u_hat = estimator(features)
    assert u_hat.shape == (2, 1, 16, 16), f"Expected (2,1,16,16), got {u_hat.shape}"
    assert u_hat.min() >= 0 and u_hat.max() <= 1, "Density should be in [0,1]"
    print(f"  Output shape: {u_hat.shape}")
    print(f"  Value range: [{u_hat.min().item():.4f}, {u_hat.max().item():.4f}]")
    print("  PASSED")


def test_losses():
    print("Testing BiophysicsInformedLoss...")
    criterion = BiophysicsInformedLoss(lambda_pde=1.0, lambda_bc=1.0)

    pred = torch.randn(2, 4, 128, 128, requires_grad=True)
    target = torch.zeros(2, 4, 128, 128)
    target[:, 0] = 1.0  # all background
    u_hat = torch.sigmoid(torch.randn(2, 1, 16, 16, requires_grad=True))

    loss, loss_dict = criterion(pred, target, u_hat)
    assert loss.requires_grad, "Loss should require grad"
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")
    print(f"  PDE loss: {loss_dict['pde']:.6f}")
    print(f"  BC loss: {loss_dict['bc']:.6f}")
    print("  PASSED")


def test_full_pipeline():
    print("Testing full forward pipeline...")
    import yaml

    config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from src.train import BiophysicsSegModel

    model = BiophysicsSegModel(cfg)
    x = torch.randn(2, 4, 128, 128)

    logits, u_hat = model(x, return_density=True)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Density shape: {u_hat.shape}")

    # Test loss computation
    target = torch.zeros(2, 4, 128, 128)
    target[:, 0] = 1.0
    criterion = BiophysicsInformedLoss()
    loss, loss_dict = criterion(logits, target, u_hat)
    loss.backward()
    print(f"  Backward pass successful")
    print(f"  Total loss: {loss_dict['total']:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Biophysics Informed Segmentation - Smoke Test")
    print("=" * 50)
    print()

    test_unet()
    print()
    test_density_estimator()
    print()
    test_losses()
    print()
    test_full_pipeline()

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)

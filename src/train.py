import os
import sys
import csv
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.unet2d import UNet2D
from src.model.density_estimator import DensityEstimator
from src.losses import BiophysicsInformedLoss
from src.dataset import BraTS2DDataset, get_case_ids, split_cases
from src.dataset_fast import BraTS2DDatasetFast, get_case_ids_from_metadata


def log(msg):
    print(msg, flush=True)


class BiophysicsSegModel(nn.Module):
    """Full model: UNet2D + Density Estimator (pruned at inference)."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        self.unet = UNet2D(
            in_channels=cfg["data"]["num_channels"],
            num_classes=cfg["data"]["num_classes"],
            features=model_cfg["features"],
        )
        self.density_estimator = DensityEstimator(
            in_channels=self.unet.bottleneck_channels,
            hidden_dim=model_cfg["density_estimator"]["hidden_dim"],
            num_layers=model_cfg["density_estimator"]["num_layers"],
            feature_size=tuple(model_cfg["density_estimator"]["feature_size"]),
        )

    def forward(self, x, return_density=True):
        if return_density:
            logits, features = self.unet(x, return_features=True)
            u_hat = self.density_estimator(features)
            return logits, u_hat
        else:
            return self.unet(x, return_features=False)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    use_biophysics=True, epoch=0, log_interval=10):
    model.train()
    epoch_losses = {"dice": 0, "pde": 0, "bc": 0, "total": 0}
    num_batches = 0
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            if use_biophysics:
                logits, u_hat = model(images, return_density=True)
                loss, loss_dict = criterion(logits, targets, u_hat)
            else:
                logits = model(images, return_density=False)
                loss = criterion(logits, targets)
                loss_dict = {"dice": loss.item(), "pde": 0, "bc": 0, "total": loss.item()}

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k]
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            log(f"    [Epoch {epoch+1} Batch {batch_idx+1}/{len(loader)}] "
                f"loss={loss_dict['total']:.4f} dice={loss_dict['dice']:.4f} "
                f"pde={loss_dict['pde']:.6f} bc={loss_dict['bc']:.6f} "
                f"lr={lr:.2e} elapsed={elapsed:.1f}s")

    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    epoch_time = time.time() - t0
    return epoch_losses, epoch_time


@torch.no_grad()
def validate(model, loader, criterion, device, use_biophysics=True):
    model.eval()
    epoch_losses = {"dice": 0, "pde": 0, "bc": 0, "total": 0}
    dice_scores = []
    num_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if use_biophysics:
            logits, u_hat = model(images, return_density=True)
            loss, loss_dict = criterion(logits, targets, u_hat)
        else:
            logits = model(images, return_density=False)
            loss = criterion(logits, targets)
            loss_dict = {"dice": loss.item(), "pde": 0, "bc": 0, "total": loss.item()}

        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k]

        pred = torch.softmax(logits, dim=1)
        pred_mask = pred.argmax(dim=1)
        for c in range(1, 4):
            p = (pred_mask == c).float()
            t = targets[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                dice = (2.0 * intersection / union).item()
            else:
                dice = 1.0
            dice_scores.append(dice)

        num_batches += 1

    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    return epoch_losses, mean_dice


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log("=" * 60)
    log("Biophysics-Informed Segmentation Training")
    log("=" * 60)
    log(f"Config: {config_path}")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Device - safe CUDA detection (avoids segfault on driver mismatch)
    log("Checking device...")
    import subprocess
    cuda_ok = False
    try:
        ret = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
            capture_output=True, text=True, timeout=10
        )
        if ret.returncode == 0:
            cuda_ok = True
            gpu_name = ret.stdout.strip()
    except Exception:
        pass

    if cuda_ok:
        device = torch.device("cuda")
        log(f"\n[Device]")
        log(f"  Device: cuda")
        log(f"  GPU: {gpu_name}")
        log(f"  CUDA (PyTorch built): {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        log(f"\n[Device]")
        log(f"  Device: cpu (CUDA unavailable or driver incompatible)")
        log(f"  TIP: upgrade PyTorch to match your driver, e.g.:")
        log(f"    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    log(f"  PyTorch: {torch.__version__}")

    # Data
    log(f"\n[Data]")
    use_fast = cfg["data"].get("use_fast_loader", False)

    if use_fast:
        preprocessed_dir = cfg["data"]["preprocessed_dir"]
        if not Path(preprocessed_dir).exists():
            log(f"ERROR: Preprocessed directory not found: {preprocessed_dir}")
            log("Run: python src/preprocess.py --data_dir ./data/BraTS2023 --output_dir ./data/preprocessed")
            sys.exit(1)

        case_ids = get_case_ids_from_metadata(preprocessed_dir)
        log(f"  [Fast loader] Found {len(case_ids)} cases from preprocessed data")

        train_ids, val_ids, test_ids = split_cases(
            case_ids,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
            seed=cfg["seed"],
        )

        train_dataset = BraTS2DDatasetFast(preprocessed_dir, case_ids=train_ids, augment=True)
        val_dataset = BraTS2DDatasetFast(preprocessed_dir, case_ids=val_ids, augment=False)
    else:
        data_dir = cfg["data"]["data_dir"]
        if not Path(data_dir).exists():
            log(f"ERROR: Data directory not found: {data_dir}")
            log("Please download BraTS 2023 dataset and place it in the data directory.")
            log("See data/README.md for instructions.")
            sys.exit(1)

        case_ids = get_case_ids(data_dir)
        log(f"  Found {len(case_ids)} cases")

        train_ids, val_ids, test_ids = split_cases(
            case_ids,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
            seed=cfg["seed"],
        )

        input_size = tuple(cfg["data"]["input_size"])
        train_dataset = BraTS2DDataset(data_dir, train_ids, input_size=input_size, augment=True)
        val_dataset = BraTS2DDataset(data_dir, val_ids, input_size=input_size, augment=False)

    log(f"  Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    log(f"  Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")
    log(f"  Batch size: {cfg['training']['batch_size']}")
    log(f"  Num workers: {cfg['data']['num_workers']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    log(f"  Train batches/epoch: {len(train_loader)}")
    log(f"  Val batches/epoch: {len(val_loader)}")

    # Model
    log(f"\n[Model]")
    model = BiophysicsSegModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Architecture: UNet2D + DensityEstimator")
    log(f"  Features: {cfg['model']['features']}")
    log(f"  Total parameters: {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")

    # Loss
    log(f"\n[Loss]")
    loss_cfg = cfg["loss"]
    use_biophysics = loss_cfg.get("use_biophysics", True)

    if use_biophysics:
        criterion = BiophysicsInformedLoss(
            lambda_pde=loss_cfg["lambda_pde"],
            lambda_bc=loss_cfg["lambda_bc"],
            d_range=tuple(loss_cfg["d_range"]),
            rho_range=tuple(loss_cfg["rho_range"]),
        ).to(device)
        log(f"  Loss: Dice + PDE + BC (biophysics regularisation)")
        log(f"  lambda_pde={loss_cfg['lambda_pde']}, lambda_bc={loss_cfg['lambda_bc']}")
        log(f"  d_range={loss_cfg['d_range']}, rho_range={loss_cfg['rho_range']}")
    else:
        from src.losses import DiceLoss
        criterion = DiceLoss().to(device)
        log(f"  Loss: Dice only (baseline)")

    # Optimizer
    log(f"\n[Optimizer]")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    log(f"  AdamW: lr={cfg['training']['lr']}, weight_decay={cfg['training']['weight_decay']}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    log(f"  Scheduler: CosineAnnealing, T_max={cfg['training']['epochs']}")

    # AMP
    use_amp = cfg["training"]["amp"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    log(f"  AMP: {'enabled' if use_amp else 'disabled'}")

    # Output
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV log file
    csv_path = output_dir / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_total", "train_dice", "train_pde", "train_bc",
        "val_total", "val_dice", "val_mean_dice", "lr", "epoch_time_s"
    ])
    log(f"\n[Output]")
    log(f"  Output dir: {output_dir}")
    log(f"  Training log: {csv_path}")

    # Save config copy
    config_save_path = output_dir / "config_used.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log(f"  Config saved: {config_save_path}")

    # Training loop
    log(f"\n{'=' * 60}")
    log(f"Starting training: {cfg['training']['epochs']} epochs")
    log(f"{'=' * 60}")

    best_dice = 0.0
    total_train_time = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        log(f"\n--- Epoch {epoch + 1}/{cfg['training']['epochs']} ---")

        train_losses, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
            use_biophysics, epoch=epoch, log_interval=10
        )
        val_losses, val_dice = validate(model, val_loader, criterion, device, use_biophysics)

        scheduler.step()
        total_train_time += epoch_time
        current_lr = optimizer.param_groups[0]["lr"]

        log(f"  [Summary] Train loss={train_losses['total']:.4f} "
            f"(dice={train_losses['dice']:.4f} pde={train_losses['pde']:.6f} bc={train_losses['bc']:.6f})")
        log(f"  [Summary] Val   loss={val_losses['total']:.4f} "
            f"(dice={val_losses['dice']:.4f}) | Mean Dice={val_dice:.4f}")
        log(f"  [Summary] LR={current_lr:.2e} | Epoch time={epoch_time:.1f}s | "
            f"Total time={total_train_time:.0f}s")

        # Write to CSV
        csv_writer.writerow([
            epoch + 1,
            f"{train_losses['total']:.6f}",
            f"{train_losses['dice']:.6f}",
            f"{train_losses['pde']:.6f}",
            f"{train_losses['bc']:.6f}",
            f"{val_losses['total']:.6f}",
            f"{val_losses['dice']:.6f}",
            f"{val_dice:.6f}",
            f"{current_lr:.8f}",
            f"{epoch_time:.2f}",
        ])
        csv_file.flush()

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": cfg,
            }, output_dir / "best_model.pth")
            log(f"  ** New best model saved (Dice: {best_dice:.4f}) **")

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "config": cfg,
            }, ckpt_path)
            log(f"  Checkpoint saved: {ckpt_path}")

        # GPU memory info
        if device.type == "cuda":
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            log(f"  GPU peak memory: {mem_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    csv_file.close()

    # Final summary
    log(f"\n{'=' * 60}")
    log(f"Training complete!")
    log(f"  Best validation Dice: {best_dice:.4f}")
    log(f"  Total training time: {total_train_time:.0f}s ({total_train_time/60:.1f} min)")
    log(f"  Outputs saved to: {output_dir}")
    log(f"  - best_model.pth")
    log(f"  - training_log.csv")
    log(f"  - config_used.yaml")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import BiophysicsSegModel
from src.dataset import BraTS2DDataset, get_case_ids, split_cases


def compute_dice(pred_mask, target_mask):
    """Compute Dice coefficient for a single class."""
    intersection = (pred_mask & target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()


def compute_hd95(pred_mask, target_mask):
    """Compute 95th percentile Hausdorff Distance."""
    pred_np = pred_mask.cpu().numpy().astype(bool)
    target_np = target_mask.cpu().numpy().astype(bool)

    if not pred_np.any() and not target_np.any():
        return 0.0
    if not pred_np.any() or not target_np.any():
        return 373.13  # max possible for 128x128

    # Distance from pred boundary to target
    pred_boundary = pred_np ^ distance_transform_edt(pred_np) <= 1
    target_boundary = target_np ^ distance_transform_edt(target_np) <= 1

    # Use distance transforms
    dt_target = distance_transform_edt(~target_np)
    dt_pred = distance_transform_edt(~pred_np)

    # Distances from pred surface to target
    if pred_boundary.any():
        d_pred_to_target = dt_target[pred_boundary]
    else:
        d_pred_to_target = np.array([0.0])

    # Distances from target surface to pred
    if target_boundary.any():
        d_target_to_pred = dt_pred[target_boundary]
    else:
        d_target_to_pred = np.array([0.0])

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    hd95 = np.percentile(all_distances, 95)
    return float(hd95)


def evaluate_model(model, loader, device, num_classes=4):
    """
    Evaluate model on test set.
    Reports Dice and HD95 for:
      - TC (Tumour Core): classes 1 + 3
      - WT (Whole Tumour): classes 1 + 2 + 3
      - ET (Enhancing Tumour): class 3
    """
    model.eval()

    metrics = {
        "TC": {"dice": [], "hd95": []},
        "WT": {"dice": [], "hd95": []},
        "ET": {"dice": [], "hd95": []},
    }

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images, return_density=False)
            pred = torch.softmax(logits, dim=1).argmax(dim=1)

            for b in range(pred.shape[0]):
                pred_b = pred[b]
                # Reconstruct region masks from class predictions
                # TC = NCR (1) + ET (3)
                pred_tc = (pred_b == 1) | (pred_b == 3)
                target_tc = (targets[b, 1] > 0.5) | (targets[b, 3] > 0.5)

                # WT = NCR (1) + ED (2) + ET (3)
                pred_wt = (pred_b == 1) | (pred_b == 2) | (pred_b == 3)
                target_wt = (targets[b, 1] > 0.5) | (targets[b, 2] > 0.5) | (targets[b, 3] > 0.5)

                # ET = class 3
                pred_et = (pred_b == 3)
                target_et = (targets[b, 3] > 0.5)

                # Compute metrics
                metrics["TC"]["dice"].append(compute_dice(pred_tc, target_tc))
                metrics["WT"]["dice"].append(compute_dice(pred_wt, target_wt))
                metrics["ET"]["dice"].append(compute_dice(pred_et, target_et))

                metrics["TC"]["hd95"].append(compute_hd95(pred_tc, target_tc))
                metrics["WT"]["hd95"].append(compute_hd95(pred_wt, target_wt))
                metrics["ET"]["hd95"].append(compute_hd95(pred_et, target_et))

    # Aggregate
    results = {}
    for region in ["TC", "WT", "ET"]:
        results[region] = {
            "dice_mean": np.mean(metrics[region]["dice"]) * 100,
            "dice_std": np.std(metrics[region]["dice"]) * 100,
            "hd95_mean": np.mean(metrics[region]["hd95"]),
            "hd95_std": np.std(metrics[region]["hd95"]),
        }

    return results


def main():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    data_dir = cfg["data"]["data_dir"]
    case_ids = get_case_ids(data_dir)
    _, _, test_ids = split_cases(
        case_ids,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"],
    )

    input_size = tuple(cfg["data"]["input_size"])
    test_dataset = BraTS2DDataset(data_dir, test_ids, input_size=input_size, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Test slices: {len(test_dataset)}")

    # Load model
    model = BiophysicsSegModel(cfg).to(device)
    checkpoint_path = Path(cfg["output_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} (Dice: {checkpoint['best_dice']:.4f})")

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"{'Region':<8} {'Dice (%)':<20} {'HD95 (px)':<20}")
    print("-" * 60)
    for region in ["TC", "WT", "ET"]:
        r = results[region]
        print(f"{region:<8} {r['dice_mean']:.2f} +/- {r['dice_std']:.2f}    "
              f"{r['hd95_mean']:.2f} +/- {r['hd95_std']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

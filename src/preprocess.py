"""
Preprocess BraTS NIfTI data into individual .npy slices for fast I/O.

Converts 3D NIfTI volumes into pre-normalized 2D .npy files,
eliminating gzip decompression and redundant volume loading during training.

Output structure:
    preprocessed_dir/
        slices/
            BraTS-GLI-00000-000_z045_image.npy   # (4, 128, 128) float32
            BraTS-GLI-00000-000_z045_seg.npy     # (128, 128) int8
            ...
        metadata.npy   # slice index with case_id, z, tumor_ratio
"""
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from skimage.transform import resize as sk_resize


def normalize_slice(image_slice):
    """Z-score normalization per channel, non-zero voxels only."""
    normalized = np.zeros_like(image_slice, dtype=np.float32)
    for c in range(image_slice.shape[0]):
        ch = image_slice[c]
        mask = ch != 0
        if mask.sum() > 0:
            mean = ch[mask].mean()
            std = ch[mask].std() + 1e-8
            normalized[c][mask] = (ch[mask] - mean) / std
            normalized[c] = np.clip(normalized[c], -5.0, 5.0)
    return normalized


def remap_labels(seg_slice):
    """Remap BraTS labels {0,1,2,4} -> {0,1,2,3}."""
    remapped = np.zeros_like(seg_slice, dtype=np.int8)
    remapped[seg_slice == 1] = 1
    remapped[seg_slice == 2] = 2
    remapped[seg_slice == 4] = 3
    return remapped


def resize_slice(image, seg, target_size=(128, 128)):
    """Resize image and segmentation to target size."""
    H, W = target_size
    resized_image = np.zeros((image.shape[0], H, W), dtype=np.float32)
    for c in range(image.shape[0]):
        resized_image[c] = sk_resize(
            image[c], (H, W), order=1, preserve_range=True, anti_aliasing=True
        )
    resized_seg = sk_resize(
        seg.astype(np.float64), (H, W), order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.int8)
    return resized_image, resized_seg


def process_case(case_path, output_dir, target_size=(128, 128), min_tumor_ratio=0.01):
    """Process a single BraTS case into 2D .npy slices."""
    case_id = case_path.name
    modality_suffixes = ["-t1n", "-t1c", "-t2w", "-t2f"]
    seg_suffix = "-seg"

    def find_file(suffix):
        for f in case_path.iterdir():
            if f.name.endswith(".nii.gz") and suffix in f.name:
                return f
        return None

    # Load all volumes once
    seg_file = find_file(seg_suffix)
    if seg_file is None:
        return []

    seg_vol = nib.load(str(seg_file)).get_fdata()
    volumes = []
    for suffix in modality_suffixes:
        mod_file = find_file(suffix)
        if mod_file is None:
            return []
        volumes.append(nib.load(str(mod_file)).get_fdata())

    slices_info = []
    num_slices = seg_vol.shape[2]

    for z in range(num_slices):
        seg_slice = seg_vol[:, :, z]
        tumor_ratio = (seg_slice > 0).sum() / seg_slice.size

        if tumor_ratio < min_tumor_ratio:
            continue

        # Stack modalities
        image = np.stack([vol[:, :, z].astype(np.float32) for vol in volumes], axis=0)

        # Normalize
        image = normalize_slice(image)

        # Remap labels
        seg_remapped = remap_labels(seg_slice)

        # Resize
        image, seg_remapped = resize_slice(image, seg_remapped, target_size)

        # Save
        slice_name = f"{case_id}_z{z:03d}"
        np.save(output_dir / f"{slice_name}_image.npy", image)
        np.save(output_dir / f"{slice_name}_seg.npy", seg_remapped)

        slices_info.append({
            "case_id": case_id,
            "z": z,
            "tumor_ratio": tumor_ratio,
            "filename": slice_name,
        })

    return slices_info


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS data to .npy slices")
    parser.add_argument("--data_dir", type=str, default="./data/BraTS2023",
                        help="Path to raw BraTS dataset")
    parser.add_argument("--output_dir", type=str, default="./data/preprocessed",
                        help="Output directory for .npy files")
    parser.add_argument("--size", type=int, nargs=2, default=[128, 128],
                        help="Target slice size (H W)")
    parser.add_argument("--min_tumor", type=float, default=0.01,
                        help="Minimum tumor pixel ratio to keep a slice")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / "slices"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Find all cases
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")])
    print(f"Found {len(cases)} cases")
    print(f"Output: {output_dir}")
    print(f"Target size: {args.size}")

    all_slices = []
    for case_path in tqdm(cases, desc="Processing cases"):
        slices_info = process_case(
            case_path, output_dir,
            target_size=tuple(args.size),
            min_tumor_ratio=args.min_tumor,
        )
        all_slices.extend(slices_info)

    # Save metadata
    metadata_path = Path(args.output_dir) / "metadata.npy"
    np.save(str(metadata_path), all_slices, allow_pickle=True)

    print(f"\nDone! Generated {len(all_slices)} slices from {len(cases)} cases")
    print(f"Metadata saved to: {metadata_path}")

    # Print storage estimate
    single_image_bytes = 4 * 128 * 128 * 4  # float32, 4 channels
    single_seg_bytes = 128 * 128  # int8
    total_bytes = len(all_slices) * (single_image_bytes + single_seg_bytes)
    print(f"Estimated storage: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

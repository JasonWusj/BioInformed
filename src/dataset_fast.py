"""Fast dataset loader for preprocessed .npy slices."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BraTS2DDatasetFast(Dataset):
    """
    Fast dataset that loads pre-extracted .npy slices.
    ~50x faster I/O compared to loading from NIfTI.

    Requires running preprocess.py first:
        python src/preprocess.py --data_dir ./data/BraTS2023 --output_dir ./data/preprocessed
    """

    def __init__(self, preprocessed_dir, case_ids=None, augment=False):
        """
        Args:
            preprocessed_dir: path to preprocessed directory (contains slices/ and metadata.npy)
            case_ids: optional list of case IDs to filter (for train/val/test split)
            augment: whether to apply data augmentation
        """
        self.slices_dir = Path(preprocessed_dir) / "slices"
        self.augment = augment

        # Load metadata
        metadata_path = Path(preprocessed_dir) / "metadata.npy"
        all_slices = np.load(str(metadata_path), allow_pickle=True)

        # Filter by case_ids if provided
        if case_ids is not None:
            case_id_set = set(case_ids)
            self.slices = [s for s in all_slices if s["case_id"] in case_id_set]
        else:
            self.slices = list(all_slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        info = self.slices[idx]
        filename = info["filename"]

        # Direct .npy load — no decompression, no header parsing
        image = np.load(self.slices_dir / f"{filename}_image.npy")  # (4, 128, 128) float32
        seg = np.load(self.slices_dir / f"{filename}_seg.npy")      # (128, 128) int8

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                image = image[:, :, ::-1].copy()
                seg = seg[:, ::-1].copy()
            if np.random.rand() > 0.5:
                image = image[:, ::-1, :].copy()
                seg = seg[::-1, :].copy()

        # Convert to one-hot
        seg_onehot = np.zeros((4, *seg.shape), dtype=np.float32)
        for c in range(4):
            seg_onehot[c] = (seg == c).astype(np.float32)

        return torch.from_numpy(image), torch.from_numpy(seg_onehot)


def get_case_ids_from_metadata(preprocessed_dir):
    """Extract unique case IDs from preprocessed metadata."""
    metadata = np.load(Path(preprocessed_dir) / "metadata.npy", allow_pickle=True)
    return sorted(set(s["case_id"] for s in metadata))

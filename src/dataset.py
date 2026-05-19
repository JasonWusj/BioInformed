import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BraTS2DDataset(Dataset):
    """
    BraTS 2023 dataset adapted for 2D slice-based training.
    Extracts axial slices from 3D volumes.

    Expected directory structure:
        data_dir/
            BraTS-GLI-00000-000/
                BraTS-GLI-00000-000-t1n.nii.gz
                BraTS-GLI-00000-000-t1c.nii.gz
                BraTS-GLI-00000-000-t2w.nii.gz
                BraTS-GLI-00000-000-t2f.nii.gz
                BraTS-GLI-00000-000-seg.nii.gz
            ...
    """

    MODALITY_SUFFIXES = ["-t1n", "-t1c", "-t2w", "-t2f"]
    SEG_SUFFIX = "-seg"

    def __init__(self, data_dir, case_ids, input_size=(128, 128), augment=False,
                 min_tumor_ratio=0.01):
        """
        Args:
            data_dir: path to BraTS dataset root
            case_ids: list of case folder names to use
            input_size: (H, W) target size for each slice
            augment: whether to apply data augmentation
            min_tumor_ratio: minimum fraction of tumor pixels to include a slice
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        self.min_tumor_ratio = min_tumor_ratio

        # Build slice index: (case_id, slice_idx)
        self.slices = []
        for case_id in case_ids:
            case_path = self.data_dir / case_id
            seg_file = self._find_file(case_path, self.SEG_SUFFIX)
            if seg_file is None:
                continue
            seg = nib.load(str(seg_file)).get_fdata()
            num_slices = seg.shape[2]

            for z in range(num_slices):
                seg_slice = seg[:, :, z]
                tumor_ratio = (seg_slice > 0).sum() / seg_slice.size
                if tumor_ratio >= self.min_tumor_ratio:
                    self.slices.append((case_id, z))

    def _find_file(self, case_path, suffix):
        """Find a NIfTI file matching the given suffix in case directory."""
        for f in case_path.iterdir():
            if f.name.endswith(".nii.gz") and suffix in f.name:
                return f
        return None

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_id, z = self.slices[idx]
        case_path = self.data_dir / case_id

        # Load all modalities
        channels = []
        for mod_suffix in self.MODALITY_SUFFIXES:
            mod_file = self._find_file(case_path, mod_suffix)
            vol = nib.load(str(mod_file)).get_fdata()
            slice_2d = vol[:, :, z].astype(np.float32)
            channels.append(slice_2d)

        # Load segmentation
        seg_file = self._find_file(case_path, self.SEG_SUFFIX)
        seg_vol = nib.load(str(seg_file)).get_fdata()
        seg_slice = seg_vol[:, :, z].astype(np.int64)

        # Stack modalities: (4, H, W)
        image = np.stack(channels, axis=0)

        # Z-score normalization per channel (non-zero voxels)
        for c in range(image.shape[0]):
            mask = image[c] != 0
            if mask.sum() > 0:
                mean = image[c][mask].mean()
                std = image[c][mask].std() + 1e-8
                image[c][mask] = (image[c][mask] - mean) / std
                # Clip outliers
                image[c] = np.clip(image[c], -5.0, 5.0)

        # Convert BraTS labels to 4-class:
        # 0: background, 1: NCR (label 1), 2: ED (label 2), 3: ET (label 4->3)
        seg_remapped = np.zeros_like(seg_slice)
        seg_remapped[seg_slice == 1] = 1  # NCR -> TC component
        seg_remapped[seg_slice == 2] = 2  # ED -> WT component
        seg_remapped[seg_slice == 4] = 3  # ET

        # Resize to target size
        image, seg_remapped = self._resize(image, seg_remapped)

        # Data augmentation
        if self.augment:
            image, seg_remapped = self._augment(image, seg_remapped)

        # Convert to one-hot: (num_classes, H, W)
        seg_onehot = np.zeros((4, *self.input_size), dtype=np.float32)
        for c in range(4):
            seg_onehot[c] = (seg_remapped == c).astype(np.float32)

        image = torch.from_numpy(image)
        seg_onehot = torch.from_numpy(seg_onehot)

        return image, seg_onehot

    def _resize(self, image, seg):
        """Resize image and segmentation to target size."""
        from skimage.transform import resize as sk_resize

        H, W = self.input_size
        # image: (C, H_orig, W_orig) -> (C, H, W)
        resized_image = np.zeros((image.shape[0], H, W), dtype=np.float32)
        for c in range(image.shape[0]):
            resized_image[c] = sk_resize(image[c], (H, W), order=1,
                                         preserve_range=True, anti_aliasing=True)

        # seg: (H_orig, W_orig) -> (H, W), nearest interpolation
        resized_seg = sk_resize(seg.astype(np.float64), (H, W), order=0,
                                preserve_range=True, anti_aliasing=False).astype(np.int64)

        return resized_image, resized_seg

    def _augment(self, image, seg):
        """Simple data augmentation: random flips and rotation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = image[:, :, ::-1].copy()
            seg = seg[:, ::-1].copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = image[:, ::-1, :].copy()
            seg = seg[::-1, :].copy()

        return image, seg


def get_case_ids(data_dir):
    """Get all valid case IDs from the data directory."""
    data_path = Path(data_dir)
    case_ids = []
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and d.name.startswith("BraTS"):
            case_ids.append(d.name)
    return case_ids


def split_cases(case_ids, train_ratio=0.7, val_ratio=0.1, seed=42):
    """Split case IDs into train/val/test sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(case_ids))

    n_train = int(len(case_ids) * train_ratio)
    n_val = int(len(case_ids) * val_ratio)

    train_ids = [case_ids[i] for i in indices[:n_train]]
    val_ids = [case_ids[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [case_ids[i] for i in indices[n_train + n_val:]]

    return train_ids, val_ids, test_ids

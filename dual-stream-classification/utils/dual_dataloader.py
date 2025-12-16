"""
Data loading module for dual-stream mammography classification.
"""

from typing import Dict, Any, Optional, Tuple
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from utils.augmentation import get_augmentation


class DualStreamDataset(Dataset):
    """
    Dataset for paired MLO and CC mammography images.
    
    Matches images by StudyInstanceUID and Side (L/R).
    Supports augmentation and class balancing through mirroring.
    """
    
    def __init__(
        self,
        mlo_dir: str,
        cc_dir: str,
        mlo_labels_csv: str,
        cc_labels_csv: str,
        split_type: str = 'Train',
        mirror_bad: bool = False,
        use_augmentation: bool = True
    ):
        self.mlo_dir = mlo_dir
        self.cc_dir = cc_dir
        self.split_type = split_type
        self.mirror_bad = mirror_bad and split_type == 'Train'
        
        self.augmentation = get_augmentation(
            split_type=split_type,
            use_augmentation=use_augmentation
        )
        
        self.paired_data = self._load_and_pair_data(mlo_labels_csv, cc_labels_csv)
        self._apply_mirror_augmentation()
        
        self._print_stats()
    
    def _load_and_pair_data(self, mlo_csv: str, cc_csv: str) -> pd.DataFrame:
        """Load and pair MLO/CC data."""
        mlo_df = pd.read_csv(mlo_csv)
        cc_df = pd.read_csv(cc_csv)
        
        mlo_df = mlo_df[mlo_df['Split'] == self.split_type].copy()
        cc_df = cc_df[cc_df['Split'] == self.split_type].copy()
        
        mlo_df = mlo_df.drop_duplicates(subset=['SOPInstanceUID'], keep='first')
        cc_df = cc_df.drop_duplicates(subset=['SOPInstanceUID'], keep='first')
        
        mlo_df['Side'] = mlo_df['SeriesDescription'].str.split('-').str[0]
        cc_df['Side'] = cc_df['SeriesDescription'].str.split('-').str[0]
        
        paired = pd.merge(
            mlo_df, cc_df,
            on=['StudyInstanceUID', 'Side'],
            how='inner',
            suffixes=('_mlo', '_cc')
        )
        
        paired = paired[paired['qualitativeLabel_mlo'] == 'Good'].copy()
        
        if len(paired) == 0:
            paired = pd.merge(
                mlo_df, cc_df,
                on=['StudyInstanceUID'],
                how='inner',
                suffixes=('_mlo', '_cc')
            )
        
        return paired
    
    def _apply_mirror_augmentation(self) -> None:
        """Apply horizontal mirror augmentation to Bad samples."""
        self.paired_data['mirror'] = False
        
        if self.mirror_bad:
            base = self.paired_data.copy()
            is_bad = base['qualitativeLabel_cc'].str.lower() == 'bad'
            mirrored = base[is_bad].copy()
            mirrored['mirror'] = True
            self.paired_data = pd.concat([base, mirrored], ignore_index=True)
    
    def _print_stats(self) -> None:
        """Print dataset statistics."""
        good_count = len(self.paired_data[self.paired_data['qualitativeLabel_cc'] == 'Good'])
        bad_count = len(self.paired_data[self.paired_data['qualitativeLabel_cc'] == 'Bad'])
        
        print(f"[{self.split_type}] Samples: {len(self.paired_data)} (Good: {good_count}, Bad: {bad_count})")
        if self.mirror_bad:
            print(f"[{self.split_type}] Mirror augmentation: ACTIVE")
    
    def __len__(self) -> int:
        return len(self.paired_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        row = self.paired_data.iloc[idx]
        
        mlo_img = self._load_image(row['SOPInstanceUID_mlo'], self.mlo_dir)
        cc_img = self._load_image(row['SOPInstanceUID_cc'], self.cc_dir)
        
        if row.get('mirror', False):
            mlo_img = torch.flip(mlo_img, dims=[2])
            cc_img = torch.flip(cc_img, dims=[2])
        
        mlo_img = self.augmentation(mlo_img)
        cc_img = self.augmentation(cc_img)
        
        label = 1 if row['qualitativeLabel_cc'] == 'Good' else 0
        
        metadata = {
            'study_id': row['StudyInstanceUID'],
            'side': row.get('Side', 'Unknown'),
            'mlo_sop': row['SOPInstanceUID_mlo'],
            'cc_sop': row['SOPInstanceUID_cc']
        }
        
        return mlo_img, cc_img, label, metadata
    
    def _load_image(self, sop_uid: str, directory: str) -> torch.Tensor:
        """Load and preprocess image."""
        path = os.path.join(directory, f"{sop_uid}.npy")
        
        try:
            img = np.load(path)
        except FileNotFoundError:
            img = np.zeros((512, 512), dtype=np.float32)
        
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        
        return torch.from_numpy(img).float()


def get_dual_dataloader(
    mlo_dir: str,
    cc_dir: str,
    mlo_labels_csv: str,
    cc_labels_csv: str,
    split_type: str = 'Train',
    batch_size: int = 8,
    num_workers: int = 4,
    mirror_bad: bool = False,
    weighted_sampler: bool = False,
    use_augmentation: bool = True
) -> DataLoader:
    """
    Create dual-stream dataloader.
    
    Args:
        mlo_dir: Directory containing MLO images
        cc_dir: Directory containing CC images
        mlo_labels_csv: Path to MLO labels CSV
        cc_labels_csv: Path to CC labels CSV
        split_type: Data split ('Train', 'Validation', 'Test')
        batch_size: Batch size
        num_workers: Number of data loading workers
        mirror_bad: Apply mirror augmentation to Bad samples
        weighted_sampler: Use weighted random sampling for class balance
        use_augmentation: Apply data augmentation
        
    Returns:
        DataLoader instance
    """
    dataset = DualStreamDataset(
        mlo_dir, cc_dir, mlo_labels_csv, cc_labels_csv,
        split_type, mirror_bad, use_augmentation
    )
    
    sampler = None
    if split_type == 'Train' and weighted_sampler:
        sampler = _create_weighted_sampler(dataset)
    
    shuffle = split_type == 'Train' and sampler is None
    num_workers = min(num_workers, 4) if num_workers > 0 else 2
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )


def _create_weighted_sampler(dataset: DualStreamDataset) -> WeightedRandomSampler:
    """Create weighted random sampler for class balancing."""
    labels = (dataset.paired_data['qualitativeLabel_cc'].str.lower() == 'good').astype(int).values
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.from_numpy(class_weights[labels]).double()
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

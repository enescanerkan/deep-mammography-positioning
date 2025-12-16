"""
Data Loading Utilities for CC Nipple Detection

This module provides dataset and dataloader classes for loading
preprocessed CC mammography images and their nipple annotations.

Unlike MLO, CC model uses both Good and Bad quality images for training
since nipple detection is needed regardless of positioning quality.
"""

import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import json


class CCDataset(Dataset):
    """
    Dataset for CC nipple detection.
    
    Loads preprocessed mammography images (.npy) and their corresponding
    nipple coordinates from a DataFrame.
    
    Args:
        dataframe (DataFrame): DataFrame with image paths and landmarks
        base_image_dir (str): Directory containing .npy image files
        target_task (str): Task type - always 'nipple' for CC
    """
    def __init__(self, dataframe, base_image_dir, target_task='nipple'):
        self.dataframe = dataframe
        self.base_image_dir = base_image_dir
        self.target_task = target_task

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Load image and coordinates for given index.
        
        Returns:
            tuple: (image tensor, coordinates tensor)
        """
        img_name = self.dataframe.iloc[idx]['SOPInstanceUID'] + '.npy'
        img_path = os.path.join(self.base_image_dir, img_name)
        
        # Load image
        image = np.load(img_path)
        image_height, image_width = image.shape[:2]
        
        # Parse landmarks
        adjusted_landmarks_str = self.dataframe.iloc[idx]['adjusted_landmarks']
        adjusted_landmarks = json.loads(adjusted_landmarks_str)
        
        # Get normalized coordinates (nipple only)
        coordinates = self._parse_coordinates(adjusted_landmarks, image_width, image_height)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        
        # Convert image to tensor with channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        return image, coordinates

    def _parse_coordinates(self, landmarks, width, height):
        """
        Parse and normalize nipple coordinates.
        
        Args:
            landmarks (dict): Landmark dictionary
            width (int): Image width
            height (int): Image height
            
        Returns:
            list: Normalized coordinates [nipple_x, nipple_y]
        """
        if self.target_task == 'nipple':
            return [
                landmarks["nipple"]["x"] / width,
                landmarks["nipple"]["y"] / height
            ]
        else:
            raise ValueError(f"Invalid target_task for CC: {self.target_task}")


def preprocess_data(config):
    """
    Load and merge label and transformation data.
    
    Args:
        config (dict): Configuration with file paths
        
    Returns:
        DataFrame: Merged data with labels and transformations
    """
    split_df = pd.read_csv(config['split_file'])
    details_df = pd.read_csv(config['details_file'])
    
    print(f"Split DF shape: {split_df.shape}")
    print(f"Details DF shape: {details_df.shape}")

    # Merge on SOPInstanceUID
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')
    
    print(f"Unified DF shape: {unified_df.shape}")

    return unified_df


def create_dataloaders(unified_df, config, return_test_df=False):
    """
    Create train, validation, and test dataloaders.
    
    Note: CC model uses both Good and Bad quality images since
    nipple detection is needed regardless of positioning quality.
    
    Args:
        unified_df (DataFrame): Merged data
        config (dict): Configuration dictionary
        return_test_df (bool): Whether to return test DataFrame
        
    Returns:
        dict: Dictionary of DataLoaders
    """
    # Filter by labelName if column exists
    if 'labelName' in unified_df.columns:
        df = unified_df[unified_df['labelName'] == 'Nipple'].copy()
    else:
        df = unified_df.copy()
    
    print(f"After labelName filtering: {df.shape}")
    
    datasets = {}
    test_df = None

    # Use existing splits from CSV
    split_col = 'Split'
    train_df = df[df[split_col].isin(['Train', 'Training'])].copy()
    val_df = df[df[split_col].isin(['Val', 'Validation'])].copy()
    test_df_temp = df[df[split_col].isin(['Test'])].copy()

    print(f"Splits - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df_temp)}")

    # Create datasets (CC uses all quality levels)
    datasets['Train'] = CCDataset(train_df, config['base_image_dir'], config['target_task'])
    datasets['Validation'] = CCDataset(val_df, config['base_image_dir'], config['target_task'])
    datasets['Test'] = CCDataset(test_df_temp, config['base_image_dir'], config['target_task'])

    if return_test_df:
        test_df = test_df_temp

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=config['batch_size'],
            shuffle=(x == 'Train')
        )
        for x in datasets.keys()
    }

    if return_test_df:
        return dataloaders, test_df
    return dataloaders

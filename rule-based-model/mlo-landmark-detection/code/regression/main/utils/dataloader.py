"""
Data Loading Utilities for MLO Landmark Detection

This module provides dataset and dataloader classes for loading
preprocessed MLO mammography images and their landmark annotations.

The dataloader filters training data to include only "Good" quality
images, ensuring the model learns from correctly positioned mammograms.
"""

import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import json


class MLODataset(Dataset):
    """
    Dataset for MLO landmark detection.
    
    Loads preprocessed mammography images (.npy) and their corresponding
    landmark coordinates from a DataFrame.
    
    Args:
        dataframe (DataFrame): DataFrame with image paths and landmarks
        base_image_dir (str): Directory containing .npy image files
        target_task (str): Task type - 'pectoralis', 'nipple', or 'all'
    """
    def __init__(self, dataframe, base_image_dir, target_task='all'):
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
        
        # Get normalized coordinates
        coordinates = self._parse_coordinates(adjusted_landmarks, image_width, image_height)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        
        # Convert image to tensor with channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        return image, coordinates

    def _parse_coordinates(self, landmarks, width, height):
        """
        Parse and normalize landmark coordinates.
        
        Args:
            landmarks (dict): Landmark dictionary
            width (int): Image width
            height (int): Image height
            
        Returns:
            list: Normalized coordinates [0, 1]
        """
        if self.target_task == 'pectoralis':
            return [
                landmarks["pectoral_line"]["x1"] / width,
                landmarks["pectoral_line"]["y1"] / height,
                landmarks["pectoral_line"]["x2"] / width,
                landmarks["pectoral_line"]["y2"] / height
            ]
        elif self.target_task == 'nipple':
            return [
                landmarks["nipple"]["x"] / width,
                landmarks["nipple"]["y"] / height
            ]
        elif self.target_task == 'all':
            return [
                landmarks["pectoral_line"]["x1"] / width,
                landmarks["pectoral_line"]["y1"] / height,
                landmarks["pectoral_line"]["x2"] / width,
                landmarks["pectoral_line"]["y2"] / height,
                landmarks["nipple"]["x"] / width,
                landmarks["nipple"]["y"] / height
            ]
        else:
            raise ValueError(f"Invalid target_task: {self.target_task}")


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
    
    # Merge on SOPInstanceUID
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left')
    
    return unified_df


def create_dataloaders(unified_df, config, return_test_df=False):
    """
    Create train, validation, and test dataloaders.
    
    Important: Training and validation use only "Good" quality images.
    Test set includes all images for comprehensive evaluation.
    
    Args:
        unified_df (DataFrame): Merged data
        config (dict): Configuration dictionary
        return_test_df (bool): Whether to return test DataFrame
        
    Returns:
        dict: Dictionary of DataLoaders
    """
    # Filter for Pectoralis label (MLO images)
    df = unified_df[unified_df['labelName'] == 'Pectoralis'].copy()

    datasets = {}
    test_df = None

    for split in ['Train', 'Validation', 'Test']:
        split_df = df[df['Split'] == split].copy()

        # Filter only "Good" quality for Train and Validation
        if split in ['Train', 'Validation']:
            original_count = len(split_df)
            split_df = split_df[split_df['qualitativeLabel'] == 'Good'].copy()
            filtered_count = len(split_df)
            print(f"{split}: {original_count} -> {filtered_count} (Good only)")
        else:
            # Test set: keep all samples
            good_count = len(split_df[split_df['qualitativeLabel'] == 'Good'])
            bad_count = len(split_df[split_df['qualitativeLabel'] == 'Bad'])
            print(f"{split}: {len(split_df)} total (Good: {good_count}, Bad: {bad_count})")

        if split == 'Test':
            test_df = split_df

        datasets[split] = MLODataset(split_df, config['base_image_dir'], config['target_task'])

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

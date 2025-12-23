#!/usr/bin/env python3
"""
Complete MLO-CC Quality Evaluation Pipeline
Single script to run all evaluations: MLO, CC, and Combined Quality Assessment

Usage:
    python run_full_evaluation.py

This will:
1. Evaluate all MLO images
2. Evaluate all CC images  
3. Combine results and calculate quality metrics
4. Generate visualizations and confusion matrix
"""

import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import measure, morphology
from scipy import ndimage
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import traceback


class PathConfig:
    """Centralized path management - all paths relative to this script"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.project_root = self.script_dir.parent.parent  # test-models -> rule-based-model -> project_root
        self.rule_based_dir = self.project_root / "rule-based-model"
        self.labels_dir = self.project_root / "labels"
        self.data_dir = self.project_root / "data" / "raw"
        self.mlo_images_dir = self.data_dir / "mlo"
        self.cc_images_dir = self.data_dir / "cc"
        self.output_dir = self.script_dir / "evaluation_results"
        self.mlo_viz_dir = self.output_dir / "mlo_visualizations"
        self.cc_viz_dir = self.output_dir / "cc_visualizations"
        self.combined_viz_dir = self.output_dir / "combined_visualizations"
        
    def setup_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlo_viz_dir.mkdir(parents=True, exist_ok=True)
        self.cc_viz_dir.mkdir(parents=True, exist_ok=True)
        self.combined_viz_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def mlo_model_path(self):
        return self.rule_based_dir / "mlo-landmark-detection" / "code" / "models" / "mlo_model.pth"
    
    @property
    def cc_model_path(self):
        return self.rule_based_dir / "cc-landmark-detection" / "code" / "models" / "cc_model.pth"
    
    @property
    def mlo_labels_path(self):
        return self.labels_dir / "mlo_labels.csv"
    
    @property
    def cc_labels_path(self):
        return self.labels_dir / "cc_labels.csv"
    
    @property
    def metadata_path(self):
        return self.labels_dir / "metadata.csv"
    
    def verify_paths(self):
        errors = []
        if not self.mlo_model_path.exists():
            errors.append(f"MLO model not found: {self.mlo_model_path}")
        if not self.cc_model_path.exists():
            errors.append(f"CC model not found: {self.cc_model_path}")
        if not self.mlo_labels_path.exists():
            errors.append(f"MLO labels not found: {self.mlo_labels_path}")
        if not self.cc_labels_path.exists():
            errors.append(f"CC labels not found: {self.cc_labels_path}")
        if not self.metadata_path.exists():
            errors.append(f"Metadata not found: {self.metadata_path}")
        if not self.mlo_images_dir.exists():
            errors.append(f"MLO images directory not found: {self.mlo_images_dir}")
        if not self.cc_images_dir.exists():
            errors.append(f"CC images directory not found: {self.cc_images_dir}")
            
        if errors:
            print("PATH ERRORS:")
            for e in errors:
                print(f"  - {e}")
            return False
        
        print("All paths verified successfully!")
        return True


class AddCoordinates(nn.Module):
    """Layer to add X,Y coordinates as channels"""
    def forward(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        x_coords = torch.linspace(0, 1, width, device=input_tensor.device).repeat(batch_size, height, 1)
        y_coords = torch.linspace(0, 1, height, device=input_tensor.device).repeat(batch_size, width, 1).transpose(1, 2)
        coords = torch.stack((x_coords, y_coords), dim=1)
        return torch.cat((input_tensor, coords), dim=1)


class CoordConv(nn.Module):
    """CoordConv layer"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CoordConv, self).__init__()
        self.add_coords = AddCoordinates()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi



class MLOCRAUNet(nn.Module):
    """
    MLO-specific CRAUNet - Uses different final layers than CC version
    This matches the architecture in mlo/models_mlo.py
    """
    def __init__(self, in_channels=1, out_features=6):
        super(MLOCRAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder - MLO uses simple CoordConv for Conv1
        self.Conv1 = CoordConv(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final layers - MLO specific (different from CC!)
        self.last = nn.Conv2d(64, out_features, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.UpConv2(d2)

        # MLO-specific regression output
        out = self.last(d2)
        out = nn.functional.adaptive_avg_pool2d(out, (16, 16))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



class CCCRAUNet(nn.Module):
    """
    CC-specific CRAUNet - Uses global pooling approach
    This matches the architecture in models.py
    """
    def __init__(self, in_channels=1, out_features=2):
        super(CCCRAUNet, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        # Encoder
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CC uses Sequential Conv1 with extra Conv2d
        self.Conv1 = nn.Sequential(
            CoordConv(in_channels, filters[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        # Decoder
        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], n_coefficients=filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], n_coefficients=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], n_coefficients=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], n_coefficients=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        # CC-specific regression head (global pooling)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters[0], 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # CC-specific regression
        out = self.global_pool(d2)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out




class ImagePreprocessor:
    """DICOM preprocessing class"""
    TARGET_SIZE = (512, 512)
    _dicom_index_cache = {}

    def load_dicom(self, path):
        dicom = pydicom.dcmread(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        data = data.astype(np.float32)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.max(data) - data
        
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return data, dicom

    def _find_largest_rectangle(self, img):
        thresh_val = img.mean()
        binary_image = img > thresh_val
        cleaned_image = morphology.opening(binary_image, morphology.disk(3))
        labeled_image, _ = ndimage.label(cleaned_image)
        regions = measure.regionprops(labeled_image, intensity_image=img)
        if not regions:
            return 0, img.shape[0]-1, 0, img.shape[1]-1
        largest_region = max(regions, key=lambda x: x.area)
        minr, minc, maxr, maxc = largest_region.bbox
        return minr, maxr, minc, maxc

    def _crop_image(self, img):
        rmin, rmax, cmin, cmax = self._find_largest_rectangle(img)
        return img[rmin:rmax+1, cmin:cmax+1], (rmin, rmax, cmin, cmax)

    def _pad_and_resize(self, img, series_description):
        target_size = self.TARGET_SIZE
        height, width = img.shape[:2]
        max_side = max(height, width)
        
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top
        
        total_lr_padding = max_side - width
        
        if "L-MLO" in series_description or "L-CC" in series_description or "LCC" in series_description:
            pad_left = 0
            pad_right = total_lr_padding
        elif "R-MLO" in series_description or "R-CC" in series_description or "RCC" in series_description:
            pad_left = total_lr_padding
            pad_right = 0
        else:
            pad_left = total_lr_padding // 2
            pad_right = max_side - width - pad_left

        img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        img_resized = cv2.resize(img_padded, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img_padded, img_resized, (pad_left, pad_right, pad_top, pad_bottom)

    def extract_pixel_spacing(self, dicom):
        try:
            if hasattr(dicom, 'ImagerPixelSpacing'):
                spacing = dicom.ImagerPixelSpacing
                return float(spacing[0]), float(spacing[1])
            elif hasattr(dicom, 'PixelSpacing'):
                spacing = dicom.PixelSpacing
                return float(spacing[0]), float(spacing[1])
            else:
                return 0.085, 0.085
        except (AttributeError, IndexError, ValueError):
            return 0.085, 0.085

    def process(self, dicom_path):
        img, dicom_obj = self.load_dicom(dicom_path)
        original_shape = img.shape
        
        try:
            laterality = getattr(dicom_obj, 'ImageLaterality', 'L')
            view_position = getattr(dicom_obj, 'ViewPosition', 'MLO')
            series_description = f"{laterality}-{view_position}"
        except:
            series_description = 'L-MLO'
        
        cropped_img, crop_coords = self._crop_image(img)
        padded_img, resized_img, pad_coords = self._pad_and_resize(cropped_img, series_description)
        
        original_pixel_spacing = self.extract_pixel_spacing(dicom_obj)
        scale_x = resized_img.shape[1] / padded_img.shape[1]
        scale_y = resized_img.shape[0] / padded_img.shape[0]
        
        transformation_info = {
            'original_shape': original_shape,
            'crop_coords': crop_coords,
            'pad_coords': pad_coords,
            'series_description': series_description,
            'original_pixel_spacing': original_pixel_spacing,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
        
        return resized_img, original_shape, dicom_obj, transformation_info

    def calculate_scaled_pixel_spacing(self, original_spacing, transformation_info):
        rmin, rmax, cmin, cmax = transformation_info['crop_coords']
        cropped_h = rmax - rmin + 1
        cropped_w = cmax - cmin + 1
        padded_size = max(cropped_h, cropped_w)
        scale_factor = padded_size / 512.0
        scaled_spacing = original_spacing[0] * scale_factor
        return scaled_spacing, scale_factor

    @classmethod
    def build_dicom_index(cls, directory_path):
        directory_path = str(Path(directory_path).resolve())
        if directory_path in cls._dicom_index_cache:
            return cls._dicom_index_cache[directory_path]

        index = {}
        if not os.path.exists(directory_path):
            cls._dicom_index_cache[directory_path] = index
            return index

        for name in os.listdir(directory_path):
            if not name.lower().endswith('.dicom'):
                continue
            fp = os.path.join(directory_path, name)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                sop = getattr(ds, 'SOPInstanceUID', None)
                if sop:
                    index[str(sop)] = fp
            except Exception:
                continue

        cls._dicom_index_cache[directory_path] = index
        return index

    @classmethod
    def get_dicom_path_for_sop(cls, directory_path, sop_uid):
        idx = cls.build_dicom_index(directory_path)
        
        full_path = idx.get(str(sop_uid))
        if full_path:
            return full_path
        
        sop_prefix = str(sop_uid)[:8]
        for full_sop, file_path in idx.items():
            if full_sop.startswith(sop_prefix):
                return file_path
        
        return None

    def transform_coords_to_original(self, coords_512, original_shape, crop_coords, pad_coords):
        is_single_coord = len(np.array(coords_512).shape) == 1
        
        if is_single_coord:
            coords_512 = [coords_512]
        
        rmin, rmax, cmin, cmax = crop_coords
        pad_left, pad_right, pad_top, pad_bottom = pad_coords
        
        cropped_height = rmax - rmin + 1
        cropped_width = cmax - cmin + 1
        padded_size = max(cropped_height, cropped_width)
        scale_factor = padded_size / 512.0
        
        original_coords = []
        for coord in coords_512:
            x_512, y_512 = coord[0], coord[1]
            x_padded = x_512 * scale_factor
            y_padded = y_512 * scale_factor
            x_cropped = x_padded - pad_left
            y_cropped = y_padded - pad_top
            x_original = x_cropped + cmin
            y_original = y_cropped + rmin
            original_coords.append([x_original, y_original])
        
        if is_single_coord:
            return np.array(original_coords[0])
        else:
            return np.array(original_coords)

    def transform_coords_to_512(self, coords_original, transformation_info):
        is_single_coord = len(np.array(coords_original).shape) == 1
        
        if is_single_coord:
            coords_original = [coords_original]
        
        crop_coords = transformation_info['crop_coords']
        pad_coords = transformation_info['pad_coords']
        
        rmin, rmax, cmin, cmax = crop_coords
        pad_left, pad_right, pad_top, pad_bottom = pad_coords
        
        cropped_height = rmax - rmin + 1
        cropped_width = cmax - cmin + 1
        padded_size = max(cropped_height, cropped_width)
        scale_factor = 512.0 / padded_size
        
        coords_512 = []
        for coord in coords_original:
            x_original, y_original = coord[0], coord[1]
            x_cropped = x_original - cmin
            y_cropped = y_original - rmin
            x_padded = x_cropped + pad_left
            y_padded = y_cropped + pad_top
            x_512 = x_padded * scale_factor
            y_512 = y_padded * scale_factor
            coords_512.append([x_512, y_512])
        
        if is_single_coord:
            return np.array(coords_512[0])
        else:
            return np.array(coords_512)

    def transform_landmarks_to_original(self, landmarks_512, transformation_info):
        return self.transform_coords_to_original(
            landmarks_512,
            transformation_info['original_shape'],
            transformation_info['crop_coords'],
            transformation_info['pad_coords']
        )




class MLOEvaluator:
    """MLO landmark detection and distance calculation"""
    
    def __init__(self, paths: PathConfig, device: torch.device):
        self.paths = paths
        self.device = device
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load MLO model with correct architecture"""
        try:
            # Use MLO-specific model architecture!
            self.model = MLOCRAUNet(in_channels=1, out_features=6).to(self.device)
            self.model.load_state_dict(
                torch.load(self.paths.mlo_model_path, map_location=self.device), 
                strict=False
            )
            self.model.eval()
            print(f"MLO model loaded: {self.paths.mlo_model_path}")
        except Exception as e:
            print(f"MLO model loading error: {e}")
            traceback.print_exc()
            self.model = None
    
    def predict_landmarks(self, processed_image):
        """Predict MLO landmarks"""
        if self.model is None:
            return None
        
        try:
            img_tensor = torch.from_numpy(processed_image).float()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(img_tensor).cpu().numpy().flatten()
            
            # MLO: 6 values -> (3, 2) for 3 landmarks
            predicted_landmarks = np.reshape(predictions, (-1, 2))
            predicted_landmarks[:, 0] *= 512
            predicted_landmarks[:, 1] *= 512
            
            return predicted_landmarks
        except Exception as e:
            print(f"MLO prediction error: {e}")
            return None
    
    @staticmethod
    def calculate_perpendicular_distance(point1, point2, nipple):
        """Calculate perpendicular distance from nipple to pectoral line"""
        line_vec = point2 - point1
        point_vec = nipple - point1
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec), point1.copy()
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj = proj_length * line_unitvec
        perp_vec = point_vec - proj
        perp_distance = np.linalg.norm(perp_vec)
        intersection = point1 + proj
        
        return perp_distance, intersection
    
    def calculate_distance(self, landmarks, pixel_spacing):
        """Calculate MLO distance in mm"""
        pectoral1 = landmarks[0]
        pectoral2 = landmarks[1]
        nipple = landmarks[2]
        
        distance_px, intersection = self.calculate_perpendicular_distance(pectoral1, pectoral2, nipple)
        distance_mm = distance_px * pixel_spacing
        
        return distance_mm, distance_px, intersection
    
    @staticmethod
    def parse_ground_truth(sop_uid, mlo_labels):
        """Parse ground truth landmarks from labels"""
        try:
            pectoral_rows = mlo_labels[
                (mlo_labels['SOPInstanceUID'] == sop_uid) & 
                (mlo_labels['labelName'] == 'Pectoralis')
            ]
            nipple_rows = mlo_labels[
                (mlo_labels['SOPInstanceUID'] == sop_uid) & 
                (mlo_labels['labelName'] == 'Nipple')
            ]
            
            if len(pectoral_rows) == 0 or len(nipple_rows) == 0:
                return None
            
            pectoral_data = ast.literal_eval(pectoral_rows.iloc[0]['data'])
            pectoral_vertices = pectoral_data['vertices']
            pectoral1 = np.array(pectoral_vertices[0])
            pectoral2 = np.array(pectoral_vertices[1])
            
            nipple_data = ast.literal_eval(nipple_rows.iloc[0]['data'])
            nipple_x = nipple_data['x'] + nipple_data['width'] / 2
            nipple_y = nipple_data['y'] + nipple_data['height'] / 2
            nipple = np.array([nipple_x, nipple_y])
            
            return np.array([pectoral1, pectoral2, nipple])
        except Exception as e:
            return None
    
    def create_visualization(self, original_image, pred_landmarks_512, gt_landmarks, 
                           transformation_info, distance_mm, sop_uid, output_path):
        """Create MLO visualization on original DICOM"""
        try:
            pred_landmarks_original = self.preprocessor.transform_landmarks_to_original(
                pred_landmarks_512, transformation_info
            )
            
            pectoral1_512 = pred_landmarks_512[0]
            pectoral2_512 = pred_landmarks_512[1]
            nipple_512 = pred_landmarks_512[2]
            
            line_vec = pectoral2_512 - pectoral1_512
            point_vec = nipple_512 - pectoral1_512
            line_len = np.linalg.norm(line_vec)
            if line_len > 0:
                line_unitvec = line_vec / line_len
                proj_length = np.dot(point_vec, line_unitvec)
                proj = proj_length * line_unitvec
                intersection_512 = pectoral1_512 + proj
            else:
                intersection_512 = pectoral1_512
            
            intersection_original = self.preprocessor.transform_coords_to_original(
                intersection_512, 
                transformation_info['original_shape'],
                transformation_info['crop_coords'],
                transformation_info['pad_coords']
            )
            
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(original_image, cmap='gray')
            
            if gt_landmarks is not None:
                ax.plot(gt_landmarks[0][0], gt_landmarks[0][1], 'co', markersize=8, 
                       label='GT Pectoral 1', markeredgecolor='white', markeredgewidth=1)
                ax.plot(gt_landmarks[1][0], gt_landmarks[1][1], 'co', markersize=8, 
                       label='GT Pectoral 2', markeredgecolor='white', markeredgewidth=1)
                ax.plot(gt_landmarks[2][0], gt_landmarks[2][1], 'bo', markersize=12, 
                       label='GT Nipple', markeredgecolor='white', markeredgewidth=2)
                ax.plot([gt_landmarks[0][0], gt_landmarks[1][0]], 
                       [gt_landmarks[0][1], gt_landmarks[1][1]], 
                       'c-', linewidth=2, alpha=0.7, label='GT Pectoral Line')
            
            ax.plot(pred_landmarks_original[0][0], pred_landmarks_original[0][1], 'ro', markersize=8, 
                   label='Pred Pectoral 1', markeredgecolor='white', markeredgewidth=1)
            ax.plot(pred_landmarks_original[1][0], pred_landmarks_original[1][1], 'ro', markersize=8, 
                   label='Pred Pectoral 2', markeredgecolor='white', markeredgewidth=1)
            ax.plot(pred_landmarks_original[2][0], pred_landmarks_original[2][1], 'go', markersize=12, 
                   label='Pred Nipple', markeredgecolor='white', markeredgewidth=2)
            
            ax.plot([pred_landmarks_original[0][0], pred_landmarks_original[1][0]], 
                   [pred_landmarks_original[0][1], pred_landmarks_original[1][1]], 
                   'r-', linewidth=2, label='Pred Pectoral Line')
            
            ax.plot([pred_landmarks_original[2][0], intersection_original[0]], 
                   [pred_landmarks_original[2][1], intersection_original[1]], 
                   'r--', linewidth=2, label='Perpendicular Distance')
            
            mid_x = (pred_landmarks_original[2][0] + intersection_original[0]) / 2
            mid_y = (pred_landmarks_original[2][1] + intersection_original[1]) / 2
            ax.annotate(f"{distance_mm:.1f}mm", xy=(mid_x, mid_y), xytext=(15, 15), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
            ax.set_title(f'MLO - {sop_uid[:8]}... | Distance: {distance_mm:.2f}mm', fontsize=12)
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.axis('off')
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(output_path)
        except Exception as e:
            print(f"MLO visualization error: {e}")
            return None




class CCEvaluator:
    """CC landmark detection and distance calculation"""
    
    def __init__(self, paths: PathConfig, device: torch.device):
        self.paths = paths
        self.device = device
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load CC model with correct architecture"""
        try:
            # Use CC-specific model architecture!
            self.model = CCCRAUNet(in_channels=1, out_features=2).to(self.device)
            self.model.load_state_dict(
                torch.load(self.paths.cc_model_path, map_location=self.device), 
                strict=False
            )
            self.model.eval()
            print(f"CC model loaded: {self.paths.cc_model_path}")
        except Exception as e:
            print(f"CC model loading error: {e}")
            traceback.print_exc()
            self.model = None
    
    def predict_landmarks(self, processed_image):
        """Predict CC nipple landmark"""
        if self.model is None:
            return None
        
        try:
            img_tensor = torch.from_numpy(processed_image).float()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(img_tensor).cpu().numpy().flatten()
            
            pred_x = predictions[0] * 512
            pred_y = predictions[1] * 512
            
            return np.array([pred_x, pred_y])
        except Exception as e:
            print(f"CC prediction error: {e}")
            return None
    
    def calculate_distance(self, nipple_coords, laterality, pixel_spacing):
        """Calculate CC distance from nipple to chest wall edge"""
        nipple_x = nipple_coords[0]
        
        if laterality == "L":
            distance_px = nipple_x
            direction = "Left"
        else:
            distance_px = 512 - nipple_x
            direction = "Right"
        
        distance_mm = distance_px * pixel_spacing
        return direction, distance_mm, distance_px
    
    @staticmethod
    def parse_ground_truth(sop_uid, cc_labels):
        """Parse ground truth nipple from labels"""
        try:
            nipple_rows = cc_labels[
                (cc_labels['SOPInstanceUID'] == sop_uid) & 
                (cc_labels['labelName'] == 'Nipple')
            ]
            
            if len(nipple_rows) == 0:
                return None
            
            nipple_data = ast.literal_eval(nipple_rows.iloc[0]['data'])
            nipple_x = nipple_data['x'] + nipple_data['width'] / 2
            nipple_y = nipple_data['y'] + nipple_data['height'] / 2
            
            return np.array([nipple_x, nipple_y])
        except Exception as e:
            return None
    
    def create_visualization(self, original_image, pred_nipple_512, gt_nipple, 
                           transformation_info, laterality, distance_mm, error_mm,
                           sop_uid, output_path):
        """Create CC visualization on original DICOM"""
        try:
            pred_nipple_original = self.preprocessor.transform_coords_to_original(
                pred_nipple_512,
                transformation_info['original_shape'],
                transformation_info['crop_coords'],
                transformation_info['pad_coords']
            )
            
            original_width = transformation_info['original_shape'][1]
            if laterality == "L":
                edge_point = np.array([0, pred_nipple_original[1]])
            else:
                edge_point = np.array([original_width, pred_nipple_original[1]])
            
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(original_image, cmap='gray')
            
            if gt_nipple is not None:
                ax.plot(gt_nipple[0], gt_nipple[1], 'bo', markersize=12, 
                       label='GT Nipple', markeredgecolor='white', markeredgewidth=2)
                ax.plot([pred_nipple_original[0], gt_nipple[0]], 
                       [pred_nipple_original[1], gt_nipple[1]],
                       'y-', linewidth=2, alpha=0.8, label='Prediction Error')
            
            ax.plot(pred_nipple_original[0], pred_nipple_original[1], 'go', markersize=12, 
                   label='Pred Nipple', markeredgecolor='white', markeredgewidth=2)
            
            ax.plot([pred_nipple_original[0], edge_point[0]], 
                   [pred_nipple_original[1], edge_point[1]],
                   'r--', linewidth=2, label=f'Distance to Edge')
            
            mid_x = (pred_nipple_original[0] + edge_point[0]) / 2
            mid_y = (pred_nipple_original[1] + edge_point[1]) / 2
            ax.annotate(f"{distance_mm:.1f}mm", xy=(mid_x, mid_y), xytext=(15, 15), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
            if error_mm is not None:
                ax.set_title(f'CC ({laterality}) - {sop_uid[:8]}... | Distance: {distance_mm:.2f}mm | Error: {error_mm:.2f}mm', fontsize=12)
            else:
                ax.set_title(f'CC ({laterality}) - {sop_uid[:8]}... | Distance: {distance_mm:.2f}mm', fontsize=12)
            
            ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.axis('off')
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(output_path)
        except Exception as e:
            print(f"CC visualization error: {e}")
            return None




class CombinedQualityEvaluator:
    """Combined MLO-CC quality evaluation with 10mm rule"""
    
    def __init__(self, paths: PathConfig, mlo_results: list, cc_results: list):
        self.paths = paths
        self.mlo_results = mlo_results
        self.cc_results = cc_results
        self.combined_results = []
        self.threshold_mm = 10.0
    
    def match_pairs(self):
        """Match MLO and CC pairs by patient_id and laterality"""
        print("\nMatching MLO-CC pairs...")
        
        mlo_dict = {}
        for mlo in self.mlo_results:
            key = f"{mlo['patient_id']}_{mlo['laterality']}"
            mlo_dict[key] = mlo
        
        matched_pairs = []
        for cc in self.cc_results:
            key = f"{cc['patient_id']}_{cc['laterality']}"
            
            if key in mlo_dict:
                mlo = mlo_dict[key]
                
                mlo_distance = mlo['distance_mm']
                cc_distance = cc['distance_mm']
                distance_diff = abs(mlo_distance - cc_distance)
                
                predicted_quality = "good" if distance_diff <= self.threshold_mm else "bad"
                gt_quality = str(cc.get('cc_quality', 'unknown')).lower()
                
                pair = {
                    'patient_id': cc['patient_id'],
                    'laterality': cc['laterality'],
                    'mlo_sop': mlo['sop_uid'],
                    'cc_sop': cc['cc_sop'],
                    'mlo_distance_mm': mlo_distance,
                    'cc_distance_mm': cc_distance,
                    'distance_diff_mm': distance_diff,
                    'predicted_quality': predicted_quality,
                    'gt_quality': gt_quality,
                    'mlo_viz_path': mlo.get('visualization_path', ''),
                    'cc_viz_path': cc.get('visualization_path', ''),
                    'correct_prediction': predicted_quality == gt_quality
                }
                matched_pairs.append(pair)
        
        self.combined_results = matched_pairs
        print(f"Matched pairs: {len(matched_pairs)}")
        return matched_pairs
    
    def calculate_metrics(self):
        """Calculate classification metrics"""
        if not self.combined_results:
            return None
        
        y_pred = [r['predicted_quality'] for r in self.combined_results]
        y_true = [r['gt_quality'] for r in self.combined_results]
        
        y_pred_encoded = [1 if p == 'good' else 0 for p in y_pred]
        y_true_encoded = [1 if t == 'good' else 0 for t in y_true]
        
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_encoded, y_pred_encoded, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'total_pairs': len(self.combined_results),
            'correct_predictions': sum(r['correct_prediction'] for r in self.combined_results)
        }
    
    def create_visualizations(self, metrics):
        """Create all visualizations"""
        print("\nCreating visualizations...")
        
        self._create_confusion_matrix(metrics['confusion_matrix'])
        self._create_distance_correlation()
        self._create_difference_histogram()
        self._create_paired_visualizations()
    
    def _create_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        plt.xlabel('Predicted Quality')
        plt.ylabel('Ground Truth Quality')
        plt.title('Quality Classification - Confusion Matrix')
        plt.savefig(self.paths.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_distance_correlation(self):
        mlo_distances = [r['mlo_distance_mm'] for r in self.combined_results]
        cc_distances = [r['cc_distance_mm'] for r in self.combined_results]
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if not r['correct_prediction'] else 'blue' for r in self.combined_results]
        plt.scatter(mlo_distances, cc_distances, c=colors, alpha=0.6)
        
        min_dist = min(min(mlo_distances), min(cc_distances))
        max_dist = max(max(mlo_distances), max(cc_distances))
        plt.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', alpha=0.5, label='Perfect Correlation')
        
        x_range = np.linspace(min_dist, max_dist, 100)
        plt.fill_between(x_range, x_range - 10, x_range + 10, alpha=0.2, color='green', 
                        label='+/- 10mm Good Zone')
        
        plt.xlabel('MLO Distance (mm)')
        plt.ylabel('CC Distance (mm)')
        plt.title('MLO vs CC Distance Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        correlation = np.corrcoef(mlo_distances, cc_distances)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.savefig(self.paths.output_dir / 'distance_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_difference_histogram(self):
        differences = [r['distance_diff_mm'] for r in self.combined_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10mm Threshold')
        
        plt.xlabel('Distance Difference (mm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of MLO-CC Distance Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        plt.text(0.7, 0.8, f'Mean: {mean_diff:.2f}mm\nMedian: {median_diff:.2f}mm', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.savefig(self.paths.output_dir / 'difference_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_paired_visualizations(self):
        print(f"Creating paired visualizations: {len(self.combined_results)} pairs")
        
        for i, result in enumerate(self.combined_results):
            try:
                mlo_path = result.get('mlo_viz_path', '')
                cc_path = result.get('cc_viz_path', '')
                
                if not mlo_path or not cc_path:
                    continue
                if not os.path.exists(mlo_path) or not os.path.exists(cc_path):
                    continue
                
                mlo_img = Image.open(mlo_path)
                cc_img = Image.open(cc_path)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                ax1.imshow(mlo_img)
                ax1.set_title(f'MLO ({result["laterality"]})\nDistance: {result["mlo_distance_mm"]:.1f}mm')
                ax1.axis('off')
                
                ax2.imshow(cc_img)
                ax2.set_title(f'CC ({result["laterality"]})\nDistance: {result["cc_distance_mm"]:.1f}mm')
                ax2.axis('off')
                
                color = 'green' if result['correct_prediction'] else 'red'
                fig.suptitle(
                    f'Predicted: {result["predicted_quality"].upper()} | '
                    f'GT: {result["gt_quality"].upper()} | '
                    f'Diff: {result["distance_diff_mm"]:.1f}mm',
                    fontsize=14, fontweight='bold', color=color
                )
                
                filename = f"pair_{i+1:03d}_{result['patient_id'][:8]}_{result['laterality']}.png"
                plt.savefig(self.paths.combined_viz_dir / filename, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                
                if (i + 1) % 20 == 0:
                    print(f"  Created {i+1}/{len(self.combined_results)} paired visualizations")
                    
            except Exception as e:
                print(f"  Pair {i+1} error: {e}")
    
    def generate_report(self, metrics):
        print(f"\n{'='*60}")
        print("COMBINED MLO-CC QUALITY EVALUATION RESULTS")
        print(f"{'='*60}")
        
        print(f"Total pairs evaluated: {metrics['total_pairs']}")
        print(f"Correct predictions: {metrics['correct_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        
        differences = [r['distance_diff_mm'] for r in self.combined_results]
        print(f"\nDistance Difference Statistics:")
        print(f"  Mean: {np.mean(differences):.2f}mm")
        print(f"  Median: {np.median(differences):.2f}mm")
        print(f"  Std: {np.std(differences):.2f}mm")
        
        with open(self.paths.output_dir / 'combined_results.json', 'w') as f:
            json.dump(self.combined_results, f, indent=2)
        
        print(f"\nOutput files saved to: {self.paths.output_dir}")
        print("EVALUATION COMPLETE!")



def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("MLO-CC QUALITY EVALUATION PIPELINE")
    print("="*60)
    
    paths = PathConfig()
    if not paths.verify_paths():
        print("\nPlease fix path errors before continuing.")
        return
    
    paths.setup_directories()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading labels...")
    mlo_labels = pd.read_csv(paths.mlo_labels_path)
    cc_labels = pd.read_csv(paths.cc_labels_path)
    metadata = pd.read_csv(paths.metadata_path)
    print(f"MLO labels: {len(mlo_labels)}, CC labels: {len(cc_labels)}, Metadata: {len(metadata)}")
    
    mlo_evaluator = MLOEvaluator(paths, device)
    cc_evaluator = CCEvaluator(paths, device)
    preprocessor = ImagePreprocessor()
    
    print("\nBuilding DICOM indexes...")
    mlo_dicom_index = preprocessor.build_dicom_index(paths.mlo_images_dir)
    cc_dicom_index = preprocessor.build_dicom_index(paths.cc_images_dir)
    print(f"MLO DICOM files indexed: {len(mlo_dicom_index)}")
    print(f"CC DICOM files indexed: {len(cc_dicom_index)}")
    
    test_mlo = mlo_labels[
        (mlo_labels['Split'] == 'Test') & 
        (mlo_labels['qualitativeLabel'] == 'Good')
    ].drop_duplicates(subset=['SOPInstanceUID'])
    print(f"\nTest MLO (Good quality): {len(test_mlo)}")
    
    # Process MLO images
    print("\n" + "="*60)
    print("STEP 1: PROCESSING MLO IMAGES")
    print("="*60)
    
    mlo_results = []
    for idx, row in test_mlo.iterrows():
        try:
            sop_uid = row['SOPInstanceUID']
            patient_id = row['StudyInstanceUID']
            
            mlo_meta = metadata[metadata['SOP Instance UID'] == sop_uid]
            if len(mlo_meta) == 0:
                continue
            laterality = mlo_meta.iloc[0]['Image Laterality']
            
            dicom_path = preprocessor.get_dicom_path_for_sop(paths.mlo_images_dir, sop_uid)
            if not dicom_path:
                continue
            
            processed_image, original_shape, dicom_data, transform_info = preprocessor.process(dicom_path)
            
            pred_landmarks = mlo_evaluator.predict_landmarks(processed_image)
            if pred_landmarks is None:
                continue
            
            pixel_spacing = preprocessor.extract_pixel_spacing(dicom_data)
            scaled_spacing, _ = preprocessor.calculate_scaled_pixel_spacing(pixel_spacing, transform_info)
            distance_mm, distance_px, intersection = mlo_evaluator.calculate_distance(pred_landmarks, scaled_spacing)
            
            gt_landmarks = mlo_evaluator.parse_ground_truth(sop_uid, mlo_labels)
            
            original_image, _ = preprocessor.load_dicom(dicom_path)
            viz_path = mlo_evaluator.create_visualization(
                original_image, pred_landmarks, gt_landmarks, transform_info,
                distance_mm, sop_uid, paths.mlo_viz_dir / f"mlo_{sop_uid[:8]}.png"
            )
            
            result = {
                'patient_id': patient_id,
                'sop_uid': sop_uid,
                'laterality': laterality,
                'distance_mm': float(distance_mm),
                'visualization_path': viz_path
            }
            mlo_results.append(result)
            
            if len(mlo_results) % 20 == 0:
                print(f"  Processed {len(mlo_results)} MLO images")
                
        except Exception as e:
            print(f"  MLO error ({sop_uid[:8] if 'sop_uid' in dir() else 'unknown'}): {e}")
    
    print(f"MLO processing complete: {len(mlo_results)} images")
    
    # Process CC images
    print("\n" + "="*60)
    print("STEP 2: PROCESSING CC IMAGES")
    print("="*60)
    
    cc_results = []
    for mlo in mlo_results:
        try:
            patient_id = mlo['patient_id']
            laterality = mlo['laterality']
            
            patient_cc_sops = cc_labels[cc_labels['StudyInstanceUID'] == patient_id]['SOPInstanceUID']
            cc_meta = metadata[
                (metadata['SOP Instance UID'].isin(patient_cc_sops)) & 
                (metadata['Image Laterality'] == laterality)
            ]
            
            if len(cc_meta) == 0:
                continue
            
            cc_sop = cc_meta.iloc[0]['SOP Instance UID']
            cc_row = cc_labels[cc_labels['SOPInstanceUID'] == cc_sop].iloc[0]
            cc_quality = cc_row.get('qualitativeLabel', 'unknown')
            
            dicom_path = preprocessor.get_dicom_path_for_sop(paths.cc_images_dir, cc_sop)
            if not dicom_path:
                continue
            
            processed_image, original_shape, dicom_data, transform_info = preprocessor.process(dicom_path)
            
            pred_nipple = cc_evaluator.predict_landmarks(processed_image)
            if pred_nipple is None:
                continue
            
            pixel_spacing = preprocessor.extract_pixel_spacing(dicom_data)
            scaled_spacing, _ = preprocessor.calculate_scaled_pixel_spacing(pixel_spacing, transform_info)
            direction, distance_mm, distance_px = cc_evaluator.calculate_distance(pred_nipple, laterality, scaled_spacing)
            
            gt_nipple = cc_evaluator.parse_ground_truth(cc_sop, cc_labels)
            
            error_mm = None
            if gt_nipple is not None:
                gt_nipple_512 = preprocessor.transform_coords_to_512(gt_nipple, transform_info)
                error_mm = float(np.linalg.norm(pred_nipple - gt_nipple_512) * scaled_spacing)
            
            original_image, _ = preprocessor.load_dicom(dicom_path)
            viz_path = cc_evaluator.create_visualization(
                original_image, pred_nipple, gt_nipple, transform_info,
                laterality, distance_mm, error_mm, cc_sop,
                paths.cc_viz_dir / f"cc_{cc_sop[:8]}.png"
            )
            
            result = {
                'patient_id': patient_id,
                'cc_sop': cc_sop,
                'laterality': laterality,
                'cc_quality': cc_quality,
                'distance_mm': float(distance_mm),
                'error_mm': error_mm,
                'visualization_path': viz_path
            }
            cc_results.append(result)
            
            if len(cc_results) % 20 == 0:
                print(f"  Processed {len(cc_results)} CC images")
                
        except Exception as e:
            print(f"  CC error: {e}")
    
    print(f"CC processing complete: {len(cc_results)} images")
    
    # Combined evaluation
    print("\n" + "="*60)
    print("STEP 3: COMBINED QUALITY EVALUATION")
    print("="*60)
    
    combined_evaluator = CombinedQualityEvaluator(paths, mlo_results, cc_results)
    combined_evaluator.match_pairs()
    metrics = combined_evaluator.calculate_metrics()
    
    if metrics:
        combined_evaluator.create_visualizations(metrics)
        combined_evaluator.generate_report(metrics)
    else:
        print("No metrics calculated - check matching pairs")
    
    with open(paths.output_dir / 'mlo_results.json', 'w') as f:
        json.dump(mlo_results, f, indent=2)
    with open(paths.output_dir / 'cc_results.json', 'w') as f:
        json.dump(cc_results, f, indent=2)
    
    print("\nAll results saved!")


if __name__ == "__main__":
    main()

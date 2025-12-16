import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import numpy as np
from PIL import Image
import json 
import os
import matplotlib.pyplot as plt
from PIL import ImageDraw
import csv

def read_dicom_image(path):
    """Load and return a DICOM image as a PIL Image."""
    dicom = pydicom.dcmread(path)
    image = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = np.max(image) - image
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def load_csv_data(csv_path):
    """Load and return CSV data."""
    return pd.read_csv(csv_path)

def extract_bbox_data(data_str):
    """Extract and return bbox data from a CSV row data string."""
    data = eval(data_str)
    return data

def is_cc_series(series_description):
    """Check if the series description indicates a CC view."""
    return 'CC' in series_description.upper() if series_description else False

def midpoint_from_bbox(bbox):
    """Calculate and return the midpoint of a bounding box."""
    midpoint = (bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2)
    return midpoint

def save_nipple_landmark_as_json(nipple_coord, path):
    """
    Save the nipple coordinate to a JSON file.
    Parameters:
        nipple_coord (tuple): The (x, y) coordinates of the nipple.
        path (str): File path where the JSON will be saved.
    """
    data = {
        "nipple": {
            "x": nipple_coord[0],
            "y": nipple_coord[1]
        }
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

def verify_cc_landmarks(dicom_paths, landmark_paths, fig_save_path, sop_instance_uids):
    """Visualize and verify nipple landmarks on CC DICOM images with red dots for nipple points."""
    for dicom_path, landmark_path, sop_instance_uid in zip(dicom_paths, landmark_paths, sop_instance_uids):
        # Load DICOM image as PIL Image
        dicom_image = read_dicom_image(dicom_path)

        # Convert the image to RGB to support colored drawings
        dicom_image_rgb = dicom_image.convert("RGB")

        # Load landmark data from JSON
        with open(landmark_path, 'r') as json_file:
            landmarks = json.load(json_file)
        
        # Draw landmarks on the RGB image
        draw = ImageDraw.Draw(dicom_image_rgb)
        nipple = landmarks['nipple']

        # Uniform dot size for nipple point
        dot_radius = 10 

        # Use red color for high visibility
        dot_color = 'red'
        
        # Draw a red dot for the nipple
        draw.ellipse([(nipple['x'] - dot_radius, nipple['y'] - dot_radius), 
                      (nipple['x'] + dot_radius, nipple['y'] + dot_radius)], fill=dot_color)

        # Save the annotated RGB image
        save_fig_path = os.path.join(fig_save_path, f"CC_{sop_instance_uid}_annotated.png")
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        dicom_image_rgb.save(save_fig_path)
        print(f"Saved CC annotated image for {sop_instance_uid} at {save_fig_path}")
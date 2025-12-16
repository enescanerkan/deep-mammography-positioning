"""
CC Landmark Generation Script
=============================
Generates nipple landmark JSON files from CC mammography labels.
"""

from utils import (read_dicom_image, load_csv_data, extract_bbox_data,
                   midpoint_from_bbox, save_nipple_landmark_as_json, is_cc_series)
import os
import pandas as pd
from tqdm import tqdm

def process_cc_images(csv_path, dicom_base_path, landmarks_save_path):
    """
    Process CC images and save nipple landmarks as JSON files.
    
    Args:
        csv_path: Path to CC labels CSV
        dicom_base_path: Directory containing raw DICOM files
        landmarks_save_path: Output directory for landmark JSONs
    """
    df = load_csv_data(csv_path)
    
    # Filter CC images and Nipple labels only
    cc_df = df[df['SeriesDescription'].str.contains('CC', case=False, na=False)]
    nipple_df = cc_df[cc_df['labelName'] == 'Nipple']
    
    print(f"Found {len(nipple_df)} CC nipple labels")
    
    processed_count = 0
    skipped_count = 0

    for _, row in tqdm(nipple_df.iterrows(), total=nipple_df.shape[0], desc="Processing CC Images"):
        # Try .dicom first, fallback to .dcm
        dicom_path = os.path.join(dicom_base_path, f"{row['SOPInstanceUID']}.dicom")
        if not os.path.exists(dicom_path):
            dicom_path = os.path.join(dicom_base_path, f"{row['SOPInstanceUID']}.dcm")
        
        json_path = os.path.join(landmarks_save_path, f"{row['SOPInstanceUID']}.json")

        # Check if DICOM file exists
        if not os.path.exists(dicom_path):
            print(f"DICOM file not found: {dicom_path}")
            skipped_count += 1
            continue

        try:
            # Load image and get dimensions
            image = read_dicom_image(dicom_path)
            image_width, image_height = image.size

            # Extract bounding box data
            bbox_data = extract_bbox_data(row['data'])
            nipple_point = midpoint_from_bbox(bbox_data)

            # Save as JSON
            save_nipple_landmark_as_json(nipple_point, json_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {row['SOPInstanceUID']}: {str(e)}")
            skipped_count += 1
            continue

    print(f"\nProcessing completed!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")

def main():
    # Dynamic path resolution
    # preprocessing/ -> regression/ -> code/ -> cc-landmark-detection/ -> rule-based-model/ -> project_root/
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).resolve().parent        # preprocessing/
    CC_ROOT = SCRIPT_DIR.parent.parent.parent           # cc-landmark-detection/
    RULE_BASED_ROOT = CC_ROOT.parent                    # rule-based-model/
    PROJECT_ROOT = RULE_BASED_ROOT.parent               # deep-mammography-positioning/
    
    # Use centralized data structure
    csv_path = str(PROJECT_ROOT / 'labels' / 'cc_labels.csv')
    dicom_base_path = str(PROJECT_ROOT / 'data' / 'raw' / 'cc')   # Raw DICOM files
    landmarks_save_path = str(CC_ROOT / 'landmark_coords')
    
    print("=" * 60)
    print("CC NIPPLE LANDMARK GENERATION")
    print("=" * 60)
    print(f"Labels CSV : {csv_path}")
    print(f"DICOM Dir  : {dicom_base_path}")
    print(f"Output     : {landmarks_save_path}")
    print("=" * 60)
    
    process_cc_images(csv_path, dicom_base_path, landmarks_save_path)
    print("[OK] CC Landmark Generation completed!")

if __name__ == "__main__":
    main()

from utils import (read_dicom_image, load_csv_data,
                    extract_line_data, midpoint_from_bbox,
                      save_landmarks_as_json, reorder_landmarks,
                      adjust_pectoralis_line, determine_side)
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Dynamic path resolution
# preprocessing/ -> regression/ -> code/ -> mlo-landmark-detection/ -> rule-based-model/ -> project_root/
SCRIPT_DIR = Path(__file__).resolve().parent            # preprocessing/
MLO_ROOT = SCRIPT_DIR.parent.parent.parent              # mlo-landmark-detection/
RULE_BASED_ROOT = MLO_ROOT.parent                       # rule-based-model/
PROJECT_ROOT = RULE_BASED_ROOT.parent                   # deep-mammography-positioning/

def process_images(csv_path, dicom_base_path, landmarks_save_path):
    df = load_csv_data(csv_path)
    
    pectoralis_df = df[df['labelName'] == 'Pectoralis']
    nipple_df = df[df['labelName'] == 'Nipple']
    
    print(f"Total Pectoralis labels: {len(pectoralis_df)}")
    print(f"Total Nipple labels: {len(nipple_df)}")

    processed_count = 0
    for _, row in tqdm(pectoralis_df.iterrows(), total=pectoralis_df.shape[0], desc="Generating MLO Landmarks"):
        # Try .dicom first, fallback to .dcm
        dicom_path = os.path.join(dicom_base_path, f"{row['SOPInstanceUID']}.dicom")
        if not os.path.exists(dicom_path):
            dicom_path = os.path.join(dicom_base_path, f"{row['SOPInstanceUID']}.dcm")
        
        json_path = os.path.join(landmarks_save_path, f"{row['SOPInstanceUID']}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        if row['SOPInstanceUID'] not in nipple_df['SOPInstanceUID'].values:
            continue
        
        if not os.path.exists(dicom_path):
            print(f"DICOM not found: {dicom_path}")
            continue

        image = read_dicom_image(dicom_path)
        image_width, image_height = image.size

        vertices = extract_line_data(row)
        pectoralis_line = reorder_landmarks([(float(vertices[0][0]), float(vertices[0][1])), (float(vertices[1][0]), float(vertices[1][1]))])
        side = determine_side(row['SeriesDescription'])

        adjusted_pectoralis_line = adjust_pectoralis_line(pectoralis_line, image_width, image_height, side)

        nipple_record = nipple_df[nipple_df['SOPInstanceUID'] == row['SOPInstanceUID']].iloc[0]
        bbox_data = eval(nipple_record['data'])
        nipple_point = midpoint_from_bbox(bbox_data)

        save_landmarks_as_json(adjusted_pectoralis_line, nipple_point, json_path)
        processed_count += 1

    print(f"\n[OK] Completed! {processed_count} landmarks generated.")

if __name__ == "__main__":
    # Use centralized data structure
    csv_path = PROJECT_ROOT / 'labels' / 'mlo_labels.csv'
    dicom_base_path = PROJECT_ROOT / 'data' / 'raw' / 'mlo'
    landmarks_save_path = MLO_ROOT / 'landmark_coords'
    
    print("=" * 60)
    print("MLO LANDMARK GENERATION")
    print("=" * 60)
    print(f"Labels CSV : {csv_path}")
    print(f"DICOM Dir  : {dicom_base_path}")
    print(f"Output     : {landmarks_save_path}")
    print("=" * 60)
    
    process_images(str(csv_path), str(dicom_base_path), str(landmarks_save_path))
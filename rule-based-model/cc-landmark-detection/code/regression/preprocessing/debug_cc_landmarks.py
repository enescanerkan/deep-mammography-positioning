from utils import verify_cc_landmarks, read_dicom_image
import os
import pandas as pd
import json

def debug_cc_landmarks():
    """Visualize and verify nipple landmarks in CC images."""
    
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    csv_path = os.path.join(base_dir, 'labels', 'positioning_labels_cc_generated_matched.csv')
    dicom_base_path = os.path.join(base_dir, 'data', 'cc_nipple_dicom_images')
    landmarks_base_path = os.path.join(base_dir, 'landmark_coords')
    debug_output_path = os.path.join(base_dir, 'debug_output')
    
    # Create debug output directory
    os.makedirs(debug_output_path, exist_ok=True)
    
    # Load CSV and select sample images
    df = pd.read_csv(csv_path)
    cc_df = df[df['SeriesDescription'].str.contains('CC', case=False, na=False)]
    sample_df = cc_df.head(50)  # First 50 samples
    
    print(f"Selected {len(sample_df)} CC images for debugging")
    
    dicom_paths = []
    landmark_paths = []
    sop_uids = []
    
    for _, row in sample_df.iterrows():
        sop_uid = row['SOPInstanceUID']
        dicom_path = os.path.join(dicom_base_path, f"{sop_uid}.dicom")
        landmark_path = os.path.join(landmarks_base_path, f"{sop_uid}.json")
        
        # Check if files exist
        if os.path.exists(dicom_path) and os.path.exists(landmark_path):
            dicom_paths.append(dicom_path)
            landmark_paths.append(landmark_path)
            sop_uids.append(sop_uid)
            
            # Print landmark data
            with open(landmark_path, 'r') as f:
                landmark_data = json.load(f)
            print(f"SOPInstanceUID: {sop_uid}")
            print(f"  Nipple: x={landmark_data['nipple']['x']:.1f}, y={landmark_data['nipple']['y']:.1f}")
            print(f"  Series: {row['SeriesDescription']}")
            print()
        else:
            print(f"Files not found: {sop_uid}")
            if not os.path.exists(dicom_path):
                print(f"  DICOM missing: {dicom_path}")
            if not os.path.exists(landmark_path):
                print(f"  Landmark missing: {landmark_path}")
    
    if dicom_paths:
        print("Starting visualization...")
        verify_cc_landmarks(dicom_paths, landmark_paths, debug_output_path, sop_uids)
        print(f"Debug images saved to: {debug_output_path}")
    else:
        print("Uygun dosya bulunamadı!")

def check_coordinate_ranges():
    """CC görüntülerindeki nipple koordinat aralıklarını kontrol et."""
    
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    landmarks_base_path = os.path.join(base_dir, 'landmark_coords')
    csv_path = os.path.join(base_dir, 'labels', 'positioning_labels_cc_generated_matched.csv')
    
    df = pd.read_csv(csv_path)
    cc_df = df[df['SeriesDescription'].str.contains('CC', case=False, na=False)]
    
    x_coords = []
    y_coords = []
    
    for _, row in cc_df.iterrows():
        sop_uid = row['SOPInstanceUID']
        landmark_path = os.path.join(landmarks_base_path, f"{sop_uid}.json")
        
        if os.path.exists(landmark_path):
            with open(landmark_path, 'r') as f:
                landmark_data = json.load(f)
            
            x_coords.append(landmark_data['nipple']['x'])
            y_coords.append(landmark_data['nipple']['y'])
    
    if x_coords and y_coords:
        print(f"\nCC Nipple Koordinat İstatistikleri ({len(x_coords)} örnek):")
        print(f"X koordinatları - Min: {min(x_coords):.1f}, Max: {max(x_coords):.1f}, Ortalama: {sum(x_coords)/len(x_coords):.1f}")
        print(f"Y koordinatları - Min: {min(y_coords):.1f}, Max: {max(y_coords):.1f}, Ortalama: {sum(y_coords)/len(y_coords):.1f}")
    else:
        print("Koordinat verisi bulunamadı!")

if __name__ == "__main__":
    print("CC Landmarks Debug Script")
    print("=" * 40)
    
    print("1. Koordinat aralıklarını kontrol ediliyor...")
    check_coordinate_ranges()
    
    print("\n2. Örnek görüntüler görselleştiriliyor...")
    debug_cc_landmarks() 
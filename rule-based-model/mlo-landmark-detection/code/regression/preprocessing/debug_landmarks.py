# debug_landmarks.py

from utils import verify_landmarks
import os
import pandas as pd
from pathlib import Path

# Dynamic path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
MLO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
PROJECT_ROOT = MLO_ROOT.parent.parent

# Configure paths
unified_df = pd.read_csv(PROJECT_ROOT / 'labels' / 'mlo_labels.csv')
dicom_base_path = str(MLO_ROOT / 'data' / 'mlo')
landmark_save_path = str(MLO_ROOT / 'landmark_coords')
fig_save_path = str(MLO_ROOT / 'figures' / 'debug_landmarks')

sop_instance_uids = [
    'ac0f58c8690f16fe4dcd300bb3e7ee0b',
    '3c5b8c9aa17dd132db42747ad3d4f10e',
    '47c8858666bcce92bcbd57974b5ce522',
    'b664cf1e7c968896144a3a2005cd3eb4',
    '552a67df6d68b8f9498db7cadd23b0e0',
    'e7ad5afd2fb520aca0409c734315e385',
    '3944b33b99c98e763763b9da54b4f41c',
    '20ff52d0631b27a5387abcd5920b3e5d',
    '0e31855d02eadf8670ffaeeaeddbf229',
    'f9e67324d12f697ff0c00f327fc4fca9',
    'e42cba584142f92dc42ef3fb95b5878d',
    'cfba6992d0d0da02d516039af80a0549',
    'c72f340f53c8a8acd3566cf239af8463'
]

dicom_paths = []
landmark_paths = []

for sop_instance_uid in sop_instance_uids:
    # Build DICOM file path
    dicom_path = os.path.join(dicom_base_path, f"{sop_instance_uid}.dcm")
    dicom_paths.append(dicom_path)
    
    landmark_path = os.path.join(landmark_save_path, f"{sop_instance_uid}.json")
    landmark_paths.append(landmark_path)

print("Verifying landmarks...")
verify_landmarks(dicom_paths, landmark_paths, fig_save_path, sop_instance_uids)

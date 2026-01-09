import torch
import nibabel as nib
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.umamba_3d import create_umamba_bot_3d
from utils.dataset_3d import NIfTISegmentationDataset3D

def save_sample_prediction(model_path, data_dir, output_name="debug_result.nii.gz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load 1 sample
    img_dir = os.path.join(data_dir, "val/images")
    lbl_dir = os.path.join(data_dir, "val/labels")
    
    dataset = NIfTISegmentationDataset3D(img_dir, lbl_dir, patch_size=(128, 128, 128))
    img_tensor, lbl_tensor = dataset[0] # Get first sample
    
    # Load Model
    model = create_umamba_bot_3d(input_channels=1, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    
    # Inference
    with torch.no_grad():
        input_data = img_tensor.unsqueeze(0).to(device)
        output = model(input_data)
        if isinstance(output, list): output = output[0]
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Prepare data for saving
    # We save Image, Ground Truth, and Prediction in one 4D file or separate
    img_raw = img_tensor.numpy()[0]
    gt_raw = lbl_tensor.numpy()
    
    # Create NIfTI header (using identity affine for simplicity in debug)
    affine = np.eye(4)
    
    # Save Prediction
    res_nii = nib.Nifti1Image(pred.astype(np.uint8), affine)
    nib.save(res_nii, f"pred_{output_name}")
    
    # Save Image (for reference)
    img_nii = nib.Nifti1Image(img_raw, affine)
    nib.save(img_nii, f"img_{output_name}")
    
    # Save GT
    gt_nii = nib.Nifti1Image(gt_raw.astype(np.uint8), affine)
    nib.save(gt_nii, f"gt_{output_name}")

    print(f"\n✅ Results saved!")
    print(f"1. pred_{output_name} (Prediction)")
    print(f"2. img_{output_name} (Source Image)")
    print(f"3. gt_{output_name} (Manual Label)")
    print("Open these in ITK-SNAP to compare.")

if __name__ == "__main__":
    save_sample_prediction(
        model_path="test_epoch.pth",
        data_dir="../dataset/organized_data"
    )

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import glob

class SegmentationDataset(Dataset):
    """
    A simple dataset for medical image segmentation (NIfTI format).
    Assumes structure:
    images/
        case_001.nii.gz
        case_002.nii.gz
    labels/
        case_001.nii.gz
        case_002.nii.gz
    """
    def __init__(self, image_dir, label_dir, transform=None, target_size=(128, 128)):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        lbl_path = self.label_files[idx]

        # Load NIfTI files
        img = nib.load(img_path).get_fdata()
        lbl = nib.load(lbl_path).get_fdata()

        # Simple normalization
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        
        # Convert to tensor and add channel dim
        # Assuming 2D slices for now, or 3D volumes
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if len(img.shape) == 2:
            img = img.unsqueeze(0) # (1, H, W)
        elif len(img.shape) == 3:
            img = img.unsqueeze(0) # (1, D, H, W)

        return img, lbl

def get_dataloader(image_dir, label_dir, batch_size=2, shuffle=True):
    dataset = SegmentationDataset(image_dir, label_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

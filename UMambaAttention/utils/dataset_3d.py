"""
Dataset utilities for medical image segmentation (NIfTI format)
Supports both 2D slices and 3D volumes
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import glob
from typing import Tuple, Optional, Callable


class NIfTISegmentationDataset3D(Dataset):
    """
    3D Medical Image Segmentation Dataset for NIfTI files (.nii.gz)
    
    Expected folder structure:
    data_root/
        images/
            case_001_0000.nii.gz  # _0000 is modality suffix (e.g., CT, T1, T2)
            case_002_0000.nii.gz
        labels/
            case_001.nii.gz
            case_002.nii.gz
    """
    def __init__(self, image_dir, label_dir, patch_size=(128, 128, 128),
                 transform=None, normalize=True, modality_suffix='_0000', is_train=False, label_mapping=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.transform = transform
        self.normalize = normalize
        self.modality_suffix = modality_suffix
        self.is_train = is_train
        self.label_mapping = label_mapping
        
        # Find all image files (.nii or .nii.gz)
        self.image_files = []
        for ext in ['.nii', '.nii.gz']:
            self.image_files.extend(glob.glob(os.path.join(image_dir, f"*{modality_suffix}{ext}")))
        self.image_files = sorted(self.image_files)
        
        # Create corresponding label file paths
        self.label_files = []
        for img_path in self.image_files:
            # Flexible basename extraction for both .nii and .nii.gz
            fname = os.path.basename(img_path)
            if fname.endswith('.nii.gz'):
                core_name = fname.replace(modality_suffix + '.nii.gz', '')
            else:
                core_name = fname.replace(modality_suffix + '.nii', '')
            
            # Check for label with or without .gz
            label_path = os.path.join(label_dir, f"{core_name}.nii")
            if not os.path.exists(label_path):
                label_path += ".gz"
            self.label_files.append(label_path)
        
        print(f"[Dataset] Found {len(self.image_files)} image-label pairs")
        if len(self.image_files) > 0:
            print(f"[Dataset] Sample: {os.path.basename(self.image_files[0])}")

    def __len__(self):
        return len(self.image_files)

    def normalize_image(self, img):
        """Z-score normalization"""
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)

    def crop_or_pad_to_size(self, img, target_size):
        """Center crop or pad image to target size"""
        current_size = img.shape
        
        # Calculate crop/pad for each dimension
        new_img = img
        for i in range(3):
            if current_size[i] > target_size[i]:
                # Crop
                start = (current_size[i] - target_size[i]) // 2
                sl = [slice(None)] * 3
                sl[i] = slice(start, start + target_size[i])
                new_img = new_img[tuple(sl)]
            elif current_size[i] < target_size[i]:
                # Pad
                pad_width = [(0, 0)] * 3
                total_pad = target_size[i] - current_size[i]
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width[i] = (pad_before, pad_after)
                new_img = np.pad(new_img, pad_width, mode='constant', constant_values=0)
        
        return new_img

    def random_crop_or_pad(self, img, lbl, target_size):
        """Random crop or pad image and label to target size"""
        current_size = img.shape
        
        # Determine crop/pad for each dimension
        sl_img = [slice(None)] * 3
        sl_lbl = [slice(None)] * 3
        pad_width = [(0, 0)] * 3
        need_pad = False
        
        for i in range(3):
            if current_size[i] > target_size[i]:
                # Random Crop
                diff = current_size[i] - target_size[i]
                start = np.random.randint(0, diff + 1)
                sl_img[i] = slice(start, start + target_size[i])
                sl_lbl[i] = slice(start, start + target_size[i])
            elif current_size[i] < target_size[i]:
                # Pad
                total_pad = target_size[i] - current_size[i]
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width[i] = (pad_before, pad_after)
                need_pad = True
                
        # Apply crop
        new_img = img[tuple(sl_img)]
        new_lbl = lbl[tuple(sl_lbl)]
        
        # Apply pad if needed
        if need_pad:
            new_img = np.pad(new_img, pad_width, mode='constant', constant_values=0)
            new_lbl = np.pad(new_lbl, pad_width, mode='constant', constant_values=0)
            
        return new_img, new_lbl

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            lbl_path = self.label_files[idx]

            # Load NIfTI files
            img_nii = nib.load(img_path)
            lbl_nii = nib.load(lbl_path)

            # Force both to canonical orientation (RAS+)
            img_nii = nib.as_closest_canonical(img_nii)
            lbl_nii = nib.as_closest_canonical(lbl_nii)

            img = img_nii.get_fdata().astype(np.float32)
            lbl = lbl_nii.get_fdata().astype(np.int64)

            # IMPORTANT: Nibabel loads as (H, W, D), but PyTorch 3D models expect (D, H, W)
            # Transpose to (D, H, W)
            img = np.transpose(img, (2, 0, 1))
            lbl = np.transpose(lbl, (2, 0, 1))

            # Keep original labels (do not binarize)
            # lbl = (lbl > 0).astype(np.int64)

            # Handle extreme case: if shapes still don't match, check if they are just permuted
            if img.shape != lbl.shape:
                if sorted(img.shape) == sorted(lbl.shape):
                    source_shape = list(lbl.shape)
                    target_shape = list(img.shape)
                    perm = []
                    temp_source = source_shape.copy()
                    for ts in target_shape:
                        idx_in_source = temp_source.index(ts)
                        perm.append(idx_in_source)
                        temp_source[idx_in_source] = -1
                    lbl = np.transpose(lbl, perm)

            # Map labels if mapping is provided
            if self.label_mapping is not None:
                lbl_new = np.zeros_like(lbl)
                for old_val, new_val in self.label_mapping.items():
                    lbl_new[lbl == old_val] = new_val
                lbl = lbl_new

            # Normalize image
            if self.normalize:
                img = self.normalize_image(img)
            
            # Crop or pad to patch size
            if self.is_train:
                img, lbl = self.random_crop_or_pad(img, lbl, self.patch_size)
            else:
                img = self.crop_or_pad_to_size(img, self.patch_size)
                lbl = self.crop_or_pad_to_size(lbl, self.patch_size)
            
            # Convert to tensors
            img = torch.from_numpy(img).float().unsqueeze(0)  # (1, D, H, W)
            lbl = torch.from_numpy(lbl).long()  # (D, H, W)

            # Apply transforms if any
            if self.transform:
                img, lbl = self.transform(img, lbl)

            return img, lbl

        except Exception as e:
            print(f"\n⚠️ Error loading sample {idx} ({os.path.basename(self.image_files[idx])}): {e}")
            # Return a different random sample instead of crashing
            new_idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(new_idx)


class SimpleAugmentation3D:
    """
    Simple 3D augmentation: random flips
    """
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img, lbl):
        # Random flip along each axis
        for axis in [1, 2, 3]:  # D, H, W (skip channel dim)
            if torch.rand(1).item() < self.flip_prob:
                img = torch.flip(img, dims=[axis])
                lbl = torch.flip(lbl, dims=[axis-1])  # lbl doesn't have channel dim
        
        return img, lbl


def get_dataloaders_3d(train_img_dir, train_lbl_dir, val_img_dir=None, val_lbl_dir=None,
                       batch_size=2, patch_size=(128, 128, 128), num_workers=0,
                       use_augmentation=True, label_mapping=None, distributed=False):
    """
    Create train and validation dataloaders for 3D medical image segmentation
    
    Args:
        train_img_dir: Directory containing training images
        train_lbl_dir: Directory containing training labels
        val_img_dir: Directory containing validation images (optional)
        val_lbl_dir: Directory containing validation labels (optional)
        batch_size: Batch size for training
        patch_size: Size of 3D patches
        num_workers: Number of workers for data loading
        use_augmentation: Whether to use data augmentation
        label_mapping: Dictionary mapping original labels to [0, num_classes-1]
        distributed: Whether to use DistributedSampler for DDP
    
    Returns:
        train_loader, val_loader (or just train_loader if no validation data)
    """
    # Training dataset
    train_transform = SimpleAugmentation3D() if use_augmentation else None
    train_ds = NIfTISegmentationDataset3D(
        train_img_dir, train_lbl_dir, 
        patch_size=patch_size, 
        transform=train_transform,
        is_train=True,
        label_mapping=label_mapping
    )
    
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        sampler=train_sampler
    )
    
    # Validation dataset (optional)
    val_loader = None
    if val_img_dir and val_lbl_dir:
        val_ds = NIfTISegmentationDataset3D(
            val_img_dir, val_lbl_dir,
            patch_size=patch_size,
            transform=None,  # No augmentation for validation
            label_mapping=label_mapping
        )
        
        # For validation in DDP, we can use a DistributedSampler if we want to parallelize validation
        # or just run on rank 0. Here we'll support DDP validation if requested.
        val_sampler = None
        if distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
            
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            sampler=val_sampler
        )
    
    return train_loader, val_loader


# Test/Example usage
if __name__ == "__main__":
    # Example: Create a dataset
    train_loader, val_loader = get_dataloaders_3d(
        train_img_dir="data/train/images",
        train_lbl_dir="data/train/labels",
        val_img_dir="data/val/images",
        val_lbl_dir="data/val/labels",
        batch_size=2,
        patch_size=(64, 64, 64)
    )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for imgs, lbls in train_loader:
        print(f"Batch - Images: {imgs.shape}, Labels: {lbls.shape}")
        break

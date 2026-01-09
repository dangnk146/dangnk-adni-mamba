import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import os
import sys
import argparse
import cv2
import SimpleITK as sitk
from tqdm import tqdm
from skimage import exposure
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.umamba_3d import create_umamba_bot_3d, create_umamba_enc_3d

# --- 1. CORE INFERENCE LOGIC ---

def sliding_window_inference(model, image, patch_size=(128, 128, 128), overlap=0.5, num_classes=2, device='cuda'):
    model.eval()
    image = image.to(device)
    _, D, H, W = image.shape
    pD, pH, pW = patch_size
    
    stride_d = int(pD * (1 - overlap))
    stride_h = int(pH * (1 - overlap))
    stride_w = int(pW * (1 - overlap))
    
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
    
    _, D_pad, H_pad, W_pad = image.shape
    
    output_probs = torch.zeros((num_classes, D_pad, H_pad, W_pad), device=device)
    count_map = torch.zeros((num_classes, D_pad, H_pad, W_pad), device=device)
    
    dz = list(range(0, D_pad - pD + 1, stride_d))
    if dz[-1] != D_pad - pD: dz.append(D_pad - pD)
    dy = list(range(0, H_pad - pH + 1, stride_h))
    if dy[-1] != H_pad - pH: dy.append(H_pad - pH)
    dx = list(range(0, W_pad - pW + 1, stride_w))
    if dx[-1] != W_pad - pW: dx.append(W_pad - pW)
    
    total = len(dz) * len(dy) * len(dx)
    
    with torch.no_grad():
        with tqdm(total=total, desc="Inference") as pbar:
            for z in dz:
                for y in dy:
                    for x in dx:
                        patch = image[:, z:z+pD, y:y+pH, x:x+pW].unsqueeze(0)
                        outputs = model(patch)
                        if isinstance(outputs, list): outputs = outputs[0]
                        probs = torch.softmax(outputs, dim=1)
                        output_probs[:, z:z+pD, y:y+pH, x:x+pW] += probs[0]
                        count_map[:, z:z+pD, y:y+pH, x:x+pW] += 1.0
                        pbar.update(1)
    
    avg_probs = output_probs / count_map
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        avg_probs = avg_probs[:, :D, :H, :W]
    return avg_probs

# --- 2. ANATOMICAL SLICE EXTRACTION (USER PREFERRED LOGIC) ---

def normalize_intensity(slice_img):
    """
    Chuẩn hóa cường độ ảnh bằng percentile normalization.
    Tốt hơn z-score vì loại bỏ outliers và tăng contrast.
    """
    # Chỉ tính trên voxels > 0 (bỏ background)
    nonzero_values = slice_img[slice_img > 0]
    
    if len(nonzero_values) == 0:
        return np.zeros_like(slice_img, dtype=np.float32)
    
    # Percentile clipping (1% và 99%)
    p1, p99 = np.percentile(nonzero_values, [1, 99])
    
    if p99 > p1:
        # Clip và normalize về [0, 1]
        normalized = np.clip(slice_img, p1, p99)
        normalized = (normalized - p1) / (p99 - p1)
    else:
        normalized = np.zeros_like(slice_img, dtype=np.float32)
    
    return normalized

def calculate_slice_entropy(slice_img):
    """
    Tính entropy của slice để đánh giá thông tin.
    Entropy cao = nhiều thông tin = slice tốt.
    """
    # Normalize với percentile
    normalized = normalize_intensity(slice_img)
    
    # Convert sang uint8 để tính histogram
    slice_uint8 = (normalized * 255).astype(np.uint8)
    
    # Tính histogram
    hist, _ = np.histogram(slice_uint8, bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist_sum = hist.sum()
    
    if hist_sum <= 0:
        return 0.0
    
    # Normalize histogram
    hist = hist / hist_sum
    
    # Loại bỏ bins = 0
    hist = hist[hist > 0]
    
    # Tính entropy: -sum(p * log2(p))
    entropy_value = -np.sum(hist * np.log2(hist))
    
    return entropy_value

def find_brain_center(volume_np, threshold_percentile=50):
    """
    Tìm trung tâm não dựa trên intensity.
    """
    # Threshold để tìm vùng não
    threshold = np.percentile(volume_np[volume_np > 0], threshold_percentile)
    brain_mask = volume_np > threshold
    
    # Tìm center of mass
    if brain_mask.sum() == 0:
        return None
    
    coords = np.argwhere(brain_mask)
    center = coords.mean(axis=0).astype(int)
    
    return tuple(center)  # (z, y, x)

def select_best_slices(volume_np, axis, num_slices):
    """
    Fallback: Chọn slices có entropy cao nhất trong toàn bộ volume.
    """
    total_slices = volume_np.shape[axis]
    
    entropies = []
    for i in range(total_slices):
        if axis == 0:
            slice_img = volume_np[i, :, :]
        elif axis == 1:
            slice_img = volume_np[:, i, :]
        else:
            slice_img = volume_np[:, :, i]
        
        ent = calculate_slice_entropy(slice_img)
        entropies.append((i, ent))
    
    # Sắp xếp theo entropy giảm dần
    entropies.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top num_slices
    selected = [idx for idx, _ in entropies[:num_slices]]
    selected.sort()
    
    return selected

def select_anatomical_slices(volume_np, axis, num_slices, view_name, spacing_mm=1.0):
    """
    Chọn slices trong vùng giải phẫu quan trọng + entropy cao, 
    với offset cho hippocampus.
    """
    total_slices = volume_np.shape[axis]
    center = find_brain_center(volume_np)
    
    # Fallback nếu không tìm được trung tâm
    if center is None:
        print(f"  [Fallback] Dùng entropy toàn bộ cho {view_name}")
        return select_best_slices(volume_np, axis, num_slices)
    
    cz, cy, cx = center

    # Cấu hình vùng lấy slice theo view, với offset cho hippocampus
    # (dựa trên MNI/ADNI research)
    if view_name == 'axial':
        center_idx = cz - 20   # Offset inferior 20mm cho hippocampus
        window_mm = 30         # ±30mm → bao phủ hippocampal body
    elif view_name == 'sagittal':
        center_idx = cx        # Midline, no offset
        window_mm = 25         # ±25mm → medial temporal lobe
    elif view_name == 'oblique_coronal':
        center_idx = cy - 25   # Offset posterior 25mm cho hippocampus
        window_mm = 30         # ±30mm → perpendicular to long axis
    else:
        center_idx = total_slices // 2
        window_mm = 40

    # Chuyển mm → slice (với round để chính xác)
    window_slices = int(round(window_mm / spacing_mm))
    start = max(0, center_idx - window_slices)
    end = min(total_slices, center_idx + window_slices + 1)  # +1 để bao gồm đầy đủ range

    # Kiểm tra nếu region rỗng → fallback
    if start >= end:
        print(f"  [Fallback] Region rỗng cho {view_name} → dùng entropy toàn bộ")
        return select_best_slices(volume_np, axis, num_slices)

    # Tính entropy trong vùng
    entropies = []
    for i in range(start, end):
        if axis == 0:
            slice_img = volume_np[i, :, :]
        elif axis == 1:
            slice_img = volume_np[:, i, :]
        else:
            slice_img = volume_np[:, :, i]
        
        ent = calculate_slice_entropy(slice_img)
        entropies.append((i, ent))

    # Chọn top entropy
    entropies.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in entropies[:num_slices]]
    selected.sort()
    
    return selected

def save_cropped_slice(img_slice, mask_slice, save_path, target_size=(256, 256)):
    """
    Save CLEAN slice: Skull-strip using mask, Crop to brain, Resize & Enhance.
    """
    # 1. Normalize Intensity
    norm = normalize_intensity(img_slice)
    img_uint8 = (norm * 255).astype(np.uint8)
    
    # 2. Apply Mask (Skull Stripping)
    if mask_slice is not None:
        # Mask > 0 is brain
        # Simple resize of mask likely needed if we operate on slices, but here we assume 
        # img_slice and mask_slice are 1:1 pixel matched from same volume.
        binary_mask = (mask_slice > 0).astype(np.uint8)
        masked_img = img_uint8 * binary_mask
    else:
        # Fallback
        masked_img = img_uint8
        binary_mask = (img_uint8 > 15).astype(np.uint8)

    # 3. Find Bounding Box & Crop
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if np.any(rows) and np.any(cols):
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        
        # Add padding
        pad = 5
        r_min = max(0, r_min - pad)
        r_max = min(masked_img.shape[0], r_max + pad + 1)
        c_min = max(0, c_min - pad)
        c_max = min(masked_img.shape[1], c_max + pad + 1)
        
        img_crop = masked_img[r_min:r_max, c_min:c_max]
    else:
        img_crop = masked_img

    # 4. Resize to Target Size (256x256)
    pil_img = Image.fromarray(img_crop)
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    # 5. Contrast Enhancement (CLAHE - identical to old code)
    img_array = np.array(pil_img)
    img_enh = exposure.equalize_adapthist(img_array / 255.0, clip_limit=0.03) * 255
    img_enh = img_enh.astype(np.uint8)
    
    # Save
    final_img = Image.fromarray(img_enh)
    final_img.save(save_path)


def extract_anatomical_result(vol_np, pred_vol, spacing, output_dir, num_slices=5):
    """
    Bio-medical slicing using the provided volume arrays (Z, Y, X layout).
    Aligns exactly with data/extract_anatomical_slices.py logic.
    """
    print(f"[EXTRACT] Extracting anatomical slices to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    slice_counter = 0
    
    # Spacing from Nibabel header is (dx, dy, dz) -> (X, Y, Z) dimensions in physical world
    # Our volume vol_np is (Z, Y, X)
    sp_z = spacing[2]
    sp_y = spacing[1]
    sp_x = spacing[0]
    
    # 1. Axial (Axis 0 in ZYX layout is Z)
    # View name 'axial', spacing is Z-spacing
    indices = select_anatomical_slices(vol_np, 0, num_slices, 'axial', spacing_mm=sp_z)
    for idx in indices:
        s_img = vol_np[idx, :, :]
        s_mask = pred_vol[idx, :, :] if pred_vol is not None else None
        
        # Axial rotation needed for viewing
        s_img = np.rot90(s_img)
        if s_mask is not None: s_mask = np.rot90(s_mask)
        
        save_cropped_slice(s_img, s_mask, os.path.join(output_dir, f"{slice_counter:02d}_axial.png"))
        slice_counter+=1
        
    # 2. Sagittal (Axis 2 in ZYX layout is X)
    # View name 'sagittal', spacing is X-spacing
    indices = select_anatomical_slices(vol_np, 2, num_slices, 'sagittal', spacing_mm=sp_x)
    for idx in indices:
        s_img = vol_np[:, :, idx]
        s_mask = pred_vol[:, :, idx] if pred_vol is not None else None
        
        # Sagittal rotation 
        s_img = np.rot90(s_img)
        if s_mask is not None: s_mask = np.rot90(s_mask)
        
        save_cropped_slice(s_img, s_mask, os.path.join(output_dir, f"{slice_counter:02d}_sagittal.png"))
        slice_counter+=1
        
    # 3. Coronal (Axis 1 in ZYX layout is Y)
    # View name 'oblique_coronal' to use the specific offset logic in user's func
    indices = select_anatomical_slices(vol_np, 1, num_slices, 'oblique_coronal', spacing_mm=sp_y)
    for idx in indices:
        s_img = vol_np[:, idx, :]
        s_mask = pred_vol[:, idx, :] if pred_vol is not None else None
        
        # Coronal rotation
        s_img = np.rot90(s_img)
        if s_mask is not None: s_mask = np.rot90(s_mask)
        
        save_cropped_slice(s_img, s_mask, os.path.join(output_dir, f"{slice_counter:02d}_coronal.png"))
        slice_counter+=1

# --- 3. MAIN EVAL FUNCTION ---

def evaluate_sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # Load Model
    print(f"[INFO] Loading model: {args.model_path}")
    if args.model_type == "umamba_bot":
        model = create_umamba_bot_3d(1, args.num_classes)
    else:
        model = create_umamba_enc_3d(tuple(args.patch_size), 1, args.num_classes)
        
    checkpoint = torch.load(args.model_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    
    # Load Image (Nibabel Only to force Canonical Orientation)
    print(f"[INFO] Loading image: {args.image_path}")
    if not os.path.exists(args.image_path):
        print(f"[ERROR] File not found: {args.image_path}")
        return

    img_nii = nib.load(args.image_path)
    img_nii = nib.as_closest_canonical(img_nii) # Force RAS (X, Y, Z)
    img_data = img_nii.get_fdata().astype(np.float32)
    spacing = img_nii.header.get_zooms() # (dx, dy, dz)
    
    # Normalize
    mean = np.mean(img_data)
    std = np.std(img_data)
    img_norm = (img_data - mean) / (std + 1e-8)
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)
    
    # Inference
    print("[INFO] Running Inference...")
    probs = sliding_window_inference(model, img_tensor.squeeze(0), patch_size=tuple(args.patch_size), num_classes=args.num_classes, device=device)
    preds = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8) # (X, Y, Z) matching img_data
    
    # Save Prediction NIfTI
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        pred_path = os.path.join(args.save_dir, "prediction.nii.gz")
        nib.save(nib.Nifti1Image(preds, img_nii.affine), pred_path)
        print(f"[INFO] Saved NIfTI prediction to: {pred_path}")
        
        # Prepare Data for Slicing (Transpose X,Y,Z -> Z,Y,X)
        # Standard medical imaging libraries (and code logic) iterate Z as first dimension (slices)
        vol_zyx = img_data.transpose(2, 1, 0)
        pred_zyx = preds.transpose(2, 1, 0)
        
        # Extract Visual Slices using SAME arrays
        extract_anatomical_result(vol_zyx, pred_zyx, spacing, os.path.join(args.save_dir, "visual_slices"))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label_path", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_type", default="umamba_bot")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    
    args = parser.parse_args()
    evaluate_sample(args)

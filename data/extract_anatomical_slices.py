"""
Trích xuất slices 2D từ MRI 3D tập trung vào vùng Hippocampus
Dành cho phân loại Alzheimer's Disease (CN, MCI, AD)

Chiến lược:
- Axial: Tập trung vào hippocampus (offset inferior 20mm)
- Sagittal: Gần midline (±20mm) 
- Oblique Coronal: Xoay 30° vuông góc với trục dài hippocampus (offset posterior 25mm)
- Mỗi view: 5 slices có entropy cao nhất trong vùng giải phẫu
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import rotate
from skimage import exposure
import gc
from tqdm import tqdm


def calculate_slice_entropy(slice_img):
    """
    Tính entropy của slice để đánh giá thông tin.
    Entropy cao = nhiều thông tin = slice tốt.
    
    Args:
        slice_img: 2D numpy array
    
    Returns:
        float: Entropy value
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
    
    Returns:
        (z, y, x): Tọa độ trung tâm não
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
    
    Args:
        volume_np: 3D numpy array (D, H, W)
        axis: 0=axial, 1=coronal, 2=sagittal
        num_slices: Số lượng slices cần chọn
    
    Returns:
        List of slice indices (sorted)
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
    
    Args:
        volume_np: 3D numpy array (D, H, W)
        axis: 0=axial, 1=coronal, 2=sagittal
        num_slices: Số lượng slices cần chọn (5)
        view_name: 'axial', 'sagittal', 'oblique_coronal'
        spacing_mm: Khoảng cách giữa các slices (mm)
    
    Returns:
        List of slice indices (sorted)
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


def normalize_intensity(slice_img):
    """
    Chuẩn hóa cường độ ảnh bằng percentile normalization.
    Tốt hơn z-score vì loại bỏ outliers và tăng contrast.
    
    Args:
        slice_img: 2D numpy array
    
    Returns:
        Normalized array [0, 1]
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


def process_and_save_slice(slice_img, output_path, target_size=(256, 256)):
    """
    Xử lý và lưu slice thành PNG với cropping tự động và normalization.
    
    Args:
        slice_img: 2D numpy array
        output_path: Đường dẫn lưu file
        target_size: Kích thước output (H, W)
    """
    # 1. Normalize với percentile
    normalized = normalize_intensity(slice_img)
    
    # 2. Convert sang uint8
    slice_uint8 = (normalized * 255).astype(np.uint8)
    
    # 3. CROPPING TỰ ĐỘNG - Loại bỏ nền đen xung quanh
    # Tạo binary mask từ threshold thấp (5th percentile)
    threshold = np.percentile(slice_uint8[slice_uint8 > 0], 5) if np.any(slice_uint8 > 0) else 0
    binary_mask = slice_uint8 > threshold
    
    # Tìm bounding box của vùng não
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if np.any(rows) and np.any(cols):
        # Tìm indices của vùng não
        row_idxs = np.where(rows)[0]
        col_idxs = np.where(cols)[0]
        
        y_min, y_max = row_idxs[0], row_idxs[-1] + 1
        x_min, x_max = col_idxs[0], col_idxs[-1] + 1
        
        # Thêm padding nhỏ (5 pixels) để không crop quá sát
        padding = 5
        h, w = slice_uint8.shape
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Crop
        cropped = slice_uint8[y_min:y_max, x_min:x_max]
    else:
        # Nếu không tìm thấy vùng não, giữ nguyên
        cropped = slice_uint8
    
    # 4. Resize về target size
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    # 5. Contrast enhancement (CLAHE)
    img_array = np.array(pil_img)
    img_enhanced = exposure.equalize_adapthist(img_array / 255.0, clip_limit=0.03) * 255
    img_enhanced = img_enhanced.astype(np.uint8)
    
    # 6. Save
    final_img = Image.fromarray(img_enhanced)
    final_img.save(output_path)




def extract_slices_from_nifti(
    nifti_path, 
    output_subject_dir, 
    num_slices=5, 
    target_size=(256, 256)
):
    """
    Trích xuất slices từ file NIfTI theo 3 views: axial, sagittal, oblique_coronal.
    
    Args:
        nifti_path: Đường dẫn đến file .nii hoặc .nii.gz
        output_subject_dir: Thư mục output cho subject này
        num_slices: Số slices mỗi view (mặc định 5)
        target_size: Kích thước output (H, W)
    
    Returns:
        (success, total_slices): Tuple (bool, int)
    """
    try:
        print(f"\nProcessing: {nifti_path}")
        
        # Load volume với SimpleITK (để xử lý oblique rotation)
        volume = sitk.ReadImage(nifti_path)
        spacing = volume.GetSpacing()  # (x, y, z) spacing in mm
        spacing_mm = spacing[2]  # z-spacing for axial slices
        
        # Convert to numpy
        volume_np = sitk.GetArrayFromImage(volume)  # (D, H, W)
        
        # Tạo thư mục output cho bệnh nhân (không có thư mục con)
        os.makedirs(output_subject_dir, exist_ok=True)
        
        slice_counter = 0  # Đếm từ 00 đến 14
        
        # === 1. AXIAL: LẤY GẦN HIPPOCAMPUS (offset inferior 20mm) ===
        # Lưu ảnh 00.png đến 04.png
        print("  Extracting AXIAL slices (00-04)...")
        
        selected_indices = select_anatomical_slices(
            volume_np, 
            axis=0, 
            num_slices=num_slices, 
            view_name='axial',
            spacing_mm=spacing_mm
        )
        
        for i, idx in enumerate(selected_indices):
            slice_img = volume_np[idx, :, :]
            output_path = os.path.join(output_subject_dir, f'{slice_counter:02d}.png')
            process_and_save_slice(slice_img, output_path, target_size)
            slice_counter += 1
        
        print(f"    Saved {len(selected_indices)} axial slices: {selected_indices}")

        # === 2. SAGITTAL: LẤY GẦN MIDLINE (±25mm) ===
        # Lưu ảnh 05.png đến 09.png
        print("  Extracting SAGITTAL slices (05-09)...")
        
        selected_indices = select_anatomical_slices(
            volume_np, 
            axis=2, 
            num_slices=num_slices, 
            view_name='sagittal',
            spacing_mm=spacing[0]  # x-spacing
        )
        
        for i, idx in enumerate(selected_indices):
            slice_img = volume_np[:, :, idx]
            output_path = os.path.join(output_subject_dir, f'{slice_counter:02d}.png')
            process_and_save_slice(slice_img, output_path, target_size)
            slice_counter += 1
        
        print(f"    Saved {len(selected_indices)} sagittal slices: {selected_indices}")

        del volume_np
        gc.collect()

        # === 3. OBLIQUE CORONAL: XOAY 30° + LẤY GẦN HIPPOCAMPUS (offset posterior 25mm) ===
        # Lưu ảnh 10.png đến 14.png
        print("  Extracting OBLIQUE CORONAL slices (10-14, 30° rotation)...")

        # Prepare transform with center
        rotator = sitk.Euler3DTransform()
        angle = np.deg2rad(30)
        rotator.SetRotation(0.0, angle, 0.0)  # Rotate around Y-axis
        
        # Set center to image physical center
        img_center_idx = np.array(volume.GetSize()) / 2.0
        center_phys = volume.TransformContinuousIndexToPhysicalPoint(img_center_idx.tolist())
        rotator.SetCenter(center_phys)

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(volume)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(rotator)
        resampler.SetDefaultPixelValue(0)
        rotated_volume = resampler.Execute(volume)

        del volume
        gc.collect()

        rotated_np = sitk.GetArrayFromImage(rotated_volume)
        del rotated_volume
        gc.collect()

        selected_indices = select_anatomical_slices(
            rotated_np, 
            axis=1, 
            num_slices=num_slices, 
            view_name='oblique_coronal',
            spacing_mm=spacing[1]  # y-spacing
        )
        
        for i, idx in enumerate(selected_indices):
            slice_img = rotated_np[:, idx, :]
            output_path = os.path.join(output_subject_dir, f'{slice_counter:02d}.png')
            process_and_save_slice(slice_img, output_path, target_size)
            slice_counter += 1
        
        print(f"    Saved {len(selected_indices)} oblique coronal slices: {selected_indices}")

        del rotated_np
        gc.collect()

        print(f"  ✓ Total slices extracted: {slice_counter} (saved as 00.png to {slice_counter-1:02d}.png)")
        return True, slice_counter

    except Exception as e:
        print(f"  ✗ Error processing {nifti_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def process_dataset(
    input_root, 
    output_root, 
    num_slices=5, 
    target_size=(256, 256),
    test_mode=False
):
    """
    Xử lý toàn bộ dataset.
    
    Args:
        input_root: Thư mục input
        output_root: Thư mục output
        num_slices: Số slices mỗi view (5)
        target_size: Kích thước ảnh output (256, 256)
        test_mode: Nếu True, chỉ xử lý 1 bệnh nhân để test
    
    Expected structure:
        input_root/
            train/
                CN/
                    patient001.nii.gz
                    ...
                MCI/
                    ...
                AD/
                    ...
            val/
                ...
            test/
                ...
    
    Output structure:
        output_root/
            train/
                CN/
                    patient001/
                        00.png (axial 1)
                        01.png (axial 2)
                        ...
                        04.png (axial 5)
                        05.png (sagittal 1)
                        ...
                        09.png (sagittal 5)
                        10.png (oblique 1)
                        ...
                        14.png (oblique 5)
                    patient002/
                        00.png to 14.png
                    ...
                MCI/
                    ...
                AD/
                    ...
            val/
                ...
            test/
                ...
    """
    splits = ['train', 'val', 'test']
    classes = ['CN', 'MCI', 'AD']
    
    total_processed = 0
    total_failed = 0
    
    if test_mode:
        print("\n" + "="*80)
        print("🧪 TEST MODE: Chỉ xử lý 1 bệnh nhân đầu tiên")
        print("="*80 + "\n")
    
    for split in splits:
        for cls in classes:
            input_dir = os.path.join(input_root, split, cls)
            
            if not os.path.exists(input_dir):
                print(f"Skipping {input_dir} (not found)")
                continue
            
            # Tìm tất cả file .nii và .nii.gz
            nifti_files = []
            for f in os.listdir(input_dir):
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    nifti_files.append(f)
            
            if len(nifti_files) == 0:
                continue
            
            # Test mode: chỉ lấy 1 file
            if test_mode:
                nifti_files = nifti_files[:1]
            
            print(f"\n{'='*80}")
            print(f"Processing {split}/{cls}: {len(nifti_files)} file{'s' if len(nifti_files) > 1 else ''}")
            print(f"{'='*80}")
            
            # Sử dụng tqdm để theo dõi tiến trình
            for nifti_file in tqdm(nifti_files, desc=f"{split}/{cls}", unit="patient"):
                nifti_path = os.path.join(input_dir, nifti_file)
                
                # Tạo tên subject (loại bỏ extension)
                subject_name = nifti_file.replace('.nii.gz', '').replace('.nii', '')
                
                # Output directory
                output_subject_dir = os.path.join(output_root, split, cls, subject_name)
                
                # Extract slices
                success, num_slices_extracted = extract_slices_from_nifti(
                    nifti_path,
                    output_subject_dir,
                    num_slices=num_slices,
                    target_size=target_size
                )
                
                if success:
                    total_processed += 1
                else:
                    total_failed += 1
            
            # Test mode: dừng sau 1 bệnh nhân
            if test_mode and total_processed > 0:
                print(f"\n✓ Test mode: Đã xử lý 1 bệnh nhân thành công!")
                print(f"  Output: {output_subject_dir}")
                print(f"  Kiểm tra thư mục để xem 15 ảnh (00.png đến 14.png)")
                return
    
    print(f"\n{'='*80}")
    print(f"Dataset Processing Complete!")
    print(f"{'='*80}")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Total slices per subject: {num_slices * 3} (5 per view × 3 views)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Trích xuất slices 2D từ MRI 3D tập trung vào vùng Hippocampus'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default="/mnt/c/Users/ADMIN/Desktop/hoinghi-dnk/dataset",
        help='Thư mục input chứa dataset'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default="/mnt/c/Users/ADMIN/Desktop/hoinghi-dnk/dataset_2d_slices",
        help='Thư mục output để lưu slices'
    )
    parser.add_argument(
        '--num-slices', 
        type=int, 
        default=5,
        help='Số slices mỗi view (mặc định: 5)'
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=256,
        help='Kích thước ảnh output (mặc định: 256x256)'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test mode: chỉ xử lý 1 bệnh nhân đầu tiên'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TRÍCH XUẤT SLICES 2D TỪ MRI 3D")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Slices per view: {args.num_slices}")
    print(f"Image size: {args.size}x{args.size}")
    print(f"Test mode: {'Yes' if args.test else 'No'}")
    print("="*80 + "\n")
    
    process_dataset(
        input_root=args.input,
        output_root=args.output,
        num_slices=args.num_slices,
        target_size=(args.size, args.size),
        test_mode=args.test
    )

import os
import sys
import argparse
import SimpleITK as sitk
import numpy as np
import cv2
import subprocess
import csv
from scipy import ndimage
from tqdm import tqdm
import gc

# ==============================================================================
# CẤU HÌNH VÀ UTILITIES
# ==============================================================================
def find_brain_center(volume_np):
    """Tìm trung tâm khối não sau skull stripping."""
    thresh = np.percentile(volume_np, 10)
    mask = volume_np > thresh
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    cz = int(np.mean(coords[0]))
    cy = int(np.mean(coords[1]))
    cx = int(np.mean(coords[2]))
    return (cz, cy, cx)

def select_anatomical_slices(volume_np, axis, num_slices, view_name, spacing_mm=1.0):
    """Chọn slice trong vùng giải phẫu quan trọng + entropy cao, với offset cho hippocampus."""
    total_slices = volume_np.shape[axis]
    center = find_brain_center(volume_np)
    
    # Fallback nếu không tìm được trung tâm
    if center is None:
        print(f"  [Fallback] Dùng entropy toàn bộ cho {view_name}")
        return select_best_slices(volume_np, axis, num_slices)
    
    cz, cy, cx = center

    # Cấu hình vùng lấy slice theo view, với offset cho hippocampus (dựa trên MNI/ADNI research)
    if view_name == 'axial':
        center_idx = cz - 20   # Tăng offset inferior lên 30mm cho hippocampus (tăng 10 so với trước)
        window_mm = 30         # ±25mm → bao phủ hippocampal body
    elif view_name == 'sagittal':
        center_idx = cx - 15        # Midline, no offset
        window_mm = 25         # ±20mm → medial temporal lobe
    elif view_name == 'oblique_coronal':
        center_idx = cy - 25   # Offset posterior 15mm cho hippocampus
        window_mm = 30         # ±25mm → perpendicular to long axis
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
        if axis == 0: slice_img = volume_np[i, :, :]
        elif axis == 1: slice_img = volume_np[:, i, :]
        else: slice_img = volume_np[:, :, i]
        entropy = calculate_slice_entropy(slice_img)
        entropies.append((i, entropy))

    # Chọn top entropy
    entropies.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in entropies[:num_slices]]
    selected.sort()
    return selected

def find_dicom_series_dirs(root_dir):
    """Tìm các thư mục chứa DICOM files."""
    dicom_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(dirpath, f))
    return sorted(list(set(os.path.dirname(p) for p in dicom_files)))

def resample_image_to_1mm(image_sitk):
    """Resample ảnh về độ phân giải 1x1x1 mm."""
    original_spacing = image_sitk.GetSpacing()  # (x, y, z)
    original_size = image_sitk.GetSize()        # (sx, sy, sz)
    target_spacing = (1.0, 1.0, 1.0)
    
    new_size = [int(round(osz * (ospc / tspc)))
                for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # Identity transform
    resampler.SetDefaultPixelValue(0)  # Background value đúng
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image_sitk)

def skull_strip_with_hdbet(input_path, output_path, device, use_gpu=True):
    """Loại bỏ hộp sọ bằng HD-BET."""
    try:
        command = ['hd-bet', '-i', input_path, '-o', output_path, '-device', device, '--disable_tta']
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        print(f"LỖI HD-BET: {e}", file=sys.stderr)
        return False

def normalize_intensity(image_np):
    """Chuẩn hóa cường độ ảnh bằng percentile normalization."""
    p1, p99 = np.percentile(image_np, (1, 99))
    if p99 > p1:
        normalized = np.clip((image_np - p1) / (p99 - p1), 0, 1)
    else:
        normalized = np.zeros_like(image_np, dtype=np.float32)
    return normalized

def calculate_slice_entropy(slice_img):
    """Tính entropy của một slice để đánh giá mức độ thông tin."""
    normalized = normalize_intensity(slice_img)
    slice_uint8 = (normalized * 255).astype(np.uint8)
    hist, _ = np.histogram(slice_uint8, bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum <= 0:
        return 0.0
    hist = hist / hist_sum
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def select_best_slices(volume_np, axis, num_slices):
    total_slices = volume_np.shape[axis]
    if num_slices >= total_slices:
        return list(range(total_slices))
    entropies = []
    for i in range(total_slices):
        if axis == 0:
            slice_img = volume_np[i, :, :]
        elif axis == 1:
            slice_img = volume_np[:, i, :]
        else:
            slice_img = volume_np[:, :, i]
        entropy = calculate_slice_entropy(slice_img)
        entropies.append(entropy)
    sorted_indices = np.argsort(entropies)[::-1]
    selected_indices = sorted(sorted_indices[:num_slices])
    return selected_indices

def process_and_save_slice(slice_img, output_path, target_size):
    normalized = normalize_intensity(slice_img)
    slice_uint8 = (normalized * 255).astype(np.uint8)
    threshold = np.percentile(slice_uint8, 5)
    binary_mask = slice_uint8 > threshold
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if np.any(rows) and np.any(cols):
        row_idxs = np.where(rows)[0]
        col_idxs = np.where(cols)[0]
        y_min, y_max = row_idxs[0], row_idxs[-1] + 1  # +1 để bao gồm hàng cuối
        x_min, x_max = col_idxs[0], col_idxs[-1] + 1  # +1 để bao gồm cột cuối
        padding = 5
        h, w = slice_uint8.shape
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        cropped = slice_uint8[y_min:y_max, x_min:x_max]
    else:
        cropped = slice_uint8
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, resized)

def process_dicom_to_png(dicom_dir, output_subject_dir, num_slices=20, target_size=(224, 224), device='cuda'):
    os.makedirs(output_subject_dir, exist_ok=True)

    # === 1. ĐỌC DICOM ===
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"Không tìm thấy Series ID nào trong thư mục DICOM: {dicom_dir}")

    series_file_names = [sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, sid) for sid in series_ids]
    if not any(series_file_names):
        raise RuntimeError(f"Không lấy được danh sách file cho bất kỳ Series ID nào trong {dicom_dir}")

    best_files = max(series_file_names, key=len)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    volume = reader.Execute()

    # === 2. CHUẨN HÓA: RAS + Resample 1mm ===
    volume = sitk.DICOMOrient(volume, 'RAS')
    volume = resample_image_to_1mm(volume)

    # === 3. SKULL STRIPPING ===
    temp_nifti = os.path.join(output_subject_dir, 'temp_volume.nii.gz')
    temp_skullstripped = os.path.join(output_subject_dir, 'temp_skullstripped.nii.gz')
    sitk.WriteImage(volume, temp_nifti)
    if not skull_strip_with_hdbet(temp_nifti, temp_skullstripped, device):
        raise RuntimeError("HD-BET thất bại")
    volume = sitk.ReadImage(temp_skullstripped)
    for f in [temp_nifti, temp_skullstripped]:
        if os.path.exists(f): os.remove(f)

    total_slices = 0
    volume_np = sitk.GetArrayFromImage(volume)

    # === 4. AXIAL: LẤY GẦN TRUNG TÂM NÃO (±25mm, offset inferior) ===
    view_dir = os.path.join(output_subject_dir, 'slices', 'axial')
    os.makedirs(view_dir, exist_ok=True)
    selected_indices = select_anatomical_slices(volume_np, axis=0, num_slices=num_slices, view_name='axial')
    for i, idx in enumerate(selected_indices):
        slice_img = volume_np[idx, :, :]
        output_path = os.path.join(view_dir, f'{i:02d}.png')
        process_and_save_slice(slice_img, output_path, target_size)
        total_slices += 1

    # === 5. SAGITTAL: LẤY GẦN MIDLINE (±20mm) ===
    view_dir = os.path.join(output_subject_dir, 'slices', 'sagittal')
    os.makedirs(view_dir, exist_ok=True)
    selected_indices = select_anatomical_slices(volume_np, axis=2, num_slices=num_slices, view_name='sagittal')
    for i, idx in enumerate(selected_indices):
        slice_img = volume_np[:, :, idx]
        output_path = os.path.join(view_dir, f'{i:02d}.png')
        process_and_save_slice(slice_img, output_path, target_size)
        total_slices += 1

    del volume_np
    gc.collect()

    # === 6. OBLIQUE CORONAL: XOAY 30° + LẤY GẦN TRUNG TÂM Y (±25mm, offset posterior) ===
    view_dir = os.path.join(output_subject_dir, 'slices', 'oblique_coronal')
    os.makedirs(view_dir, exist_ok=True)

    # Prepare transform with center
    rotator = sitk.Euler3DTransform()
    angle = np.deg2rad(30)
    rotator.SetRotation(0.0, angle, 0.0)
    # Set center to image physical center
    img_center_idx = np.array(volume.GetSize()) / 2.0
    center_phys = volume.TransformContinuousIndexToPhysicalPoint(img_center_idx.tolist())
    rotator.SetCenter(center_phys)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(rotator)
    resampler.SetDefaultPixelValue(0)  # Background value đúng
    rotated_volume = resampler.Execute(volume)

    del volume
    gc.collect()

    rotated_np = sitk.GetArrayFromImage(rotated_volume)
    del rotated_volume
    gc.collect()

    selected_indices = select_anatomical_slices(rotated_np, axis=1, num_slices=num_slices, view_name='oblique_coronal')
    for i, idx in enumerate(selected_indices):
        slice_img = rotated_np[:, idx, :]
        output_path = os.path.join(view_dir, f'{i:02d}.png')
        process_and_save_slice(slice_img, output_path, target_size)
        total_slices += 1

    del rotated_np
    gc.collect()

    return True, total_slices

# ==============================================================================
# HÀM XỬ LÝ MỘT SUBJECT
# ==============================================================================

def process_single_subject(subject_info, args):
    subject_id = subject_info['subject_id']
    input_dir = subject_info['input_dir']
    output_dir = subject_info['output_dir']
    split = subject_info['split']
    label = subject_info['label']
    
    tqdm.write(f" -> Bắt đầu xử lý bệnh nhân {subject_id} ({split}/{label})...")
    
    log_row = {
        'SubjectID': subject_id,
        'Split': split,
        'Label': label,
        'Slices_Extracted': 0,
        'Status': 'FAILED',
        'Error': '',
        'OutputDir': ''
    }
    
    try:
        dicom_dirs = find_dicom_series_dirs(input_dir)
        if not dicom_dirs:
            raise FileNotFoundError(f"Không tìm thấy DICOM files trong {input_dir}")
        
        tqdm.write(f" -> [1] Tìm thấy DICOM series tại: {dicom_dirs[0]}")
        
        success, num_saved = process_dicom_to_png(
            dicom_dirs[0], 
            output_dir, 
            num_slices=args.num_slices,
            target_size=args.target_size,
            device=args.device
        )
        
        if success:
            log_row['Status'] = 'SUCCESS'
            log_row['Slices_Extracted'] = num_saved
            log_row['OutputDir'] = os.path.join(output_dir, 'slices')
            tqdm.write(f" -> Hoàn thành bệnh nhân {subject_id}: {num_saved} slices")
        else:
            log_row['Error'] = 'Xử lý DICOM thất bại không rõ nguyên nhân'
            tqdm.write(f" -> LỖI bệnh nhân {subject_id}: Xử lý DICOM thất bại không rõ nguyên nhân")
            
    except Exception as e:
        log_row['Error'] = str(e)
        tqdm.write(f" -> LỖI bệnh nhân {subject_id}: {str(e)}")
        import traceback
        tqdm.write(f" -> Chi tiết lỗi: {traceback.format_exc()}")
    
    return log_row

# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH – ĐÃ SỬA ĐỂ HỖ TRỢ CẤU TRÚC MỚI
# ==============================================================================

def run_simple_preprocessing(args):
    output_root = args.output_root or os.path.join(os.path.dirname(args.dataset_root), "preprocessed_data")
    os.makedirs(output_root, exist_ok=True)
    
    log_file_path = os.path.join(output_root, 'preprocessing_log.csv')
    fieldnames = ['SubjectID', 'Split', 'Label', 'Slices_Extracted', 'Status', 'Error', 'OutputDir']
    
    # Đọc log cũ để skip
    processed_subjects = set()
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get('Status') == 'SUCCESS':
                        processed_subjects.add(row['SubjectID'])  # Lưu raw SubjectID
            print(f"Đã tìm thấy {len(processed_subjects)} subjects đã xử lý thành công")
        except Exception as e:
            print(f"Cảnh báo: Không thể đọc log file cũ: {e}")
    
    with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
            csv_writer.writeheader()
        
        # --- PHẦN MỚI: HỖ TRỢ CẤU TRÚC TỰ TẠO + ĐỔI + → -- ---
        target_split = None
        target_label = None
        if args.label:
            path_parts = args.label.replace('\\', '/').split('/')
            if len(path_parts) == 1:
                target_label = path_parts[0]
            elif len(path_parts) == 2:
                target_split, target_label = path_parts
            else:
                print(f"Lỗi: Định dạng --label không hợp lệ: '{args.label}'.")
                return
        
        subjects_to_process = []
        all_splits = [d for d in os.listdir(args.dataset_root) 
                     if os.path.isdir(os.path.join(args.dataset_root, d))]
        
        splits_to_scan = [target_split] if target_split else all_splits
        if target_split and target_split not in all_splits:
            print(f"Không tìm thấy split '{target_split}' trong {args.dataset_root}.")
            return

        for split in splits_to_scan:
            split_path = os.path.join(args.dataset_root, split)
            if not os.path.isdir(split_path):
                continue
            all_labels_in_split = [l for l in os.listdir(split_path) 
                                 if os.path.isdir(os.path.join(split_path, l))]
            
            labels_to_scan = [target_label] if target_label and target_label in all_labels_in_split else all_labels_in_split

            for label in labels_to_scan:
                label_path = os.path.join(split_path, label)
                subjects = [s for s in os.listdir(label_path) 
                           if os.path.isdir(os.path.join(label_path, s))]
                
                for raw_subject_id in subjects:
                    # Chuẩn hóa output_dir với --
                    subject_id_for_log = raw_subject_id  # Lưu raw cho log
                    output_subject_dir = os.path.join(output_root, split, label, raw_subject_id.replace('+', '--'))
                    
                    # Bỏ qua nếu đã xử lý (dùng raw cho so sánh)
                    if subject_id_for_log in processed_subjects:
                        continue
                    
                    input_subject_dir = os.path.join(label_path, raw_subject_id)
                    
                    subjects_to_process.append({
                        'subject_id': subject_id_for_log,  # Raw cho log
                        'input_dir': input_subject_dir,
                        'output_dir': output_subject_dir,  # Với --
                        'split': split,
                        'label': label
                    })
        
        total_subjects = len(subjects_to_process)
        if total_subjects == 0:
            print("Không có subjects mới để xử lý")
            return
        
        print(f"Bắt đầu xử lý {total_subjects} subjects...")
        print(f"Output: {output_root}")
        print(f"Log file: {log_file_path}")
        print("=" * 80)
        
        with tqdm(total=total_subjects, desc="Processing subjects", unit="subject") as pbar:
            for subject_info in subjects_to_process:
                log_row = process_single_subject(subject_info, args)
                csv_writer.writerow(log_row)
                csvfile.flush()
                pbar.update(1)
    
    print(f"\n{'='*80}")
    print(f"Hoàn tất! Kết quả tại: {output_root}")
    print(f"Log file: {log_file_path}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quy trình tiền xử lý đơn giản cho dataset MRI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                       help="Đường dẫn đến thư mục gốc dataset (chứa train/val/test)")
    parser.add_argument("--output_root", type=str, default=None,
                       help="Thư mục output (mặc định: preprocessed_data)")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'],
                       help="Thiết bị cho HD-BET")
    parser.add_argument("--num_slices", type=int, default=20,
                       help="Số slices cho mỗi view")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                       help="Kích thước output (width height)")
    parser.add_argument("--label", type=str, default=None,
                       help="(Tùy chọn) Chỉ xử lý một class cụ thể (VD: MCI) hoặc một tập con (VD: test/MCI).")
    
    args = parser.parse_args()
    args.target_size = tuple(args.target_size)
    
    run_simple_preprocessing(args)
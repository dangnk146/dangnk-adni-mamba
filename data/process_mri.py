#!/usr/bin/env python3
"""
ADNI Hippocampus Segmentation - FINAL + nibabel.processing FIX
"""

import os
import sys
import nibabel as nib
from nibabel.processing import resample_from_to  # <-- ĐÚNG CÁCH
import numpy as np
import subprocess
import tempfile
import re
from pathlib import Path
import glob

# ================================
# CẤU HÌNH
# ================================
INPUT_DIR = "/mnt/c/Users/ADMIN/Desktop/sdh-dnk/dataset/AD-ADNI1"
MASK_DIR  = "/mnt/c/Users/ADMIN/Desktop/sdh-dnk/dataset/Mask-AD-ADNI1"
HIPPODDEEP_SCRIPT = os.path.join(os.path.dirname(__file__), "hippodeep.py")
OUTPUT_ROOT = Path("output/AD")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ================================
# TRÍCH XUẤT INFO
# ================================
def extract_info_from_path(nii_path):
    path = Path(nii_path)
    parts = path.parts
    subject = next((p for p in parts if re.match(r"\d{3}_S_\d{4}", p)), None)
    if not subject: return None, None, None
    date = next((p for p in parts if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d", p)), None)
    if not date: return subject, None, None
    session_match = re.search(r"_S(\d+)_", path.name)
    if not session_match: return subject, date, None
    session_id = f"S{session_match.group(1)}"
    return subject, date, session_id

# ================================
# TÌM MASK
# ================================
def find_corresponding_mask(image_path, mask_root):
    subject, date, session_id = extract_info_from_path(image_path)
    if not all([subject, date, session_id]):
        print(f"   [Lỗi] Không trích xuất info")
        return None

    for root, dirs, files in os.walk(mask_root):
        if subject not in root or date not in root:
            continue
        for file in files:
            if file.endswith((".nii", ".nii.gz")) and session_id in file and "Mask" in file:
                return os.path.join(root, file)
    print(f"   [Cảnh báo] Không tìm thấy mask")
    return None

# ================================
# RESAMPLE MASK THEO ẢNH GỐC
# ================================
def resample_mask_to_image(image_path, mask_path, output_path):
    print(f"   [Resample] Căn chỉnh mask → ảnh gốc...")
    try:
        img_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        # Dùng nibabel.processing.resample_from_to
        mask_resampled = resample_from_to(mask_nii, (img_nii.shape, img_nii.affine), order=0)

        nib.save(mask_resampled, output_path)
        print(f"   → ĐÃ CĂN CHỈNH: {output_path}")
        print(f"     Shape: {img_nii.shape} ← {mask_nii.shape}")
        return output_path
    except Exception as e:
        print(f"   [Lỗi] Resample thất bại: {e}")
        return None

# ================================
# SKULL-STRIP
# ================================
def skull_strip_image(image_path, aligned_mask_path, output_path):
    print(f"   [Skull-strip] {Path(image_path).name}")
    try:
        img = nib.load(image_path)
        mask = nib.load(aligned_mask_path)
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        if img_data.shape != mask_data.shape:
            print(f"   [Lỗi] Shape vẫn không khớp sau resample!")
            return None

        mask_bin = (mask_data > 0.5).astype(np.float32)
        brain_data = img_data * mask_bin
        brain_img = nib.Nifti1Image(brain_data, img.affine, img.header)
        nib.save(brain_img, output_path)
        print(f"   → Brain extracted: {output_path}")
        return output_path
    except Exception as e:
        print(f"   [Lỗi] Skull-strip: {e}")
        return None

# ================================
# CHẠY HIPPODDEEP
# ================================
def run_hippodeep(image_path, work_dir):
    print(f"   [Hippodeep] Đang xử lý...")
    cmd = [sys.executable, HIPPODDEEP_SCRIPT, image_path]
    try:
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"   [Lỗi] Timeout")
        return False
    if result.returncode != 0:
        print(f"   [Lỗi] Hippodeep: {result.stderr[:300]}...")
        return False
    print(f"   [Hippodeep] Thành công!")
    return True

# ================================
# TÌM MASK HIPPOCAMPUS
# ================================
def find_hippo_masks(work_dir, base_stem):
    pattern_l = os.path.join(work_dir, f"{base_stem}_mask_L.nii*")
    pattern_r = os.path.join(work_dir, f"{base_stem}_mask_R.nii*")
    mask_l = glob.glob(pattern_l)
    mask_r = glob.glob(pattern_r)
    if not mask_l or not mask_r:
        return None, None
    return mask_l[0], mask_r[0]

# ================================
# TẠO HIPPOCAMPUS ONLY
# ================================
def create_hippocampus_only(brain_path, mask_l_path, mask_r_path, output_path):
    print(f"   [Hippocampus-only] → {output_path.name}")
    try:
        brain = nib.load(brain_path)
        l_img = nib.load(mask_l_path)
        r_img = nib.load(mask_r_path)
        brain_data = brain.get_fdata()
        l_data = l_img.get_fdata() / 255.0
        r_data = r_img.get_fdata() / 255.0
        l_bin = (l_data > 0.125).astype(np.float32)
        r_bin = (r_data > 0.125).astype(np.float32)
        hippo_data = brain_data * (l_bin + r_bin)
        hippo_img = nib.Nifti1Image(hippo_data, brain.affine, brain.header)
        nib.save(hippo_img, output_path)
        print(f"   → ĐÃ LƯU: {output_path}")
    except Exception as e:
        print(f"   [Lỗi] Lưu thất bại: {e}")

# ================================
# MAIN
# ================================
def main():
    print("="*70)
    print("ADNI HIPPOCAMPUS BATCH SEGMENTATION (FINAL - nibabel.processing)")
    print(f"Input : {INPUT_DIR}")
    print(f"Mask  : {MASK_DIR}")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*70)

    if not os.path.exists(HIPPODDEEP_SCRIPT):
        raise FileNotFoundError(f"Không tìm thấy hippodeep.py")

    image_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith((".nii", ".nii.gz")) and "Mask" not in file:
                image_files.append(os.path.join(root, file))
    print(f"Tìm thấy {len(image_files)} ảnh T1")

    success_count = 0
    for img_path in image_files:
        subject, date, session_id = extract_info_from_path(img_path)
        if not all([subject, date, session_id]):
            continue

        output_filename = f"ADNI1-{subject}-{date}.nii"
        final_output = OUTPUT_ROOT / output_filename
        if final_output.exists():
            continue

        print(f"\n[{success_count+1}/{len(image_files)}] Xử lý: {subject} - {date}")

        mask_path = find_corresponding_mask(img_path, MASK_DIR)
        if not mask_path:
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # 1. Resample mask
            aligned_mask = temp_dir / "mask_aligned.nii.gz"
            aligned_mask = resample_mask_to_image(img_path, mask_path, aligned_mask)
            if not aligned_mask:
                continue

            # 2. Skull-strip
            brain_path = temp_dir / "brain.nii.gz"
            brain_path = skull_strip_image(img_path, aligned_mask, brain_path)
            if not brain_path:
                continue

            # 3. Hippodeep
            if not run_hippodeep(img_path, temp_dir):
                continue

            # 4. Tìm mask L/R
            base_stem = Path(img_path).stem.replace(".nii", "")
            mask_l, mask_r = find_hippo_masks(str(temp_dir), base_stem)
            if not mask_l or not mask_r:
                print(f"   [Lỗi] Không tìm thấy _mask_L/R")
                continue

            # 5. Tạo output
            create_hippocampus_only(brain_path, mask_l, mask_r, final_output)
            success_count += 1

    print("="*70)
    print(f"HOÀN TẤT! Thành công: {success_count}/{len(image_files)}")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*70)

if __name__ == "__main__":
    main()
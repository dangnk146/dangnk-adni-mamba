# U-Mamba 3D - Simplified Implementation

A simplified but **complete** implementation of U-Mamba for 3D medical image segmentation, focusing solely on 3D architectures.

## 📁 Project Structure

```
UMambaAttention/
├── model/
│   ├── umamba.py          # Original 2D implementation (legacy)
│   └── umamba_3d.py       # ✨ Complete 3D U-Mamba implementation
├── utils/
│   ├── dataset.py         # Original 2D/3D dataset (simple)
│   └── dataset_3d.py      # ✨ Full-featured 3D NIfTI dataset loader
├── train.py               # Original 2D training script (legacy)
└── train_3d.py            # ✨ Complete 3D training script
```

## 🎯 Key Features

### Two Model Variants

1. **UMambaBot3d** (Recommended for most tasks)
   - Mamba module **only at bottleneck**
   - Faster training and inference
   - Lower memory requirements
   - Good for standard segmentation tasks

2. **UMambaEnc3d** (For complex tasks)
   - Mamba modules at **all encoder stages**
   - More powerful but computationally expensive
   - Better for complex multi-organ segmentation
   - Adaptive token strategy (spatial/channel tokens based on feature map size)

### What's Included

✅ **Complete architecture following original U-Mamba paper**:
- Residual blocks with Instance Normalization
- Mamba SSM layers with FP32 conversion
- Deep supervision support
- Skip connections
- Configurable stages and features

✅ **Full dataset handling**:
- NIfTI (.nii.gz) file loading
- Automatic crop/pad to patch size
- Z-score normalization
- 3D augmentation (random flips)
- Multi-modality support

✅ **Production-ready training**:
- Mixed precision training (AMP)
- Combined Cross-Entropy + Dice Loss
- Cosine annealing learning rate
- Validation loop with metrics
- Automatic checkpointing
- Best model saving

## 🚀 Quick Start

### Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Mamba
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir

# Install other dependencies
pip install nibabel tqdm
```

### Prepare Your Data

Organize your NIfTI files as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── case_001_0000.nii.gz
│   │   ├── case_002_0000.nii.gz
│   │   └── ...
│   └── labels/
│       ├── case_001.nii.gz
│       ├── case_002.nii.gz
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

**Note**: The `_0000` suffix indicates modality (e.g., CT, T1, T2). You can customize this in the dataset loader.

### Train the Model

#### Simple Training (UMambaBot - Recommended)

```bash
python train_3d.py \
    --model umamba_bot \
    --train_img_dir data/train/images \
    --train_lbl_dir data/train/labels \
    --val_img_dir data/val/images \
    --val_lbl_dir data/val/labels \
    --num_classes 3 \
    --epochs 100 \
    --batch_size 2 \
    --use_amp
```

#### Advanced Training (UMambaEnc)

```bash
python train_3d.py \
    --model umamba_enc \
    --train_img_dir data/train/images \
    --train_lbl_dir data/train/labels \
    --val_img_dir data/val/images \
    --val_lbl_dir data/val/labels \
    --num_classes 3 \
    --n_stages 5 \
    --patch_size 96 96 96 \
    --deep_supervision \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --use_amp
```

### Test the Model

```python
import torch
from model.umamba_3d import create_umamba_bot_3d

# Load model
model = create_umamba_bot_3d(input_channels=1, num_classes=3, n_stages=6)
model.load_state_dict(torch.load("umamba_3d_best.pth"))
model.eval()

# Inference
with torch.no_grad():
    x = torch.randn(1, 1, 128, 128, 128)  # (B, C, D, H, W)
    output = model(x)
    pred = output.argmax(dim=1)  # (B, D, H, W)
```

## ⚙️ Configuration Options

### Model Arguments

- `--model`: Choose `umamba_bot` (faster) or `umamba_enc` (more powerful)
- `--input_channels`: Number of input modalities (default: 1)
- `--num_classes`: Number of segmentation classes
- `--n_stages`: Encoder/decoder depth (default: 6)
- `--patch_size`: 3D patch size, e.g., `128 128 128`
- `--deep_supervision`: Enable multi-scale supervision

### Training Arguments

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--use_amp`: Enable mixed precision training (recommended for faster training)

### Data Arguments

- `--train_img_dir`, `--train_lbl_dir`: Training data paths
- `--val_img_dir`, `--val_lbl_dir`: Validation data paths (optional)

## 📊 Comparison: Original vs Simplified

| Feature | Original U-Mamba | This Implementation |
|---------|------------------|---------------------|
| Framework | nnUNet (complex) | Standalone PyTorch (simple) |
| Dependencies | 20+ packages | 4 core packages |
| Code size | ~50 files | 3 files |
| Preprocessing | nnUNet pipeline | Built-in normalization |
| Training | nnUNet trainer | Custom trainer with AMP |
| Lines of code | ~10,000+ | ~1,000 |
| Flexibility | Limited | High |
| Learning curve | Steep | Gentle |

## 🎓 Architecture Details

### UMambaBot3d Architecture

```
Input (B, 1, D, H, W)
  ↓
Stem Block
  ↓
Encoder Stage 1 (Conv3d + ResBlocks)
  ↓ skip connection 1
Encoder Stage 2 (Downsample + ResBlocks)
  ↓ skip connection 2
...
  ↓
Encoder Stage N (Bottleneck)
  ↓
🔥 Mamba Layer (SSM sequence modeling)
  ↓
Decoder Stage N-1 (Upsample + Concat + ResBlocks)
  ↓
...
  ↓
Decoder Stage 1
  ↓
Output (B, num_classes, D, H, W)
```

### UMambaEnc3d Architecture

Same as UMambaBot3d, but **Mamba is applied after every encoder stage**, not just at the bottleneck.

## 💡 Tips & Best Practices

1. **Memory Management**:
   - Start with `batch_size=1` and `patch_size=64 64 64`
   - Gradually increase if you have more GPU memory
   - Use `--use_amp` to reduce memory usage by ~40%

2. **Model Selection**:
   - Use **UMambaBot** for most tasks (faster, less memory)
   - Use **UMambaEnc** only if you need maximum performance

3. **Hyperparameters**:
   - Learning rate: `1e-4` works well for most datasets
   - Batch size: `2` is typical for 3D medical imaging
   - Epochs: `100-200` depending on dataset size

4. **Data Augmentation**:
   - Currently includes random flips only
   - Add more augmentations in `utils/dataset_3d.py` if needed

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce patch size
--patch_size 64 64 64

# Reduce batch size
--batch_size 1

# Enable AMP
--use_amp
```

### NaN Loss

```bash
# Disable AMP (Mamba can be sensitive to FP16)
# Remove --use_amp flag

# Or use the NoAMP trainer (if you implement it)
```

### No Data Found

Check that:
- Image files have `_0000` suffix (or customize in dataset)
- Label files don't have the suffix
- Paths are correct

## 📚 References

- **U-Mamba Paper**: [arXiv:2401.04722](https://arxiv.org/abs/2401.04722)
- **Original Code**: [github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)
- **Mamba**: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

## 📝 License

This simplified implementation follows the same license as the original U-Mamba (Apache 2.0).

---

**Note**: This is a simplified, educational implementation. For production use on medical datasets, consider the original nnUNet-based U-Mamba with its full preprocessing pipeline.

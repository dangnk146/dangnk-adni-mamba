"""
Training script for U-Mamba 3D medical image segmentation
Simplified but complete implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import sys
from tqdm import tqdm
import argparse
import nibabel as nib
import numpy as np
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.umamba_3d import create_umamba_bot_3d, create_umamba_enc_3d
from utils.dataset_3d import get_dataloaders_3d


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, D, H, W), target: (B, D, H, W)
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # If we have more than 1 class, we usually want to ignore the background (class 0)
        # for the dice score to prevent it from dominating the loss.
        # For 'removing bone/skull', we really care about the brain mask accuracy.
        
        if pred.shape[1] > 1:
            # Return mean of foreground classes only (indices 1:)
            return 1 - dice[:, 1:].mean()
        else:
            return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        dist.barrier()
        print(f"[INFO] Initialized process group on rank {rank}")
        return local_rank, rank, world_size
    else:
        print("[INFO] Not using distributed mode")
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=True, rank=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                # Handle deep supervision
                if isinstance(outputs, list):
                    loss = sum([criterion(o, masks) for o in outputs])
                else:
                    loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, list):
                loss = sum([criterion(o, masks) for o in outputs])
            else:
                loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
        
        # Reduce loss for logging
        loss_val = loss.item()
        if dist.is_initialized():
            loss_tensor = torch.tensor([loss_val], device=device)
            dist.all_reduce(loss_tensor)
            loss_val = loss_tensor.item() / dist.get_world_size()

        total_loss += loss_val
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, rank=0):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
            
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            # Handle deep supervision - use only the highest resolution output
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            loss = criterion(outputs, masks)
            
            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            batch_correct = (pred == masks).sum().item()
            batch_total = masks.numel()

            # Reduce metrics
            loss_val = loss.item()
            if dist.is_initialized():
                # Loss
                loss_tensor = torch.tensor([loss_val], device=device)
                dist.all_reduce(loss_tensor)
                loss_val = loss_tensor.item() / dist.get_world_size()
                
                # Accuracy components
                acc_tensor = torch.tensor([batch_correct, batch_total], device=device)
                dist.all_reduce(acc_tensor)
                batch_correct = acc_tensor[0].item()
                batch_total = acc_tensor[1].item()

            total_loss += loss_val
            correct += batch_correct
            total += batch_total
            
            if rank == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{100.*correct/total:.2f}%"})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(args):
    """Main training function"""
    
    # Distributed setup
    local_rank, rank, world_size = setup_distributed()
    args.distributed = world_size > 1
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    if rank == 0:
        print(f"[INFO] Using device: {device} (World Size: {world_size})")
    
    # Create data loaders
    # Auto-detect number of classes (only on rank 0, then broadcast or scan on all)
    
    label_mapping = None
    if args.auto_detect_classes or args.num_classes == 0:
        if rank == 0:
            print(f"[INFO] Auto-detecting number of classes from {args.train_lbl_dir}...")
        
        label_files = []
        for ext in ['*.nii', '*.nii.gz']:
            label_files.extend(glob.glob(os.path.join(args.train_lbl_dir, ext)))
        
        if not label_files:
            if rank == 0: print("[WARNING] No label files found for detection! Using default num_classes.")
        else:
            scan_files = label_files[:20] if len(label_files) > 20 else label_files
            all_labels = set()
            
            if rank == 0: print(f"[INFO] Scanning {len(scan_files)} label files to determine classes...")
            # Simple sync: run on all ranks to avoid broadcasting logic overhead
            for fpath in (tqdm(scan_files, desc="Scanning labels") if rank==0 else scan_files):
                try:
                    lbl = nib.load(fpath).get_fdata()
                    unique = np.unique(lbl).astype(int)
                    all_labels.update(unique)
                except Exception as e:
                    if rank == 0: print(f"[WARNING] could not read {fpath}: {e}")
            
            sorted_labels = sorted(list(all_labels))
            if rank == 0: print(f"[INFO] Found labels: {sorted_labels}")
            
            label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}
            if rank == 0: print(f"[INFO] Created label mapping: {label_mapping}")
            
            detected_classes = len(sorted_labels)
            if rank == 0: print(f"[INFO] Detected {detected_classes} unique classes. Setting num_classes = {detected_classes}")
            args.num_classes = detected_classes
            
    # Create data loaders
    if rank == 0: print(f"[INFO] Loading dataset from {args.train_img_dir}")
    
    train_loader, val_loader = get_dataloaders_3d(
        train_img_dir=args.train_img_dir,
        train_lbl_dir=args.train_lbl_dir,
        val_img_dir=args.val_img_dir if args.val_img_dir else None,
        val_lbl_dir=args.val_lbl_dir if args.val_lbl_dir else None,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        num_workers=args.num_workers,
        use_augmentation=True,
        label_mapping=label_mapping,
        distributed=args.distributed
    )
    if rank == 0: print(f"[INFO] Training with Data Augmentation and Random Crop enabled.")

    if len(train_loader) == 0:
        if rank == 0: print("[ERROR] No training data found! Please check your data paths.")
        return
    
    if rank == 0:
        print(f"[INFO] Train batches: {len(train_loader)}")
        if val_loader:
            print(f"[INFO] Val batches: {len(val_loader)}")
    
    # Create model
    if rank == 0: print(f"[INFO] Creating {args.model} model...")
    if args.model == "umamba_bot":
        model = create_umamba_bot_3d(
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            n_stages=args.n_stages,
            deep_supervision=args.deep_supervision
        )
    elif args.model == "umamba_enc":
        model = create_umamba_enc_3d(
            input_size=tuple(args.patch_size),
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            n_stages=args.n_stages,
            deep_supervision=args.deep_supervision
        )
    elif args.model == "mambavision":
        # MambaVision is now implemented as a configuration of UMambaEnc
        model = create_umamba_enc_3d(
            input_size=tuple(args.patch_size),
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            n_stages=args.n_stages,
            deep_supervision=args.deep_supervision,
            use_hybrid=True
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Wrap with DDP
    if args.distributed:
        # Convert BatchNorm to SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    
    # Count parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss and optimizer
    criterion = CombinedLoss(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP scaler
    scaler = GradScaler() if args.use_amp and device.type == "cuda" else None
    
    # Training loop
    best_val_loss = float('inf')
    
    if rank == 0: print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        if rank == 0: print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.use_amp, rank)
        if rank == 0: print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
            
            if rank == 0:
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save underlying model (module)
                    model_to_save = model.module if args.distributed else model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, args.save_path.replace('.pth', '_best.pth'))
                    print(f"[INFO] Best model saved with val_loss: {val_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            model_to_save = model.module if args.distributed else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, args.save_path.replace('.pth', f'_epoch{epoch+1}.pth'))
    
    # Save final model
    if rank == 0:
        model_to_save = model.module if args.distributed else model
        torch.save(model_to_save.state_dict(), args.save_path)
        print(f"\n[INFO] Training finished! Model saved to {args.save_path}")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Mamba 3D")
    
    # Data
    parser.add_argument("--train_img_dir", type=str, default="data/train/images", help="Training images directory")
    parser.add_argument("--train_lbl_dir", type=str, default="data/train/labels", help="Training labels directory")
    parser.add_argument("--val_img_dir", type=str, default=None, help="Validation images directory")
    parser.add_argument("--val_lbl_dir", type=str, default=None, help="Validation labels directory")
    
    # Model
    parser.add_argument("--model", type=str, default="umamba_bot", choices=["umamba_bot", "umamba_enc", "mambavision"],
                        help="Model type: umamba_bot (Mamba at bottleneck), umamba_enc (Mamba at all stages) or mambavision (Hybrid)")
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=0, help="Number of output classes (0 for auto-detect)")
    parser.add_argument("--auto_detect_classes", action="store_true", help="Auto-detect number of classes from data", default=True)
    parser.add_argument("--n_stages", type=int, default=6, help="Number of encoder/decoder stages")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128], help="Patch size (D H W)")
    parser.add_argument("--deep_supervision", action="store_true", help="Use deep supervision")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Saving
    parser.add_argument("--save_path", type=str, default="umamba_3d.pth", help="Path to save model")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    # DDP (Optional, usually passed via env but keeping for compatibility)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()
    
    # Print configuration (rank 0 only)
    if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
        print("="*60)
        print("U-Mamba 3D Training Configuration")
        print("="*60)
        for arg, value in vars(args).items():
            print(f"{arg:20s}: {value}")
        print("="*60)
    
    train(args)

"""
U-Mamba 3D - Simplified but Complete Implementation
Supports both UMambaBot (Mamba at bottleneck) and UMambaEnc (Mamba at all encoder stages)
"""
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm import Mamba
from torch.cuda.amp import autocast


class MambaLayer(nn.Module):
    """Mamba sequence modeling layer with automatic FP32 conversion"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        """Standard spatial tokens"""
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        n_tokens = x.shape[2:].numel()
        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

    def forward_channel_token(self, x):
        """Use channels as tokens (for small spatial dims)"""
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        x_flat = x.flatten(2)  # (B, C, D*H*W)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    def forward(self, x):
        # Ensure FP32 for Mamba (it doesn't support FP16/BF16 well)
        original_dtype = x.dtype
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        
        # Convert back to original dtype if needed
        if out.dtype != original_dtype:
            out = out.type(original_dtype)
        
        return out


class BasicResBlock3d(nn.Module):
    """Basic 3D Residual Block with Instance Normalization"""
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, 
                 padding=1, use_1x1conv=False):
        super().__init__()
        
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, 
                               stride=stride, padding=padding, bias=True)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True, eps=1e-5)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, 
                               stride=1, padding=padding, bias=True)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True, eps=1e-5)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        if use_1x1conv:
            self.skip = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.skip = None

    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        
        out = self.conv1(x)
        out = self.act1(self.norm1(out))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.act2(out)


class UpsampleLayer3d(nn.Module):
    """3D Upsampling layer with 1x1x1 conv"""
    def __init__(self, input_channels, output_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return self.conv(x)


class UNetResEncoder3d(nn.Module):
    """3D U-Net Residual Encoder"""
    def __init__(self, input_channels, n_stages, features_per_stage, strides, 
                 n_blocks_per_stage, kernel_sizes=3):
        super().__init__()
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
            
        self.output_channels = features_per_stage
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        
        # Calculate padding
        self.conv_pad_sizes = [[k // 2 for k in (ks if isinstance(ks, (list, tuple)) else [ks]*3)] 
                               for ks in kernel_sizes]
        
        # Stem
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock3d(input_channels, stem_channels, kernel_size=kernel_sizes[0],
                           padding=self.conv_pad_sizes[0][0], use_1x1conv=True)
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        
        for s in range(n_stages):
            stage_blocks = [
                BasicResBlock3d(in_ch, features_per_stage[s], kernel_size=kernel_sizes[s],
                               stride=strides[s] if isinstance(strides[s], int) else strides[s][0],
                               padding=self.conv_pad_sizes[s][0], use_1x1conv=True)
            ]
            
            # Additional blocks in stage
            for _ in range(max(0, n_blocks_per_stage[s] - 1)):
                stage_blocks.append(
                    BasicResBlock3d(features_per_stage[s], features_per_stage[s],
                                   kernel_size=kernel_sizes[s], padding=self.conv_pad_sizes[s][0])
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            in_ch = features_per_stage[s]

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class ResidualMambaEncoder3d(nn.Module):
    """3D Encoder with Mamba at every stage (UMambaEnc variant)"""
    def __init__(self, input_size, input_channels, n_stages, features_per_stage, 
                 strides, n_blocks_per_stage, kernel_sizes=3):
        super().__init__()
        
        # Same initialization as UNetResEncoder3d
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
            
        self.output_channels = features_per_stage
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        
        # Determine whether to use channel tokens based on spatial size
        feature_map_sizes = []
        feature_map_size = list(input_size)
        do_channel_token = []
        
        for s in range(n_stages):
            stride = strides[s] if isinstance(strides[s], int) else strides[s][0]
            feature_map_size = [d // stride for d in feature_map_size]
            feature_map_sizes.append(feature_map_size.copy())
            
            # Use channel tokens if spatial dims are small
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token.append(True)
            else:
                do_channel_token.append(False)
        
        print(f"[UMambaEnc] Feature map sizes: {feature_map_sizes}")
        print(f"[UMambaEnc] Channel token: {do_channel_token}")
        
        # Same encoder structure but with Mamba after each stage
        self.conv_pad_sizes = [[k // 2 for k in (ks if isinstance(ks, (list, tuple)) else [ks]*3)] 
                               for ks in kernel_sizes]
        
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock3d(input_channels, stem_channels, kernel_size=kernel_sizes[0],
                           padding=self.conv_pad_sizes[0][0], use_1x1conv=True)
        )
        
        self.stages = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        in_ch = stem_channels
        
        for s in range(n_stages):
            stage_blocks = [
                BasicResBlock3d(in_ch, features_per_stage[s], kernel_size=kernel_sizes[s],
                               stride=strides[s] if isinstance(strides[s], int) else strides[s][0],
                               padding=self.conv_pad_sizes[s][0], use_1x1conv=True)
            ]
            
            for _ in range(max(0, n_blocks_per_stage[s] - 1)):
                stage_blocks.append(
                    BasicResBlock3d(features_per_stage[s], features_per_stage[s],
                                   kernel_size=kernel_sizes[s], padding=self.conv_pad_sizes[s][0])
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            
            # Add Mamba layer
            mamba_dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s]
            self.mamba_layers.append(
                MambaLayer(dim=mamba_dim, channel_token=do_channel_token[s])
            )
            
            in_ch = features_per_stage[s]

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = stage(x)
            x = mamba(x)  # Apply Mamba after each stage
            skips.append(x)
        return skips


class UNetResDecoder3d(nn.Module):
    """3D U-Net Residual Decoder with skip connections"""
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder.output_channels)
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        self.stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            
            # Upsample
            self.upsample_layers.append(
                UpsampleLayer3d(input_features_below, input_features_skip, 
                               stride_for_upsampling)
            )
            
            # Decoder block
            stage_blocks = [
                BasicResBlock3d(2 * input_features_skip, input_features_skip,
                               kernel_size=encoder.kernel_sizes[-(s + 1)],
                               padding=encoder.conv_pad_sizes[-(s + 1)][0],
                               use_1x1conv=True)
            ]
            
            for _ in range(max(0, n_conv_per_stage[s-1] - 1)):
                stage_blocks.append(
                    BasicResBlock3d(input_features_skip, input_features_skip,
                                   kernel_size=encoder.kernel_sizes[-(s + 1)],
                                   padding=encoder.conv_pad_sizes[-(s + 1)][0])
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            self.seg_layers.append(
                nn.Conv3d(input_features_skip, num_classes, kernel_size=1, bias=True)
            )

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        
        for i, (upsample, stage, seg_layer) in enumerate(zip(
            self.upsample_layers, self.stages, self.seg_layers)):
            
            x = upsample(lres_input)
            x = torch.cat([x, skips[-(i+2)]], dim=1)
            x = stage(x)
            
            if self.deep_supervision:
                seg_outputs.append(seg_layer(x))
            elif i == len(self.stages) - 1:
                seg_outputs.append(seg_layer(x))
            
            lres_input = x
        
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class UMambaBot3d(nn.Module):
    """
    U-Mamba Bottleneck 3D: Mamba only at the bottleneck
    - Simpler and faster than UMambaEnc
    - Good for most segmentation tasks
    """
    def __init__(self, input_channels, num_classes, n_stages=6, 
                 features_per_stage=None, strides=None, 
                 n_blocks_per_stage=None, deep_supervision=True):
        super().__init__()
        
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
        if strides is None:
            strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)][:n_stages]
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2, 2, 2, 2, 2, 2][:n_stages]
            # Reduce blocks in later stages
            for s in range(math.ceil(n_stages / 2), n_stages):
                n_blocks_per_stage[s] = 1
        
        self.encoder = UNetResEncoder3d(
            input_channels, n_stages, features_per_stage, strides, n_blocks_per_stage
        )
        
        # Mamba at bottleneck only
        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])
        
        # Decoder
        n_conv_decoder = n_blocks_per_stage[:-1].copy()
        for s in range(math.ceil((n_stages - 1) / 2), n_stages - 1):
            n_conv_decoder[s] = 1
        
        self.decoder = UNetResDecoder3d(self.encoder, num_classes, n_conv_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])  # Apply Mamba at bottleneck
        return self.decoder(skips)


class UMambaEnc3d(nn.Module):
    """
    U-Mamba Encoder 3D: Mamba at all encoder stages
    - More powerful but computationally expensive
    - Better for complex segmentation tasks
    """
    def __init__(self, input_size, input_channels, num_classes, n_stages=6,
                 features_per_stage=None, strides=None,
                 n_blocks_per_stage=None, deep_supervision=True):
        super().__init__()
        
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
        if strides is None:
            strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)][:n_stages]
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2, 2, 2, 2, 2, 2][:n_stages]
            for s in range(math.ceil(n_stages / 2), n_stages):
                n_blocks_per_stage[s] = 1
        
        self.encoder = ResidualMambaEncoder3d(
            input_size, input_channels, n_stages, features_per_stage, 
            strides, n_blocks_per_stage
        )
        
        # Decoder
        n_conv_decoder = n_blocks_per_stage[:-1].copy()
        for s in range(math.ceil((n_stages - 1) / 2), n_stages - 1):
            n_conv_decoder[s] = 1
        
        self.decoder = UNetResDecoder3d(self.encoder, num_classes, n_conv_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


# Factory functions for easy model creation
def create_umamba_bot_3d(input_channels=1, num_classes=2, **kwargs):
    """Create U-Mamba Bot 3D model (Mamba at bottleneck only)"""
    return UMambaBot3d(input_channels, num_classes, **kwargs)


def create_umamba_enc_3d(input_size, input_channels=1, num_classes=2, **kwargs):
    """Create U-Mamba Enc 3D model (Mamba at all encoder stages)"""
    return UMambaEnc3d(input_size, input_channels, num_classes, **kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing UMambaBot3d...")
    model_bot = create_umamba_bot_3d(input_channels=1, num_classes=3, n_stages=4)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model_bot(x)
    print(f"Input: {x.shape}, Output: {out.shape if not isinstance(out, list) else [o.shape for o in out]}")
    
    print("\nTesting UMambaEnc3d...")
    model_enc = create_umamba_enc_3d(input_size=(64, 64, 64), input_channels=1, num_classes=3, n_stages=4)
    out = model_enc(x)
    print(f"Input: {x.shape}, Output: {out.shape if not isinstance(out, list) else [o.shape for o in out]}")

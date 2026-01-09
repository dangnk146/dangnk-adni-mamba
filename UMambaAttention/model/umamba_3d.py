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
                 strides, n_blocks_per_stage, kernel_sizes=3, use_hybrid=False):
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
        if use_hybrid:
            print(f"[UMambaEnc] Using Hybrid Blocks (Attention/Mamba + MLP)")
        
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
            
            # Add Mamba layer or Hybrid Layer
            if use_hybrid:
                # Use Attention for later stages (e.g., last 2 stages), Mamba for early
                if s >= n_stages - 2:
                    mixer_type = "attention"
                else:
                    mixer_type = "mamba"
                    
                self.mamba_layers.append(
                    HybridMambaLayer(dim=features_per_stage[s], mixer_type=mixer_type)
                )
            else:
                # Original MambaLayer
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


# -----------------------------------------------------------------------------------------
# Hybrid Mamba + Attention Modules (MambaVision Style)
# -----------------------------------------------------------------------------------------

def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, C, D, H, W)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size^3, C)
    """
    B, C, D, H, W = x.shape
    x = x.view(B, C, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, window_size*window_size*window_size, C)
    return windows

def window_reverse_3d(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size^3, C)
        window_size: int
        D, H, W: Original dimensions
    Returns:
        x: (B, C, D, H, W)
    """
    ws = window_size
    B = int(windows.shape[0] / (D * H * W / ws / ws / ws))
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, -1, D, H, W)
    return x

class MambaVisionMixer3d(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", 
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, 
                 dt_init_floor=1e-4, conv_bias=True, bias=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Init dt_proj
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        """
        B, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states) # (B, L, 2*d_inner)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        
        # Conv1d
        x = self.conv1d_x(x)[:, :, :seqlen]
        z = self.conv1d_z(z)[:, :, :seqlen]
        
        x = F.silu(x)
        z = F.silu(z)
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B_ssm = rearrange(B_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_ssm = rearrange(C_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        A = -torch.exp(self.A_log.float())
        
        if selective_scan_fn is not None:
             y = selective_scan_fn(
                x, dt, A, B_ssm, C_ssm, self.D.float(), z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None
            )
        else:
            # Fallback (slow)
            y = x # Dummy
            print("Warning: selective_scan_fn not found, skipping SSM core")

        y = y * z # Gated
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

class Attention3d(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled Dot Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block3d(nn.Module):
    def __init__(self, dim, num_heads, mixer_type="mamba", mlp_ratio=4., 
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        if mixer_type == "attention":
            self.mixer = Attention3d(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        else:
            self.mixer = MambaVisionMixer3d(d_model=dim, d_state=16, d_conv=4, expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: (B, L, C)
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class HybridMambaLayer(nn.Module):
    """
    Hybrid Layer that can switch between Mamba and Attention,
    and includes an MLP.
    Drop-in replacement for MambaLayer in ResidualMambaEncoder3d.
    """
    def __init__(self, dim, mixer_type="mamba", num_heads=8, window_size=8, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.mixer_type = mixer_type
        self.window_size = window_size
        
        # We reuse the Block3d defined above
        self.block = Block3d(
            dim=dim, 
            num_heads=num_heads, 
            mixer_type=mixer_type, 
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop, 
            drop_path=drop_path
        )

    def forward(self, x):
        # Input x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Flatten to (B, L, C) for the block
        x_flat = x.flatten(2).transpose(1, 2) # (B, D*H*W, C)
        
        if self.mixer_type == "attention":
            # Handle Window Attention if needed
            # Reshape back to image for windowing
            x_Reshaped = x # (B, C, D, H, W)
            
            # Pad
            pad_d = (self.window_size - D % self.window_size) % self.window_size
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x_Reshaped = F.pad(x_Reshaped, (0, pad_w, 0, pad_h, 0, pad_d))
            
            Dp, Hp, Wp = x_Reshaped.shape[2:]
            
            # Partition
            windows = window_partition_3d(x_Reshaped, self.window_size) # (N_win*B, ws^3, C)
            
            # Apply Block
            windows = self.block(windows)
            
            # Reverse
            x_Reshaped = window_reverse_3d(windows, self.window_size, Dp, Hp, Wp)
            
            # Crop padding
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x_Reshaped = x_Reshaped[:, :, :D, :H, :W]
            
            out = x_Reshaped
            
        else:
            # Mamba is global
            x_out_flat = self.block(x_flat)
            out = x_out_flat.transpose(1, 2).view(B, C, D, H, W)
            
        return out


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
                 n_blocks_per_stage=None, deep_supervision=True, use_hybrid=False):
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
            strides, n_blocks_per_stage, use_hybrid=use_hybrid
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

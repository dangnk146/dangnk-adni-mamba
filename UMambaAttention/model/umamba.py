import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        n_tokens = x.shape[2:].numel()
        
        # Reshape to (B, L, C) where L is sequence length
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        
        # Reshape back to original spatial dimensions
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, norm_op=nn.InstanceNorm2d, act_op=nn.LeakyReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, affine=True)
        self.act1 = act_op(inplace=True)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = norm_op(output_channels, affine=True)
        self.act2 = act_op(inplace=True)
        
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.act2(out)

class UpsampleLayer(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.conv(x)

class UMambaBot(nn.Module):
    """
    Simplified U-Mamba Bot implementation (2D)
    """
    def __init__(self, input_channels, num_classes, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Encoder
        curr_channels = input_channels
        for feat in features:
            self.encoder.append(BasicResBlock(curr_channels, feat, stride=2 if curr_channels != input_channels else 1))
            curr_channels = feat
            
        # Bottleneck Mamba
        self.mamba_bottleneck = MambaLayer(features[-1])
        
        # Decoder
        for feat in reversed(features[:-1]):
            self.upsamples.append(UpsampleLayer(curr_channels, feat))
            self.decoder.append(BasicResBlock(feat * 2, feat))
            curr_channels = feat
            
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        
        # Mamba at bottleneck
        x = self.mamba_bottleneck(x)
        
        skips = skips[:-1][::-1]
        for upsample, decoder, skip in zip(self.upsamples, self.decoder, skips):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            
        return self.final_conv(x)

# 3D version follows same logic but with Conv3d and InstanceNorm3d
class BasicResBlock3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, norm_op=nn.InstanceNorm3d, act_op=nn.LeakyReLU):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, affine=True)
        self.act1 = act_op(inplace=True)
        
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = norm_op(output_channels, affine=True)
        self.act2 = act_op(inplace=True)
        
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.act2(out)

class UMambaBot3d(nn.Module):
    def __init__(self, input_channels, num_classes, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Encoder
        curr_channels = input_channels
        for feat in features:
            self.encoder.append(BasicResBlock3d(curr_channels, feat, stride=2 if curr_channels != input_channels else 1))
            curr_channels = feat
            
        # Bottleneck Mamba
        self.mamba_bottleneck = MambaLayer(features[-1])
        
        # Decoder
        for feat in reversed(features[:-1]):
            self.upsamples.append(nn.ConvTranspose3d(curr_channels, feat, kernel_size=2, stride=2))
            self.decoder.append(BasicResBlock3d(feat * 2, feat))
            curr_channels = feat
            
        self.final_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        
        x = self.mamba_bottleneck(x)
        
        skips = skips[:-1][::-1]
        for upsample, decoder, skip in zip(self.upsamples, self.decoder, skips):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            
        return self.final_conv(x)

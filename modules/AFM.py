import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class SE(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        hidden_channels = in_c // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_c, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        pool = self.avg_pool(x)
        weight = self.se(pool)
        return x * weight   #Q_xl
    
class SPA(nn.Module):
    def __init__(self, in_c, bins=(1, 4, 7), reduction=4):
        super().__init__()
        self.bins = bins
        inter_channels = max(in_c // reduction, 16)

        self.branches = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(bin),
            nn.Conv2d(in_c=in_c, out_c=inter_channels, kernel_size=bin, bias=False),
            nn.ReLU(inplace=True)
        ) for bin in bins])

        self.fuse = nn.Sequential(
            ConvBNReLU(in_c=inter_channels * len(bins), out_c=in_c, kernel_size=1),
            nn.Conv2d(in_c, in_c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = []

        for branch in self.branches:
            feat = branch(x)
            
            feats.append(feat)
        
        feat = torch.cat(feats, dim=1)

        weight = self.fuse(feat)
        return x * weight   #Q_xs

class AFM(nn.Module):
    def __init__(self, rgb_channels, depth_channels=None, out_channels=None, reduction=16,
                 bins=(1, 4, 7), use_fuse_conv=True):
        super().__init__()
        if depth_channels is None:
            depth_channels = rgb_channels
        if out_channels is None:
            out_channels = rgb_channels
        
        self.out_channels = out_channels

        #如果RGB特征通道数、Depth特征通道数和输出特征通道数不相等，则进行通道变换
        if rgb_channels != out_channels:
            self.rgb_proj = ConvBNReLU(in_c=rgb_channels, out_c=out_channels, kernel_size=1)
        else:
            self.rgb_proj = nn.Identity()
        if depth_channels != out_channels:
            self.depth_proj = ConvBNReLU(in_c=depth_channels, out_c=out_channels, kernel_size=1)
        else:
            self.depth_proj = nn.Identity()
        
        #RGB分支
        self.rgb_attn = SE(in_c=out_channels, reduction=reduction)
        #Depth分支
        self.depth_attn = SPA(in_c=out_channels, bins=bins, reduction=reduction)

        if use_fuse_conv:  #融合特征再做一次卷积整合
            self.fuse_conv = ConvBNReLU(in_c=out_channels, out_c=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.fuse_conv = nn.Identity()
    
    def forward(self, rgb_feat, depth_feat):
        #如果空间尺寸不一致，直接报错
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            raise ValueError(
                f"AFM requires rgb_feat and depth_feat to have the same spatial size, "
                f"but got rgb_feat={rgb_feat.shape}, depth_feat={depth_feat.shape}. "
                f"Please align them in the encoder instead of using interpolate inside AFM."
            )
        #通道对齐
        rgb_feat = self.rgb_proj(rgb_feat)
        depth_feat = self.depth_proj(depth_feat)

        #depth过SPA
        depth_enhanced = self.depth_attn(depth_feat)
        #RGB过SE
        rgb_enhanced = self.rgb_attn(rgb_feat)
        fused = rgb_enhanced + depth_enhanced
        return fused if isinstance(self.fuse_conv, nn.Identity) else self.fuse_conv(fused)

if __name__ == "__main__":
    rgb_feat = torch.randn(2, 128, 60, 80)
    depth_feat = torch.randn(2, 64, 60, 80)

    afm = AFM(
        rgb_channels=128,
        depth_channels=64,
        out_channels=128,
        reduction=16,
        bins=(1, 4, 7)
    )

    out = afm(rgb_feat, depth_feat)

    print("rgb_feat:", rgb_feat.shape)
    print("depth_feat:", depth_feat.shape)
    print("out:", out.shape)

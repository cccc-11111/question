import torch
import torch.nn as nn
import torch.nn.functional as F

class CONV(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(CONV, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.CBR(x)
    
class CS(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1):
        super().__init__()
        self.CS = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.CS(x)

class GFE(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        #分支1
        self.b_1 = CONV(in_c, out_c, kernel_size=3, stride=1, padding=1)
        #分支2
        self.b_2 = CONV(in_c, out_c, kernel_size=5, stride=2, padding=2)
        #分支3
        self.b_3 = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.CS = CS(out_c*3, out_c, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]

        f1 = self.b_1(x)
        #print(f1.shape)
        f2 = self.b_2(x)
        print(f2.shape)
        f3 = self.b_3(x)
        print(f3.shape)

        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
        #print(f2.shape)
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False)
        #print(f3.shape)
        f = torch.cat([f1, f2, f3], dim=1)
        #print(f.shape)

        out = self.CS(f)

        return out

class CCS(nn.Module):
    def __init__(self, in_c=2, out_c=1, kernel_size=1, **kwargs):
        super().__init__()
        self.CCS = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, bias=False),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.CCS(x)

class FCCA(nn.Module):
    def __init__(self, channels, reduction=16, pool_type='avg', **kwargs):
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

        #FC + RELU + FC
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        #全局池化
        pool = self.pool(x)
        #全连接层
        fc = self.fc(pool)
        #Sigmoid
        sig = F.sigmoid(fc)
        #Softmax
        soft = F.softmax(fc, dim=1)
        #相加
        weight = sig + soft
        #通道加权
        out = x * weight
        return out
    
class CA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_weight = self.mlp(self.avg_pool(x))
        max_weight = self.mlp(self.max_pool(x))
        weight = self.sigmoid(avg_weight + max_weight)
        out = x * weight
        return out
    
class TAEM(nn.Module):
    def __init__(self, channels, reduction=16, use_residual=True):
        super().__init__()

        self.use_residual = use_residual

        #预处理 CBR
        self.rgb_cbr = CBR(channels, channels, kernel_size=3, stride=1, padding=1)
        self.depth_cbr = CBR(channels, channels, kernel_size=3, stride=1, padding=1)
        #CS
        self.rgb_cs = CS(channels, channels, kernel_size=1)
        self.depth_cs = CS(channels, channels, kernel_size=1)
        #GFE
        self.rgb_GFE = GFE(channels, channels)
        self.depth_GFE = GFE(channels, channels)

        #RGB分支
        self.rgb_ccs = CCS()
        self.fcca = FCCA(channels, reduction)

        #depth分支
        self.depth_conv = CONV(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ca = CA(channels, reduction)

    def forward(self, rgb, depth):
        '''
        Forward pass for the TAEM model.
        Args:
            rgb (torch.Tensor): Input RGB image.   [B, C, H, W]
            depth (torch.Tensor): Input depth image.   [B, C, H, W]
        Returns:
            tuple: A tuple containing the output RGB and depth images.
        '''
        assert rgb.shape == depth.shape, f"Input shapes must be the same, but got {rgb.shape} and {depth.shape}"
        #预处理
        rgb_cbr = self.rgb_cbr(rgb)
        rgb_cs = self.rgb_cs(rgb_cbr)

        depth_cbr = self.depth_cbr(depth)
        depth_cs = self.depth_cs(depth_cbr)

        #RGB分支
        depth_guide_rgb = rgb_cs * depth_cs
        rgb_spatail_chMax = torch.max(depth_guide_rgb, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        rgb_spatial_chAvg = torch.mean(depth_guide_rgb, dim=1, keepdim=True)    # [B, 1, H, W]
        rgb_spatial_weight = torch.cat([rgb_spatail_chMax, rgb_spatial_chAvg], dim=1)  # [B, 2, H, W]
        rgb_ccs = self.rgb_ccs(rgb_spatial_weight)    # [B, 1, H, W]
        rgb_out = self.fcca(rgb_ccs * rgb_cs)

        #depth分支
        depth_gfe = self.depth_GFE(depth_cs)
        rgb_cfe = self.rgb_GFE(rgb_cs)
        depth_weight = torch.sigmoid(depth_gfe) * depth_gfe
        rgb_weight = torch.sigmoid(rgb_cfe) * rgb_cfe

        rgb_guide_depth = depth_weight + rgb_weight + depth
        depth_out = self.ca(self.depth_conv(rgb_guide_depth))

        return rgb_out, depth_out


if __name__ == "__main__":
    x_rgb = torch.randn(2, 3, 640, 640)
    x_depth = torch.randn(2, 3, 640, 640)
    taem = TAEM(channels=3)
    y_rgb, y_depth = taem(x_rgb, x_depth)
    
    print("RGB out:", y_rgb.shape)
    print("Depth out:", y_depth.shape)

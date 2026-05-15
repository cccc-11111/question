import torch
import torch.nn as nn
import torch.nn.functional as F

class CONV(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(CONV, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

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
        print(f1.shape)
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
    
if __name__ == "__main__":
    model = GFE(3, 128)
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    print(out.shape)

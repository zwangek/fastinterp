import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                  padding = padding, dilation = dilation, bias= True),
        nn.LeakyReLU(0.1))

def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

class LateralBlock(nn.Module):
    def __init__(self, channels, n=2):
        super().__init__()
        convlist = []
        for _ in range(n):
            convlist.append(conv(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1))
        self.convlist = nn.ModuleList(convlist)
    
    def forward(self, x):
        for _conv in self.convlist:
            x = _conv(x) + x
        return x


class AggregationBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = conv(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv1_0 = conv(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.aggregate = conv_bn(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = conv(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = conv(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x0, x1):
        f0 = self.conv0_0(x0) + x0
        f1 = self.conv1_0(x1) + x1
        f = torch.cat((f0,f1), dim=1)
        f = self.aggregate(f)
        out0 = self.conv0_1(f) + f 
        out1 = self.conv1_1(f) + f
        return out0, out1


class FlowEstimator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # downsample track
        self.conv0_0 = conv(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = conv(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=2, padding=1)
        self.conv0_2 = conv(in_channels=channels*2, out_channels=channels*4, kernel_size=3, stride=2, padding=1)

        self.conv1_0 = conv(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = conv(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = conv(in_channels=channels*2, out_channels=channels*4, kernel_size=3, stride=2, padding=1)

        # lateral conv for down sample
        self.lateral0_0a = LateralBlock(channels, 2)
        self.lateral0_1a = LateralBlock(channels*2, 2)
        self.lateral0_2a = LateralBlock(channels*4, 2)

        self.lateral1_0a = LateralBlock(channels, 2)
        self.lateral1_1a = LateralBlock(channels*2, 2)
        self.lateral1_2a = LateralBlock(channels*4, 2)

        # lateral conv for up sample
        self.lateral0_0b = LateralBlock(channels, 2)
        self.lateral0_1b = LateralBlock(channels*2, 2)

        self.lateral1_0b = LateralBlock(channels, 2)
        self.lateral1_1b = LateralBlock(channels*2, 2)

        # aggregation blocks
        self.aggregate_0 = AggregationBlock(channels)
        self.aggregate_1 = AggregationBlock(channels*2)
        self.aggregate_2 = AggregationBlock(channels*4)

        # upsample
        self.deconv0_1 = deconv(in_channels=channels*2+3+2, out_channels=channels, kernel_size=4, stride=2, padding=1)
        self.deconv0_2 = deconv(in_channels=channels*4+3+2, out_channels=channels*2, kernel_size=4, stride=2, padding=1)

        self.deconv1_1 = deconv(in_channels=channels*2+3+2, out_channels=channels, kernel_size=4, stride=2, padding=1)
        self.deconv1_2 = deconv(in_channels=channels*4+3+2, out_channels=channels*2, kernel_size=4, stride=2, padding=1)

        # flow estimation
        self.flow0_2 = conv(in_channels=channels*4*2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.flow1_2 = conv(in_channels=channels*4*2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.flow0_1 = conv(in_channels=channels*2*2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.flow1_1 = conv(in_channels=channels*2*2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.flow0_0 = conv(in_channels=channels*1*2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.flow1_0 = conv(in_channels=channels*1*2, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, img0, img1, return_flow=False):
        img0_0, img0_1, img0_2 = img0
        img1_0, img1_1, img1_2 = img1

        # downsample
        feat0_0a = self.lateral0_0a(self.conv0_0(img0_0)) # C H
        feat0_1a = self.lateral0_1a(self.conv0_1(feat0_0a)) # 2C H/2
        feat0_2a = self.lateral0_2a(self.conv0_2(feat0_1a)) # 4C H/4

        feat1_0a = self.lateral1_0a(self.conv1_0(img1_0))
        feat1_1a = self.lateral1_1a(self.conv1_1(feat1_0a)) 
        feat1_2a = self.lateral1_2a(self.conv1_2(feat1_1a))

        # aggregation & estimate flow on 1/4 resoluion
        feat0_2, feat1_2 = self.aggregate_2(feat0_2a, feat1_2a) # 4C H/4
        flow0_2 = self.flow0_2(torch.cat((feat0_2, feat1_2), dim=1)) # 2 H/4
        flow1_2 = self.flow1_2(torch.cat((feat0_2, feat1_2), dim=1))
        frame0_2 = self.warp(img0_2, flow0_2) # 3 H/4
        frame1_2 = self.warp(img1_2, flow1_2)
        frame_2 = (frame0_2 + frame1_2) / 2

        # upsample to 1/2 resolution
        feat0_1b = self.lateral0_1b(self.deconv0_2(torch.cat((feat0_2, frame_2, flow0_2), dim=1)) + feat0_1a) # 2C H/2
        feat1_1b = self.lateral1_1b(self.deconv1_2(torch.cat((feat1_2, frame_2, flow1_2), dim=1)) + feat1_1a)
        feat0_1, feat1_1 = self.aggregate_1(feat0_1b, feat1_1b) # 2C H/2
        flow0_1 = self.flow0_1(torch.cat((feat0_1, feat1_1), dim=1)) # 2 H/2
        flow1_1 = self.flow1_1(torch.cat((feat0_1, feat1_1), dim=1))
        frame0_1 = self.warp(img0_1, flow0_1) # 3 H/2
        frame1_1 = self.warp(img1_1, flow1_1)
        frame_1 = (frame0_1 + frame1_1) / 2

        # upsample to 1/1 resolution
        feat0_0b = self.lateral0_0b(self.deconv0_1(torch.cat((feat0_1, frame_1, flow0_1), dim=1)) + feat0_0a) # C H
        feat1_0b = self.lateral1_0b(self.deconv1_1(torch.cat((feat1_1, frame_1, flow1_1), dim=1)) + feat1_0a)
        feat0_0, feat1_0 = self.aggregate_0(feat0_0b, feat1_0b)
        flow0_0 = self.flow0_0(torch.cat((feat0_0, feat1_0), dim=1))
        flow1_0 = self.flow1_0(torch.cat((feat0_0, feat1_0), dim=1))
        frame0_0 = self.warp(img0_0, flow0_0)
        frame1_0 = self.warp(img0_0, flow1_0)
        frame_0 = (frame0_0 + frame1_0) / 2

        if not return_flow:
            return frame_0, frame0_0, frame1_0, frame_1, frame0_1, frame1_1, frame_2, frame0_2, frame1_2 
        else:
            return frame_0, frame0_0, frame1_0, flow0_0, flow1_0
        
    
    def warp(self, x, flow):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)

        vgrid = torch.autograd.Variable(grid) + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

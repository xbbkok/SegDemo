'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-23 20:52:25
 # @ Description: 网络模型搭建
 '''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, c1, c2, s=1, downsample= None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.depths = [2, 2, 2, 2]
        self.channels = [64, 128, 256, 512]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, s=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, s=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, s=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, s=2)

    def _make_layer(self, block, planes, depth, s=1) -> nn.Sequential:
            downsample = None
            if s != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            layers = nn.Sequential(
                block(self.inplanes, planes, s, downsample),
                *[block(planes * block.expansion, planes) for _ in range(1, depth)]
            )
            self.inplanes = planes * block.expansion
            return layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))  
        x1 = self.layer1(x)   
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2)  
        x4 = self.layer4(x3)  
        return x1, x2, x3, x4

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class FPNHead(nn.Module):

    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])
        
        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out


class SegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet18()
        self.head = FPNHead(in_channels=[64, 128, 256, 512],num_classes=9)
    def forward(self, x):
        feature = self.backbone(x)
        out = self.head(feature)
        output = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return output
        

class SegModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet18()
        self.head = FPNHead(in_channels=[64, 128, 256, 512],num_classes=9)
    def forward(self, x):
        feature = self.backbone(x)
        out = self.head(feature)
        output = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        y = torch.argmax(output, dim=1)  
        y = torch.squeeze(y)
        return y

if __name__ == '__main__':
    model = SegModel()
    model = SegModel2()
    x = torch.randn(1,3,256,320)
    print(model(x).shape)

import torch
from torch import nn
from torch.nn import functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip_connection):
        x = self.up_conv(x)
        
        # # Need to pad things
        # diffY = skip_connection.size()[2] - x.size()[2]
        # diffX = skip_connection.size()[3] - x.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip_connection, x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class ColorblindEncoder(nn.Module):
    def __init__(self):
        super(ColorblindEncoder, self).__init__()
        
        # Input convolutions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Down blocks
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        
        # Only used if we're just trying to get a feature vector out of here
        self.pool = nn.AdaptiveAvgPool1d((1024))
    
    def forward(self, x, return_intermediates=False):
        x1 = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        
        # Flatten and reshape for feature vector
        flatten_x = x.flatten().reshape(x.shape[0], -1)
        emb_x = self.pool(flatten_x)
        
        return_vals = [x, emb_x] 
        if return_intermediates:
            return_vals.extend((x1, x2, x3, x4))
        
        return return_vals


class RGB2GreyUNet(nn.Module):
    def __init__(self):
        super(RGB2GreyUNet, self).__init__()
        
        self.encoder = ColorblindEncoder()
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        

    def forward(self, x):
        x, emb_x, x1, x2, x3, x4 = self.encoder(x, return_intermediates=True)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.sigmoid(self.out_conv(x))
        
        return x, emb_x

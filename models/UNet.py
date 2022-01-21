# import deep learning framework, pytorch
import torch
from torch import nn

# U-Net
class UNet(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = self.down_conv_layer(3, 64, 7)
    self.conv2 = self.down_conv_layer(64, 128, 5)
    self.conv3 = self.down_conv_layer(128, 256, 3)
    self.conv4 = self.down_conv_layer(256, 512, 3)

    self.upconv3 = self.up_conv_layer(512, 256, 3)
    self.iconv3 = self.conv_layer(512, 256, 3)

    self.upconv2 = self.up_conv_layer(256, 128, 3)
    self.iconv2 = self.conv_layer(256, 128, 3)

    self.upconv1 = self.up_conv_layer(128, 64, 3)
    self.iconv1 = self.conv_layer(128, 64, 3)

    self.upconv0 = self.up_conv_layer(64, 64, 3)
    self.out = nn.Conv2d(64, 3, 3, padding=1)



  def forward(self, x):
    # x: N x 3 x H x W
    conv1 = self.conv1(x) # N x 64 x H/2 x W/2
    conv2 = self.conv2(conv1) # N x 128 x H/4 x W/4
    conv3 = self.conv3(conv2) # N x 256 x H/8 x W/8
    conv4 = self.conv4(conv3) # N x 512 x H/16 x W/16

    upconv3 = self.upconv3(conv4) # N x 256 x H/8 x W/8
    iconv3 = self.iconv3(torch.cat((upconv3, conv3), 1)) # N x 256 x H/8 x W/8

    upconv2 = self.upconv2(iconv3) # N x 128 x H/4 x W/4
    iconv2 = self.iconv2(torch.cat((upconv2, conv2), 1)) # N x 128 x H/4 x W/4

    upconv1 = self.upconv1(iconv2) # N x 64 x H/2 x W/2
    iconv1 = self.iconv1(torch.cat((upconv1, conv1), 1)) # N x 64 x H/2 x W/2

    upconv0 = self.upconv0(iconv1) # N x 64 x H x W
    _out = self.out(upconv0) # N x 3 x H x W
    out = torch.sigmoid(_out)

    return out

  def down_conv_layer(self, in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
    
  def up_conv_layer(self, in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )

  def conv_layer(self, in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x, testing=False):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    
if __name__ == '__main__':
    device = 'cpu'
    model = BallTrackerNet().to(device)
    inp = torch.rand(1, 9, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))

# class Conv2DBlock(nn.Module):
#     """ Conv2D + BN + ReLU """
#     def __init__(self, in_dim, out_dim, **kwargs):
#         super(Conv2DBlock, self).__init__(**kwargs)
#         self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
#         self.bn = nn.BatchNorm2d(out_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
# class Double2DConv(nn.Module):
#     """ Conv2DBlock x 2 """
#     def __init__(self, in_dim, out_dim):
#         super(Double2DConv, self).__init__()
#         self.conv_1 = Conv2DBlock(in_dim, out_dim)
#         self.conv_2 = Conv2DBlock(out_dim, out_dim)
#
#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         return x
#
# class Triple2DConv(nn.Module):
#     """ Conv2DBlock x 3 """
#     def __init__(self, in_dim, out_dim):
#         super(Triple2DConv, self).__init__()
#         self.conv_1 = Conv2DBlock(in_dim, out_dim)
#         self.conv_2 = Conv2DBlock(out_dim, out_dim)
#         self.conv_3 = Conv2DBlock(out_dim, out_dim)
#
#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.conv_3(x)
#         return x
#
# class TrackNet(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(TrackNet, self).__init__()
#         self.down_block_1 = Double2DConv(in_dim, 64)
#         self.down_block_2 = Double2DConv(64, 128)
#         self.down_block_3 = Triple2DConv(128, 256)
#         self.bottleneck = Triple2DConv(256, 512)
#         self.up_block_1 = Triple2DConv(768, 256)
#         self.up_block_2 = Double2DConv(384, 128)
#         self.up_block_3 = Double2DConv(192, 64)
#         self.predictor = nn.Conv2d(64, out_dim, (1, 1))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
#         x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
#         x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
#         x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
#         x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
#         x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
#         x = self.bottleneck(x)                                          # (N,  512,   36,    64)
#         x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
#         x = self.up_block_1(x)                                          # (N,  256,   72,   128)
#         x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
#         x = self.up_block_2(x)                                          # (N,  128,  144,   256)
#         x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
#         x = self.up_block_3(x)                                          # (N,   64,  288,   512)
#         x = self.predictor(x)                                           # (N,    3,  288,   512)
#         x = self.sigmoid(x)                                             # (N,    3,  288,   512)
#         return x
#
#
# class Conv1DBlock(nn.Module):
#     """ Conv1D + LeakyReLU"""
#     def __init__(self, in_dim, out_dim, **kwargs):
#         super(Conv1DBlock, self).__init__(**kwargs)
#         self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
#         self.relu = nn.LeakyReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
# class Double1DConv(nn.Module):
#     """ Conv1DBlock x 2"""
#     def __init__(self, in_dim, out_dim):
#         super(Double1DConv, self).__init__()
#         self.conv_1 = Conv1DBlock(in_dim, out_dim)
#         self.conv_2 = Conv1DBlock(out_dim, out_dim)
#
#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         return x
#
# class InpaintNet(nn.Module):
#     def __init__(self):
#         super(InpaintNet, self).__init__()
#         self.down_1 = Conv1DBlock(3, 32)
#         self.down_2 = Conv1DBlock(32, 64)
#         self.down_3 = Conv1DBlock(64, 128)
#         self.buttleneck = Double1DConv(128, 256)
#         self.up_1 = Conv1DBlock(384, 128)
#         self.up_2 = Conv1DBlock(192, 64)
#         self.up_3 = Conv1DBlock(96, 32)
#         self.predictor = nn.Conv1d(32, 2, 3, padding='same')
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, m):
#         x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
#         x = x.permute(0, 2, 1)                                         # (N,   3,   L)
#         x1 = self.down_1(x)                                            # (N,  16,   L)
#         x2 = self.down_2(x1)                                           # (N,  32,   L)
#         x3 = self.down_3(x2)                                           # (N,  64,   L)
#         x = self.buttleneck(x3)                                        # (N,  256,  L)
#         x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
#         x = self.up_1(x)                                               # (N,  128,  L)
#         x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
#         x = self.up_2(x)                                               # (N,   64,  L)
#         x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
#         x = self.up_3(x)                                               # (N,   32,  L)
#         x = self.predictor(x)                                          # (N,   2,   L)
#         x = self.sigmoid(x)                                            # (N,   2,   L)
#         x = x.permute(0, 2, 1)                                         # (N,   L,   2)
#         return x
#

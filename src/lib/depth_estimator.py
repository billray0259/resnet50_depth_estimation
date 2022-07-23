from torch import nn
import torch

from torch.nn import functional as F

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, relu_leak=0.1):
        super().__init__()
        self.convTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # Apply convolution that does not change the size
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1,  padding=2)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu_leak = relu_leak # we didn't use this for our experiments
    
    def forward(self, x):
        x = self.convTranspose(x)
        # x = F.leaky_relu(x, self.relu_leak)
        x = self.conv(x)
        # x = self.batch_norm(x)
        return x


class DepthEstimator(nn.Module):

    def __init__(self, resnet, dropout_p=0.25, probing=False):
        super().__init__()
        self.probing = probing
        self.dropout_p = dropout_p

        # encoder
        self.input_bn = nn.BatchNorm2d(3)
        self.resnet = resnet

        # dropout layer we don't actually use in any of our experiments... other than our rough hyperparameter search
        if self.dropout_p > 0:
            self.dropout = nn.Dropout2d(p=dropout_p)

        # decoder
        self.block1 = UpsampleBlock(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2 = UpsampleBlock(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block3 = UpsampleBlock(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block4 = UpsampleBlock(32, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block5 = UpsampleBlock(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.output_bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        # x (batch_size, 3, 224, 224)
        
        x = self.input_bn(x)

        # if we're probing don't train the reset encoder
        if self.probing:
            with torch.no_grad():
                x = self.resnet(x) # (batch_size, 2048, 7, 7)
        else:
            x = self.resnet(x)
        
        if self.dropout_p > 0:
            x = self.dropout(x) # (batch_size, 2048, 7, 7)

        x = self.block1(x) # (batch_size, 512, 14, 14)
        x = self.block2(x) # (batch_size, 128, 28, 28)
        x = self.block3(x) # (batch_size, 32, 56, 56)
        x = self.block4(x) # (batch_size, 8, 112, 112)
        x = self.block5(x) # (batch_size, 1, 224, 224)

        # x = self.output_bn(x)

        return x
        
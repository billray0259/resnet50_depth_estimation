import sys
sys.path.append('simclr-pytorch')

import torch
from torch import nn
import models
from torchvision.models import resnet50


class MyResnet50(nn.Module):

    def __init__(self, pretrained=True, return_feature_map=True, device=None):
        super().__init__()
        self.return_feature_map = return_feature_map
        self.resnet = resnet50(pretrained=pretrained)
        if device is not None:
            self.resnet = self.resnet.to(device)
        
        if return_feature_map:
            self.resnet.avgpool = nn.Identity()
            self.resnet.fc = nn.Identity()
    
    def forward(self, x):
        x = self.resnet(x)
        if self.return_feature_map:
            return x.view(x.shape[0], -1, 7, 7)


def load_classifier_resnet50(return_feature_map=True, device="cuda"):
    resnet = MyResnet50(pretrained=True, return_feature_map=return_feature_map, device=device)
    return resnet


def load_contrastive_resnet50(checkpoint_path, return_feature_map=True, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint['hparams'].return_feature_map = return_feature_map
    checkpoint['hparams'].dist = "dp"
    encoder = models.REGISTERED_MODELS[checkpoint['hparams'].problem].load(checkpoint, device=device)
    resnet = encoder.model.module.convnet
    return resnet
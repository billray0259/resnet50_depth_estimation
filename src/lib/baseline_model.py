# import torch
# from torch import nn
# from torch.nn import functional as F


# class BaselineModel(nn.Module):

#     def __init__(self):
#         # 9 3x3 convolutional layers
#         super(BaselineModel, self).__init__()
#         layers = []
#         layers.append(nn.Conv2d(3, 32, 3, padding=1))
#         layers.append(nn.ReLU())
        
#         for i in range(8):
#             layers.append(nn.Conv2d(32, 32, 3, padding=1))
#             layers.append(nn.ReLU())
        
#         layers.append(nn.Conv2d(32, 1, 3, padding=1))

#         self.layers = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.layers(x)
    

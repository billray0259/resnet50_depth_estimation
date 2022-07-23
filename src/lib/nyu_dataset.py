import sys
sys.path.append('nyuv2-python-toolbox')

import os
import h5py
import numpy as np
import torch
from torchvision.transforms import Resize
import random

# from nyuv2 import synchronise_frames, read_pgm, read_ppm
# pgm = depth
# ppm = RGB

from torch.utils.data import Dataset


class NYUDataset(Dataset):
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path, transform=None):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path, mode='r')
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        self.transform = transform

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def __getitem__(self, idx):
        color_map = self.color_maps[idx].transpose(0, 2, 1)
        depth_map = self.depth_maps[idx].transpose(1, 0)

        if self.transform is not None:
            color_map, depth_map = self.transform(color_map, depth_map)

        return color_map, depth_map


# class NYUDatasetRaw(Dataset):

#     # Images shapes
#     # Color: (480, 640, 3)
#     # Depth: (480, 640)

#     def __init__(self, data_dir, transform=None):
#         self.transform = transform
#         frame_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn]
#         self.frames = synchronise_frames(frame_names)
    
#     def __len__(self):
#         return len(self.frames)
    
#     def __getitem__(self, idx):
#         depth_file_name, color_file_name = self.frames[idx]

#         with open(depth_file_name, 'rb') as f:
#             depth_image = read_pgm(f)
        
#         with open(color_file_name, 'rb') as f:
#             color_image = read_ppm(f)
        
#         if self.transform:
#             depth_image = self.transform(depth_image)
#             color_image = self.transform(color_image)
        
#         return color_image, depth_image


def transform(color_map, depth_map, seed=None):
    color_map = torch.from_numpy(color_map).float()
    depth_map = torch.from_numpy(depth_map).float().unsqueeze(0)

    h, w = color_map.shape[1]//2, color_map.shape[2]//2

    color_map = Resize((h, w))(color_map)
    depth_map = Resize((h, w))(depth_map)

    # crop a random 224x224 patch
    if seed is not None:
        np.random.seed(seed)
    
    h0 = np.random.randint(0, h - 224)
    w0 = np.random.randint(0, w - 224)
    h1 = h0 + 224
    w1 = w0 + 224

    color_map = color_map[:, h0:h1, w0:w1]
    depth_map = depth_map[:, h0:h1, w0:w1]

    color_map /= 255.0

    # the 2 and 1 are constants used in other places, don't change them without modifying functions in util.py
    # Don't change them anyways, these make the math work out well in the other places
    depth_map /= 2.0
    depth_map -= 1

     # flip horizontally half the time
    if np.random.rand() > 0.5:
        color_map = torch.flip(color_map, [2])
        depth_map = torch.flip(depth_map, [2])

    return color_map, depth_map


def convert_depth_to_m(depth_map):
    return (depth_map + 1) * 2.0

def convert_depth_squared_to_m(depth_map):
    return np.sqrt(depth_map + 1) * 2.0
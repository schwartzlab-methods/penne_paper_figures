'''
The PanNuke dataset is stored in two numpy files:
- masks/fold1/images.npy: size (N, 256, 256, 3)
- images/fold1/masks.npy: size (N, 256, 256, 6), where each channel is the mask for one class
'''

from torch.utils.data import Dataset
import os
import numpy as np

class PanNukeDataset(Dataset):
    def __init__(self, root_dir):
        self.images = np.load(os.path.join(root_dir, 'images/fold1/images.npy'))
        self.masks = np.load(os.path.join(root_dir, 'masks/fold1/masks.npy'))
        self.type = np.load(os.path.join(root_dir, 'images/fold1/types.npy'))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.type[idx]
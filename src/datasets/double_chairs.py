from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class DoubleChairs(Dataset):
    def __init__(self, path, dim):
        super().__init__()
        print("Chairs dataset initialized")
        self.theta_1 = "008"
        self.theta_2 = "011"
        self.path = path
        self.dim = dim
        self.chairs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(self.chairs)
        
        
    def __len__(self):
        # For testing last.
        return len(self.chairs)

    def __getitem__(self, idx):

        chair_folder = self.chairs[idx]
        chair_path = os.path.join(self.path, chair_folder)

        chair_1 = os.path.join(chair_path, f"{self.theta_1}.png")
        chair_2 = os.path.join(chair_path, f"{self.theta_2}.png")

        chair_1 = Image.open(chair_1).convert("RGB")
        chair_2 = Image.open(chair_2).convert("RGB")

        transform = Compose([ Resize((self.dim, self.dim)), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        chair_1, chair_2 = transform(chair_1), transform(chair_2)
        pixels_1, pixels_2 = chair_1.permute(1, 2, 0).view(-1, 3), chair_2.permute(1, 2, 0).view(-1, 3)
        coords_1, coords_2 = get_mgrid(self.dim, 2), get_mgrid(self.dim, 2)

        return coords_1, pixels_1, coords_2, pixels_2

    def get_chair_name_by_idx(self, idx):
        return self.chairs[idx]
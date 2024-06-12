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


class Chairs(Dataset):
    def __init__(self, path):
        super().__init__()
        print("Chairs dataset initialized")
        self.theta_1 = "007"
        self.theta_2 = "010"
        self.path = path
        self.chairs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):

        chair_folder = self.chairs[idx]
        chair_path = os.path.join(self.path, chair_folder)

        chair_1 = os.path.join(chair_path, f"{self.theta_1}.png")
        chair_2 = os.path.join(chair_path, f"{self.theta_2}.png")

        chair_1 = Image.open(chair_1).convert("RGB")
        chair_2 = Image.open(chair_2).convert("RGB")

        transform = Compose([ Resize((512, 512)), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        chair_1 = transform(chair_1)
        # print(f"Chair 1 shape {chair_1.shape}")
        pixels = chair_1.permute(1, 2, 0).view(-1, 3)
        coords = get_mgrid(512, 2)
        
        # print(f"Coordinates shapes {coords.shape}")
        # print(f"c1 shape {pixels.shape}")

        return coords, pixels


        

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch

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

        chair_1_tensor = torch.tensor(np.array(chair_1), dtype=torch.float32).reshape(-1, 3)
        chair_2_tensor = torch.tensor(np.array(chair_2), dtype=torch.float32).reshape(-1, 3)

        return chair_1_tensor, chair_2_tensor


        

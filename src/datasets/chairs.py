from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class Chairs(Dataset):
    def __init__(self, path):
        super().__init__()
        print("Chairs dataset initialized")
        self.theta_1 = "007"
        self.theta_2 = "010"
        self.path = path
        self.chairs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        
    def __len__(self):
        return len(self.chairs * 512 * 512)

    def __getitem__(self, idx):
        chair_idx = idx // (512 * 512)
        idx %= 512 * 512
        x = idx % 512
        y = idx // 512

        chair_folder = self.chairs[chair_idx]
        chair_path = os.path.join(self.path, chair_folder)

        chair_1 = os.path.join(chair_path, f"{self.theta_1}.png")
        chair_2 = os.path.join(chair_path, f"{self.theta_2}.png")

        chair_1 = np.array(Image.open(chair_1).convert("RGB"))
        chair_2 = np.array(Image.open(chair_2).convert("RGB"))

        return chair_1[x][y], chair_2[x][y], x, y


        

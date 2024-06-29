from torch.utils.data import DataLoader
from src.datasets.chairs import Chairs
from src.datasets.double_chairs import DoubleChairs
import pytorch_lightning as pl

class ChairsDatamodule(pl.LightningDataModule):
    def __init__(self, path, dim, batch_size = 1, double = False):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        

        if double:
            self.train_dataset = DoubleChairs(self.path, dim)
        else:
            self.train_dataset = Chairs(self.path, dim)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):

        self.train_dataloader =  DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=30,
            pin_memory=True
            )
        
        return self.train_dataloader
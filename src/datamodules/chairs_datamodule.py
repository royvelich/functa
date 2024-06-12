from torch.utils.data import DataLoader
from src.datasets.chairs import Chairs
import pytorch_lightning as pl

class ChairsDatamodule(pl.LightningDataModule):
    def __init__(self, path, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.path = path

        self.train_dataset = Chairs(self.path)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):

        self.train_dataloader =  DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=8,
            )
        
        return self.train_dataloader